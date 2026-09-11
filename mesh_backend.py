#!/usr/bin/env python3
"""Rigged-mesh backend: drive any skinned humanoid from the editor's skeleton.

Why this rather than MPFB2 itself
---------------------------------
MPFB2 is a Blender add-on. It imports bpy and cannot run outside Blender, and
its code is GPLv3, so it cannot be linked into this editor. What it produces,
though, is ordinary rigged geometry, and MakeHuman's core assets (base mesh,
targets, skeletons) are CC0 with reuse explicitly encouraged.

So the workflow is: build the character in Blender with MPFB2, export .glb with
the armature, and load it here. The editor keeps owning the pose; this module
skins the mesh to it. Nothing MakeHuman-specific is assumed, so Mixamo, Daz and
VRoid exports work the same way.

    python3 mesh_backend.py --inspect body.glb     # list bones, check mapping
    python3 mesh_backend.py --selftest             # no files needed

Export settings that matter (Blender glTF exporter):
    Include > Limit to Selected Objects: mesh + armature
    Data > Mesh: apply modifiers, no shape keys needed (bake the body first)
    Data > Skinning: on
    Transform > +Y Up: on (the default)
"""

from __future__ import annotations

import json
import math
import os
import struct

import numpy as np

# Bone roles the retarget drives, and the name fragments seen in the wild.
# MakeHuman/MPFB2's default rig, Mixamo and Daz all differ; matching is done on
# lowercased names with separators stripped, longest pattern first.
BONE_ALIASES = {
    "hips": ["pelvis", "hips", "hip", "root", "spine05"],
    "spine": ["spine01", "spine1", "spine", "abdomen"],
    "chest": ["spine03", "spine2", "chest", "spine02"],
    "neck": ["neck01", "neck", "neck1"],
    "head": ["head"],
    "l_collar": ["clavicle.l", "claviclel", "leftshoulder", "lcollar"],
    "r_collar": ["clavicle.r", "clavicler", "rightshoulder", "rcollar"],
    "l_shoulder": ["upperarm01.l", "upperarm.l", "leftarm", "lshldr", "upperarml"],
    "r_shoulder": ["upperarm01.r", "upperarm.r", "rightarm", "rshldr", "upperarmr"],
    "l_elbow": ["lowerarm01.l", "lowerarm.l", "leftforearm", "lforearm", "forearml"],
    "r_elbow": ["lowerarm01.r", "lowerarm.r", "rightforearm", "rforearm", "forearmr"],
    "l_wrist": ["wrist.l", "lefthand", "lhand", "handl"],
    "r_wrist": ["wrist.r", "righthand", "rhand", "handr"],
    "l_hip": ["upperleg01.l", "upperleg.l", "leftupleg", "lthigh", "thighl"],
    "r_hip": ["upperleg01.r", "upperleg.r", "rightupleg", "rthigh", "thighr"],
    "l_knee": ["lowerleg01.l", "lowerleg.l", "leftleg", "lshin", "shinl"],
    "r_knee": ["lowerleg01.r", "lowerleg.r", "rightleg", "rshin", "shinr"],
    "l_ankle": ["foot.l", "leftfoot", "lfoot", "footl"],
    "r_ankle": ["foot.r", "rightfoot", "rfoot", "footr"],
}

# role -> (child role, editor keypoint the child should land on)
AIM_CHAIN = [
    ("l_collar", "l_shoulder", "l_shoulder"),
    ("r_collar", "r_shoulder", "r_shoulder"),
    ("l_shoulder", "l_elbow", "l_elbow"),
    ("r_shoulder", "r_elbow", "r_elbow"),
    ("l_elbow", "l_wrist", "l_wrist"),
    ("r_elbow", "r_wrist", "r_wrist"),
    ("l_hip", "l_knee", "l_knee"),
    ("r_hip", "r_knee", "r_knee"),
    ("l_knee", "l_ankle", "l_ankle"),
    ("r_knee", "r_ankle", "r_ankle"),
]

# Which bone each aimed bone takes its roll from. Roll is carried down a limb
# rather than derived for each bone on its own, so the only direction that can
# degenerate is a bone folded exactly back onto the one above it. Roots of a
# chain are absent and fall back to the pelvis.
ROLL_PARENT = {"l_shoulder": "l_collar", "r_shoulder": "r_collar",
               "l_elbow": "l_shoulder", "r_elbow": "r_shoulder",
               "l_knee": "l_hip", "r_knee": "r_hip"}

COMPONENT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
COMPONENT_SIZE = {"b": 1, "B": 1, "h": 2, "H": 2, "I": 4, "f": 4}
NUM_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
NORMALISE = {5121: 255.0, 5123: 65535.0, 5120: 127.0, 5122: 32767.0}


# ---------------------------------------------------------------------------
# maths (kept self-contained so this module only needs numpy)
# ---------------------------------------------------------------------------

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(3)


def rotation_between(a, b):
    a, b = unit(np.asarray(a, float)), unit(np.asarray(b, float))
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return np.eye(3)
    v, c = np.cross(a, b), float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = unit(np.cross(a, [1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-9:
            axis = unit(np.cross(a, [0.0, 1.0, 0.0]))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + K * s + K @ K * (1.0 - c)


def matrix_from_axis_angle(v):
    """Rodrigues, used for rolling a bone about its own axis."""
    v = np.asarray(v, float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + K * math.sin(angle) + K @ K * (1.0 - math.cos(angle))


def frame_from(forward, up):
    up = unit(np.asarray(up, float))
    forward = np.asarray(forward, float)
    forward = unit(forward - up * float(np.dot(forward, up)))
    if np.linalg.norm(forward) < 1e-9:
        forward = unit(np.cross(up, [1.0, 0.0, 0.0]))
    return np.column_stack([np.cross(up, forward), up, forward])


def quat_to_matrix(q):
    x, y, z, w = q                                   # glTF stores xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------------------
# glTF / GLB reading
# ---------------------------------------------------------------------------

def _read_glb(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] == b"glTF":
        _, _, total = struct.unpack("<III", data[:12])
        offset, gltf, binary = 12, None, b""
        while offset < min(total, len(data)):
            length, kind = struct.unpack("<II", data[offset:offset + 8])
            chunk = data[offset + 8:offset + 8 + length]
            if kind == 0x4E4F534A:
                gltf = json.loads(chunk.decode("utf-8"))
            elif kind == 0x004E4942:
                binary = chunk
            offset += 8 + length + (-length % 4)
        if gltf is None:
            raise RuntimeError("GLB has no JSON chunk: %s" % path)
        return gltf, binary
    gltf = json.loads(data.decode("utf-8"))
    return gltf, b""


def _buffer_bytes(gltf, binary, index, base_dir):
    buffer = gltf["buffers"][index]
    uri = buffer.get("uri")
    if uri is None:
        return binary
    if uri.startswith("data:"):
        import base64
        return base64.b64decode(uri.split(",", 1)[1])
    with open(os.path.join(base_dir, uri), "rb") as fh:
        return fh.read()


def _accessor(gltf, binary, index, base_dir):
    """Read an accessor into an array, honouring byteStride and normalisation."""
    acc = gltf["accessors"][index]
    count = acc["count"]
    ncomp = NUM_COMPONENTS[acc["type"]]
    fmt = COMPONENT[acc["componentType"]]
    size = COMPONENT_SIZE[fmt]
    if "bufferView" not in acc:
        return np.zeros((count, ncomp), dtype=np.float64)
    view = gltf["bufferViews"][acc["bufferView"]]
    raw = _buffer_bytes(gltf, binary, view.get("buffer", 0), base_dir)
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride") or ncomp * size
    out = np.empty((count, ncomp), dtype=np.float64)
    for i in range(count):
        chunk = raw[start + i * stride:start + i * stride + ncomp * size]
        out[i] = struct.unpack("<" + fmt * ncomp, chunk)
    if acc.get("normalized") and acc["componentType"] in NORMALISE:
        out = out / NORMALISE[acc["componentType"]]
    return out


def _node_matrix(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], float).reshape(4, 4).T   # column major
    M = np.eye(4)
    R = quat_to_matrix(node["rotation"]) if "rotation" in node else np.eye(3)
    S = np.diag(node.get("scale", [1.0, 1.0, 1.0]))
    M[:3, :3] = R @ S
    M[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
    return M


def _shell_sizes(verts, faces):
    """Vertex counts of each connected piece of geometry.

    Vertices are welded by position first: glTF splits them at UV seams, so
    raw index connectivity would report one body as dozens of pieces.
    """
    if len(faces) == 0:
        return []
    keys = {}
    weld = np.empty(len(verts), np.int64)
    for i, v in enumerate(verts):
        key = (round(v[0], 5), round(v[1], 5), round(v[2], 5))
        weld[i] = keys.setdefault(key, len(keys))
    n = len(keys)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in faces:
        a, b, c = (find(weld[t]) for t in tri)
        for x in (b, c):
            if x != a:
                parent[x] = a
    counts = {}
    for i in range(n):
        counts[find(i)] = counts.get(find(i), 0) + 1
    return sorted(counts.values(), reverse=True)


def _keep_main_shell(mesh):
    """Drop every piece of geometry except the largest.

    MakeHuman's basemesh carries helper geometry - a skirt and tights volume,
    hair helper, joint cubes and a ground plane - as loose shells around the
    body. Exported unbaked they read as clothing in a depth map.
    """
    verts, faces = mesh["vertices"], mesh["faces"]
    keys, weld = {}, np.empty(len(verts), np.int64)
    for i, v in enumerate(verts):
        key = (round(v[0], 5), round(v[1], 5), round(v[2], 5))
        weld[i] = keys.setdefault(key, len(keys))
    parent = list(range(len(keys)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in faces:
        a, b, c = (find(weld[t]) for t in tri)
        for x in (b, c):
            if x != a:
                parent[x] = a
    counts = {}
    for i in range(len(verts)):
        root = find(weld[i])
        counts[root] = counts.get(root, 0) + 1
    # Keep the largest piece and anything comparable to it, so a body exported
    # with separate clothing survives; only the small loose shells go. Helper
    # geometry is far smaller than the 13380-vertex body, so it falls out here.
    biggest = max(counts.values())
    kept_roots = {r for r, c in counts.items() if c >= 0.4 * biggest}
    keep = np.array([find(weld[i]) in kept_roots for i in range(len(verts))])
    if keep.all():
        return mesh
    remap = np.full(len(verts), -1, np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    face_keep = keep[faces].all(axis=1)
    out = dict(mesh)
    out["vertices"] = verts[keep]
    out["faces"] = remap[faces[face_keep]]
    out["skin_joints"] = mesh["skin_joints"][keep]
    out["skin_weights"] = mesh["skin_weights"][keep]
    out["dropped_vertices"] = int((~keep).sum())
    return out


def load_rigged_mesh(path, drop_loose=True):
    """Read the first skinned mesh from a glTF/GLB file."""
    gltf, binary = _read_glb(path)
    base_dir = os.path.dirname(os.path.abspath(path))
    nodes = gltf.get("nodes", [])

    # global rest transform of every node
    parent_of = {}
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            parent_of[child] = i
    globals_ = [None] * len(nodes)

    def global_of(i):
        if globals_[i] is None:
            local = _node_matrix(nodes[i])
            p = parent_of.get(i)
            globals_[i] = local if p is None else global_of(p) @ local
        return globals_[i]

    skinned = [(i, n) for i, n in enumerate(nodes)
               if "mesh" in n and "skin" in n]
    if not skinned:
        raise RuntimeError("No skinned mesh in %s (export the armature too)"
                           % os.path.basename(path))
    node_index, node = skinned[0]
    skin = gltf["skins"][node["skin"]]
    mesh = gltf["meshes"][node["mesh"]]

    verts, faces, joint_idx, weights = [], [], [], []
    for prim in mesh["primitives"]:
        attrs = prim["attributes"]
        if "JOINTS_0" not in attrs:
            continue
        offset = len(verts)
        position = _accessor(gltf, binary, attrs["POSITION"], base_dir)
        verts.extend(position)
        joint_idx.extend(_accessor(gltf, binary, attrs["JOINTS_0"], base_dir))
        weights.extend(_accessor(gltf, binary, attrs["WEIGHTS_0"], base_dir))
        if "indices" in prim:
            idx = _accessor(gltf, binary, prim["indices"], base_dir).ravel()
        else:
            idx = np.arange(len(position))
        idx = idx.astype(np.int64) + offset
        faces.extend(idx[:len(idx) // 3 * 3].reshape(-1, 3))

    verts = np.asarray(verts, float)
    if "inverseBindMatrices" in skin:
        ibm = _accessor(gltf, binary, skin["inverseBindMatrices"], base_dir)
        ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1)             # column major
    else:
        ibm = np.array([np.linalg.inv(global_of(j)) for j in skin["joints"]])

    joints = skin["joints"]
    rest_global = np.array([global_of(j) for j in joints])
    # the mesh node's own transform sits between mesh space and the world
    mesh_world = global_of(node_index)
    verts = (mesh_world[:3, :3] @ verts.T).T + mesh_world[:3, 3]
    ibm = np.array([m @ np.linalg.inv(mesh_world) for m in ibm])

    shells = _shell_sizes(verts, np.asarray(faces, np.int64))

    local_index = {node_id: i for i, node_id in enumerate(joints)}
    parents = np.array([local_index.get(parent_of.get(node_id, -1), -1)
                        for node_id in joints])
    names = [nodes[j].get("name", "joint%d" % j) for j in joints]

    result = {
        "shells": shells,
        "vertices": verts,
        "faces": np.asarray(faces, np.int64),
        "joint_names": names,
        "parents": parents,
        "rest_global": rest_global,
        "rest_position": rest_global[:, :3, 3].copy(),
        "inverse_bind": ibm,
        "skin_joints": np.asarray(joint_idx, np.int64),
        "skin_weights": np.asarray(weights, float),
    }
    if drop_loose and len(shells) > 1:
        result = _keep_main_shell(result)
    return result


# ---------------------------------------------------------------------------
# bone mapping
# ---------------------------------------------------------------------------

def _canonical(name):
    return "".join(c for c in name.lower() if c.isalnum() or c == ".")


def resolve_bones(joint_names, overrides=None):
    """Map roles onto bone indices by name. Longest alias first, so
    'upperarm01.l' wins over a bare 'arm'."""
    canon = [_canonical(n) for n in joint_names]
    found = {}
    for role, aliases in BONE_ALIASES.items():
        if overrides and role in overrides:
            name = _canonical(overrides[role])
            if name in canon:
                found[role] = canon.index(name)
            continue
        best = None
        for alias in sorted(aliases, key=len, reverse=True):
            for i, name in enumerate(canon):
                if name == alias:
                    best = (0, len(alias), i)
                    break
                if alias in name and best is None:
                    best = (1, len(alias), i)
            if best and best[0] == 0:
                break
        if best:
            found[role] = best[2]
    return found


REQUIRED_ROLES = ["hips", "neck", "l_shoulder", "r_shoulder", "l_elbow",
                  "r_elbow", "l_hip", "r_hip", "l_knee", "r_knee"]


def validate_roles(mesh, roles):
    """Check the mapping actually describes this rig.

    Aiming a joint at a child only works if the child really hangs below it.
    Without this a mis-matched name silently produces a mangled limb rather
    than an error, which is far harder to diagnose.
    """
    parents = mesh["parents"]
    names = mesh["joint_names"]
    problems = []
    for role in REQUIRED_ROLES:
        if role not in roles:
            problems.append("no bone matched the role '%s'" % role)
    for role, child_role, _target in AIM_CHAIN:
        if role not in roles or child_role not in roles:
            continue
        j, c = roles[role], roles[child_role]
        walk, seen = c, False
        while walk >= 0:
            walk = parents[walk]
            if walk == j:
                seen = True
                break
        if not seen:
            problems.append("'%s' (%s) is not below '%s' (%s) in the rig"
                            % (child_role, names[c], role, names[j]))
    return problems


# ---------------------------------------------------------------------------
# posing
# ---------------------------------------------------------------------------

def pose_globals(rest_position, parents, roles, points, stretch=True,
                 rest_orient=None):
    """Global rotation and position per bone, from the editor's keypoints.

    Same closed-form aim used for SMPL-X, but expressed purely in global terms
    so it does not care what rest orientation the rig's bones were authored
    with - only where the joints are.
    """
    n = len(rest_position)
    # Local rotation per joint. Storing only globals and flagging which were
    # set by hand loses the composition: rotating a shoulder must carry the
    # twist bone below it, and with it the elbow. That is invisible on a rig
    # whose aimed joints are direct parent and child, and wrong on any rig with
    # intermediate bones - which MakeHuman's default rig has throughout.
    local = [np.eye(3) for _ in range(n)]
    Q = [np.eye(3) for _ in range(n)]
    q = rest_position.copy()
    order = sorted(range(n), key=lambda j: _depth(parents, j))

    def propagate():
        for j in order:
            p = parents[j]
            if p < 0:
                Q[j] = local[j]
                q[j] = Q[j] @ rest_position[j]
            else:
                Q[j] = Q[p] @ local[j]
                q[j] = q[p] + Q[p] @ (rest_position[j] - rest_position[p])

    def kp(name):
        return np.asarray(points[name], float)

    hip_mid = 0.5 * (kp("r_hip") + kp("l_hip"))
    sh_mid = 0.5 * (kp("r_shoulder") + kp("l_shoulder"))

    # root: a full frame from hips and spine, so the figure faces the right way
    rest_hip_axis = rest_position[roles["l_hip"]] - rest_position[roles["r_hip"]]
    rest_up = rest_position[roles["neck"]] - 0.5 * (
        rest_position[roles["l_hip"]] + rest_position[roles["r_hip"]])
    rest_frame = frame_from(np.cross(rest_hip_axis, rest_up), rest_up)
    target_frame = frame_from(np.cross(kp("l_hip") - kp("r_hip"), sh_mid - hip_mid),
                              sh_mid - hip_mid)
    rotation = target_frame @ rest_frame.T
    for j in range(n):
        if parents[j] < 0:
            local[j] = rotation
    def recentre():
        # slice assignment, not "q +=": rebinding the name inside aim() would
        # make it a local there and shadow this array
        q[:] += hip_mid - 0.5 * (q[roles["l_hip"]] + q[roles["r_hip"]])

    propagate()
    recentre()

    def aim(role, child_role, target, positional=True):
        if role not in roles or child_role not in roles:
            return
        j, c = roles[role], roles[child_role]
        current = q[c] - q[j]
        goal = (np.asarray(target, float) - q[j]) if positional \
            else np.asarray(target, float)
        delta = rotation_between(current, goal)
        p = parents[j]
        above = Q[p] if p >= 0 else np.eye(3)
        # express the wanted global change as a change of this joint's local
        # rotation, so everything below inherits it
        local[j] = above.T @ delta @ above @ local[j]
        propagate()
        recentre()

    # spine bend, spread over whatever spine bones the rig has
    spine_roles = [r for r in ("spine", "chest") if r in roles]
    if spine_roles and "neck" in roles:
        for role in spine_roles:
            aim(role, "neck", sh_mid)
    for role, child, target in AIM_CHAIN:
        aim(role, child, kp(target))

    # Twist. Aiming only fixes where a bone points, never how it is rolled
    # about its own axis, and the leftover roll compounds down a chain: by the
    # elbow it is enough to wring the mesh into the classic skinning collapse.
    # Nothing in 18 keypoints observes twist, so tie it to the body instead -
    # each bone keeps the roll it had in the rest pose, carried by the pelvis
    # rotation. Rolling a bone about its own axis leaves the joint it points at
    # exactly where it was, so this cannot disturb the pose; the child is
    # re-aimed afterwards to bring its own subtree back.
    if rest_orient is not None:
        carried = {}            # role -> rest space to posed space, roll and all

        def roll_align(role, child_role):
            if role not in roles or child_role not in roles:
                return
            j, c = roles[role], roles[child_role]
            axis = unit(q[c] - q[j])
            if np.linalg.norm(axis) < 1e-9:
                return
            rest_dir = unit(rest_position[c] - rest_position[j])
            frame = rest_orient[j]
            k = int(np.argmin([abs(float(np.dot(frame[:, i], rest_dir)))
                               for i in range(3)]))
            reference = frame[:, k]
            current = Q[j] @ reference
            # Swing the reference onto the bone's posed axis before comparing.
            # Carrying it by the pelvis alone leaves it perpendicular to the
            # *rest* direction, so once the bone has swung away from rest the
            # part of it that survives projection onto the plane across `axis`
            # shrinks towards nothing and the angle below becomes noise: 45 deg
            # of spurious twist on an arm reaching forward and down, a clean
            # 180 on one across the body, and the forearm of a bent elbow
            # wrung right over. The swing is the minimal rotation, so it adds
            # no twist of its own, and it keeps the reference the same angle
            # off the axis as it was off rest_dir - at least 54 deg for the
            # least-aligned column of an orthonormal frame, so never degenerate.
            base = carried.get(ROLL_PARENT.get(role), rotation)
            swing = rotation_between(base @ rest_dir, axis) @ base
            carried[role] = swing
            wanted = swing @ reference
            current = current - axis * float(np.dot(current, axis))
            wanted = wanted - axis * float(np.dot(wanted, axis))
            if np.linalg.norm(current) < 1e-6 or np.linalg.norm(wanted) < 1e-6:
                return
            current, wanted = unit(current), unit(wanted)
            angle = math.atan2(float(np.dot(np.cross(current, wanted), axis)),
                               float(np.clip(np.dot(current, wanted), -1, 1)))
            delta = matrix_from_axis_angle(axis * angle)
            p = parents[j]
            above = Q[p] if p >= 0 else np.eye(3)
            local[j] = above.T @ delta @ above @ local[j]
            propagate()
            recentre()

        # Parents before children, so a bone is rolled only after whatever it
        # hangs from has settled. A second pass changes nothing measurable.
        for role, child, target in AIM_CHAIN:
            roll_align(role, child)
            aim(role, child, kp(target))         # restore the subtree below

    # Head. Without this the skull keeps whatever orientation the rig was
    # authored with, however the nose and ears are posed. The neck is aimed by
    # direction rather than position, because a rig's head bone sits at the
    # base of the skull, not at the ear midpoint.
    if "neck" in roles and "head" in roles and "r_ear" in points:
        ear_mid = 0.5 * (kp("r_ear") + kp("l_ear"))
        aim("neck", "head", ear_mid - sh_mid, positional=False)
        head = roles["head"]
        above = Q[parents[head]] if parents[head] >= 0 else np.eye(3)
        target_head = frame_from(kp("nose") - ear_mid, ear_mid - sh_mid)
        local[head] = above.T @ (target_head @ rest_frame.T)
        propagate()
        recentre()

    if stretch:
        # Aiming only transfers directions, so a rig whose thigh is 45 cm keeps
        # its own length against a 43 cm preset and the error accumulates down
        # the chain. The depth map has to line up with the pose PNG or the two
        # conditioning images disagree, so slide each mapped joint onto its
        # keypoint and carry its subtree along.
        children = {}
        for j in range(n):
            if parents[j] >= 0:
                children.setdefault(int(parents[j]), []).append(j)

        def subtree(j):
            out, stack = [], [j]
            while stack:
                k = stack.pop()
                out.append(k)
                stack.extend(children.get(k, ()))
            return out

        placed = {"l_shoulder": "l_shoulder", "r_shoulder": "r_shoulder",
                  "l_elbow": "l_elbow", "r_elbow": "r_elbow",
                  "l_wrist": "l_wrist", "r_wrist": "r_wrist",
                  "l_hip": "l_hip", "r_hip": "r_hip",
                  "l_knee": "l_knee", "r_knee": "r_knee",
                  "l_ankle": "l_ankle", "r_ankle": "r_ankle"}
        for role in sorted(placed, key=lambda r: _depth(parents, roles[r])
                           if r in roles else 0):
            if role not in roles:
                continue
            j = roles[role]
            delta = kp(placed[role]) - q[j]
            if float(np.linalg.norm(delta)) < 1e-9:
                continue
            for k in subtree(j):
                q[k] = q[k] + delta
    return np.array(Q), q


def _depth(parents, j):
    d = 0
    while parents[j] >= 0:
        j = parents[j]
        d += 1
    return d


def skin_mesh(mesh, Q, q):
    """Linear blend skinning. Each bone contributes rotate-about-its-rest-joint
    then translate, composed with the rig's own inverse bind matrix."""
    rest = mesh["rest_position"]
    n = len(rest)
    M = np.zeros((n, 4, 4))
    M[:, 3, 3] = 1.0
    M[:, :3, :3] = Q
    M[:, :3, 3] = q - np.einsum("jab,jb->ja", Q, rest)
    skinning = np.einsum("jab,jbc,jcd->jad", M, mesh["rest_global"],
                         mesh["inverse_bind"])

    verts = mesh["vertices"]
    weights = mesh["skin_weights"]
    total = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, np.where(total < 1e-9, 1.0, total))
    out = np.zeros_like(verts)
    for k in range(weights.shape[1]):
        w = weights[:, k:k + 1]
        if not np.any(w):
            continue
        mats = skinning[mesh["skin_joints"][:, k]]
        moved = np.einsum("vab,vb->va", mats[:, :3, :3], verts) + mats[:, :3, 3]
        out += w * moved
    return out


def solve_pose(mesh, points_cm, roles=None, stretch=True):
    """Solve the rig once and hand back the result keyed by bone name.

    Assets - hair, clothing, shoes - are separate skinned meshes built on the
    same armature, so they must be driven by the body's solution rather than
    solved again. Keying by name also lets an asset carry only the bones it
    actually uses, which hair rigged to the head alone does.
    """
    roles = roles or mesh.get("roles") or resolve_bones(mesh["joint_names"])
    problems = validate_roles(mesh, roles)
    if problems:
        raise RuntimeError(
            "This rig does not map cleanly:\n  " + "\n  ".join(problems)
            + "\n\nRun 'python3 mesh_backend.py --inspect <file>' to see the "
              "bone names.")
    rest = mesh["rest_position"]
    span_rig = float(np.linalg.norm(rest[roles["l_shoulder"]] - rest[roles["l_hip"]]))
    target = np.asarray(points_cm["l_shoulder"], float) - \
        np.asarray(points_cm["l_hip"], float)
    scale = float(np.linalg.norm(target)) / max(1e-9, span_rig)
    Q, q = pose_globals(rest * scale, mesh["parents"], roles, points_cm,
                        stretch=stretch,
                        rest_orient=mesh["rest_global"][:, :3, :3])
    return {"scale": scale,
            "bones": {name: (Q[i], q[i])
                      for i, name in enumerate(mesh["joint_names"])}}


def skin_with(mesh, solution):
    """Skin any mesh on the same armature using a solved pose."""
    scale = solution["scale"]
    rest = mesh["rest_position"] * scale
    n = len(rest)
    Q = np.tile(np.eye(3), (n, 1, 1))
    q = rest.copy()
    for i, name in enumerate(mesh["joint_names"]):
        found = solution["bones"].get(name)
        if found is not None:
            Q[i], q[i] = found
    scaled = dict(mesh)
    scaled["rest_position"] = rest
    scaled["vertices"] = mesh["vertices"] * scale
    scaled["rest_global"] = mesh["rest_global"].copy()
    scaled["rest_global"][:, :3, 3] *= scale
    scaled["inverse_bind"] = np.array([np.linalg.inv(g)
                                       for g in scaled["rest_global"]])
    return skin_mesh(scaled, Q, q)


def load_assets(folder):
    """Every .glb in a folder, as name -> path, for hair and clothing."""
    out = {}
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith((".glb", ".gltf")):
            out[os.path.splitext(name)[0]] = os.path.join(folder, name)
    return out


def pose_mesh(mesh, points_cm, roles=None, stretch=True):
    """Editor keypoints (centimetres) -> posed vertices in the same space."""
    roles = roles or mesh.get("roles") or resolve_bones(mesh["joint_names"])
    missing = [r for r in REQUIRED_ROLES if r not in roles]
    if missing:
        raise RuntimeError(
            "Could not identify these bones by name: %s\\nRun "
            "'python3 mesh_backend.py --inspect <file>' and supply a mapping."
            % ", ".join(missing))

    rest = mesh["rest_position"]
    span_rig = float(np.linalg.norm(rest[roles["l_shoulder"]]
                                    - rest[roles["l_hip"]]))
    target = np.asarray(points_cm["l_shoulder"], float) - \
        np.asarray(points_cm["l_hip"], float)
    scale = float(np.linalg.norm(target)) / max(1e-9, span_rig)

    scaled = dict(mesh)
    scaled["rest_position"] = rest * scale
    scaled["vertices"] = mesh["vertices"] * scale
    grow = np.eye(4)
    grow[:3, :3] *= scale
    scaled["rest_global"] = mesh["rest_global"].copy()
    scaled["rest_global"][:, :3, 3] *= scale
    scaled["inverse_bind"] = np.array([np.linalg.inv(g) for g in
                                       scaled["rest_global"]])

    Q, q = pose_globals(scaled["rest_position"], mesh["parents"], roles,
                        points_cm, stretch=stretch)
    return skin_mesh(scaled, Q, q), mesh["faces"], scale


# ---------------------------------------------------------------------------
# inspection and self-test
# ---------------------------------------------------------------------------

def inspect(path):
    mesh = load_rigged_mesh(path)
    roles = resolve_bones(mesh["joint_names"])
    print("%s\n  %d vertices, %d faces, %d bones"
          % (os.path.basename(path), len(mesh["vertices"]),
             len(mesh["faces"]), len(mesh["joint_names"])))
    size = mesh["vertices"].max(axis=0) - mesh["vertices"].min(axis=0)
    print("  bounding box: %.3f x %.3f x %.3f (units as exported)" % tuple(size))
    print("\n  matched bones:")
    for role in BONE_ALIASES:
        if role in roles:
            print("    %-12s -> %s" % (role, mesh["joint_names"][roles[role]]))
    problems = validate_roles(mesh, roles)
    if problems:
        print("\n  PROBLEMS:")
        for problem in problems:
            print("    " + problem)
    else:
        print("\n  mapping is consistent with the rig hierarchy")
    shells = mesh.get("shells")
    if shells and len(shells) > 1:
        print("\n  %d separate pieces of geometry: %s vertices" %
              (len(shells), ", ".join(str(n) for n in shells)))
        print("    MakeHuman helper geometry (skirt, tights, hair, joint cubes)"
              "\n    shows up as extra pieces. Delete helpers in MPFB2 before"
              "\n    exporting, or load with helpers dropped.")
    print("\n  all bones:")
    for i, name in enumerate(mesh["joint_names"]):
        print("    %3d %s" % (i, name))
    return 0 if not problems else 1


def _write_test_glb(path, lift=0.0):
    """A small rigged humanoid, so the loader and skinning can be tested with
    no external assets. Bone names follow MakeHuman's default rig.

    `lift` displaces the geometry, which lets a test build an asset that is
    distinguishable from the body instead of an exact overlap.
    """
    bones = [
        ("pelvis", -1, (0.0, 0.95, 0.0)),
        ("spine01", 0, (0.0, 1.05, 0.0)),
        ("spine03", 1, (0.0, 1.20, 0.0)),
        ("neck01", 2, (0.0, 1.40, 0.0)),
        ("head", 3, (0.0, 1.52, 0.0)),
        ("clavicle.L", 2, (0.04, 1.38, 0.0)),
        ("upperarm01.L", 5, (0.18, 1.38, 0.0)),
        ("upperarm02.L", 6, (0.31, 1.38, 0.0)),      # twist bone
        ("lowerarm01.L", 7, (0.45, 1.38, 0.0)),
        ("lowerarm02.L", 8, (0.58, 1.38, 0.0)),      # twist bone
        ("wrist.L", 9, (0.70, 1.38, 0.0)),
        ("clavicle.R", 2, (-0.04, 1.38, 0.0)),
        ("upperarm01.R", 11, (-0.18, 1.38, 0.0)),
        ("upperarm02.R", 12, (-0.31, 1.38, 0.0)),
        ("lowerarm01.R", 13, (-0.45, 1.38, 0.0)),
        ("lowerarm02.R", 14, (-0.58, 1.38, 0.0)),
        ("wrist.R", 15, (-0.70, 1.38, 0.0)),
        ("upperleg01.L", 0, (0.09, 0.92, 0.0)),
        ("upperleg02.L", 17, (0.10, 0.71, 0.0)),     # twist bone
        ("lowerleg01.L", 18, (0.10, 0.50, 0.0)),
        ("lowerleg02.L", 19, (0.10, 0.29, 0.0)),
        ("foot.L", 20, (0.10, 0.08, 0.0)),
        ("upperleg01.R", 0, (-0.09, 0.92, 0.0)),
        ("upperleg02.R", 22, (-0.10, 0.71, 0.0)),
        ("lowerleg01.R", 23, (-0.10, 0.50, 0.0)),
        ("lowerleg02.R", 24, (-0.10, 0.29, 0.0)),
        ("foot.R", 25, (-0.10, 0.08, 0.0)),
    ]
    verts, joints, weights, faces = [], [], [], []
    for b, (_name, parent, pos) in enumerate(bones):
        cx, cy, cz = pos
        base = len(verts)
        for dx, dy, dz in ((0.05, 0, 0.05), (-0.05, 0, 0.05), (0, 0.05, -0.05),
                           (0, -0.05, -0.05)):
            verts.append((cx + dx, cy + dy + lift, cz + dz))
            joints.append((b, 0, 0, 0))
            weights.append((1.0, 0.0, 0.0, 0.0))
        faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3),
                      (base, base + 3, base + 1), (base + 1, base + 3, base + 2)])

    verts = np.asarray(verts, np.float32)
    faces = np.asarray(faces, np.uint32)
    joints = np.asarray(joints, np.uint8)
    weights = np.asarray(weights, np.float32)
    ibm = np.zeros((len(bones), 4, 4), np.float32)
    for b, (_n, _p, pos) in enumerate(bones):
        M = np.eye(4, dtype=np.float32)
        M[:3, 3] = -np.asarray(pos, np.float32)
        ibm[b] = M.T                                    # column major on disk

    blobs = [verts.tobytes(), faces.tobytes(), joints.tobytes(),
             weights.tobytes(), ibm.tobytes()]
    offsets, cursor, payload = [], 0, b""
    for blob in blobs:
        pad = (-len(payload)) % 4
        payload += b"\0" * pad
        offsets.append(len(payload))
        payload += blob
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0, "scenes": [{"nodes": [0, 1]}],
        "buffers": [{"byteLength": len(payload)}],
        "bufferViews": [{"buffer": 0, "byteOffset": offsets[i],
                         "byteLength": len(blobs[i])} for i in range(5)],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(verts),
             "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": faces.size,
             "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5121, "count": len(joints),
             "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": len(weights),
             "type": "VEC4"},
            {"bufferView": 4, "componentType": 5126, "count": len(bones),
             "type": "MAT4"}],
        "meshes": [{"primitives": [{"attributes": {
            "POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3}, "indices": 1}]}],
        "skins": [{"joints": list(range(2, 2 + len(bones))),
                   "inverseBindMatrices": 4}],
        "nodes": [{"name": "body", "mesh": 0, "skin": 0}],
    }
    children = {}
    for b, (_n, parent, _p) in enumerate(bones):
        if parent >= 0:
            children.setdefault(parent, []).append(b + 2)
    root_bones = [b + 2 for b, (_n, p, _q) in enumerate(bones) if p < 0]
    gltf["nodes"].append({"name": "armature", "children": root_bones})
    for b, (name, parent, pos) in enumerate(bones):
        parent_pos = bones[parent][2] if parent >= 0 else (0.0, 0.0, 0.0)
        node = {"name": name,
                "translation": [pos[i] - parent_pos[i] for i in range(3)]}
        if b in children:
            node["children"] = children[b]
        gltf["nodes"].append(node)

    raw = json.dumps(gltf).encode("utf-8")
    raw += b" " * ((-len(raw)) % 4)
    payload += b"\0" * ((-len(payload)) % 4)
    total = 12 + 8 + len(raw) + 8 + len(payload)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(raw), 0x4E4F534A))
        fh.write(raw)
        fh.write(struct.pack("<II", len(payload), 0x004E4942))
        fh.write(payload)
    return path


def _selftest():
    import tempfile
    ok = True

    def check(label, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(("PASS " if cond else "FAIL ") + label
              + ("  " + extra if extra else ""))

    path = _write_test_glb(os.path.join(tempfile.gettempdir(), "rig_test.glb"))
    mesh = load_rigged_mesh(path)
    solid = load_rigged_mesh(path, drop_loose=False)
    check("loose shells detected", len(solid["shells"]) >= 20,
          "%d pieces: %s" % (len(solid["shells"]), solid["shells"][:4]))
    check("equal-sized pieces are all kept (nothing to call a helper)",
          len(mesh["vertices"]) == len(solid["vertices"]))
    # a body with small loose bits: the bits must go, the body must stay
    import copy
    fake = copy.deepcopy(solid)
    body = np.array([[math.cos(t), math.sin(t), 0.0] for t in
                     np.linspace(0, 2 * math.pi, 400, endpoint=False)])
    tri = np.array([[i, (i + 1) % 400, (i + 2) % 400] for i in range(398)])
    junk = np.array([[9.0, 9.0, 9.0], [9.1, 9.0, 9.0], [9.0, 9.1, 9.0]])
    fake["vertices"] = np.vstack([body, junk])
    fake["faces"] = np.vstack([tri, [[400, 401, 402]]])
    fake["skin_joints"] = np.zeros((403, 4), np.int64)
    fake["skin_weights"] = np.zeros((403, 4))
    fake["skin_weights"][:, 0] = 1.0
    cleaned = _keep_main_shell(fake)
    check("small loose shell dropped, body kept",
          len(cleaned["vertices"]) == 400 and cleaned.get("dropped_vertices") == 3,
          "%d vertices left" % len(cleaned["vertices"]))
    check("GLB parsed", len(mesh["vertices"]) == len(mesh["faces"]),
          "%d verts, %d faces" % (len(mesh["vertices"]), len(mesh["faces"])))
    check("bones read with hierarchy", len(mesh["joint_names"]) == 27
          and mesh["parents"][0] == -1, "%d bones" % len(mesh["joint_names"]))
    check("rest positions rebuilt from the node tree",
          abs(mesh["rest_position"][4][1] - 1.52) < 1e-6,
          "head at y=%.3f" % mesh["rest_position"][4][1])
    check("weights normalised",
          abs(mesh["skin_weights"].sum() - len(mesh["vertices"])) < 1e-6)

    roles = resolve_bones(mesh["joint_names"])
    check("MakeHuman rig names all matched",
          all(r in roles for r in REQUIRED_ROLES),
          "missing " + str([r for r in REQUIRED_ROLES if r not in roles]))
    bn = mesh["joint_names"]
    bp = mesh["parents"]
    elbow = bn.index("lowerarm01.L")
    check("rig has twist bones between the aimed joints, as MakeHuman does",
          bp[elbow] != bn.index("upperarm01.L"),
          "elbow's parent is %s" % bn[bp[elbow]])
    check("mapping validates against the hierarchy",
          validate_roles(mesh, roles) == [], str(validate_roles(mesh, roles)))
    check("left/right not confused",
          mesh["joint_names"][roles["l_shoulder"]].endswith(".L")
          and mesh["joint_names"][roles["r_shoulder"]].endswith(".R"))
    alt = resolve_bones(["Hips", "Spine", "Spine1", "Neck", "Head",
                         "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
                         "RightShoulder", "RightArm", "RightForeArm",
                         "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot",
                         "RightUpLeg", "RightLeg", "RightFoot"])
    check("Mixamo rig names also matched",
          all(r in alt for r in REQUIRED_ROLES),
          "missing " + str([r for r in REQUIRED_ROLES if r not in alt]))

    # pose it from a synthetic editor skeleton, in centimetres
    names = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder",
             "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip",
             "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]
    pts = {"neck": (0, 0, 0), "nose": (0, 16, 5),
           "r_eye": (-3, 20, 7), "l_eye": (3, 20, 7),
           "r_ear": (-7.5, 19, 1), "l_ear": (7.5, 19, 1),
           "r_shoulder": (-19, -2, 0), "l_shoulder": (19, -2, 0),
           "r_elbow": (-23, -30, 0), "l_elbow": (23, -30, 0),
           "r_wrist": (-25, -56, 2), "l_wrist": (25, -56, 2),
           "r_hip": (-10, -52, 0), "l_hip": (10, -52, 0),
           "r_knee": (-11, -95, 0), "l_knee": (11, -95, 0),
           "r_ankle": (-11, -138, -3), "l_ankle": (11, -138, -3)}
    verts, faces, scale = pose_mesh(mesh, pts)
    check("rig auto-scaled from metres to the editor's centimetres",
          80.0 < scale < 120.0, "scale %.1f" % scale)

    rest = mesh["rest_position"] * scale
    limbs = (("l_shoulder", "l_elbow"), ("r_shoulder", "r_elbow"),
             ("l_elbow", "l_wrist"), ("r_elbow", "r_wrist"),
             ("l_hip", "l_knee"), ("r_hip", "r_knee"),
             ("l_knee", "l_ankle"), ("r_knee", "r_ankle"))

    Q, q = pose_globals(rest, mesh["parents"], roles, pts, stretch=False)
    # without stretch each bone aims from wherever the rig's own joint sits, so
    # the exact property is "points at the keypoint", not "parallel to the
    # editor's bone" - those differ once the rig's proportions do
    worst = 0.0
    for a, b in limbs:
        got = unit(q[roles[b]] - q[roles[a]])
        want = unit(np.asarray(pts[b], float) - q[roles[a]])
        worst = max(worst, float(np.linalg.norm(got - want)))
    check("every bone aims at its keypoint", worst < 1e-9,
          "worst direction error %.2e" % worst)

    drift = max(float(np.linalg.norm(q[roles[r]] - np.asarray(pts[r], float)))
                for _a, r in limbs)
    check("without stretch, drift is only the rig's own bone lengths",
          1.0 < drift < 8.0, "%.2f cm" % drift)

    Q, q = pose_globals(rest, mesh["parents"], roles, pts, stretch=True)
    worst = max(float(np.linalg.norm(q[roles[r]] - np.asarray(pts[r], float)))
                for _a, r in limbs)
    check("with stretch, joints land exactly on the keypoints", worst < 1e-9,
          "worst %.2e cm" % worst)
    still = 0.0
    for a, b in limbs:
        got = unit(q[roles[b]] - q[roles[a]])
        want = unit(np.asarray(pts[b], float) - np.asarray(pts[a], float))
        still = max(still, float(np.linalg.norm(got - want)))
    check("stretching did not disturb the directions", still < 1e-9)
    # the twist bones must ride along with the joint above them
    Q2, q2 = pose_globals(rest, mesh["parents"], roles, pts, stretch=False)
    bn = mesh["joint_names"]
    twist, upper = bn.index("upperarm02.L"), bn.index("upperarm01.L")
    check("twist bone inherits the rotation of the joint above it",
          np.allclose(Q2[twist], Q2[upper], atol=1e-12),
          "max difference %.2e" % np.abs(Q2[twist] - Q2[upper]).max())
    on_line = np.linalg.norm(np.cross(q2[twist] - q2[upper],
                                      unit(q2[bn.index("lowerarm01.L")] - q2[upper])))
    check("and stays on the bone between shoulder and elbow", on_line < 1e-9,
          "%.2e cm off" % on_line)

    # a bone bound 1:1 to its vertices must carry them exactly
    j = roles["l_elbow"]
    verts, faces, scale = pose_mesh(mesh, pts)
    bound = np.nonzero(mesh["skin_joints"][:, 0] == j)[0]
    centre = verts[bound].mean(axis=0)
    check("skinned vertices follow their bone",
          float(np.linalg.norm(centre - q[j])) < 8.0,
          "%.2f cm from the joint" % float(np.linalg.norm(centre - q[j])))
    check("no vertex left at the origin", np.abs(verts).sum(axis=1).min() > 1e-6)

    # rigid check: distances inside one bone's vertex set are preserved
    d_rest = np.linalg.norm(mesh["vertices"][bound[0]] - mesh["vertices"][bound[1]])
    d_posed = np.linalg.norm(verts[bound[0]] - verts[bound[1]]) / scale
    check("skinning is rigid within a bone", abs(d_rest - d_posed) < 1e-6,
          "%.6f vs %.6f" % (d_rest, d_posed))

    try:
        from smplx_backend import rasterize_depth
        px = verts.copy()
        px[:, 0] = px[:, 0] * 2 + 200
        px[:, 1] = -px[:, 1] * 2 + 100
        z = rasterize_depth(px, faces, 400, 400)
        check("posed mesh rasterises", np.isfinite(z).any(),
              "%d pixels covered" % int(np.isfinite(z).sum()))
    except ImportError:
        print("SKIP rasteriser check (smplx_backend.py not alongside)")

    # an asset on the same rig must follow the body's own solution
    asset = load_rigged_mesh(path, drop_loose=False)
    solution = solve_pose(mesh, pts)
    moved = skin_with(asset, solution)
    body_verts, _f, _s = pose_mesh(mesh, pts)
    check("asset skinned by the body's solution lands on the body",
          float(np.abs(moved.mean(axis=0) - body_verts.mean(axis=0)).max()) < 30.0,
          "centres %.1f cm apart"
          % float(np.linalg.norm(moved.mean(axis=0) - body_verts.mean(axis=0))))
    partial = dict(asset)
    keep = [i for i, n in enumerate(asset["joint_names"]) if n in ("head", "neck01")]
    check("an asset rigged to a subset of bones still poses",
          len(skin_with(partial, solution)) == len(asset["vertices"]))


    # head must follow the face keypoints
    turned = dict(pts)
    turned["nose"] = (14.0, 16.0, -2.0)      # look to the figure's left
    turned["r_ear"] = (-2.0, 19.0, -7.0)
    turned["l_ear"] = (7.0, 19.0, 6.0)
    Q3, q3 = pose_globals(rest, mesh["parents"], roles, turned, stretch=False)
    head_i = mesh["joint_names"].index("head")
    facing = Q3[head_i] @ np.array([0.0, 0.0, 1.0])
    want = unit(np.asarray(turned["nose"], float)
                - 0.5 * (np.asarray(turned["r_ear"], float)
                         + np.asarray(turned["l_ear"], float)))
    # Twist.
    #
    # Aiming composes a *minimal* rotation per bone onto its parent's global,
    # which is parallel transport: swing a limb and its frame follows without
    # gaining any roll of its own. So there is no leftover roll for the reset
    # to cut, and the job of these checks is to prove the reset never invents
    # one. The version before this measured the roll against the rest
    # reference left where it was, found the large number that always falls
    # out of comparing a moved bone with an unmoved one, and "corrected" it -
    # which is what wrung the limbs in the depth map. It scored 176 degrees of
    # jump between neighbouring poses as success.
    orient = mesh["rest_global"][:, :3, :3]
    loose, _lq = pose_globals(rest, mesh["parents"], roles, pts, stretch=False)
    tight, tq = pose_globals(rest, mesh["parents"], roles, pts, stretch=False,
                             rest_orient=orient)
    moved = max(float(np.linalg.norm(tq[roles[r]] - _lq[roles[r]]))
                for r in ("l_wrist", "r_wrist", "l_ankle", "r_ankle"))
    check("rolling a bone moves no joint at all", moved < 1e-9, "%.2e cm" % moved)

    def arm_pose(upper, lower):
        out = dict(pts)
        sh = np.asarray(pts["l_shoulder"], float)
        el = sh + unit(np.asarray(upper, float)) * 28.0
        out["l_elbow"] = tuple(el)
        out["l_wrist"] = tuple(el + unit(np.asarray(lower, float)) * 26.0)
        return out

    def arm_frames(upper, lower, rest_orient):
        """Where each arm bone's cross-section points, across its own axis."""
        Q, q = pose_globals(rest, mesh["parents"], roles, arm_pose(upper, lower),
                            stretch=False, rest_orient=rest_orient)
        out = []
        for role, child in (("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist")):
            j, c = roles[role], roles[child]
            axis = unit(q[c] - q[j])
            k = int(np.argmin([abs(float(np.dot(
                orient[j][:, i], unit(rest[c] - rest[j])))) for i in range(3)]))
            ref = Q[j] @ orient[j][:, k]
            out.append(unit(ref - axis * float(np.dot(ref, axis))))
        return out

    def biggest_step(poses, rest_orient):
        worst, prev = 0.0, None
        for upper, lower in poses:
            frames = arm_frames(upper, lower, rest_orient)
            if prev is not None:
                for a, b in zip(prev, frames):
                    worst = max(worst, math.degrees(math.acos(
                        float(np.clip(np.dot(a, b), -1.0, 1.0)))))
            prev = frames
        return worst

    # A question no roll formula can satisfy by construction: a limb swung
    # smoothly must not jump. Both sweeps run right past the directions where
    # the old roll snapped - the arm along the body's own forward, and the
    # forearm of a bent elbow coming back across the upper arm.
    step = 4
    swept = [((math.sin(math.radians(d)), -math.cos(math.radians(d)), 0.0),) * 2
             for d in range(-170, 171, step)]
    folded = [((0.0, 0.0, 1.0),
               (math.cos(math.radians(d)), math.sin(math.radians(d)), 0.0))
              for d in range(0, 360, step)]
    for label, poses in (("a limb swung right round", swept),
                         ("the forearm of a bent elbow", folded)):
        for how, rest_orient in (("aimed", None), ("with the roll reset", orient)):
            jump = biggest_step(poses, rest_orient)
            check("%s does not jump (%s)" % (label, how), jump < 3.0 * step,
                  "worst step %.1f deg over %d deg moves" % (jump, step))

    # and the reset must agree with what aiming already worked out, not fight it
    drift = 0.0
    for upper, lower in swept[::5] + folded[::5]:
        a = arm_frames(upper, lower, None)
        b = arm_frames(upper, lower, orient)
        for u, v in zip(a, b):
            drift = max(drift, math.degrees(math.acos(
                float(np.clip(np.dot(u, v), -1.0, 1.0)))))
    check("the roll reset confirms the aimed roll rather than replacing it",
          drift < 1.0, "worst disagreement %.2f deg" % drift)

    check("head turns with the face keypoints",
          float(np.dot(unit(facing), want)) > 0.98,
          "cos %.4f" % float(np.dot(unit(facing), want)))
    Q4, _ = pose_globals(rest, mesh["parents"], roles, pts, stretch=False)
    check("and a level head stays level",
          float(np.dot(unit(Q4[head_i] @ np.array([0.0, 0.0, 1.0])),
                       [0.0, 0.0, 1.0])) > 0.98)

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    if "--inspect" in sys.argv:
        raise SystemExit(inspect(sys.argv[sys.argv.index("--inspect") + 1]))
    print(__doc__)
