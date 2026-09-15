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
    Data > Mesh: apply modifiers; shape keys may be left in, they are read
                 at their default weights (that is where MPFB2 keeps the body)
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

# The keypoint at the near end of each aimed bone, so a pose can be read as a
# direction between two keypoints rather than as an absolute place to put the
# far one. The collar has no keypoint of its own: OpenPose's neck is the
# midpoint of the shoulders, which is where a collar starts.
AIM_FROM = {"l_collar": "neck", "r_collar": "neck",
            "l_shoulder": "l_shoulder", "r_shoulder": "r_shoulder",
            "l_elbow": "l_elbow", "r_elbow": "r_elbow",
            "l_hip": "l_hip", "r_hip": "r_hip",
            "l_knee": "l_knee", "r_knee": "r_knee"}

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


NUMPY_TYPE = {"b": "<i1", "B": "<u1", "h": "<i2", "H": "<u2",
              "I": "<u4", "f": "<f4"}


def _typed(raw, start, count, ncomp, fmt, stride):
    """`count` elements of `ncomp` components, honouring a byte stride.

    Read a vertex at a time this used to cost one `struct.unpack` per vertex;
    a body with thirty morph targets is half a million of them before a single
    triangle is drawn.
    """
    dtype = np.dtype(NUMPY_TYPE[fmt])
    packed = ncomp * dtype.itemsize
    if stride == packed:
        flat = np.frombuffer(raw, dtype=dtype, count=count * ncomp,
                             offset=start)
        return flat.reshape(count, ncomp).astype(np.float64)
    rows = np.frombuffer(raw, dtype=np.uint8, count=count * stride,
                         offset=start).reshape(count, stride)[:, :packed]
    return rows.copy().view(dtype).reshape(count, ncomp).astype(np.float64)


def _sparse_part(gltf, binary, base_dir, part, count, ncomp, fmt):
    view = gltf["bufferViews"][part["bufferView"]]
    raw = _buffer_bytes(gltf, binary, view.get("buffer", 0), base_dir)
    start = view.get("byteOffset", 0) + part.get("byteOffset", 0)
    size = COMPONENT_SIZE[fmt]
    return _typed(raw, start, count, ncomp, fmt, ncomp * size)


def _accessor(gltf, binary, index, base_dir):
    """Read an accessor into an array, honouring byteStride, normalisation
    and sparse storage."""
    acc = gltf["accessors"][index]
    count = acc["count"]
    ncomp = NUM_COMPONENTS[acc["type"]]
    fmt = COMPONENT[acc["componentType"]]
    size = COMPONENT_SIZE[fmt]
    if "bufferView" in acc:
        view = gltf["bufferViews"][acc["bufferView"]]
        raw = _buffer_bytes(gltf, binary, view.get("buffer", 0), base_dir)
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride") or ncomp * size
        out = _typed(raw, start, count, ncomp, fmt, stride)
    else:
        out = np.zeros((count, ncomp), dtype=np.float64)
    # A sparse accessor stores only the elements that differ from that base.
    # It is how an exporter writes a morph target that moves a few hundred
    # vertices of a body, so ignoring it reads the target as no displacement
    # at all - which looks exactly like a mesh that has no targets.
    sparse = acc.get("sparse")
    if sparse:
        n = sparse["count"]
        idx_fmt = COMPONENT[sparse["indices"]["componentType"]]
        where = _sparse_part(gltf, binary, base_dir, sparse["indices"],
                             n, 1, idx_fmt).ravel().astype(np.int64)
        out[where] = _sparse_part(gltf, binary, base_dir, sparse["values"],
                                  n, ncomp, fmt)
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

    # A DCC stores a shape key as a glTF morph target with a default weight,
    # and a body built out of shape keys - which is what MakeHuman's macro
    # sliders are - carries its whole shape there. Read the base mesh alone
    # and every figure in a set loads as the identical unshaped base: a child
    # and a heavy adult come out the same 167 cm mannequin, and the only sign
    # of it is that the rig, which *is* fitted to the shape, no longer matches
    # the mesh it drives. Node weights override the mesh's own, per the spec.
    morph = node.get("weights")
    if morph is None:
        morph = mesh.get("weights") or []

    verts, faces, joint_idx, weights = [], [], [], []
    applied = 0
    for prim in mesh["primitives"]:
        attrs = prim["attributes"]
        if "JOINTS_0" not in attrs:
            continue
        offset = len(verts)
        position = _accessor(gltf, binary, attrs["POSITION"], base_dir)
        for weight, target in zip(morph, prim.get("targets", [])):
            if abs(weight) < 1e-6 or "POSITION" not in target:
                continue
            position = position + weight * _accessor(
                gltf, binary, target["POSITION"], base_dir)
            applied += 1
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
        "morphs_applied": applied,
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

def _limb_direction(kp, role, target):
    """Which way a bone points, from the two keypoints at its ends."""
    import numpy as np
    start = AIM_FROM.get(role, target)
    return np.asarray(kp(target), float) - np.asarray(kp(start), float)


def pose_globals(rest_position, parents, roles, points, rest_orient=None,
                 rest_points=None):
    """Global rotation and position per bone: the rig, posed by the keypoints.

    Same closed-form aim used for SMPL-X, expressed purely in global terms so
    it does not care what rest orientation the bones were authored with.

    Every bone is ROTATED; none is moved or resized. A bone is told which way
    to point, taken from the line between the two keypoints at its ends, and
    keeps the length it was authored with. Eighteen keypoints are a good
    witness to direction and a poor one to size, and this asks them only for
    what they know.

    What this replaced aimed each bone at the absolute keypoint, then slid the
    joint onto it and scaled the segment until it landed, so the rig came out
    wearing the keypoint skeleton's proportions - measured on a walking figure,
    +7% on a humerus and -10% on a shin, thigh and shin pulling opposite ways
    on the same leg. Reconciling the two conventions was most of the length of
    this file: which landmark a shoulder keypoint is, how tall the figure is,
    how the arm splits. None of it is needed in order to rotate a bone.
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
        # rotation, so everything below inherits it. Once segments have been
        # scaled, Q carries a uniform scale and its transpose is no longer its
        # inverse - so take the rotation out first, or every re-aim multiplies
        # the limb's scale by the square of it.
        above = _rotation_of(above)
        local[j] = above.T @ delta @ above @ local[j]
        propagate()
        recentre()

    # spine bend, spread over whatever spine bones the rig has
    spine_roles = [r for r in ("spine", "chest") if r in roles]
    if spine_roles and "neck" in roles:
        for role in spine_roles:
            aim(role, "neck", sh_mid - hip_mid, positional=False)
    for role, child, target in AIM_CHAIN:
        aim(role, child, _limb_direction(kp, role, target), positional=False)

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

    # Aiming has the last word. A roll about a bone's own axis must not move
    # the joint it points at, and it does not - but it does move whatever
    # hangs below that joint, and until now a scaling pass at the end quietly
    # put those back. With the scaling gone the re-aim has to be explicit, or
    # the reset ends up overruling what aiming already worked out, which is
    # the exact failure that used to wring the limbs in the depth map.
    if rest_orient is not None:
        for role, child, target in AIM_CHAIN:
            aim(role, child, _limb_direction(kp, role, target), positional=False)

    # Feet.
    #
    # There is no keypoint past the ankle, so a foot rides its shin rigidly
    # and points wherever the shin does - which is right for a foot in the
    # air and wrong for one standing on something. A lunge came out on
    # pointed toes and a seated figure dangled its feet, because the shin is
    # tilted and the foot went with it. So where the shin is near enough to
    # upright that the foot must be taking weight, put the foot back to the
    # pitch the rig authored it at - flat - while keeping the heading the leg
    # gave it. Beyond 40 degrees the leg is not standing on anything: a
    # kneeling figure's shins point backwards and a lying one's sideways, and
    # both keep the rig's own relationship.
    for role in ("l_ankle", "r_ankle"):
        knee_role = role.replace("ankle", "knee")
        j = roles.get(role)
        if j is None or knee_role not in roles:
            continue
        # measured on the keypoints, not on `q`: the rig has not been slid
        # onto them yet, so its own bone lengths put the knee somewhere else
        # and its shin reads 11 degrees steeper than the pose asked for
        shin = unit(kp(role) - kp(knee_role))
        if np.linalg.norm(shin) < 1e-9 or -shin[1] < math.cos(math.radians(40.0)):
            continue
        toes = [i for i in range(n) if parents[i] == j]
        if not toes:
            continue
        rest_dir = unit(np.mean([rest_position[t] for t in toes], axis=0)
                        - rest_position[j])
        posed = unit(Q[j] @ rest_dir)
        heading = np.array([posed[0], 0.0, posed[2]])
        if np.linalg.norm(rest_dir) < 1e-9 or np.linalg.norm(heading) < 1e-6:
            continue
        rise = float(np.clip(rest_dir[1], -1.0, 1.0))
        want = unit(unit(heading) * math.sqrt(max(0.0, 1.0 - rise * rise))
                    + np.array([0.0, rise, 0.0]))
        p = parents[j]
        above = Q[p] if p >= 0 else np.eye(3)
        local[j] = above.T @ rotation_between(posed, want) @ above @ local[j]
    propagate()
    recentre()

    # Head.
    #
    # Turn the neck by how far the head has moved *from the body's own rest
    # pose*, rather than aiming the neck bone at an absolute direction. There
    # is no keypoint on the skull to aim at: the nearest is the ear midpoint,
    # and a neck bone does not point at the ears. MakeHuman's runs from C7
    # forward and up to the base of the skull, 20 degrees off vertical, while
    # the shoulder-to-ear line of the editor's figure leans 3 - so aiming the
    # bone at the ears tipped every skull on every MPFB2 body 17 degrees back,
    # in the rest pose as much as any other. Against a moved reference the
    # rest pose is a rotation of zero and the rig keeps the neck it was
    # authored with, which is the whole point of having a rig.
    #
    # Measured in the body's own frame on both sides, so that a figure which
    # has also turned or leaned does not count the torso's rotation twice; and
    # applied to the neck rather than to the skull, because it is a neck that
    # bends when someone looks down.
    if ("neck" in roles and "r_ear" in points and rest_points
            and "r_ear" in rest_points):

        def head_frame(at):
            get = lambda name: np.asarray(at[name], float)
            shoulders = 0.5 * (get("r_shoulder") + get("l_shoulder"))
            ears = 0.5 * (get("r_ear") + get("l_ear"))
            hips = 0.5 * (get("r_hip") + get("l_hip"))
            body = frame_from(np.cross(get("l_hip") - get("r_hip"),
                                       shoulders - hips), shoulders - hips)
            head = frame_from(get("nose") - ears, ears - shoulders)
            return body, head

        body_now, head_now = head_frame(points)
        body_rest, head_rest = head_frame(rest_points)
        # How the head sits on the torso, then and now; the change between
        # them, carried back into world by the torso's current orientation.
        # The order matters: `now @ rest.T`, not `rest.T @ now` - rotations do
        # not commute, and the reversed product is 4 degrees out on a head
        # turned 40.
        was = body_rest.T @ head_rest
        is_ = body_now.T @ head_now
        turned = body_now @ (is_ @ was.T) @ body_now.T
        j = roles["neck"]
        p = parents[j]
        above = Q[p] if p >= 0 else np.eye(3)
        local[j] = above.T @ turned @ above @ local[j]
        propagate()
        recentre()

    return np.array(Q), q


def _rotation_of(M):
    """The rotation part of a rotation-times-uniform-scale."""
    scale = float(abs(np.linalg.det(M))) ** (1.0 / 3.0)
    return M / scale if scale > 1e-9 else M


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


def pose_rig(mesh, points_cm, roles=None, rest_points=None, stature=None):
    """Pose the body's OWN rig, rather than fitting it to the keypoints.

    A pose is a set of joint ANGLES, and
    that is the one thing eighteen keypoints report well - the line from a
    shoulder to an elbow says which way the upper arm points no matter whose
    arm it is. So this takes the directions and nothing else: every bone is
    rotated, none is moved or resized, and the body that comes out is exactly
    the body that went in. The rig's own hands, feet and spine chain come along
    for free, where a retarget had nothing to say about them.

    `stature` still sizes the rig, because a figure has to be the right height
    to stand next to another one; it is one uniform scale over the whole body
    and it changes no proportion.
    """
    roles = roles or mesh.get("roles") or resolve_bones(mesh["joint_names"])
    problems = validate_roles(mesh, roles)
    if problems:
        raise RuntimeError(
            "This rig does not map cleanly:\n  " + "\n  ".join(problems))
    rest = mesh["rest_position"]
    if stature:
        height = float(mesh["vertices"][:, 1].max()
                       - mesh["vertices"][:, 1].min())
        scale = float(stature) / max(1e-9, height)
    else:
        # No stature given - an outside rig, or a bare self-test. Size it on
        # the torso, which both conventions describe and neither disagrees
        # about much: hip midpoint up to the neck. One uniform scale, so no
        # proportion moves. Leaving it at 1.0 would hand back a rig still in
        # metres beside keypoints in centimetres, and every joint would sit a
        # metre from the keypoint it was posed by.
        at = lambda r: rest[roles[r]]
        kp = lambda n: np.asarray(points_cm[n], float)
        rig_span = float(np.linalg.norm(
            at("neck") - 0.5 * (at("l_hip") + at("r_hip"))))
        kp_span = float(np.linalg.norm(
            kp("neck") - 0.5 * (kp("l_hip") + kp("r_hip"))))
        scale = kp_span / max(1e-9, rig_span)
    Q, q = pose_globals(rest * scale, mesh["parents"], roles, points_cm,
                        rest_orient=mesh["rest_global"][:, :3, :3],
                        rest_points=rest_points)
    return {"scale": scale,
            "bones": {name: (Q[i], q[i])
                      for i, name in enumerate(mesh["joint_names"])}}


# Reading the eighteen keypoints back off a posed rig.
#
# Most of them are joints the rig already has, and those are simply read: the
# elbow keypoint is where the forearm bone starts, and no description of it can
# beat the thing itself. Only the ones the rig has no joint for need carrying -
# the face, which rides the skull, and the shoulder, which OpenPose puts at the
# acromion, a bony corner the rig does not model because nothing rotates there.
JOINT_KEYPOINTS = ("l_elbow", "r_elbow", "l_wrist", "r_wrist",
                   "l_hip", "r_hip", "l_knee", "r_knee",
                   "l_ankle", "r_ankle")
FACE_KEYPOINTS = ("nose", "l_eye", "r_eye", "l_ear", "r_ear")


def keypoint_riders(mesh, rest_points, roles=None, stature=None):
    """What each keypoint needs in order to be read off a posed rig.

    A joint keypoint needs only the bone it is. The face keypoints need their
    offset from the skull, and the shoulder its offset from the joint the arm
    swings from - both taken once, against the rig's own rest pose, with the
    two coordinate systems brought together by the torso, which is the one
    thing they both describe.
    """
    roles = roles or mesh.get("roles") or resolve_bones(mesh["joint_names"])
    rest = mesh["rest_position"]
    if stature:
        height = float(mesh["vertices"][:, 1].max()
                       - mesh["vertices"][:, 1].min())
        rest = rest * (float(stature) / max(1e-9, height))
    at = lambda role: rest[roles[role]]
    rig_hips = 0.5 * (at("l_hip") + at("r_hip"))
    rig_up = rest[roles["neck"]] - rig_hips
    rig_frame = frame_from(np.cross(at("l_hip") - at("r_hip"), rig_up), rig_up)

    kp = lambda name: np.asarray(rest_points[name], float)
    kp_hips = 0.5 * (kp("l_hip") + kp("r_hip"))
    kp_up = 0.5 * (kp("l_shoulder") + kp("r_shoulder")) - kp_hips
    kp_frame = frame_from(np.cross(kp("l_hip") - kp("r_hip"), kp_up), kp_up)
    into_rig = rig_frame @ kp_frame.T
    put = lambda name: into_rig @ (kp(name) - kp_hips) + rig_hips

    riders = {}
    for name in JOINT_KEYPOINTS:
        if name in roles:
            riders[name] = ("joint", roles[name], None)
    for name in FACE_KEYPOINTS:
        if "head" in roles and name in rest_points:
            riders[name] = ("ride", roles["head"], put(name) - at("head"))
    for side in ("l", "r"):
        role = side + "_shoulder"
        collar = side + "_collar"
        if role in roles and collar in roles and role in rest_points:
            # the acromion, as an offset from the joint the arm swings from,
            # carried by the collar - which is the bone it actually sits on
            riders[role] = ("ride", roles[collar], put(role) - at(collar))
    return riders


def keypoints_of(solution, riders, mesh):
    """The eighteen keypoints, read off a posed rig.

    The other direction from everything else here, and the honest one: the
    mesh is what the depth map shows, so the skeleton beside it should be a
    description of that mesh and not an independent claim about the same
    figure. Read this way the two cannot disagree, whatever the rig's
    proportions turn out to be.
    """
    names = mesh["joint_names"]
    out = {}
    for name, (how, j, offset) in riders.items():
        Q, q = solution["bones"][names[j]]
        out[name] = q if how == "joint" else q + _rotation_of(Q) @ offset
    # OpenPose has no neck of its own: it is the midpoint of the shoulders,
    # and it has to stay that way or the format is not the format.
    if "l_shoulder" in out and "r_shoulder" in out:
        out["neck"] = 0.5 * (out["l_shoulder"] + out["r_shoulder"])
    return out


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


def pose_mesh(mesh, points_cm, roles=None):
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
                        points_cm)
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


def _write_test_glb(path, lift=0.0, morph=False):
    """A small rigged humanoid, so the loader and skinning can be tested with
    no external assets. Bone names follow MakeHuman's default rig.

    `lift` displaces the geometry, which lets a test build an asset that is
    distinguishable from the body instead of an exact overlap.

    `morph` adds two shape keys with non-zero default weights - one written
    plainly, one sparse - which is how a DCC stores a body built out of
    sliders. A loader that reads the base mesh alone gets neither.
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
        ("toe.L", 21, (0.10, 0.02, 0.14)),          # so a foot has a pitch
        ("upperleg01.R", 0, (-0.09, 0.92, 0.0)),
        ("upperleg02.R", 23, (-0.10, 0.71, 0.0)),
        ("lowerleg01.R", 24, (-0.10, 0.50, 0.0)),
        ("lowerleg02.R", 25, (-0.10, 0.29, 0.0)),
        ("foot.R", 26, (-0.10, 0.08, 0.0)),
        ("toe.R", 27, (-0.10, 0.02, 0.14)),
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
    if morph:
        raised = np.zeros_like(verts)
        raised[:, 1] = np.float32(0.10)                  # a plain target
        moved = np.array([[0.0, 0.0, 0.20]] * 3, np.float32)
        blobs += [raised.tobytes(), np.arange(3, dtype=np.uint32).tobytes(),
                  moved.tobytes()]                       # and a sparse one
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
                         "byteLength": len(blobs[i])}
                        for i in range(len(blobs))],
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
        "morph_placeholder": None,
        "skins": [{"joints": list(range(2, 2 + len(bones))),
                   "inverseBindMatrices": 4}],
        "nodes": [{"name": "body", "mesh": 0, "skin": 0}],
    }
    del gltf["morph_placeholder"]
    if morph:
        gltf["accessors"] += [
            {"bufferView": 5, "componentType": 5126, "count": len(verts),
             "type": "VEC3"},
            # no bufferView of its own: the base is all zeros and only the
            # three substituted elements say anything, which is how an
            # exporter writes a target that moves a handful of vertices
            {"componentType": 5126, "count": len(verts), "type": "VEC3",
             "sparse": {"count": 3,
                        "indices": {"bufferView": 6, "componentType": 5125},
                        "values": {"bufferView": 7}}}]
        gltf["meshes"][0]["primitives"][0]["targets"] = [{"POSITION": 5},
                                                         {"POSITION": 6}]
        gltf["meshes"][0]["weights"] = [0.5, 1.0]

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
    check("bones read with hierarchy", len(mesh["joint_names"]) == 29
          and mesh["parents"][0] == -1, "%d bones" % len(mesh["joint_names"]))
    check("rest positions rebuilt from the node tree",
          abs(mesh["rest_position"][4][1] - 1.52) < 1e-6,
          "head at y=%.3f" % mesh["rest_position"][4][1])
    check("weights normalised",
          abs(mesh["skin_weights"].sum() - len(mesh["vertices"])) < 1e-6)

    # A shape key reaches glTF as a morph target with a default weight, and a
    # body built out of sliders - MakeHuman's macros, a Daz morph dial - keeps
    # its whole shape there. Read the base alone and every figure in a set
    # loads as the same unshaped mannequin, while the rig, which *is* fitted
    # to the shape, still differs: five MPFB2 bodies came in as one 167 cm
    # base mesh with five different skeletons stretching it, and the only
    # sign of it was a limb the skinning appeared to pinch.
    shaped = load_rigged_mesh(
        _write_test_glb(os.path.join(tempfile.gettempdir(), "rig_morph.glb"),
                        morph=True), drop_loose=False)
    check("a mesh with morph targets is loaded at its own shape",
          shaped["morphs_applied"] == 2, "%d applied"
          % shaped["morphs_applied"])
    rise = shaped["vertices"][:, 1] - solid["vertices"][:, 1]
    check("a plain target moves every vertex by its weight",
          abs(rise.max() - 0.05) < 1e-6 and abs(rise.min() - 0.05) < 1e-6,
          "%.4f..%.4f, wanted 0.5 x 0.10" % (rise.min(), rise.max()))
    shift = shaped["vertices"][:, 2] - solid["vertices"][:, 2]
    check("and a sparse target moves only the vertices it names",
          abs(shift[:3] - 0.20).max() < 1e-6 and abs(shift[3:]).max() < 1e-9,
          "%d of %d vertices moved" % (int((abs(shift) > 1e-9).sum()),
                                       len(shift)))
    check("a mesh without targets is unchanged by any of this",
          solid["morphs_applied"] == 0)

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

    Q, q = pose_globals(rest, mesh["parents"], roles, pts)

    # The contract now: a bone points the way the keypoints say, and keeps the
    # length the rig gave it. Nothing lands ON a keypoint, and it should not -
    # the keypoints describe a body with different proportions, and chasing
    # them is what used to stretch the mesh.
    worst = 0.0
    for a, b in limbs:
        got = unit(q[roles[b]] - q[roles[a]])
        want = unit(np.asarray(pts[b], float) - np.asarray(pts[a], float))
        worst = max(worst, float(np.linalg.norm(got - want)))
    check("every bone points the way its two keypoints do", worst < 1e-9,
          "worst direction error %.2e" % worst)

    grew = 0.0
    for a, b in limbs:
        was = float(np.linalg.norm(rest[roles[b]] - rest[roles[a]]))
        now = float(np.linalg.norm(q[roles[b]] - q[roles[a]]))
        grew = max(grew, abs(now / max(1e-9, was) - 1.0))
    check("and keeps the length the rig authored, exactly", grew < 1e-9,
          "worst %.2e" % grew)

    drift = max(float(np.linalg.norm(q[roles[r]] - np.asarray(pts[r], float)))
                for _a, r in limbs)
    check("so a joint sits off its keypoint by the difference in proportions",
          0.5 < drift < 20.0, "%.2f cm" % drift)

    # the twist bones must ride along with the joint above them
    Q2, q2 = pose_globals(rest, mesh["parents"], roles, pts)
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

    # A bone's own geometry keeps its *shape*.
    #
    # It used to keep its size as well, and that check had to go: a rig is
    # fitted to the figure segment by segment now, so a bone whose segment is
    # 62% of the keypoints' is 62% of its own length - and 62% as thick, which
    # is the point. What must not change is the shape. A uniform scale leaves
    # every ratio of distances alone; shear, a torn joint, or a limb dragged
    # onto its keypoint do not, which is what this is watching for.
    trio = bound[:3]
    def sides(points):
        return np.array([np.linalg.norm(points[a] - points[b])
                         for a, b in ((0, 1), (1, 2), (2, 0))])
    rest_sides = sides(mesh["vertices"][trio])
    posed_sides = sides(verts[trio])
    shape_rest = rest_sides / rest_sides.sum()
    shape_posed = posed_sides / posed_sides.sum()
    check("a bone's geometry keeps its shape under skinning",
          float(np.abs(shape_rest - shape_posed).max()) < 1e-6,
          "worst side ratio off by %.2e"
          % float(np.abs(shape_rest - shape_posed).max()))
    grew = float(posed_sides.sum() / max(1e-12, rest_sides.sum()) / scale)
    check("and its size follows the segment it belongs to",
          0.3 < grew < 3.0, "scaled %.3f against the rig's own" % grew)

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
    solution = pose_rig(mesh, pts)
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


    # Head.
    #
    # It must turn by as much as the face keypoints turned, and by nothing at
    # all when they have not moved. The check this replaces asserted that the
    # head bone's own +Z ends up along the gaze, which assumes the rig's head
    # bone is authored square to the body - and then the code was written to
    # satisfy it, by pointing the neck at the ear midpoint, the nearest
    # keypoint to a skull that has none. MakeHuman's neck leans 20 degrees
    # forward and the editor's shoulder-to-ear line leans 3, so every skull on
    # every MPFB2 body sat 17 degrees back, in the rest pose as much as any
    # other, and the test said nothing because the mock rig's neck is
    # vertical. Comparing a moved head with a moved reference cannot be
    # satisfied that way.
    turned = dict(pts)
    turned["nose"] = (14.0, 16.0, -2.0)      # look to the figure's left
    turned["r_ear"] = (-2.0, 19.0, -7.0)
    turned["l_ear"] = (7.0, 19.0, 6.0)
    head_i = mesh["joint_names"].index("head")

    def face(at):
        ears = 0.5 * (np.asarray(at["r_ear"], float)
                      + np.asarray(at["l_ear"], float))
        shoulders = 0.5 * (np.asarray(at["r_shoulder"], float)
                           + np.asarray(at["l_shoulder"], float))
        return frame_from(np.asarray(at["nose"], float) - ears,
                          ears - shoulders)

    Q_still, _ = pose_globals(rest, mesh["parents"], roles, pts, rest_points=pts)
    Q3, q3 = pose_globals(rest, mesh["parents"], roles, turned, rest_points=pts)
    unmoved = math.degrees(math.acos(max(-1.0, min(1.0,
              (np.trace(Q_still[head_i]) - 1.0) / 2.0))))
    check("a head whose keypoints have not moved does not turn",
          unmoved < 0.5, "%.2f deg" % unmoved)
    got = Q3[head_i] @ Q_still[head_i].T
    wanted = face(turned) @ face(pts).T
    gap = math.degrees(math.acos(max(-1.0, min(1.0,
          (np.trace(got @ wanted.T) - 1.0) / 2.0))))
    check("and one whose keypoints turned turns with them",
          gap < 1.0, "%.2f deg apart" % gap)
    # Feet. There is no keypoint past the ankle, so a foot rides its shin and
    # points wherever the shin does - a lunge on pointed toes, a seated figure
    # dangling. A foot whose shin is near upright is taking weight, so it goes
    # back to the pitch the rig authored; one whose shin is not stays put,
    # because a kneeling figure's shins point backwards and a lying one's
    # sideways and neither is standing on anything.
    foot_i = mesh["joint_names"].index("foot.L")
    toe = mesh["rest_position"][mesh["joint_names"].index("toe.L")]
    rest_toe = unit(toe - mesh["rest_position"][foot_i])

    def foot_pitch(where):
        Qf, _ = pose_globals(rest, mesh["parents"], roles, where, rest_points=pts)
        return math.degrees(math.asin(float(unit(Qf[foot_i] @ rest_toe)[1])))

    flat = foot_pitch(pts)
    tilted = dict(pts)                       # knee forward: shin 30 deg back
    tilted["l_knee"] = (11.0, -95.0, 21.0)
    lifted = dict(pts)                       # shin out sideways, 75 deg over
    lifted["l_ankle"] = (52.0, -85.0, 0.0)
    check("a foot under a near-upright shin keeps its authored pitch",
          abs(foot_pitch(tilted) - flat) < 1.0,
          "%.1f deg vs %.1f at rest" % (foot_pitch(tilted), flat))
    check("and one under a shin that is not standing on anything is left "
          "alone", abs(foot_pitch(lifted) - flat) > 10.0,
          "%.1f deg vs %.1f at rest" % (foot_pitch(lifted), flat))

    facing = (Q3[head_i] @ Q_still[head_i].T) @ face(pts) @ np.array([0.0, 0.0, 1.0])
    want = face(turned) @ np.array([0.0, 0.0, 1.0])
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
    loose, _lq = pose_globals(rest, mesh["parents"], roles, pts)
    tight, tq = pose_globals(rest, mesh["parents"], roles, pts,
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
        Q, q = pose_globals(rest, mesh["parents"], roles, arm_pose(upper, lower), rest_orient=rest_orient)
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

    # And the reset must not fight what aiming already worked out. It used to
    # agree to 0.00 degrees, because a scaling pass ran afterwards and
    # converged the two; with the rig posed rather than fitted there is no
    # such pass, aiming has the last word, and what is left is the reset
    # nudging the frame a few degrees on its way past. That it is a *few* is
    # the point - the bug this guards against read 176 degrees of jump as
    # success - and the two continuity checks above now return byte-identical
    # numbers to a third of a degree with the reset and without it, which is
    # the stronger statement: on a rig posed this way it changes nothing
    # that matters.
    drift = 0.0
    for upper, lower in swept[::5] + folded[::5]:
        a = arm_frames(upper, lower, None)
        b = arm_frames(upper, lower, orient)
        for u, v in zip(a, b):
            drift = max(drift, math.degrees(math.acos(
                float(np.clip(np.dot(u, v), -1.0, 1.0)))))
    check("the roll reset confirms the aimed roll rather than replacing it",
          drift < 10.0, "worst disagreement %.2f deg" % drift)
    apart = max(abs(biggest_step(poses, None) - biggest_step(poses, orient))
                for poses in (swept, folded))
    check("and leaves the frame's continuity where aiming left it",
          apart < 0.5, "worst step differs by %.3f deg" % apart)

    check("head turns with the face keypoints",
          float(np.dot(unit(facing), want)) > 0.98,
          "cos %.4f" % float(np.dot(unit(facing), want)))
    Q4, _ = pose_globals(rest, mesh["parents"], roles, pts)
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
