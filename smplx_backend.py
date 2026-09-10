#!/usr/bin/env python3
"""SMPL-X backend for the 3D OpenPose editor.

Replaces the built-in capsule/profile body with a real scanned-template mesh
for depth export. The editor keeps owning the pose; this module only converts
18 COCO keypoints into SMPL-X parameters, runs the model, and rasterises the
resulting 10,475-vertex mesh into a depth map.

Why there is no optimiser here
------------------------------
SMPLify-X fits SMPL-X to *2D* keypoints, which needs gradient descent and a
pose prior. You already have full 3D joint positions, so every body joint's
rotation is determined in closed form: rotate the rest bone onto the target
bone, walking down the kinematic tree. It is exact, instant and deterministic.

Setup
-----
    pip install smplx torch numpy pillow

Download from https://smpl-x.is.tue.mpg.de (your account, "SMPL-X v1.1"),
unpack, and arrange as the smplx package expects:

    <model_dir>/smplx/SMPLX_NEUTRAL.npz
    <model_dir>/smplx/SMPLX_MALE.npz
    <model_dir>/smplx/SMPLX_FEMALE.npz

Then point the editor at <model_dir>, or set SMPLX_MODEL_DIR.

Self-test (works without the model files, exercises the maths on a mock body):
    python3 smplx_backend.py --selftest
"""

from __future__ import annotations

import math
import os

import numpy as np

# 21 body joints carry body_pose in SMPL-X; the rest are hands, jaw and eyes.
SMPLX_BODY_JOINTS = 21

# Canonical ordering from smplx/joint_names.py. Loaded from the package when
# available; this copy keeps the module usable (and testable) without it.
SMPLX_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "jaw", "left_eye_smplhf", "right_eye_smplhf",
]

# Our COCO-18 names -> SMPL-X landmark names. The editor's "neck" is the
# midpoint of the shoulders, which is not SMPL-X's neck joint, so it is handled
# separately rather than mapped here.
KEYPOINT_TO_SMPLX = {
    "nose": "nose", "r_eye": "right_eye", "l_eye": "left_eye",
    "r_ear": "right_ear", "l_ear": "left_ear",
    "r_shoulder": "right_shoulder", "l_shoulder": "left_shoulder",
    "r_elbow": "right_elbow", "l_elbow": "left_elbow",
    "r_wrist": "right_wrist", "l_wrist": "left_wrist",
    "r_hip": "right_hip", "l_hip": "left_hip",
    "r_knee": "right_knee", "l_knee": "left_knee",
    "r_ankle": "right_ankle", "l_ankle": "left_ankle",
}

# Which SMPL-X joint is rotated, which of its children is aimed, and the
# keypoint that child should land on. Ordered parents before children.
AIM_CHAIN = [
    ("left_collar", "left_shoulder", "l_shoulder"),
    ("right_collar", "right_shoulder", "r_shoulder"),
    ("left_shoulder", "left_elbow", "l_elbow"),
    ("right_shoulder", "right_elbow", "r_elbow"),
    ("left_elbow", "left_wrist", "l_wrist"),
    ("right_elbow", "right_wrist", "r_wrist"),
    ("left_hip", "left_knee", "l_knee"),
    ("right_hip", "right_knee", "r_knee"),
    ("left_knee", "left_ankle", "l_ankle"),
    ("right_knee", "right_ankle", "r_ankle"),
]


# ---------------------------------------------------------------------------
# rotation helpers
# ---------------------------------------------------------------------------

def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(3)


def rotation_between(a, b):
    """Minimal rotation taking direction a onto direction b."""
    a, b = unit(np.asarray(a, float)), unit(np.asarray(b, float))
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return np.eye(3)
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = unit(np.cross(a, np.array([1.0, 0.0, 0.0])))
        if np.linalg.norm(axis) < 1e-9:
            axis = unit(np.cross(a, np.array([0.0, 1.0, 0.0])))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = v / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + K * s + K @ K * (1.0 - c)


def axis_angle_from_matrix(R):
    """Rotation matrix -> axis-angle vector, the form SMPL-X expects."""
    R = np.asarray(R, float)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(c)
    if angle < 1e-8:
        return np.zeros(3)
    if angle > math.pi - 1e-5:                 # near 180 degrees
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] > 1e-8:
            axis = A[:, k] / axis[k]
        axis = unit(axis)
        return axis * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return unit(axis) * angle


def matrix_from_axis_angle(v):
    v = np.asarray(v, float)
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + K * math.sin(angle) + K @ K * (1.0 - math.cos(angle))


def scale_rotation(R, factor):
    """A fraction of a rotation, for spreading torso bend over the spine."""
    return matrix_from_axis_angle(axis_angle_from_matrix(R) * factor)


def frame_from(forward, up):
    """Orthonormal 3x3 whose columns are (left, up, forward)."""
    up = unit(up)
    forward = unit(forward - up * float(np.dot(forward, up)))
    if np.linalg.norm(forward) < 1e-9:
        forward = unit(np.cross(up, np.array([1.0, 0.0, 0.0])))
    left = np.cross(up, forward)
    return np.column_stack([unit(left), up, forward])


# ---------------------------------------------------------------------------
# forward kinematics on joints only (no skinning) - used for alignment and tests
# ---------------------------------------------------------------------------

def fk_joints(rest, parents, rotations):
    """Forward kinematics over the kinematic tree only.

    A model's joint array is longer than its parent array: SMPL-X returns 127
    joints, of which only the first 55 are skeleton joints with parents. The
    rest are landmarks (nose, ears, toes, finger tips) regressed from vertices
    and posed by skinning, not by the tree. Iterate over the parents, not the
    joints.

    rest: (J,3) rest joints. rotations: dict joint index -> local 3x3.
    """
    n = len(parents)
    rest = np.asarray(rest, float)[:n]
    globals_ = [np.eye(3)] * n
    posed = np.zeros_like(rest)
    for j in range(n):
        p = parents[j]
        R = rotations.get(j, np.eye(3))
        if p < 0:
            globals_[j] = R
            posed[j] = rest[j]
        else:
            globals_[j] = globals_[p] @ R
            posed[j] = posed[p] + globals_[p] @ (rest[j] - rest[p])
    return posed, globals_


# ---------------------------------------------------------------------------
# shape: solve betas so SMPL-X bone lengths match the edited skeleton
# ---------------------------------------------------------------------------

def fit_betas(joint_template, joint_shapedirs, parents, targets, index_of,
              iterations=8, regulariser=0.05, limit=4.0):
    """Gauss-Newton on bone lengths.

    Joint positions are linear in beta, J(b) = J0 + S b, but bone *length* is a
    norm of a difference, so it is linearised and iterated. Betas are standard
    deviations of the scan population, so they are clamped to a sane range
    rather than allowed to run off into implausible bodies.

    joint_template : (J,3)      joints at beta = 0
    joint_shapedirs: (J,3,B)    d(joint)/d(beta)
    targets        : {(parent_name, child_name): length in metres}
    """
    n_betas = joint_shapedirs.shape[2]
    betas = np.zeros(n_betas)
    bones = [(index_of[p], index_of[c], L) for (p, c), L in targets.items()
             if p in index_of and c in index_of]
    if not bones:
        return betas
    for _ in range(iterations):
        J = joint_template + np.einsum("jdb,b->jd", joint_shapedirs, betas)
        rows, res = [], []
        for pi, ci, target in bones:
            vec = J[ci] - J[pi]
            length = float(np.linalg.norm(vec))
            if length < 1e-9:
                continue
            u = vec / length
            grad = np.einsum("d,db->b", u,
                             joint_shapedirs[ci] - joint_shapedirs[pi])
            rows.append(grad)
            res.append(target - length)
        if not rows:
            break
        A = np.asarray(rows)
        r = np.asarray(res)
        H = A.T @ A + regulariser * np.eye(n_betas)
        betas = np.clip(betas + np.linalg.solve(H, A.T @ r), -limit, limit)
    return betas


# ---------------------------------------------------------------------------
# pose: closed-form retarget from the editor's keypoints
# ---------------------------------------------------------------------------

def retarget_pose(rest_joints, parents, index_of, points):
    """Solve global_orient and body_pose from 3D keypoints.

    rest_joints : (J,3) SMPL-X joints in its own rest space
    index_of    : joint name -> row in rest_joints
    points      : our keypoint name -> (3,) position in SMPL-X space

    Returns (global_orient (3,), body_pose (21,3), rotations dict).
    Rotations are minimal, so twist about a bone's own axis is not recovered -
    nothing in an 18-keypoint skeleton observes it.
    """
    needed = set(KEYPOINT_TO_SMPLX) | {"neck"}
    absent = sorted(needed - set(points))
    if absent:
        raise RuntimeError("retarget needs all 18 keypoints, missing %s" % absent)

    def kp(name):
        return np.asarray(points[name], float)

    hip_mid = 0.5 * (kp("r_hip") + kp("l_hip"))
    sh_mid = 0.5 * (kp("r_shoulder") + kp("l_shoulder"))
    ear_mid = 0.5 * (kp("r_ear") + kp("l_ear"))

    rest_hip_mid = 0.5 * (rest_joints[index_of["right_hip"]]
                          + rest_joints[index_of["left_hip"]])
    rotations = {}

    # pelvis: a full frame, so no ambiguity at the root
    rest_frame = frame_from(
        np.cross(rest_joints[index_of["left_hip"]]
                 - rest_joints[index_of["right_hip"]],
                 rest_joints[index_of["neck"]] - rest_hip_mid),
        rest_joints[index_of["neck"]] - rest_hip_mid)
    target_frame = frame_from(
        np.cross(kp("l_hip") - kp("r_hip"), sh_mid - hip_mid), sh_mid - hip_mid)
    rotations[0] = target_frame @ rest_frame.T

    def state():
        return fk_joints(rest_joints, parents, rotations)

    def global_of(joint):
        _, globals_ = state()
        j = parents[index_of[joint]]
        return globals_[j] if j >= 0 else np.eye(3)

    def aim(joint, child, target, positional=True):
        """Rotate joint so its child bone points at (or towards) the target.

        Positional aiming makes the child land exactly on the keypoint when the
        bone length matches, which directional aiming does not: an intermediate
        joint like the collar has no keypoint of its own, so aiming its bone
        along a direction measured from somewhere else leaves the shoulder
        displaced and every joint below it inherits the error."""
        posed, globals_ = state()
        ji, ci = index_of[joint], index_of[child]
        current = posed[ci] - posed[ji]
        goal = (target - posed[ji]) if positional else target
        delta = rotation_between(current, goal)
        parent = parents[ji]
        G_parent = globals_[parent] if parent >= 0 else np.eye(3)
        rotations[ji] = G_parent.T @ delta @ globals_[ji]

    # spine: spread the pelvis-to-neck bend over spine1/2/3
    spine_rest = rest_joints[index_of["neck"]] - rest_hip_mid
    spine_delta = rotation_between(rotations[0] @ spine_rest, sh_mid - hip_mid)
    third = scale_rotation(spine_delta, 1.0 / 3.0)
    for name in ("spine1", "spine2", "spine3"):
        j = index_of[name]
        G_parent = global_of(name)
        rotations[j] = G_parent.T @ third @ G_parent

    # everything is solved in the model's own frame, so shift the targets to
    # sit on the posed pelvis before aiming at them
    posed, _ = state()
    shift = 0.5 * (posed[index_of["right_hip"]] + posed[index_of["left_hip"]]) \
        - hip_mid
    goal = {name: kp(name) + shift for name in points}

    for joint, child, target in AIM_CHAIN:
        aim(joint, child, goal[target])

    # the neck is aimed by direction: SMPL-X's head joint is the base of the
    # skull, not the ear midpoint, so making it land there would tilt the head
    aim("neck", "head", ear_mid - sh_mid, positional=False)
    head_rest = frame_from(
        rest_joints[index_of["nose"]] - rest_joints[index_of["head"]],
        rest_joints[index_of["head"]] - rest_joints[index_of["neck"]])
    head_target = frame_from(kp("nose") - ear_mid, ear_mid - sh_mid)
    G_parent = global_of("head")
    rotations[index_of["head"]] = G_parent.T @ (head_target @ head_rest.T)
    del posed

    body_pose = np.zeros((SMPLX_BODY_JOINTS, 3))
    for j in range(1, SMPLX_BODY_JOINTS + 1):
        body_pose[j - 1] = axis_angle_from_matrix(rotations.get(j, np.eye(3)))
    return axis_angle_from_matrix(rotations[0]), body_pose, rotations


# ---------------------------------------------------------------------------
# depth rasteriser
# ---------------------------------------------------------------------------

def rasterize_depth(verts, faces, width, height, cull=True):
    """Z-buffer a triangle mesh given in pixel space (x, y, z all in pixels).

    Returns a float array of nearest depth per pixel, +inf where empty.
    """
    zbuf = np.full((height, width), np.inf)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
            - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
    keep = np.abs(area) > 1e-12
    if cull:
        # winding and camera handedness both flip the sign of the screen-space
        # area, so decide empirically: of the two groups, the front-facing one
        # is the one whose triangles sit nearer the camera on average
        neg = keep & (area < 0)
        pos = keep & (area > 0)
        mid = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0
        if neg.any() and pos.any():
            keep &= neg if mid[neg].mean() < mid[pos].mean() else pos
    idx = np.nonzero(keep)[0]

    lo_x = np.maximum(np.floor(np.minimum(np.minimum(v0[:, 0], v1[:, 0]),
                                          v2[:, 0])).astype(int), 0)
    hi_x = np.minimum(np.ceil(np.maximum(np.maximum(v0[:, 0], v1[:, 0]),
                                         v2[:, 0])).astype(int) + 1, width)
    lo_y = np.maximum(np.floor(np.minimum(np.minimum(v0[:, 1], v1[:, 1]),
                                          v2[:, 1])).astype(int), 0)
    hi_y = np.minimum(np.ceil(np.maximum(np.maximum(v0[:, 1], v1[:, 1]),
                                         v2[:, 1])).astype(int) + 1, height)

    for t in idx:
        x0, x1 = lo_x[t], hi_x[t]
        y0, y1 = lo_y[t], hi_y[t]
        if x0 >= x1 or y0 >= y1:
            continue
        a, b, c = v0[t], v1[t], v2[t]
        inv = 1.0 / area[t]
        ys, xs = np.mgrid[y0:y1, x0:x1]
        xs = xs + 0.5
        ys = ys + 0.5
        w0 = ((b[0] - a[0]) * (ys - a[1]) - (xs - a[0]) * (b[1] - a[1])) * inv
        w1 = ((xs - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (ys - a[1])) * inv
        inside = (w0 >= 0.0) & (w1 >= 0.0) & (w0 + w1 <= 1.0)
        if not inside.any():
            continue
        z = a[2] + w1 * (b[2] - a[2]) + w0 * (c[2] - a[2])
        window = zbuf[y0:y1, x0:x1]
        np.minimum(window, np.where(inside, z, np.inf), out=window)
    return zbuf


def depth_to_image(zbuf, near=255, far=45, background=0):
    """ControlNet convention: nearest surface white, background black."""
    from PIL import Image
    covered = np.isfinite(zbuf)
    img = np.full(zbuf.shape, float(background))
    if covered.any():
        z = zbuf[covered]
        lo, hi = z.min(), z.max()
        span = hi - lo
        if span < 1e-6:                 # flat surface: it is all "nearest"
            img[covered] = near
            span = None
        if span is not None:
            img[covered] = far + (near - far) * (hi - z) / span
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"), mode="L")


# ---------------------------------------------------------------------------
# the model wrapper
# ---------------------------------------------------------------------------

class SmplxBody:
    """Holds a loaded SMPL-X model and converts editor skeletons into meshes."""

    def __init__(self, model_dir=None, gender="neutral", num_betas=10):
        import torch
        import smplx
        self.torch = torch
        model_dir = model_dir or os.environ.get("SMPLX_MODEL_DIR")
        if not model_dir:
            raise RuntimeError(
                "No SMPL-X model directory. Pass one, or set SMPLX_MODEL_DIR "
                "to the folder that contains smplx/SMPLX_NEUTRAL.npz")
        self.model = smplx.create(model_dir, model_type="smplx", gender=gender,
                                  use_face_contour=False, use_pca=False,
                                  flat_hand_mean=True, num_betas=num_betas,
                                  ext="npz")
        self.model.eval()
        try:
            from smplx.joint_names import JOINT_NAMES
            names = list(JOINT_NAMES)
        except Exception:
            names = list(SMPLX_JOINT_NAMES)
        self.index_of = {n: i for i, n in enumerate(names)}
        self.parents = self.model.parents.detach().cpu().numpy()
        self.faces = np.asarray(self.model.faces, dtype=np.int64)
        self.num_betas = num_betas
        self.betas = np.zeros(num_betas)
        self.n_kinematic = len(self.parents)
        missing = [n for n in ("pelvis", "neck", "head", "nose", "left_hip")
                   if n not in self.index_of]
        if missing:
            raise RuntimeError("Unexpected SMPL-X joint layout, missing %s"
                               % missing)
        if self.n_kinematic <= SMPLX_BODY_JOINTS:
            raise RuntimeError(
                "This model has %d kinematic joints; SMPL-X needs more than %d. "
                "Check model_type is smplx." % (self.n_kinematic,
                                                SMPLX_BODY_JOINTS))

    # -- rest geometry -----------------------------------------------------
    def _rest_joints(self, betas):
        """Joints at a given shape, before posing (metres)."""
        torch = self.torch
        with torch.no_grad():
            out = self.model(betas=torch.tensor(betas, dtype=torch.float32)[None],
                             body_pose=torch.zeros(1, SMPLX_BODY_JOINTS * 3),
                             global_orient=torch.zeros(1, 3),
                             return_verts=True)
        return out.joints[0].numpy()

    def joint_shape_jacobian(self):
        """d(joint)/d(beta) for the kinematic joints, via the joint regressor."""
        torch = self.torch
        with torch.no_grad():
            reg = self.model.J_regressor.numpy()          # (J, V)
            shapedirs = self.model.shapedirs.numpy()      # (V, 3, B)
            v_template = self.model.v_template.numpy()    # (V, 3)
        n_betas = min(self.num_betas, shapedirs.shape[2])
        template = reg @ v_template
        jac = np.einsum("jv,vdb->jdb", reg, shapedirs[:, :, :n_betas])
        return template, jac

    # -- fitting -----------------------------------------------------------
    def fit_shape_to_lengths(self, bone_lengths_m):
        template, jac = self.joint_shape_jacobian()
        index_of = {n: i for i, n in enumerate(SMPLX_JOINT_NAMES)}
        betas = fit_betas(template, jac, self.parents, bone_lengths_m, index_of)
        self.betas = np.pad(betas, (0, self.num_betas - len(betas)))[:self.num_betas]
        return self.betas

    def pose_from_points(self, points_m):
        """points_m: our keypoint name -> (3,) in SMPL-X space, metres."""
        rest = self._rest_joints(self.betas)
        return retarget_pose(rest, self.parents, self.index_of, points_m)

    def vertices(self, points_m):
        torch = self.torch
        global_orient, body_pose, _ = self.pose_from_points(points_m)
        with torch.no_grad():
            out = self.model(
                betas=torch.tensor(self.betas, dtype=torch.float32)[None],
                global_orient=torch.tensor(global_orient, dtype=torch.float32)[None],
                body_pose=torch.tensor(body_pose.reshape(1, -1), dtype=torch.float32),
                return_verts=True)
        verts = out.vertices[0].numpy()
        joints = out.joints[0].numpy()
        # slide the mesh so the solved joints sit on the editor's keypoints
        offsets = [np.asarray(points_m[k]) - joints[self.index_of[s]]
                   for k, s in KEYPOINT_TO_SMPLX.items()
                   if s in self.index_of and k in points_m]
        if offsets:
            verts = verts + np.mean(offsets, axis=0)
        return verts


# ---------------------------------------------------------------------------
# glue to the editor's world space
# ---------------------------------------------------------------------------

def world_to_smplx_frame(body, points_cm=None):
    """Rotation mapping the editor's world into SMPL-X rest space.

    Derived from the model's own rest joints rather than assumed, so it stays
    correct whatever axis convention the model files use.
    """
    rest = body._rest_joints(np.zeros(body.num_betas))
    idx = body.index_of
    hip_mid = 0.5 * (rest[idx["left_hip"]] + rest[idx["right_hip"]])
    model_frame = frame_from(rest[idx["nose"]] - rest[idx["head"]],
                             rest[idx["neck"]] - hip_mid)
    # In any right-handed frame, cross(up, forward) is the body's own left, so
    # both frames are built the same way and the mapping is a pure rotation.
    # Verify it rather than trust it: getting this backwards would silently
    # mirror the mesh, swapping the figure's left and right.
    to_left = rest[idx["left_hip"]] - rest[idx["right_hip"]]
    if float(np.dot(model_frame[:, 0], to_left)) <= 0.0:
        raise RuntimeError(
            "SMPL-X rest pose has an unexpected handedness: the frame built "
            "from its spine and face directions disagrees with its own hip "
            "order. Refusing to build a mirrored body.")
    # editor world: +X right, +Y up, +Z towards the camera, figure faces +Z
    editor_frame = frame_from(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    return model_frame @ editor_frame.T


def points_to_smplx(points_cm, rotation, scale=0.01):
    return {k: rotation @ (np.asarray(v, float) * scale)
            for k, v in points_cm.items()}


# Bone lengths used to solve the shape. Not all are parent/child pairs -
# any pair of joints has a length that varies with beta, so shoulder and hip
# breadth can be constrained directly.
SHAPE_BONES = [
    (("left_hip", "left_knee"), ("l_hip", "l_knee")),
    (("right_hip", "right_knee"), ("r_hip", "r_knee")),
    (("left_knee", "left_ankle"), ("l_knee", "l_ankle")),
    (("right_knee", "right_ankle"), ("r_knee", "r_ankle")),
    (("left_shoulder", "left_elbow"), ("l_shoulder", "l_elbow")),
    (("right_shoulder", "right_elbow"), ("r_shoulder", "r_elbow")),
    (("left_elbow", "left_wrist"), ("l_elbow", "l_wrist")),
    (("right_elbow", "right_wrist"), ("r_elbow", "r_wrist")),
    (("left_shoulder", "right_shoulder"), ("l_shoulder", "r_shoulder")),
    (("left_hip", "right_hip"), ("l_hip", "r_hip")),
]


def bone_targets(points_m):
    """Measured lengths from the edited skeleton, for the shape solve."""
    def P(name):
        return np.asarray(points_m[name], float)

    targets = {pair: float(np.linalg.norm(P(b) - P(a)))
               for pair, (a, b) in SHAPE_BONES}
    hip_mid = 0.5 * (P("l_hip") + P("r_hip"))
    sh_mid = 0.5 * (P("l_shoulder") + P("r_shoulder"))
    targets[("pelvis", "neck")] = float(np.linalg.norm(sh_mid - hip_mid))
    return targets


def mesh_from_skeleton(body, points_cm, fit_shape=True):
    """Edited skeleton (centimetres, editor axes) -> mesh vertices in the same
    space. Returns (vertices_cm, faces)."""
    rotation = world_to_smplx_frame(body, points_cm)
    points_m = points_to_smplx(points_cm, rotation)
    if fit_shape:
        body.fit_shape_to_lengths(bone_targets(points_m))
    verts = body.vertices(points_m)
    return (rotation.T @ verts.T).T * 100.0, body.faces


# ---------------------------------------------------------------------------
# self-test on a mock body, so the maths can be checked without model files
# ---------------------------------------------------------------------------

def _mock_rest():
    """A stand-in SMPL-X rest skeleton: right ordering, plausible geometry."""
    names = list(SMPLX_JOINT_NAMES) + ["nose", "right_eye", "left_eye",
                                       "right_ear", "left_ear"]
    # parents covers only the kinematic joints; the five face landmarks that
    # follow are regressed, exactly as in the real model
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
               16, 17, 18, 19, 15, 15, 15]
    J = {
        "pelvis": (0, 0, 0), "left_hip": (0.09, -0.02, 0),
        "right_hip": (-0.09, -0.02, 0), "spine1": (0, 0.12, 0),
        "left_knee": (0.10, -0.42, 0), "right_knee": (-0.10, -0.42, 0),
        "spine2": (0, 0.25, 0), "left_ankle": (0.10, -0.84, -0.02),
        "right_ankle": (-0.10, -0.84, -0.02), "spine3": (0, 0.33, 0),
        "left_foot": (0.10, -0.88, 0.10), "right_foot": (-0.10, -0.88, 0.10),
        "neck": (0, 0.50, 0), "left_collar": (0.05, 0.44, 0),
        "right_collar": (-0.05, 0.44, 0), "head": (0, 0.62, 0.01),
        "left_shoulder": (0.17, 0.46, 0), "right_shoulder": (-0.17, 0.46, 0),
        "left_elbow": (0.44, 0.46, 0), "right_elbow": (-0.44, 0.46, 0),
        "left_wrist": (0.69, 0.46, 0), "right_wrist": (-0.69, 0.46, 0),
        "jaw": (0, 0.60, 0.05), "left_eye_smplhf": (0.03, 0.66, 0.08),
        "right_eye_smplhf": (-0.03, 0.66, 0.08),
        "nose": (0, 0.64, 0.11), "right_eye": (-0.03, 0.66, 0.08),
        "left_eye": (0.03, 0.66, 0.08), "right_ear": (-0.07, 0.65, 0.01),
        "left_ear": (0.07, 0.65, 0.01),
    }
    rest = np.array([J[n] for n in names], float)
    return names, rest, np.array(parents)


def _selftest():
    names, rest, parents = _mock_rest()
    index_of = {n: i for i, n in enumerate(names)}
    rng = np.random.default_rng(7)
    ok = True

    def check(label, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(("PASS " if cond else "FAIL ") + label + ("  " + extra if extra else ""))

    print("-- rotation algebra")
    for _ in range(200):
        a, b = rng.normal(size=3), rng.normal(size=3)
        R = rotation_between(a, b)
        err = np.linalg.norm(unit(R @ a) - unit(b))
        if err > 1e-9:
            check("rotation_between", False, "%.2e" % err)
            break
    else:
        check("rotation_between maps a onto b", True)
    for _ in range(200):
        v = rng.normal(size=3) * rng.uniform(0, math.pi)
        back = axis_angle_from_matrix(matrix_from_axis_angle(v))
        if np.linalg.norm(matrix_from_axis_angle(back)
                          - matrix_from_axis_angle(v)) > 1e-8:
            check("axis-angle round trip", False)
            break
    else:
        check("axis-angle round trip", True)
    R = rotation_between([1, 0, 0], [-1, 0, 0])
    check("180 degree case", np.linalg.norm(R @ np.array([1., 0, 0])
                                            + np.array([1., 0, 0])) < 1e-9)
    half = scale_rotation(matrix_from_axis_angle([0, 1.2, 0]), 0.5)
    check("fractional rotation composes", np.linalg.norm(
        half @ half - matrix_from_axis_angle([0, 1.2, 0])) < 1e-9)

    print("\n-- pose retarget (mock body)")
    # take a known pose, generate keypoints from it, retarget, compare
    truth = {}
    for j in (1, 2, 4, 5, 12, 16, 17, 18, 19):
        truth[j] = matrix_from_axis_angle(rng.normal(size=3) * 0.5)
    truth[0] = matrix_from_axis_angle(np.array([0.1, 0.6, -0.2]))
    posed, globals_ = fk_joints(rest, parents, truth)

    def landmark(name):
        """Face landmarks are skinned to the head rather than posed by the
        tree, so carry them with the head's global transform."""
        h = index_of["head"]
        i = index_of[name]
        if i < len(parents):
            return posed[i]
        return posed[h] + globals_[h] @ (rest[i] - rest[h])

    points = {k: landmark(s) for k, s in KEYPOINT_TO_SMPLX.items()}
    points["neck"] = 0.5 * (points["r_shoulder"] + points["l_shoulder"])

    go, body_pose, rots = retarget_pose(rest, parents, index_of, points)
    solved = {0: matrix_from_axis_angle(go)}
    for j in range(1, SMPLX_BODY_JOINTS + 1):
        solved[j] = matrix_from_axis_angle(body_pose[j - 1])
    out, _ = fk_joints(rest, parents, solved)
    out = out + (points["l_hip"] - out[index_of["left_hip"]])

    worst, worst_name = 0.0, ""
    for kp, name in KEYPOINT_TO_SMPLX.items():
        if name in ("nose", "right_eye", "left_eye", "right_ear", "left_ear"):
            continue
        if index_of[name] >= len(parents):
            continue
        err = float(np.linalg.norm(out[index_of[name]] - points[kp]))
        if err > worst:
            worst, worst_name = err, name
    check("every driven joint lands on its keypoint", worst < 1e-6,
          "worst %.2e m at %s" % (worst, worst_name))
    check("body_pose has the right shape", body_pose.shape == (21, 3))
    check("landmark rows beyond the parent array are tolerated",
          len(rest) > len(parents) and len(fk_joints(rest, parents, {})[0])
          == len(parents), "%d joints, %d parents" % (len(rest), len(parents)))
    check("wrists left unrotated (no hand keypoints)",
          np.linalg.norm(body_pose[19]) < 1e-12
          and np.linalg.norm(body_pose[20]) < 1e-12)

    print("\n-- coordinate frames")
    editor_frame = frame_from(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    check("editor frame: figure's left is +X",
          np.allclose(editor_frame[:, 0], [1.0, 0.0, 0.0]))
    mock_frame = frame_from(rest[index_of["nose"]] - rest[index_of["head"]],
                            rest[index_of["neck"]]
                            - 0.5 * (rest[index_of["left_hip"]]
                                     + rest[index_of["right_hip"]]))
    to_left = rest[index_of["left_hip"]] - rest[index_of["right_hip"]]
    check("frame's left agrees with the skeleton's own hip order",
          float(np.dot(mock_frame[:, 0], to_left)) > 0)
    R = mock_frame @ editor_frame.T
    check("world mapping is a rotation, not a reflection",
          abs(np.linalg.det(R) - 1.0) < 1e-9
          and np.allclose(R @ R.T, np.eye(3), atol=1e-9))

    print("\n-- shape fit")
    template = rest[:25].copy()
    jac = rng.normal(size=(25, 3, 4)) * 0.01
    true_betas = np.array([1.5, -0.8, 0.4, 0.0])
    shaped = template + np.einsum("jdb,b->jd", jac, true_betas)
    idx25 = {n: i for i, n in enumerate(SMPLX_JOINT_NAMES)}
    targets = {}
    for p, c in (("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                 ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
                 ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                 ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                 ("pelvis", "spine1"), ("spine3", "neck")):
        targets[(p, c)] = float(np.linalg.norm(shaped[idx25[c]] - shaped[idx25[p]]))
    fitted = fit_betas(template, jac, parents, targets, idx25, regulariser=1e-4)
    J = template + np.einsum("jdb,b->jd", jac, fitted)
    err = max(abs(np.linalg.norm(J[idx25[c]] - J[idx25[p]]) - L)
              for (p, c), L in targets.items())
    check("bone lengths recovered", err < 1e-4, "worst %.2e m" % err)
    check("betas stay in range", np.all(np.abs(fitted) <= 4.0))

    print("\n-- rasteriser")
    # a sphere of known radius: compare rasterised depth with the analytic hit
    lat, lon, R0, cx, cy, cz = 96, 192, 60.0, 100.0, 100.0, 300.0
    verts, faces = [], []
    for i in range(lat + 1):
        th = math.pi * i / lat
        for k in range(lon):
            ph = 2 * math.pi * k / lon
            verts.append((cx + R0 * math.sin(th) * math.cos(ph),
                          cy + R0 * math.cos(th),
                          cz + R0 * math.sin(th) * math.sin(ph)))
    for i in range(lat):
        for k in range(lon):
            a = i * lon + k
            b = i * lon + (k + 1) % lon
            faces.append((a, b, a + lon))
            faces.append((b, b + lon, a + lon))
    verts = np.array(verts)
    faces = np.array(faces)
    import time
    t = time.time()
    z = rasterize_depth(verts, faces, 200, 200)
    elapsed = time.time() - t
    ys, xs = np.mgrid[0:200, 0:200]
    d2 = (xs + 0.5 - cx) ** 2 + (ys + 0.5 - cy) ** 2
    exact = np.where(d2 < R0 ** 2, cz - np.sqrt(np.maximum(R0 ** 2 - d2, 0)),
                     np.inf)
    inner = np.isfinite(exact) & (d2 < (R0 - 3) ** 2)
    err = np.abs(z[inner] - exact[inner]).max()
    check("sphere depth matches the analytic surface", err < 0.6,
          "worst %.3f px over %d faces in %.2fs" % (err, len(faces), elapsed))
    filled = np.isfinite(z).sum()
    expect = math.pi * R0 ** 2
    check("silhouette area correct", abs(filled - expect) / expect < 0.02,
          "%d vs %.0f px" % (filled, expect))
    check("nothing rendered outside the sphere",
          not np.isfinite(z[d2 > (R0 + 2) ** 2]).any())

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
