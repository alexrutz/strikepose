#!/usr/bin/env python3
"""The eighteen keypoints, the limbs between them, and the pose they hold.

The invariant this file exists to keep: a bone may not change length unless
the caller explicitly asks. `move_joint` takes `stretch=False`, and a target
at the wrong distance aims the bone rather than resizing it.
"""

from __future__ import annotations

import math

from anthro import (DEFAULT_PRESET, build_rest_points, merge_body,
                    preset_params)
from vecmath import (IDENTITY, any_perpendicular, matvec, rotation_between,
                     vadd, vcross,
                     vdot, vlen, vmul, vnorm, vsub)


# Verified against lllyasviel/ControlNet annotator/openpose/util.py
# ---------------------------------------------------------------------------

KEYPOINT_NAMES = [
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]

# (parent, child) pairs in canonical OpenPose limb order; index -> COLORS index
LIMB_SEQ = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

# Viewport-only styling so people can be told apart. Blending towards a
# signature colour was not enough: a tint that already appears in the OpenPose
# ramp leaves that keypoint unchanged, so two figures could share a colour.
# A hue rotation combined with saturation and value scaling moves every
# keypoint instead. (hue turns, saturation scale, value scale); the first
# figure is the untouched palette.
# Signature colour per figure, applied to the body preview. The bones span the
# whole hue circle whatever you do to them, so recolouring them alone never
# reads at a glance; the body is a large flat area where a tint is obvious.
# Objects are one flat neutral, a little cooler than any figure tint, so a
# prop never reads as another person in the viewport.
PROP_TINT = (196, 202, 214)

FIGURE_BODY_TINTS = ((235, 235, 240),     # 1: neutral
                     (255, 165, 80),      # 2: orange
                     (105, 205, 255),     # 3: cyan
                     (150, 240, 115),     # 4: green
                     (250, 140, 230),     # 5: pink
                     (240, 225, 110),     # 6: yellow
                     (175, 160, 255))     # 7: periwinkle

# Ordered most-distinct-first, so a scene with two or three people gets the
# biggest separation: pale and dark read apart at a glance far better than a
# hue shift does.
# Kept gentle: the body tint above already identifies each figure, so the bones
# only need a secondary nudge, which matters when the body preview is off (B).
# Pushing them harder makes the keypoint colours hard to read.
FIGURE_STYLES = ((0.00, 1.00, 1.00),      # 1: the untouched OpenPose palette
                 (0.00, 0.70, 1.00),      # 2: softer
                 (-0.07, 1.00, 0.84),     # 3: cool, deeper
                 (0.09, 1.00, 1.00),      # 4: warm
                 (0.16, 0.72, 1.00),      # 5: soft warm
                 (-0.16, 1.00, 0.92),     # 6: cool
                 (0.24, 0.85, 0.80))      # 7: muted

COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

ROOT = 1  # neck
PARENT = {child: parent for parent, child in LIMB_SEQ}
CHILDREN = {}
for _p, _c in LIMB_SEQ:
    CHILDREN.setdefault(_p, []).append(_c)

ADJACENCY = {}
for _p, _c in LIMB_SEQ:
    ADJACENCY.setdefault(_p, []).append(_c)
    ADJACENCY.setdefault(_c, []).append(_p)

MIRROR_OF = {}

_REROOT_CACHE = {}


def reroot(anchor):
    """Kinematic tree re-hung from an arbitrary joint.

    The skeleton is a tree, so any joint can serve as its root: breadth-first
    from the anchor reverses the parent links along the path back to the neck
    and leaves everything else alone. Anchoring a knee therefore makes the hip
    a child of the knee, and dragging the hip swings the whole upper body about
    that knee instead of the other way round.
    """
    if anchor in _REROOT_CACHE:
        return _REROOT_CACHE[anchor]
    parents = {anchor: -1}
    order, queue = [anchor], [anchor]
    while queue:
        joint = queue.pop(0)
        for neighbour in ADJACENCY.get(joint, ()):
            if neighbour not in parents:
                parents[neighbour] = joint
                order.append(neighbour)
                queue.append(neighbour)
    children = {}
    for child, parent in parents.items():
        if parent >= 0:
            children.setdefault(parent, []).append(child)
    _REROOT_CACHE[anchor] = (parents, children, order)
    return _REROOT_CACHE[anchor]


GIRDLE = (2, 5, 8, 11)      # shoulders and hips, re-seated on preset change
MIRROR_PAIRS = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (14, 15), (16, 17)]
for _a, _b in MIRROR_PAIRS:
    MIRROR_OF[_a] = _b
    MIRROR_OF[_b] = _a


# ---------------------------------------------------------------------------

class Skeleton:
    def __init__(self, body=None):
        self.body = merge_body(body or preset_params(DEFAULT_PRESET))
        self.body_scale = 1.0
        rest = build_rest_points(self.body)
        self.points = [rest[n] for n in KEYPOINT_NAMES]
        self.visible = [True] * len(KEYPOINT_NAMES)
        self.lengths = {c: vlen(vsub(self.points[c], self.points[p]))
                        for p, c in LIMB_SEQ}
        self.anchors = []           # 0: neck-rooted, 1: pivot, 2: hinge axis
        self.assets = []            # rigged-mesh add-ons on the same armature
        # {slot: preset} of hair and clothing; empty is bare. Normalised by
        # wearables.clean on the way in rather than validated here, so this
        # module needs nothing from wearables at import time - wearables
        # imports the geometry helpers from this one.
        self.outfit = {}

    @staticmethod
    def torso_frame(pts):
        """Orthonormal (side, up, facing) built from the posed torso."""
        side = vnorm(vsub(pts[5], pts[2]))
        sh_mid = vmul(vadd(pts[2], pts[5]), 0.5)
        hip_mid = vmul(vadd(pts[8], pts[11]), 0.5)
        up_raw = vnorm(vsub(sh_mid, hip_mid))
        if vlen(side) < 1e-6 or vlen(up_raw) < 1e-6:
            return None
        facing = vnorm(vcross(side, up_raw))
        if vlen(facing) < 1e-6:
            return None
        return side, vnorm(vcross(facing, side)), facing

    def apply_body(self, body):
        """Swap in new proportions while keeping the current pose: every bone
        keeps its direction and takes the new length, walking down the tree
        from the root."""
        old = list(self.points)
        rest = build_rest_points(body)
        lengths = {c: vlen(vsub(rest[KEYPOINT_NAMES[c]], rest[KEYPOINT_NAMES[p]]))
                   for p, c in LIMB_SEQ}
        frame = self.torso_frame(old)
        for parent, child in LIMB_SEQ:      # LIMB_SEQ is ordered root-first
            if child in GIRDLE and frame is not None:
                # shoulders and hips are re-seated in the torso's own frame,
                # otherwise a wider preset would only lengthen the bone and
                # leave the width unchanged
                o = vsub(rest[KEYPOINT_NAMES[child]], rest["neck"])
                self.points[child] = vadd(
                    self.points[ROOT],
                    vadd(vadd(vmul(frame[0], o[0]), vmul(frame[1], o[1])),
                         vmul(frame[2], o[2])))
                continue
            d = vsub(old[child], old[parent])
            if vlen(d) < 1e-9:
                d = vsub(rest[KEYPOINT_NAMES[child]], rest[KEYPOINT_NAMES[parent]])
            self.points[child] = vadd(self.points[parent],
                                      vmul(vnorm(d), lengths[child]))
        self.lengths = lengths
        self.body = merge_body(body)
        self.body_scale = 1.0

    # -- anchors -----------------------------------------------------------
    @property
    def anchor(self):
        """The joint the tree hangs from. With two anchors the first one holds
        the tree together while the pair defines the hinge axis."""
        return self.anchors[0] if self.anchors else ROOT

    @anchor.setter
    def anchor(self, value):
        self.anchors = [] if value == ROOT else [value]

    @property
    def hinged(self):
        return len(self.anchors) == 2

    def hinge_axis(self):
        a, b = self.points[self.anchors[0]], self.points[self.anchors[1]]
        return a, vnorm(vsub(b, a))

    def hinge_set(self, joint):
        """Joints that swing with `joint`: everything reachable from it without
        passing through an anchor. Anchoring both knees and dragging the neck
        therefore carries the torso and thighs, while the shins and feet on the
        far side of the anchors stay put."""
        blocked = set(self.anchors)
        seen, stack = {joint}, [joint]
        while stack:
            current = stack.pop()
            for neighbour in ADJACENCY.get(current, ()):
                if neighbour in blocked or neighbour in seen:
                    continue
                seen.add(neighbour)
                stack.append(neighbour)
        return seen

    def rotate_about_axis(self, joints, origin, axis, angle):
        c, s_ = math.cos(angle), math.sin(angle)
        x, y, z = axis
        K = ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))
        KK = tuple(tuple(vdot(K[i], (K[0][j], K[1][j], K[2][j]))
                         for j in range(3)) for i in range(3))
        R = tuple(tuple(IDENTITY[i][j] + K[i][j] * s_ + KK[i][j] * (1.0 - c)
                        for j in range(3)) for i in range(3))
        for j in joints:
            self.points[j] = vadd(origin, matvec(R, vsub(self.points[j], origin)))

    def hinge_circle(self, joint):
        """Centre and the two radius vectors of the circle `joint` travels on.

        P(t) = centre + a*cos(t) + b*sin(t), with P(0) the joint's position now.
        """
        origin, axis = self.hinge_axis()
        if vlen(axis) < 1e-9:
            return None
        offset = vsub(self.points[joint], origin)
        along = vmul(axis, vdot(offset, axis))
        a = vsub(offset, along)
        if vlen(a) < 1e-6:
            return None
        return vadd(origin, along), a, vcross(axis, a)

    def hinge_angle(self, joint, cursor, project, samples=120):
        """Angle about the hinge that brings `joint` nearest the cursor.

        Solved in screen space rather than in the plane perpendicular to the
        axis: that plane is edge-on whenever the axis runs across the view (two
        knees seen from the front, say), and mapping the cursor into it then
        yields no angle at all. The joint's path projects to an ellipse, so
        sample it coarsely and refine.
        """
        circle = self.hinge_circle(joint)
        if circle is None:
            return None
        centre, a, b = circle

        def at(t):
            ct, st = math.cos(t), math.sin(t)
            return vadd(centre, vadd(vmul(a, ct), vmul(b, st)))

        def distance(t):
            sx, sy = project(at(t))[:2]
            return (sx - cursor[0]) ** 2 + (sy - cursor[1]) ** 2

        best, best_d = 0.0, distance(0.0)
        for i in range(1, samples):
            t = 2.0 * math.pi * i / samples
            d = distance(t)
            if d < best_d:
                best, best_d = t, d
        step = 2.0 * math.pi / samples
        for _ in range(20):
            step *= 0.5
            for candidate in (best - step, best + step):
                d = distance(candidate)
                if d < best_d:
                    best, best_d = candidate, d
        return best

    def hinge_screen_extent(self, joint, project):
        """How wide the joint's circle appears, so an edge-on hinge can be
        reported instead of silently snapping between two positions."""
        circle = self.hinge_circle(joint)
        if circle is None:
            return 0.0
        centre, a, b = circle
        points = [project(vadd(centre, vadd(vmul(a, math.cos(t)),
                                            vmul(b, math.sin(t)))))[:2]
                  for t in (i * math.pi / 6.0 for i in range(12))]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(max(xs) - min(xs), max(ys) - min(ys))

    def hinge_spin(self, joint, angle):
        origin, axis = self.hinge_axis()
        self.rotate_about_axis(self.hinge_set(joint), origin, axis, angle)

    def hinge_drag(self, joint, target_point):
        """Swing `joint` about the axis joining the two anchors.

        A rotation about a fixed axis is an isometry and both anchors lie on
        it, so every bone length survives exactly, including the ones crossing
        from the moving side to the fixed side.
        """
        if joint in self.anchors:
            self.translate(vsub(target_point, self.points[joint]))
            return True
        origin, axis = self.hinge_axis()
        if vlen(axis) < 1e-9:
            return False
        here = vsub(self.points[joint], origin)
        there = vsub(target_point, origin)
        here = vsub(here, vmul(axis, vdot(here, axis)))
        there = vsub(there, vmul(axis, vdot(there, axis)))
        if vlen(here) < 1e-6 or vlen(there) < 1e-6:
            return False        # the joint sits on the axis; nothing to swing
        u, v = vnorm(here), vnorm(there)
        angle = math.atan2(vdot(vcross(u, v), axis),
                           max(-1.0, min(1.0, vdot(u, v))))
        self.rotate_about_axis(self.hinge_set(joint), origin, axis, angle)
        return True

    # -- topology ----------------------------------------------------------
    def tree(self, anchor=None):
        return reroot(self.anchor if anchor is None else anchor)

    def parent_of(self, idx, anchor=None):
        return self.tree(anchor)[0].get(idx, -1)

    def bone_length(self, a, b):
        """Bones are stored by their child in the neck-rooted tree; look them
        up by unordered pair so re-rooting does not lose them."""
        return self.lengths[b] if PARENT.get(b) == a else self.lengths[a]

    def set_bone_length(self, a, b, length):
        key = b if PARENT.get(b) == a else a
        self.lengths[key] = length

    def subtree(self, idx, anchor=None):
        children = self.tree(anchor)[1]
        out, stack = [], [idx]
        while stack:
            j = stack.pop()
            out.append(j)
            stack.extend(children.get(j, ()))
        return out

    # -- symmetry ----------------------------------------------------------
    def sagittal_plane(self):
        """Origin and normal of the body's own plane of symmetry."""
        normal = vnorm(vsub(self.points[5], self.points[2]))
        if vlen(normal) < 1e-6:
            normal = vnorm(vsub(self.points[11], self.points[8]))
        if vlen(normal) < 1e-6:
            normal = (1.0, 0.0, 0.0)
        origin = vmul(vadd(vadd(self.points[2], self.points[5]),
                           vadd(self.points[8], self.points[11])), 0.25)
        return origin, normal

    @staticmethod
    def reflect(point, plane):
        origin, normal = plane
        return vsub(point, vmul(normal, 2.0 * vdot(vsub(point, origin), normal)))

    # -- editing -----------------------------------------------------------
    def translate(self, delta, indices=None):
        for j in (indices if indices is not None else range(len(self.points))):
            self.points[j] = vadd(self.points[j], delta)

    def move_joint(self, idx, target_point, anchor=None, stretch=False):
        """Aim the bone at target_point, carrying the downstream chain along.

        By default the bone keeps its length and only rotates, so a target at
        the wrong distance aims the limb rather than resizing it. Only the
        explicit free-length drag passes stretch=True. Silently resizing was
        how symmetric editing used to stretch a figure a little on every drag.
        """
        parent_idx = self.parent_of(idx, anchor)
        if parent_idx < 0:
            self.translate(vsub(target_point, self.points[idx]))
            return
        parent = self.points[parent_idx]
        old_dir = vsub(self.points[idx], parent)
        new_dir = vsub(target_point, parent)
        if vlen(new_dir) < 1e-9:
            return
        rot = rotation_between(old_dir, new_dir)
        chain = self.subtree(idx, anchor)
        for j in chain:
            self.points[j] = vadd(parent, matvec(rot, vsub(self.points[j], parent)))
        # rotation preserves length; translate the chain for any length change
        fix = vsub(target_point, self.points[idx])
        if stretch and vlen(fix) > 1e-9:
            self.translate(fix, chain)
            self.set_bone_length(parent_idx, idx,
                                 vlen(vsub(target_point, parent)))

    def solve_drag(self, idx, plane_offset, fwd, sign, free_length=False,
                   anchor=None):
        """Map an in-plane drop offset onto the constraint sphere.

        plane_offset : world vector from the parent joint, lying in the view
                       plane, taken from the cursor position.
        fwd          : unit view direction (away from the camera).
        sign         : +1 keeps the joint behind the view plane, -1 in front.
        """
        parent_idx = self.parent_of(idx, anchor)
        if parent_idx < 0:
            return vadd(self.points[idx], plane_offset)
        parent = self.points[parent_idx]
        if free_length:
            return vadd(parent, plane_offset)
        length = self.bone_length(parent_idx, idx)
        d = vlen(plane_offset)
        if d >= length:
            if d < 1e-9:
                return self.points[idx]
            return vadd(parent, vmul(plane_offset, length / d))
        depth = math.sqrt(max(0.0, length * length - d * d))
        return vadd(parent, vadd(plane_offset, vmul(fwd, sign * depth)))

    def flip_depth(self, idx, fwd, anchor=None):
        """Mirror one bone through the view plane, chain included."""
        parent_idx = self.parent_of(idx, anchor)
        if parent_idx < 0:
            return
        parent = self.points[parent_idx]
        old = vsub(self.points[idx], parent)
        new = vsub(old, vmul(fwd, 2.0 * vdot(old, fwd)))
        self.move_joint(idx, vadd(parent, new), anchor)

    def mirror_x(self):
        pts = [(-x, y, z) for (x, y, z) in self.points]
        vis = list(self.visible)
        for a, b in MIRROR_PAIRS:
            pts[a], pts[b] = pts[b], pts[a]
            vis[a], vis[b] = vis[b], vis[a]
        self.points, self.visible = pts, vis
        self.lengths = {c: vlen(vsub(self.points[c], self.points[p]))
                        for p, c in LIMB_SEQ}

    def scale(self, factor):
        root = self.points[ROOT]
        self.points = [vadd(root, vmul(vsub(p, root), factor)) for p in self.points]
        self.lengths = {k: v * factor for k, v in self.lengths.items()}
        self.body_scale *= factor       # keep the depth volumes in proportion

    # -- state -------------------------------------------------------------
    def snapshot(self):
        return (list(self.points), list(self.visible), dict(self.lengths),
                dict(self.body), self.body_scale, list(self.anchors),
                list(self.assets), dict(self.outfit))

    def restore(self, snap):
        self.points, self.visible = list(snap[0]), list(snap[1])
        self.lengths = dict(snap[2])
        if len(snap) > 4:
            self.body, self.body_scale = merge_body(snap[3]), snap[4]
        if len(snap) > 5:
            self.anchors = list(snap[5]) if isinstance(snap[5], list) \
                else ([] if snap[5] == ROOT else [snap[5]])
        if len(snap) > 6:
            self.assets = list(snap[6])
        if len(snap) > 7:
            self.outfit = dict(snap[7])
