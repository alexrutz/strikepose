#!/usr/bin/env python3
"""The pose, as the Anny armature's own joint angles.

This is the editor's document. It is a local rotation per bone on the 104-bone
MakeHuman skeleton every body in `bodies/` is built on, and nothing else - the
thing you drag in the viewport is a bone of that armature, the thing the depth
map is skinned from is the same array of rotations, and there is no
intermediate description of the pose for the two to disagree about.

What this replaced was an eighteen-point OpenPose skeleton that you dragged and
a solver that aimed the rig's bones at it. That was two models of one pose, and
the second was strictly poorer than the first: eighteen points cannot say which
way a palm faces, cannot roll a forearm, cannot bend a finger, and cannot place
a collarbone - so hands and feet needed two hand-set angles each bolted on the
side, the rest of the rig's 104 bones were unreachable, and every one of them
had to be inferred from a description that did not contain them.

Posing the rig directly needs none of that. A bone is rotated about its own
head; everything below it follows, because that is what a local rotation
composed down the tree means. No bone can change length, whatever a drag does,
because a rotation cannot change one - the invariant that `check_lengths` used
to guard at runtime is now structural.

The keypoints still exist, but only as OUTPUT: `mesh_backend.keypoints_of`
reads them off a posed rig for a pose PNG. That direction is sound, and it is
the only direction that ever was.
"""

from __future__ import annotations

import math

import numpy as np

from mesh_backend import (ANNY_BONES, CM_PER_METRE, matrix_from_axis_angle,
                          rotation_between, unit)


# Bones the viewport offers as handles, in the order a panel should list them.
# The rig has 104 and every one of them can be dragged; this is only what gets
# a name in the UI, because "upperarm01.L" is not what a person calls an arm.
FRIENDLY = {
    "root": "root", "spine05": "hips", "spine03": "chest", "spine01": "waist",
    "neck01": "neck", "head": "head",
    "clavicle.L": "left collar", "clavicle.R": "right collar",
    "upperarm01.L": "left upper arm", "upperarm01.R": "right upper arm",
    "lowerarm01.L": "left forearm", "lowerarm01.R": "right forearm",
    "wrist.L": "left hand", "wrist.R": "right hand",
    "upperleg01.L": "left thigh", "upperleg01.R": "right thigh",
    "lowerleg01.L": "left shin", "lowerleg01.R": "right shin",
    "foot.L": "left foot", "foot.R": "right foot",
    "toe1-1.L": "left toes", "toe1-1.R": "right toes",
}

# MakeHuman splits each limb segment in two and uses the second half to
# distribute ROLL, so the skin does not wring itself into a candy wrapper at
# the joint. They are not joints: a humerus does not hinge 8 cm below the
# shoulder, and swinging `upperarm02` to follow a dragged elbow kinks the arm
# in the middle of the bone.
#
# Named, not detected. The obvious rule - a bone continuing its parent's name
# one number higher - also catches every finger phalanx, every toe joint and
# the two lower cervicals, which ARE joints; and the obvious geometric rule,
# near-collinear with its parent, puts `upperleg02` (12.9 deg) on the same
# side as `finger2-2` (12.6). There is one rig and these are its eight twist
# bones, so the honest thing is to say which.
TWIST = tuple(stem + side
              for stem in ("upperarm02", "lowerarm02", "upperleg02",
                           "lowerleg02")
              for side in (".L", ".R"))

# The command vocabulary's names for parts of the body, against the rig.
# (the bone that swings, the joint at the far end of it) - the far joint is
# what `point` aims and what a bend measures its direction by.
SEGMENTS = {
    "l_upper_arm": ("upperarm01.L", "lowerarm01.L"),
    "l_forearm":   ("lowerarm01.L", "wrist.L"),
    "r_upper_arm": ("upperarm01.R", "lowerarm01.R"),
    "r_forearm":   ("lowerarm01.R", "wrist.R"),
    "l_thigh":     ("upperleg01.L", "lowerleg01.L"),
    "l_shin":      ("lowerleg01.L", "foot.L"),
    "r_thigh":     ("upperleg01.R", "lowerleg01.R"),
    "r_shin":      ("lowerleg01.R", "foot.R"),
    "l_hand":      ("wrist.L", "metacarpal3.L"),
    "r_hand":      ("wrist.R", "metacarpal3.R"),
    "l_foot":      ("foot.L", "toe3-1.L"),
    "r_foot":      ("foot.R", "toe3-1.R"),
    "torso":       ("spine05", "neck01"),
    "neck":        ("neck01", "head"),
}

# What `bend` can flex: (the bone above, the bone that swings, the joint at
# the far end, which way positive degrees carries it). The pivot bone's head
# IS the joint, so rotating that bone rotates about the joint - no separate
# pivot point and no subtree list, which is what the keypoint version needed.
BEND = {
    "l_elbow":    ("upperarm01.L", "lowerarm01.L", "wrist.L", "facing"),
    "r_elbow":    ("upperarm01.R", "lowerarm01.R", "wrist.R", "facing"),
    "l_knee":     ("upperleg01.L", "lowerleg01.L", "foot.L", "-facing"),
    "r_knee":     ("upperleg01.R", "lowerleg01.R", "foot.R", "-facing"),
    "l_shoulder": (None, "upperarm01.L", "lowerarm01.L", "facing"),
    "r_shoulder": (None, "upperarm01.R", "lowerarm01.R", "facing"),
    "l_hip":      (None, "upperleg01.L", "lowerleg01.L", "facing"),
    "r_hip":      (None, "upperleg01.R", "lowerleg01.R", "facing"),
    "l_wrist":    ("lowerarm01.L", "wrist.L", "metacarpal3.L", "facing"),
    "r_wrist":    ("lowerarm01.R", "wrist.R", "metacarpal3.R", "facing"),
    "l_ankle":    ("lowerleg01.L", "foot.L", "toe3-1.L", "facing"),
    "r_ankle":    ("lowerleg01.R", "foot.R", "toe3-1.R", "facing"),
    "neck":       ("spine01", "neck01", "head", "facing"),
    "waist":      (None, "spine05", "neck01", "facing"),
}

# The bone the torso leans from. Everything above it follows because it is
# their parent, and the legs do not because they hang off `root` beside it -
# so a lean needs no list of what it carries. The keypoint version had one,
# `ABOVE_WAIST`, and the two bugs its length check caught on the randomizer's
# first run were both about rotating a list of points about the wrong axis.
# Neither is expressible here: a bone rotation cannot move a bone it is not
# above, and cannot change a length at all.
WAIST = "spine05"

# Bones a drag should refuse to move, because moving them is not posing.
# `root` carries the whole figure: turning it is the "turn figure" control and
# dragging a limb must never reach it.
LOCKED = ("root",)


class RigPose:
    """One figure: the armature, its joint angles, and where it stands.

    `local[j]` is bone j's rotation RELATIVE TO ITS REST ORIENTATION, expressed
    in its parent's frame - the same convention `mesh_backend.pose_globals`
    composes, so a pose built here skins through exactly the same code.
    """

    def __init__(self, mesh, stature=None):
        self.mesh = mesh
        self.names = list(mesh["joint_names"])
        self.index = {name: i for i, name in enumerate(self.names)}
        self.parents = np.asarray(mesh["parents"], int)
        n = len(self.names)

        # Anny bodies are authored in metres at the figure's own stature, so
        # sizing the rig is a unit conversion. A stature is honoured anyway,
        # which keeps a body that is not from the set honest.
        height = float(mesh["vertices"][:, 1].max() -
                       mesh["vertices"][:, 1].min())
        self.scale = (float(stature) / max(1e-9, height) if stature
                      else CM_PER_METRE)
        self.rest = np.asarray(mesh["rest_position"], float) * self.scale

        self.local = np.tile(np.eye(3), (n, 1, 1))
        self.offset = np.zeros(3)
        self.stature = stature

        self.order = self._topological()
        self.kids = {}
        for c in range(n):
            p = int(self.parents[c])
            if p >= 0:
                self.kids.setdefault(p, []).append(c)
        self.roots = [j for j in range(n) if self.parents[j] < 0]
        self._solved = None

    # -- structure ---------------------------------------------------------

    def _topological(self):
        depth = []
        for j in range(len(self.names)):
            d, walk = 0, j
            while self.parents[walk] >= 0:
                walk = int(self.parents[walk])
                d += 1
            depth.append(d)
        return sorted(range(len(self.names)), key=lambda j: depth[j])

    def bone(self, name):
        """Index of a bone by rig name or by the friendly name the UI shows."""
        if name in self.index:
            return self.index[name]
        for rig, friendly in FRIENDLY.items():
            if friendly == name and rig in self.index:
                return self.index[rig]
        if name in ANNY_BONES and ANNY_BONES[name] in self.index:
            return self.index[ANNY_BONES[name]]
        raise KeyError("no bone %r in this rig" % (name,))

    def label(self, j):
        return FRIENDLY.get(self.names[j], self.names[j])

    def locked(self, j):
        return self.names[j] in LOCKED

    def subtree(self, j):
        out, stack = [], [j]
        while stack:
            k = stack.pop()
            out.append(k)
            stack.extend(self.kids.get(k, ()))
        return out

    # -- forward kinematics ------------------------------------------------

    def solve(self):
        """Global rotation and position per bone. Cached until an edit."""
        if self._solved is not None:
            return self._solved
        n = len(self.names)
        Q = np.zeros((n, 3, 3))
        q = np.zeros((n, 3))
        for j in self.order:
            p = int(self.parents[j])
            if p < 0:
                Q[j] = self.local[j]
                q[j] = self.offset + self.rest[j]
            else:
                Q[j] = Q[p] @ self.local[j]
                q[j] = q[p] + Q[p] @ (self.rest[j] - self.rest[p])
        self._solved = (Q, q)
        return self._solved

    def touched(self):
        self._solved = None

    def positions(self):
        return self.solve()[1]

    def segments(self):
        """Every drawable bone as (parent index, index, head, tail)."""
        q = self.positions()
        return [(int(self.parents[j]), j, q[int(self.parents[j])], q[j])
                for j in range(len(self.names)) if self.parents[j] >= 0]

    # -- editing -----------------------------------------------------------

    def _apply(self, j, delta):
        """Compose a WORLD-space rotation onto bone j, about its own head.

        Expressed as a change of j's local rotation, so everything below it
        inherits the motion - which is what makes a shoulder carry the whole
        arm, and a spine bone the whole upper body.
        """
        Q, _q = self.solve()
        p = int(self.parents[j])
        above = Q[p] if p >= 0 else np.eye(3)
        self.local[j] = above.T @ delta @ above @ self.local[j]
        self.touched()

    def rotate(self, j, axis, angle):
        """Turn bone j about a world axis through its own head."""
        if abs(angle) < 1e-12:
            return
        self._apply(j, matrix_from_axis_angle(unit(np.asarray(axis, float))
                                              * float(angle)))

    def aim(self, j, child, target):
        """Rotate bone j so that `child`'s head lands on `target`.

        This is the drag. The grabbed segment names both the bone that turns
        and the point that follows the cursor, so nothing has to guess which
        way a bone "points" - a question with no answer on a branch like the
        chest, which carries a spine and two collarbones.
        """
        _Q, q = self.solve()
        here = q[child] - q[j]
        goal = np.asarray(target, float) - q[j]
        if np.linalg.norm(here) < 1e-9 or np.linalg.norm(goal) < 1e-9:
            return
        self._apply(j, rotation_between(here, goal))

    def move_joint(self, idx, target):
        """Drag joint `idx` to `target`: the same call the editor always made.

        The bone that turns is the one ABOVE the joint you grabbed, so the
        gesture is identical to the one the keypoint skeleton had - grab an
        elbow, the upper arm swings. What changed is only which joints there
        are to grab: 104 of the body's own instead of eighteen inferred ones.

        A rotation is all that happens, so the bone cannot change length and
        there is no `stretch` to pass. The free-length drag is gone because
        the thing it was for - a skeleton whose proportions you could edit -
        is gone: these are a measured body's bones.
        """
        p = self.swing_above(idx)
        if p < 0 or self.locked(p):
            return
        self.aim(p, idx, target)

    def swing_above(self, idx):
        """The joint that should turn when `idx` is dragged.

        The bone directly above a joint is sometimes a twist bone, which is
        half a segment rather than a joint - so a dragged elbow walks up past
        `upperarm02.L` to the shoulder, and the whole humerus swings as the one
        rigid piece it is. Without this the arm folds in the middle of the
        bone, which the mesh shows as a crease and no drag should be able to
        produce.
        """
        p = int(self.parents[idx])
        while p >= 0 and self.names[p] in TWIST:
            p = int(self.parents[p])
        return p

    def bone_length(self, idx):
        """Distance from joint `idx` to its parent - the drag's constraint
        sphere, exactly as before."""
        p = int(self.parents[idx])
        if p < 0:
            return 0.0
        q = self.positions()
        return float(np.linalg.norm(q[idx] - q[p]))

    def roll_segment(self, idx, angle):
        """Roll the segment above `idx` - and roll it on the TWIST bone when
        the rig has one.

        This is what those bones are for. Rolling at the shoulder turns the
        deltoid with the arm and wrings the skin at the armpit; rolling at
        `upperarm02` leaves the shoulder alone and spreads the turn down the
        segment, which is how a real arm pronates.
        """
        p = int(self.parents[idx])
        if p < 0:
            return
        twist = [k for k in self.kids.get(self.swing_above(idx), ())
                 if self.names[k] in TWIST]
        self.spin(twist[0] if twist else p, angle)

    def spin(self, j, angle):
        """Roll bone j about its own axis - the twist a drag cannot reach.

        The axis is the bone's own direction, so the joint it points at does
        not move; only the roll of everything below changes. This is how a
        forearm pronates and an upper arm turns in its socket, neither of
        which any endpoint position can describe.
        """
        _Q, q = self.solve()
        kids = self.kids.get(j)
        axis = (q[kids[0]] - q[j]) if kids else None
        if axis is None or np.linalg.norm(axis) < 1e-9:
            p = int(self.parents[j])
            axis = q[j] - q[p] if p >= 0 else np.array([0.0, 1.0, 0.0])
        self.rotate(j, axis, angle)

    def reset(self, j, below=False):
        """Put bone j back to the angle the rig was authored with."""
        for k in (self.subtree(j) if below else [j]):
            self.local[k] = np.eye(3)
        self.touched()

    def reset_all(self):
        self.local[:] = np.eye(3)
        self.touched()

    def translate(self, delta):
        self.offset = self.offset + np.asarray(delta, float)
        self.touched()

    def turn(self, angle):
        """Turn the whole figure about its own vertical."""
        for j in self.roots:
            self.rotate(j, (0.0, 1.0, 0.0), angle)

    # -- placement ---------------------------------------------------------

    def lowest(self):
        """The lowest point of the SKINNED body, which is what stands on the
        floor. A rig joint is inside the body: the ankle bone sits several
        centimetres above the sole, so grounding on joints floats the figure.
        """
        from mesh_backend import skin_with
        # `skin_with`, which is what the export renders, and not `skin_mesh`
        # on the raw mesh. The two disagree: `skin_with` scales the rest pose,
        # the bind matrices and the vertices together, while this held joint
        # positions already in centimetres against a mesh still in metres. It
        # came out as a figure standing 2.2 cm into the floor in every depth
        # map, which the ground plane then cut through at the ankles - and
        # `stand()` reported 0.000, because it was measuring the other body.
        return float(skin_with(self.mesh, self.solution())[:, 1].min())

    def stand(self, y=0.0):
        """Drop the figure until its lowest point rests at `y`."""
        self.translate((0.0, y - self.lowest(), 0.0))

    # -- history and files -------------------------------------------------

    def snapshot(self):
        return (self.local.copy(), self.offset.copy())

    def restore(self, snap):
        local, offset = snap
        self.local = np.array(local, float)
        self.offset = np.array(offset, float)
        self.touched()

    def to_dict(self):
        """Only the bones that actually moved, by name.

        By name rather than by index so a scene survives a body set rebuilt
        with a different bone ORDER, and only the moved ones so a file says
        what was posed rather than restating the whole armature.
        """
        out = {}
        for j, name in enumerate(self.names):
            if not np.allclose(self.local[j], np.eye(3), atol=1e-12):
                out[name] = [round(float(v), 9)
                             for v in self.local[j].reshape(9)]
        return {"bones": out, "offset": [float(v) for v in self.offset]}

    def from_dict(self, data):
        self.reset_all()
        for name, flat in (data.get("bones") or {}).items():
            j = self.index.get(name)
            if j is not None and len(flat) == 9:
                self.local[j] = np.asarray(flat, float).reshape(3, 3)
        self.offset = np.asarray(data.get("offset") or [0.0, 0.0, 0.0], float)
        self.touched()

    # -- the export --------------------------------------------------------

    def solution(self):
        """The same shape `mesh_backend.pose_rig` used to return, so every
        consumer downstream - skinning, garments, assets - is untouched."""
        Q, q = self.solve()
        return {"scale": self.scale,
                "bones": {name: (Q[j], q[j])
                          for j, name in enumerate(self.names)}}




# Viewport colour per bone, by region. Not the OpenPose palette and not
# pretending to be: that palette is eighteen keypoints and a format, and this
# is a hundred and four bones of a real skeleton. It must never reach an
# export, and it cannot - the pose PNG is drawn from `keypoints_of`, which
# never sees these.
def bone_colour(name):
    """A bone's viewport colour as (r, g, b), by region and by side.

    Side is read off the `.L` / `.R` suffix rather than tabulated, which is the
    only thing that scales to an armature this size - and it means the two
    halves of the figure never disagree about which is which. RGB rather than
    a hex string because the viewport shades every colour by depth.
    """
    left = name.endswith(".L")
    stem = name[:-2] if name.endswith((".L", ".R")) else name
    low = stem.lower()
    if low.startswith(("finger", "metacarpal")):
        return (138, 222, 122) if left else (240, 192, 112)
    if low.startswith(("foot", "toe")):
        return (122, 200, 232) if left else (176, 138, 224)
    if low.startswith(("clavicle", "shoulder", "upperarm", "lowerarm",
                       "wrist", "elbow")):
        return (90, 200, 90) if left else (224, 160, 60)
    if low.startswith(("upperleg", "lowerleg", "knee")):
        return (60, 160, 224) if left else (154, 106, 224)
    if low.startswith(("head", "neck", "jaw", "eye", "tongue", "special",
                       "orbicularis", "levator", "risorius", "temporalis")):
        return (224, 96, 200)
    return (224, 90, 74)          # spine, pelvis, root - the trunk


def region_of(name):
    """Which coarse group a bone belongs to, for the panel's bone list."""
    left = name.endswith(".L")
    low = (name[:-2] if name.endswith((".L", ".R")) else name).lower()
    side = "left " if left else "right " if name.endswith(".R") else ""
    if low.startswith(("finger", "metacarpal")):
        return side + "hand"
    if low.startswith(("foot", "toe")):
        return side + "foot"
    if low.startswith(("clavicle", "shoulder", "upperarm", "lowerarm",
                       "wrist")):
        return side + "arm"
    if low.startswith(("upperleg", "lowerleg")):
        return side + "leg"
    if low.startswith(("head", "neck", "jaw", "eye", "tongue", "special",
                       "orbicularis", "levator", "risorius", "temporalis")):
        return "head"
    return "torso"


# ---------------------------------------------------------------------------

def mirror_map(names):
    """Bone -> its twin, by name.

    MakeHuman suffixes every paired bone `.L` / `.R`, so symmetric editing gets
    the whole armature rather than the eight keypoints that had a twin - a
    collarbone, a thumb and a big toe all mirror now, and none of them could
    before.
    """
    where = {n: i for i, n in enumerate(names)}
    out = {}
    for i, n in enumerate(names):
        twin = (n[:-2] + ".R" if n.endswith(".L") else
                n[:-2] + ".L" if n.endswith(".R") else None)
        if twin in where:
            out[i] = where[twin]
    return out


class Figure:
    """One posable person: the armature, plus who the body is.

    The surface the editor drags through - `points`, `parent_of`, `solve_drag`,
    `move_joint`, `snapshot` - is deliberately the same as the keypoint
    skeleton's was, because the gesture is the same gesture. Grab a joint, the
    bone above it swings, everything below comes along. Only the joints on
    offer changed: the body's own, all 104 of them, instead of eighteen
    inferred ones that had no collarbone, no spine to speak of and no fingers.
    """

    def __init__(self, mesh, body=None):
        self.body = dict(body or {})
        self.pose = RigPose(mesh, self.body.get("stature"))
        self.mesh = mesh
        self.mirror = mirror_map(self.pose.names)
        self._riders = None
        self.visible = [True] * len(self.pose.names)
        # Anchoring - pin a hand, move the body, let the arm follow - was a
        # re-rooting of the keypoint tree. A rig has one root and the bones
        # hang off it, so the same gesture is an IK solve rather than a
        # re-parent, and is not in yet. Kept as an empty set so the drawing
        # code that asks does not have to know.
        self.anchors = set()
        self.outfit = {}
        self.assets = []
        # What the sliders and the `hand`/`foot` commands last asked for, so a
        # panel can show it back. The POSE does not live here - it is in the
        # bones like every other rotation; this is only the control's reading.
        self.extremities = {}
        self.pose.stand()

    # -- what the editor reads --------------------------------------------

    @property
    def points(self):
        return [tuple(p) for p in self.pose.positions()]

    @property
    def names(self):
        return self.pose.names

    def label(self, idx):
        return self.pose.label(idx)

    def parent_of(self, idx, anchor=None):
        return int(self.pose.parents[idx])

    def subtree(self, idx, anchor=None):
        return self.pose.subtree(idx)

    def bone_length(self, a, b):
        q = self.pose.positions()
        return float(np.linalg.norm(np.asarray(q[a]) - np.asarray(q[b])))

    @property
    def lengths(self):
        q = self.pose.positions()
        return {j: float(np.linalg.norm(q[j] - q[int(self.pose.parents[j])]))
                for j in range(len(self.pose.names))
                if self.pose.parents[j] >= 0}

    def draggable(self, idx):
        """A joint is draggable when the bone above it is free to turn."""
        p = int(self.pose.parents[idx])
        return p >= 0 and not self.pose.locked(p)

    # -- the drag ----------------------------------------------------------

    def solve_drag(self, idx, plane_offset, fwd, sign, free_length=False,
                   anchor=None):
        """Map an in-plane cursor offset onto the constraint sphere.

        Unchanged from the keypoint editor except that `free_length` is gone:
        a bone is turned, never resized, so the sphere is the only place the
        joint can be. What used to be an Alt-drag that stretched a segment has
        nothing left to mean - these are a measured body's bones, and the one
        thing that may scale the rig is the figure's stature.
        """
        p = int(self.pose.parents[idx])
        if p < 0:
            return tuple(self.points[idx])
        q = self.pose.positions()
        parent = q[p]
        length = float(np.linalg.norm(q[idx] - parent))
        plane_offset = np.asarray(plane_offset, float)
        d = float(np.linalg.norm(plane_offset))
        if d >= length:
            if d < 1e-9:
                return tuple(q[idx])
            return tuple(parent + plane_offset * (length / d))
        depth = math.sqrt(max(0.0, length * length - d * d))
        return tuple(parent + plane_offset + np.asarray(fwd, float) * sign * depth)

    def move_joint(self, idx, target, anchor=None, stretch=False):
        self.pose.move_joint(idx, target)

    def mirror_drag(self, idx, target):
        """Give the twin limb the mirrored ANGLES, never a mirrored position.

        A reflected point only sits the right distance from the twin's parent
        while the figure is symmetric; reflecting the rotation is exact
        whatever it is doing. On a rig that is almost free: the body rests
        facing +Z with its left at +X, so the mirror is a sign flip on X, and
        a local rotation reflects as `S R S`.

        The whole limb below the bone that moved, not just that bone. Mirroring
        one leaves the twin's forearm and hand wherever a previous drag put
        them, so the two sides agree at the shoulder and nowhere else.

        The bone that moved is `swing_above`, not the joint's parent - the
        parent may be a twist bone the drag skipped, whose rotation is the
        identity, and mirroring THAT copies nothing at all while looking like
        it worked.
        """
        twin = self.mirror.get(idx)
        if twin is None:
            return
        moved = self.pose.swing_above(idx)
        if moved < 0:
            return
        S = np.diag([-1.0, 1.0, 1.0])
        for j in self.pose.subtree(moved):
            t = self.mirror.get(j)
            if t is not None:
                self.pose.local[t] = S @ self.pose.local[j] @ S
        self.pose.touched()

    def mirror_x(self):
        """Swap the figure left for right.

        Every bone takes its twin's mirrored angles, and the ones with no twin
        - the spine, the head - mirror in place. The body rests facing +Z with
        its left at +X, so the reflection is a sign flip on X.
        """
        S = np.diag([-1.0, 1.0, 1.0])
        swapped = np.empty_like(self.pose.local)
        for j in range(len(self.pose.names)):
            source = self.mirror.get(j, j)
            swapped[j] = S @ self.pose.local[source] @ S
        self.pose.local = swapped
        self.visible = [self.visible[self.mirror.get(j, j)]
                        for j in range(len(self.visible))]
        self.pose.touched()
        self.pose.stand()

    def flip_depth(self, idx, fwd, anchor=None):
        """Mirror one bone through the view plane - the ambiguity an
        orthographic drag leaves behind."""
        p = int(self.pose.parents[idx])
        if p < 0:
            return
        q = self.pose.positions()
        here = q[idx] - q[p]
        fwd = np.asarray(fwd, float)
        self.pose.aim(p, idx, q[p] + here - 2.0 * float(here @ fwd) * fwd)

    def grip(self, letter, amount, thumb=True):
        """Close a hand, 0 open to 1 a fist.

        Every finger bone turned about the joint it hangs from, which is what
        a hand does and what nothing before the armature could express: the
        `hand` command turns the WRIST, so a figure told to grip a mug got a
        flat palm aimed at it. There are fifteen bones in each set of fingers
        and they were all there the whole time.

        The three knuckles of a finger do not close equally - the middle joint
        travels furthest - so the curl is weighted down the chain, and the
        thumb opposes rather than curling with the rest.
        """
        amount = max(0.0, min(1.0, float(amount)))
        side = letter.upper()
        pose = self.pose
        for finger in range(2, 6):            # index to little
            for knuckle, share in enumerate((0.75, 1.0, 0.8)):
                name = "finger%d-%d.%s" % (finger, knuckle + 1, side)
                if name not in pose.index:
                    continue
                j = pose.bone(name)
                axis = self._curl_axis(j)
                pose.rotate(j, axis, math.radians(85.0 * share * amount))
        if thumb:
            for knuckle, share in enumerate((0.6, 0.7, 0.7)):
                name = "finger1-%d.%s" % (knuckle + 1, side)
                if name in pose.index:
                    j = pose.bone(name)
                    pose.rotate(j, self._curl_axis(j),
                                math.radians(55.0 * share * amount))
        self.extremities["%s_grip" % letter] = float(amount)

    def _curl_axis(self, j):
        """The axis a finger bone folds about: across the palm.

        Taken from the hand's own geometry - the bone's direction crossed with
        the spread of the knuckles - so it is right on both hands without a
        table, and mirrors itself because the knuckles do.
        """
        pose = self.pose
        q = pose.positions()
        kids = pose.kids.get(j)
        along = (q[kids[0]] - q[j]) if kids else (q[j] - q[int(pose.parents[j])])
        side = self.body_frame()[0]
        axis = np.cross(unit(along), np.asarray(side, float))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.asarray(self.body_frame()[1], float)
        # fold toward the palm, whichever way that is for this hand
        return unit(axis) * (1.0 if j in self._palm_side() else -1.0)

    def _palm_side(self):
        """Bone indices of the left hand, so a curl can pick its sign."""
        if getattr(self, "_left_bones", None) is None:
            self._left_bones = {j for j, n in enumerate(self.pose.names)
                                if n.endswith(".L")}
        return self._left_bones

    def set_extremity(self, part, letter, first, second):
        """Set a hand or a foot: `part` is "hand" or "foot", `letter` l or r.

        Two rotations of one bone. The AXES are derived - the segment's own
        direction and a perpendicular built from it - but the four SIGNS were
        measured on the body set rather than reasoned about, because a
        canonical axis still leaves the handedness of each rotation open and
        getting one backwards gives a control whose +30 points the left toe in
        and the right toe out, which nobody can use:

          - a positive lift raises both sets of toes 8.3 cm
          - a positive foot turn points both toes outward 7.4 cm
          - a positive hand bend curls both hands toward their own palms
          - a positive hand turn is pronation, taking both thumbs down

        The perpendicular is built from each part's OWN direction, so it is
        already mirrored between the sides and must NOT be given a side sign
        as well - doing that flips it twice on one side and not at all on the
        other, which is what made the lift raise one toe and drop the other.
        The turn axis is shared between the sides, so that one does.
        """
        pose = self.pose
        swing, far = SEGMENTS["%s_%s" % (letter, part)]
        j, c = pose.bone(swing), pose.bone(far)
        q = pose.positions()
        along = unit(q[c] - q[j])
        _side, up, _facing = (np.asarray(v, float) for v in self.body_frame())
        across = np.cross(along, up)
        across = unit(across) if np.linalg.norm(across) > 1e-6 else _side
        mirror = 1.0 if letter == "l" else -1.0
        pose.rotate(j, across, math.radians(first))
        # a foot yaws about the body's up; a hand pronates about its own axis
        pose.rotate(j, up if part == "foot" else along,
                    math.radians(second) * mirror)
        self.extremities["%s_%s" % (letter, part)] = (float(first),
                                                      float(second))

    def spin(self, idx, angle):
        """Roll the segment above this joint - on the twist bone where the rig
        has one, which is what twist bones are for."""
        self.pose.roll_segment(idx, angle)

    def sagittal_plane(self):
        q = self.pose.positions()
        j = self.pose.index.get("spine05", 0)
        return (tuple(q[j]), (1.0, 0.0, 0.0))

    def translate(self, delta, indices=None):
        self.pose.translate(delta)

    def turn(self, angle):
        self.pose.turn(angle)

    # -- history, files, export -------------------------------------------

    def snapshot(self):
        """Everything an undo has to put back.

        All of it, not just the pose: what the figure is wearing, which bones
        are hidden and what is pinned are as much a part of the scene as the
        joint angles, and an undo that restored only the angles took a coat
        off and left it off.
        """
        return (self.pose.snapshot(), dict(self.body), dict(self.outfit),
                list(self.assets), list(self.visible), set(self.anchors),
                dict(self.extremities))

    def restore(self, snap):
        pose, body, outfit, assets, visible, anchors, extremities = snap
        self.pose.restore(pose)
        self.body = dict(body)
        self.outfit = dict(outfit)
        self.assets = list(assets)
        self.visible = list(visible)
        self.anchors = set(anchors)
        self.extremities = dict(extremities)

    def to_dict(self):
        data = self.pose.to_dict()
        data["body"] = dict(self.body)
        return data

    def from_dict(self, data):
        self.pose.from_dict(data)
        if data.get("body"):
            self.body = dict(data["body"])

    def scale(self, factor):
        """Resize the whole figure. One uniform scale over the rig, which is
        the only scaling there is: a bone never changes length on its own."""
        self.pose.rest = self.pose.rest * float(factor)
        self.pose.scale *= float(factor)
        self.pose.touched()

    def solution(self):
        return self.pose.solution()

    def surface(self, every=4):
        """The skinned body, for framing and for anything that needs to know
        how far the figure actually reaches.

        Subsampled, because framing wants a bounding box and not a mesh: every
        fourth vertex of thirteen thousand still pins the extremes to well
        under a millimetre, and the box is padded by a 6% margin anyway.
        Cached on the pose, because a resize reframes and a drag redraws.
        """
        import mesh_backend
        key = (id(self.pose.local), self.pose.local.tobytes(),
               self.pose.offset.tobytes())
        if getattr(self, "_surface_key", None) != key:
            self._surface_key = key
            self._surface = mesh_backend.skin_with(
                self.mesh, self.solution())[::every]
        return self._surface

    @property
    def hinged(self):
        """Two anchors used to turn the figure into a hinge it could swing
        about. That was a re-rooting of the keypoint tree: pick a new root,
        and the chain runs outward from it. A rig has one root and every bone
        hangs off it, so the same gesture is an IK solve rather than a
        re-parent, and it is NOT implemented - `anchor` below pins a single
        joint, which is the part that carries over.
        """
        return False

    def anchor(self, joint=None):
        """Pin one joint in space: what the figure does afterwards, it does
        without moving that point.

        A translation, not a solve. Dragging a bone rotates it and everything
        below; pinning says the figure should end up with the pinned joint
        where it started, so the whole body slides back by however far the pin
        drifted. That is exactly right for "hold the hand still and lean the
        body", and it is honestly all one anchor can mean.
        """
        self.anchors = set() if joint is None else {int(joint)}

    def hold_anchors(self, before):
        """Slide the figure back so a pinned joint is where it was."""
        if not self.anchors:
            return
        j = next(iter(self.anchors))
        now = np.asarray(self.pose.positions()[j], float)
        self.pose.translate(np.asarray(before, float) - now)

    def apply_body(self, body):
        """Change who this figure is, keeping the pose.

        A preset is a different rigged body, so this swaps the mesh and
        rebuilds on it - and carries the joint angles across BY BONE NAME,
        which works because all nine bodies are built on the same armature.
        The pose survives a change of preset exactly; only the proportions
        change, which is the point.
        """
        import bodies_lib
        from anthro import preset_params
        preset = body.get("preset") or self.body.get("preset")
        params = dict(preset_params(preset)) if preset else {}
        params.update(body)
        if preset and preset != self.body.get("preset"):
            mesh = bodies_lib.load(None, [preset])[preset]
            held = self.pose.to_dict()
            self.mesh = mesh
            self.pose = RigPose(mesh, params.get("stature"))
            self.mirror = mirror_map(self.pose.names)
            self._riders = None
            self._surface_key = None
            self.pose.from_dict(held)
            self.visible = [True] * len(self.pose.names)
        self.body = params
        self.pose.stand()

    @property
    def body_scale(self):
        """How big this figure is against the body it was built from."""
        return self.pose.scale / CM_PER_METRE

    @body_scale.setter
    def body_scale(self, value):
        self.scale(float(value) / max(1e-9, self.body_scale))

    def from_keypoints(self, points):
        """Load a pose written as eighteen OpenPose keypoints.

        The one place keypoints still come IN, and it is an importer rather
        than a way of working: scene files written before the editor posed the
        rig hold eighteen points and nothing else, and refusing to open them
        would be throwing away the user's own saved work.

        `mesh_backend.pose_globals` is what the editor used to run on every
        redraw - aim each bone at the keypoint that names it. Here it runs
        once, at load, and what is kept is the joint angles it worked out. The
        file is a rig pose from then on, and everything it could not express -
        the hands, the collarbones, the roll of a limb - is editable for the
        first time.
        """
        import mesh_backend
        from exporting import build_rest_points
        from skeleton import KEYPOINT_NAMES
        named = ({name: tuple(points[i])
                  for i, name in enumerate(KEYPOINT_NAMES)}
                 if not isinstance(points, dict) else dict(points))
        Q, q = mesh_backend.pose_globals(
            self.pose.rest, self.pose.parents,
            self.mesh.get("roles") or mesh_backend.resolve_bones(
                self.pose.names),
            named,
            rest_orient=self.mesh["rest_global"][:, :3, :3],
            rest_points=build_rest_points(self.body))
        # globals back to the local rotations this holds: local = Qp^T @ Q
        for j in self.pose.order:
            p = int(self.pose.parents[j])
            self.pose.local[j] = (Q[j] if p < 0 else Q[p].T @ Q[j])
        root = self.pose.roots[0]
        self.pose.offset = q[root] - self.pose.rest[root]
        self.pose.touched()

    def at(self, name):
        """Where a named part of the body is.

        Takes a rig bone name (`upperarm01.L`), one of the eighteen role names
        the OpenPose format uses (`l_wrist`), or a friendly name (`left hand`).
        Ten of the eighteen ARE rig joints and are read straight off the
        armature; the face ones have no bone, so they come from the keypoints
        the posed rig reports.
        """
        from mesh_backend import ANNY_BONES
        q = self.pose.positions()
        if name in self.pose.index:
            return tuple(q[self.pose.index[name]])
        if name in ANNY_BONES and ANNY_BONES[name] in self.pose.index:
            return tuple(q[self.pose.index[ANNY_BONES[name]]])
        kp = self.keypoints()
        if name in kp:
            return tuple(kp[name])
        raise KeyError("no part called %r" % (name,))

    def as_skeleton(self):
        """A keypoint `Skeleton` describing this figure, for the two things
        that genuinely want one.

        The swept anatomy builds its cross-sections from eighteen keypoints,
        and `wearables` cuts a garment out of that sweep. Both are
        approximations that draw the viewport at sixty frames a second, and
        both are downstream of the rig - so they are handed a description of
        the posed body rather than being another way to pose it. Nothing
        writes back through this.
        """
        from skeleton import Skeleton, KEYPOINT_NAMES
        sk = Skeleton(self.body)
        kp = self.keypoints()
        sk.points = [tuple(kp[name]) if name in kp else sk.points[i]
                     for i, name in enumerate(KEYPOINT_NAMES)]
        sk.lengths = {i: sk.bone_length(sk.parent_of(i), i)
                      for i in range(len(sk.points))
                      if sk.parent_of(i) >= 0}
        return sk

    def body_frame(self):
        """(side, up, facing), side to the figure's left.

        The hips bone's own orientation, with nothing inferred from positions.
        Anny is authored standing upright facing +Z with its left at +X, so at
        rest this is exactly the world axes, and after any pose it is whatever
        the pelvis has been turned to - which is what "which way is this person
        facing" means, and stays right for a figure lying down, where a
        shoulder-to-hip line says nothing about it at all.

        What this replaced took `up` from the hips to a point on the spine.
        That is exact on a keypoint figure, whose torso is vertical because a
        table built it that way, and 8 degrees out on a real body, whose
        lumbar curve leans back. Every direction in the command vocabulary is
        resolved against this frame, so those 8 degrees reached all of them:
        a crate placed in front of someone at arm's length landed 12 cm high.
        """
        Q, _q = self.pose.solve()
        M = Q[self.pose.bone(WAIST)]
        return (tuple(M[:, 0]), tuple(M[:, 1]), tuple(M[:, 2]))

    def keypoints(self):
        """The eighteen, read off the posed rig - for a pose PNG and for the
        swept preview. Output only: nothing reads them back in."""
        import mesh_backend
        from exporting import build_rest_points
        if self._riders is None:
            self._riders = mesh_backend.keypoint_riders(
                self.mesh, build_rest_points(self.body),
                self.mesh.get("roles"), stature=self.body.get("stature"))
        return mesh_backend.keypoints_of(self.solution(), self._riders,
                                         self.mesh)



def figure_for(preset, folder=None, body=None):
    """A posable figure on the body set's rig for `preset`.

    There is no fallback. The editor poses Anny's armature, so without the
    body set there is nothing to pose and nothing to draw - `MissingBodies`
    names the command that builds it, which is a better answer than a
    mannequin nobody asked for.
    """
    import bodies_lib
    from anthro import preset_params
    mesh = bodies_lib.load(folder, [preset])[preset]
    params = dict(preset_params(preset))
    params.update(body or {})
    params["preset"] = preset
    return Figure(mesh, params)


# ---------------------------------------------------------------------------

def _selftest():
    import mesh_backend, tempfile, os
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("  %s %s  %s" % ("PASS" if cond else "FAIL", name, detail))

    path = os.path.join(tempfile.mkdtemp(), "rig.glb")
    mesh_backend._write_test_glb(path)
    mesh = mesh_backend.load_rigged_mesh(path)
    pose = RigPose(mesh)

    # rest
    q0 = pose.positions().copy()
    check("rest pose is the rig's own",
          np.allclose(q0, pose.rest + pose.offset),
          "every bone at its authored place")

    # a rotation carries the subtree and changes no bone length
    def lengths(p):
        q = p.positions()
        return np.array([np.linalg.norm(q[j] - q[int(p.parents[j])])
                         for j in range(len(p.names)) if p.parents[j] >= 0])

    before = lengths(pose)
    j = pose.bone("l_shoulder")
    kid = pose.kids[j][0]
    pose.rotate(j, (0.0, 0.0, 1.0), math.radians(40))
    after = lengths(pose)
    check("a rotation changes no bone length",
          np.max(np.abs(after - before)) < 1e-12,
          "worst %.2e cm" % np.max(np.abs(after - before)))
    moved = np.linalg.norm(pose.positions() - q0, axis=1)
    below = set(pose.subtree(j))
    check("the whole limb below follows",
          all(moved[k] > 1e-6 for k in pose.subtree(kid)) and
          all(moved[k] < 1e-9 for k in range(len(pose.names))
              if k not in below),
          "%d bones moved, all of them below the shoulder" %
          int((moved > 1e-6).sum()))

    # aiming puts the child exactly where it was asked to go
    pose.reset_all()
    _Q, q = pose.solve()
    reach = np.linalg.norm(q[kid] - q[j])
    target = q[j] + reach * unit(np.array([0.3, -0.9, 0.4]))
    pose.aim(j, kid, target)
    landed = pose.positions()[kid]
    check("a drag lands the joint on the cursor",
          np.linalg.norm(landed - target) < 1e-9,
          "%.2e cm away" % np.linalg.norm(landed - target))

    # a roll does not move the joint it points at
    pose.reset_all()
    Q_before, q_before = (x.copy() for x in pose.solve())
    pose.spin(j, math.radians(35))
    Q_after, q_after = pose.solve()
    check("a roll does not move the joint below it",
          np.linalg.norm(q_after[kid] - q_before[kid]) < 1e-9,
          "%.2e cm" % np.linalg.norm(q_after[kid] - q_before[kid]))
    # Measured as an ORIENTATION change, which is what a roll is. Positions
    # below only move where something hangs off the axis, so checking those
    # would be testing the mock rig's shape rather than the rotation.
    turned = math.degrees(math.acos(np.clip(
        (np.trace(Q_after[kid].T @ Q_before[kid]) - 1.0) / 2.0, -1.0, 1.0)))
    check("but it does roll what hangs off it",
          abs(turned - 35.0) < 1e-6, "the chain below turned %.1f deg" % turned)

    # a file round-trips exactly
    pose.reset_all()
    pose.rotate(pose.bone("r_elbow"), (1.0, 0.0, 0.0), 0.7)
    pose.rotate(j, (0.0, 1.0, 0.0), -0.4)
    pose.translate((3.0, 0.0, -2.0))
    saved = pose.to_dict()
    twin = RigPose(mesh)
    twin.from_dict(saved)
    check("a scene round-trips exactly",
          np.allclose(twin.positions(), pose.positions(), atol=1e-9),
          "%d bones stored of %d" % (len(saved["bones"]), len(pose.names)))

    # undo
    snap = pose.snapshot()
    pose.rotate(j, (1.0, 0.0, 0.0), 1.1)
    pose.restore(snap)
    check("undo restores the pose exactly",
          np.allclose(pose.positions(), twin.positions(), atol=1e-12))

    # standing
    pose.reset_all()
    pose.offset = np.array([0.0, 0.0, 0.0])
    pose.stand()
    check("the figure stands on the floor",
          abs(pose.lowest()) < 1e-6, "sole at %.2e cm" % pose.lowest())

    print("rigpose selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
