"""Depth export uses rigged geometry, or it does not happen.

The built-in swept anatomy is a stack of tapering cross-sections. It draws the
viewport and it gives `wearables` a profile to cut a garment out of, and it is
not what anyone wants out of an export. The trap is that falling back to it is
invisible: the PNG still appears, it is just a picture of a mannequin, and
nothing says so. So every path that writes a depth map has to refuse.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import bodies_lib
import pose_agent
from openpose3d_editor import BODY_PRESETS, KEYPOINT_NAMES

ok = True


def check(label, condition, extra=""):
    global ok
    ok = ok and bool(condition)
    print(("PASS " if condition else "FAIL ") + label
          + ("  " + extra if extra else ""))


def scene(preset="Female, average", pose="reading_a_book"):
    return pose_agent.build_scene({"figures": [
        {"preset": preset, "commands": [{"op": "stance", "name": pose}]}]})


# -- the set is found without being told where ----------------------------
bank = bodies_lib.load(required=False)
check("a body set is found with no folder given", len(bank) > 0,
      "%d in %s" % (len(bank), ", ".join(bodies_lib.folders()) or "nowhere"))
check("and it covers every body preset",
      all(p in bank for p in BODY_PRESETS),
      "missing " + str([p for p in BODY_PRESETS if p not in bank]))
check("every one of them is a real rig, not a stand-in",
      all(len(m["vertices"]) > 5000 and len(m["joint_names"]) > 20
          for m in bank.values()),
      "smallest %d vertices" % min(len(m["vertices"]) for m in bank.values()))

# -- the export is the rigged one -----------------------------------------
figures, objects, camera, _w = scene()
_pose, rigged, _rect = pose_agent.render_scene(figures, camera, 192, 288,
                                               props=objects)
_pose, swept, _rect = pose_agent.render_scene(figures, camera, 192, 288,
                                              props=objects, anatomy=True)
check("a plain render is not the swept anatomy",
      np.asarray(rigged).tobytes() != np.asarray(swept).tobytes())
# A rigged body has fingers, a face and a collarbone; the sweep is smooth
# tapering tubes. Count how much of the picture sits on an edge, per covered
# pixel rather than per frame, so the measure is about surface and not about
# how much room the body takes up - which is what it used to be measuring.
def detail(image):
    a = np.asarray(image, float)
    edge = (np.abs(np.diff(a, axis=0)).sum() + np.abs(np.diff(a, axis=1)).sum())
    return float(edge / max(1, int((a > 0).sum())))


check("and carries more surface detail than it does",
      detail(rigged) > 1.05 * detail(swept),
      "%.2f against %.2f" % (detail(rigged), detail(swept)))

# And it is the same SIZE as the sweep, which is the point the bar above used
# to be carrying by accident. The rig used to be sized by dividing the
# keypoints' shoulder-to-hip span by its own, and those two spans measure
# different things - an acromion-to-trochanter against a glenohumeral-to-
# femoral-head - so it came up 15% oversize on an adult and 41% on the child
# before a bone was aimed. A body a sixth too big for its own skeleton filled
# more of the frame and carried more edge with it, so the detail test passed
# on the strength of the bug.
area = lambda im: int((np.asarray(im) > 0).sum())
check("and fills the same frame as the sweep, not a sixth more",
      abs(area(rigged) - area(swept)) < 0.15 * area(swept),
      "%d px against %d" % (area(rigged), area(swept)))

# -- with no bodies it refuses, and says how ------------------------------
was = os.environ.get("STRIKEPOSE_BODIES")
os.environ["STRIKEPOSE_BODIES"] = os.path.join(os.path.dirname(__file__),
                                               "no-such-bodies")
bodies_lib._cache.clear()
try:
    refused = None
    try:
        pose_agent.render_scene(figures, camera, 64, 96, props=objects)
    except bodies_lib.MissingBodies as problem:
        refused = str(problem)
    check("with no body set the export refuses rather than falling back",
          refused is not None)
    check("and the refusal names the command that fixes it",
          refused and "tools/make_bodies.py" in refused,
          (refused or "").splitlines()[0] if refused else "no message")
    check("while the deliberate sweep still works, for the viewport",
          pose_agent.render_scene(figures, camera, 64, 96, props=objects,
                                  anatomy=True)[1] is not None)
finally:
    if was is None:
        os.environ.pop("STRIKEPOSE_BODIES", None)
    else:
        os.environ["STRIKEPOSE_BODIES"] = was
    bodies_lib._cache.clear()

# -- every preset renders from its own body -------------------------------
sizes = {}
for preset in BODY_PRESETS:
    figures, objects, camera, _w = scene(preset, "standing_still")
    _pose, depth, _rect = pose_agent.render_scene(figures, camera, 96, 144,
                                                  props=objects)
    sizes[preset] = int((np.asarray(depth) > 0).sum())
check("every preset exports from a rigged body of its own",
      len(sizes) == len(BODY_PRESETS) and all(v > 500 for v in sizes.values()),
      "smallest %d pixels" % min(sizes.values()))
check("and they are not all the same body",
      len(set(sizes.values())) >= len(BODY_PRESETS) - 1,
      "%d distinct silhouettes of %d" % (len(set(sizes.values())), len(sizes)))

# -- clothes go ON the rigged body -----------------------------------------
#
# `wearables` clips and pads the body's own *swept* profile, and that profile
# is thinner than the rigged mesh wherever the two disagree - a chest, a
# shoulder. Dropped straight into the buffer a coat comes out with the body
# through it, which is worse than no coat. Cloth lies on the surface instead:
# where a garment covers the body it sits a centimetre in front of whatever
# the rig put there, so it follows the rigged shape.
import openpose3d_editor as editor

figures, objects, camera, _w = scene("Female, curvy", "standing_still")
worn = figures[0]
mesh = bodies_lib.for_figures(figures)[0]
rect = editor.frame_rect(900, 700, 384 / 576.0)
worn.outfit = {}
bare = np.asarray(editor.rigged_depth_image([(worn, mesh, ())], camera, rect,
                                            384, 576), float)
worn.outfit = {"top": "coat", "hair": "long"}
dressed = np.asarray(editor.rigged_depth_image([(worn, mesh, ())], camera, rect,
                                               384, 576), float)
both = (bare > 0) & (dressed > 0)
# brighter is nearer, so a dressed pixel must never be darker than its bare one
behind = int((dressed[both] < bare[both] - 0.5).sum())
check("a coat reaches the rigged depth map at all",
      int((dressed > 0).sum()) > int((bare > 0).sum()) + 500,
      "%d px against %d" % (int((dressed > 0).sum()), int((bare > 0).sum())))
check("and the body does not come through it",
      behind < 0.02 * int(both.sum()),
      "%d of %d pixels, %.1f%%" % (behind, int(both.sum()),
                                   100.0 * behind / max(1, int(both.sum()))))

# A garment clears the rigged body by its OWN thickness, not by a flat
# centimetre. With one figure for all of them an afro came out at the same
# depth as a crew cut, a helmet as a bare head and a coat as a t-shirt: every
# thick garment there is rendered as bare skin, and since the silhouette still
# grew the PNG looked plausible, which is why nothing caught it.
#
# Reproduce the bug rather than measure a proxy for it. A pixel count of how
# far a thick garment stands proud is a proxy, and it moved the moment the
# anatomy underneath it was corrected - it read a helmet as 40 pixels against
# the bug's 29, which is not a test. Flattening every standoff to the old
# centimetre and asking whether the picture changes is the thing itself.
import wearables

def head_of(image, band=0.22):
    grey = np.asarray(image, float)
    return grey[:int(band * grey.shape[0])]


def dressed_head(outfit):
    worn.outfit = dict(outfit)
    return head_of(editor.rigged_depth_image([(worn, mesh, ())], camera, rect,
                                             384, 576))


honest = wearables.layers
try:
    flat = []
    for slot, name in (("hair", "afro"), ("hair", "curly"),
                       ("headgear", "helmet"), ("top", "coat")):
        wearables.layers = honest
        right = dressed_head({slot: name})
        # exactly the old bug: one standoff for every garment there is
        wearables.layers = (lambda *a, **k:
                            [(s, (1.0 if v is not None else None), p)
                             for s, v, p in honest(*a, **k)])
        wrong = dressed_head({slot: name})
        moved = int((np.abs(right - wrong) > 2.0).sum())
        flat.append(("%s/%s" % (slot, name), moved))
finally:
    wearables.layers = honest
    worn.outfit = {}

check("a garment's own thickness is what it clears the rigged body by",
      all(n > 150 for _g, n in flat),
      ", ".join("%s %d px" % (g, n) for g, n in flat))

# and it still has to reach the frame at all
worn.outfit = {"top": "coat"}
coated = int((np.asarray(editor.rigged_depth_image(
    [(worn, mesh, ())], camera, rect, 384, 576), float) > 0).sum())
worn.outfit = {"top": "t_shirt"}
teed = int((np.asarray(editor.rigged_depth_image(
    [(worn, mesh, ())], camera, rect, 384, 576), float) > 0).sum())
check("and a coat is bulkier in the frame than a t-shirt",
      coated > teed * 1.05, "%d px against %d" % (coated, teed))
worn.outfit = {}

# -- a rig is fitted segment by segment, not dragged -----------------------
#
# The rig used to be fitted to the figure by one number, the ratio of the two
# shoulder-to-hip spans, and one number cannot match a torso and the limbs
# hanging off it unless the bodies have the same proportions. Against these
# presets the same rig came out with a forearm 30% long on an average man and
# 54% on the child, whose thigh and shin were 58% and 61% over - and the fix
# for that was to slide each joint onto its keypoint and carry its subtree,
# which puts the *joint* right and leaves the geometry between at the rig's
# own length. A 42 cm shin pulled onto a 26 cm gap overshoots the ankle. The
# child came out with bowed shins and its feet hanging off them.
import mesh_backend
from openpose3d_editor import Skeleton, preset_params, build_rest_points

SEGMENTS = [("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
            ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
            ("r_shoulder", "r_elbow"), ("r_hip", "r_knee")]
index = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
# -1.0, not 0.0: the fit lands exactly now, and a sentinel that only records
# a value strictly greater than zero reports a perfect fit as "none measured",
# which reads like the loop never ran.
worst_segment, worst_joint = ("exactly", -1.0), ("exactly", -1.0)
for preset in BODY_PRESETS:
    mesh = bank[preset]
    skeleton = Skeleton(preset_params(preset))
    points = {n: skeleton.points[i] for i, n in enumerate(KEYPOINT_NAMES)}
    solved = mesh_backend.solve_pose(
        mesh, points, mesh["roles"],
        rest_points=build_rest_points(skeleton.body),
        stature=skeleton.body.get("stature"))
    names, roles = mesh["joint_names"], mesh["roles"]
    at = lambda role: solved["bones"][names[roles[role]]][1]
    # Where the joint was actually SENT. For most roles that is the keypoint;
    # for a landmark role it is the keypoint plus the rig's own rest offset,
    # because OpenPose's shoulder is the acromion on top of the shoulder and
    # the arm swings from the joint below and inboard of it. Measuring against
    # the bare keypoint is measuring against a point no bone was aimed at.
    sent = lambda role: (np.asarray(points[role], float)
                         + solved["shift"].get(role, 0.0))
    for top, end in SEGMENTS:
        if top not in roles or end not in roles:
            continue
        got = float(np.linalg.norm(at(end) - at(top)))
        want = float(np.linalg.norm(sent(end) - sent(top)))
        gap = abs(got - want) / max(1e-9, want)
        if gap > worst_segment[1]:
            worst_segment = ("%s %s->%s (%.1f vs %.1f cm)"
                             % (preset, top, end, got, want), gap)
    for role in ("l_shoulder", "l_elbow", "l_wrist", "l_hip", "l_knee",
                 "l_ankle"):
        if role not in roles:
            continue
        off = float(np.linalg.norm(at(role) - sent(role)))
        if off > worst_joint[1]:
            worst_joint = ("%s %s" % (preset, role), off)

check("every limb segment is scaled to the keypoints on every body",
      worst_segment[1] < 0.02,
      "worst %s off by %.1f%%" % (worst_segment[0], 100.0 * worst_segment[1]))
check("and every mapped joint still lands where it was sent",
      worst_joint[1] < 0.5, "worst %s by %.2f cm" % worst_joint)

# The shoulder is the one that is deliberately NOT sent to its keypoint, and
# the correction has to be real and small: zero means the landmark offset has
# quietly stopped being applied, and a large one means it is being computed in
# the wrong frame - which it was first time round, when subtracting two
# positions that share no origin threw the shoulders further out than leaving
# them alone did.
pulls = []
for preset in BODY_PRESETS:
    skeleton = Skeleton(preset_params(preset))
    points = {n: skeleton.points[i] for i, n in enumerate(KEYPOINT_NAMES)}
    shift = mesh_backend.solve_pose(
        bank[preset], points, bank[preset]["roles"],
        rest_points=build_rest_points(skeleton.body),
        stature=skeleton.body.get("stature"))["shift"]
    pulls.append((preset, float(np.linalg.norm(shift.get("l_shoulder", 0.0)))))
# A landmark offset says where a joint SITS, never how long the bone leaving
# it is - this project's oldest invariant, arriving somewhere new. Applied
# whole, the shoulder's offset moved the joint 6 cm down the arm while the
# elbow keypoint stayed put, so the humerus had to span a gap 6 cm shorter
# than it is: 18% of an adult's upper arm and 54% of the child's, which came
# out of the depth map as visibly deformed elbows. It shipped, because every
# check here was on the shoulder and none was on the arm hanging off it.
square, squashed = ("none", 0.0), ("none", 0.0)
for preset in BODY_PRESETS:
    mesh = bank[preset]
    skeleton = Skeleton(preset_params(preset))
    points = {n: skeleton.points[i] for i, n in enumerate(KEYPOINT_NAMES)}
    solved = mesh_backend.solve_pose(
        mesh, points, mesh["roles"],
        rest_points=build_rest_points(skeleton.body),
        stature=skeleton.body.get("stature"))
    names, roles = mesh["joint_names"], mesh["roles"]
    at = lambda role: solved["bones"][names[roles[role]]][1]
    rested = mesh["rest_position"] * solved["scale"]
    for role, bone in mesh_backend.LANDMARK_JOINTS.items():
        if bone is None or role not in roles or role not in solved["shift"]:
            continue
        along = np.asarray(points[bone], float) - np.asarray(points[role],
                                                             float)
        along = along / max(1e-9, float(np.linalg.norm(along)))
        leak = abs(float(np.dot(solved["shift"][role], along)))
        if leak > square[1]:
            square = ("%s %s by %.2f cm" % (preset, role, leak), leak)
        was = float(np.linalg.norm(rested[roles[bone]] - rested[roles[role]]))
        now = float(np.linalg.norm(at(bone) - at(role)))
        gap = abs(now / max(1e-9, was) - 1.0)
        if gap > squashed[1]:
            squashed = ("%s %s->%s (%.1f vs %.1f cm)"
                        % (preset, role, bone, now, was), gap)

check("a landmark offset never points along the bone it feeds", square[1] < 0.01,
      "worst %s" % square[0])
check("so no landmark bone is squashed to reach its keypoint",
      squashed[1] < 0.12, "worst %s off by %.0f%%"
      % (squashed[0], 100.0 * squashed[1]))

check("the shoulder is held off its keypoint, by a few centimetres",
      all(1.0 < v < 9.0 for _p, v in pulls),
      ", ".join("%s %.1f cm" % (p.split(",")[0], v) for p, v in pulls[:4]))

# The mesh must not be torn doing it: a sliding subtree leaves a seam, so
# check the limb is still one connected piece of surface by measuring the
# gap between the shin's vertices and the foot's.
child = bank["Child, about 7"]
skeleton = Skeleton(preset_params("Child, about 7"))
points = {n: skeleton.points[i] for i, n in enumerate(KEYPOINT_NAMES)}
solved = mesh_backend.solve_pose(child, points, child["roles"],
                                 rest_points=build_rest_points(skeleton.body))
posed = mesh_backend.skin_with(child, solved)
seam = (None, 0.0)


def closest(points, other):
    return min(float(np.linalg.norm(other - point, axis=1).min())
               for point in points[::7])


for a, b in (("l_knee", "l_ankle"), ("l_shoulder", "l_elbow"),
             ("l_hip", "l_knee"), ("l_elbow", "l_wrist")):
    lower, upper = child["roles"][b], child["roles"][a]
    mine = child["skin_joints"][:, 0] == lower
    theirs = child["skin_joints"][:, 0] == upper
    if mine.sum() < 3 or theirs.sum() < 3:
        continue
    # How far the two bones' own vertices sit from each other, against how far
    # they sat at rest. On an intact limb the answer is "the same": posing is
    # rotation and scale. A subtree slid onto its keypoint opens the gap by
    # however far it slid, which is the seam that was showing.
    was = closest(child["vertices"][mine] * solved["scale"],
                  child["vertices"][theirs] * solved["scale"])
    now = closest(posed[mine], posed[theirs])
    if now - was > seam[1]:
        seam = ("%s/%s" % (a, b), now - was)
check("and the limb is not torn open at the joint doing it", seam[1] < 2.0,
      "widest seam opened %.2f cm at %s" % (seam[1], seam[0]))

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
