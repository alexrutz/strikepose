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

# -- clothes are real meshes now ------------------------------------------
#
# `wearables` builds a garment out of the body's own swept profile. That still
# draws the viewport, and it is an approximation: what a rigged export wears
# is the CC0 MakeHuman library, fitted to each body by tools/make_wearables.py
# and skinned to its armature. So the thickness checks that used to live here
# moved to the swept path, where the sweep still is, and what is checked on
# the rigged path is that the real mesh arrives and that the body does not
# come through it.
import garments_lib

have = garments_lib.available()
check("the garment library is on disk", len(have) >= 12,
      "%d garments" % len(have))
missing = [(slot, name) for slot in garments_lib.CATALOGUE
           for name in garments_lib.CATALOGUE[slot]
           if not garments_lib.named(slot, name)]
check("and every garment it names is one of them", not missing,
      str(missing[:3]))

worn.outfit = {}
bare_px = int((np.asarray(editor.rigged_depth_image(
    [(worn, mesh, ())], camera, rect, 384, 576), float) > 0).sum())
for slot, name in (("top", "long_sleeve"), ("bottom", "trousers"),
                   ("shoes", "shoes"), ("hair", "bob")):
    worn.outfit = {slot: name}
    grey = np.asarray(editor.rigged_depth_image(
        [(worn, mesh, ())], camera, rect, 384, 576), float)
    check("a real %s reaches the rigged depth map" % name,
          int((grey > 0).sum()) > bare_px + 200,
          "%d px against %d bare" % (int((grey > 0).sum()), bare_px))
worn.outfit = {}

# The body must not come through what it is wearing. MakeHuman ships the list
# of body vertices a garment stands in for; without it a shirt sits a
# millimetre off a chest and the chest wins the z-test, so the garment is
# simply absent wherever it matters most.
worn.outfit = {"top": "long_sleeve", "bottom": "trousers"}
dressed = np.asarray(editor.rigged_depth_image(
    [(worn, mesh, ())], camera, rect, 384, 576), float)
worn.outfit = {}
bare = np.asarray(editor.rigged_depth_image(
    [(worn, mesh, ())], camera, rect, 384, 576), float)
both = (bare > 0) & (dressed > 0)
behind = int((dressed[both] < bare[both] - 0.5).sum())
check("and the body does not come through it",
      behind < 0.02 * int(both.sum()),
      "%d of %d pixels" % (behind, int(both.sum())))

# -- the swept garments, on the path that still uses them ------------------
sweep = lambda outfit: (setattr(worn, "outfit", dict(outfit)),
                        np.asarray(editor.anatomy_depth_image(
                            [worn], camera, rect, 384, 576), float))[1]
for slot, thin, thick in (("hair", "short", "curly"),
                          ("headgear", "none", "hat")):
    lean = sweep({slot: thin})[:int(0.22 * 576)]
    bulky = sweep({slot: thick})[:int(0.22 * 576)]
    check("a swept %s is thicker than a %s" % (thick, thin),
          int((bulky > 0).sum()) > int((lean > 0).sum()) + 40,
          "%d px against %d" % (int((bulky > 0).sum()), int((lean > 0).sum())))
worn.outfit = {}

# -- the rig is posed, not fitted -----------------------------------------
#
# Every bone points the way its two keypoints do and keeps the length the rig
# authored, so a joint sits off its keypoint by however much the two bodies'
# proportions differ and nothing is stretched to close that gap. What this
# replaced aimed each bone at the absolute keypoint and then slid and scaled
# it until it landed, which put +7% on a humerus and -10% on a shin of the
# same figure.
import mesh_backend
from openpose3d_editor import Skeleton, preset_params, build_rest_points

SEGMENTS = [("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
            ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
            ("r_shoulder", "r_elbow"), ("r_hip", "r_knee")]
worst_length, worst_aim = ("exactly", -1.0), ("exactly", -1.0)
for preset in BODY_PRESETS:
    mesh = bank[preset]
    skeleton = Skeleton(preset_params(preset))
    points = {n: skeleton.points[i] for i, n in enumerate(KEYPOINT_NAMES)}
    solved = mesh_backend.pose_rig(
        mesh, points, mesh["roles"],
        rest_points=build_rest_points(skeleton.body),
        stature=skeleton.body.get("stature"))
    names, roles = mesh["joint_names"], mesh["roles"]
    at = lambda role: solved["bones"][names[roles[role]]][1]
    rested = mesh["rest_position"] * solved["scale"]
    for top, end in SEGMENTS:
        if top not in roles or end not in roles:
            continue
        was = float(np.linalg.norm(rested[roles[end]] - rested[roles[top]]))
        now = float(np.linalg.norm(at(end) - at(top)))
        gap = abs(now / max(1e-9, was) - 1.0)
        if gap > worst_length[1]:
            worst_length = ("%s %s->%s (%.2f vs %.2f cm)"
                            % (preset, top, end, now, was), gap)
        got = at(end) - at(top)
        got = got / max(1e-9, float(np.linalg.norm(got)))
        want = (np.asarray(points[end], float)
                - np.asarray(points[top], float))
        want = want / max(1e-9, float(np.linalg.norm(want)))
        off = float(np.linalg.norm(got - want))
        if off > worst_aim[1]:
            worst_aim = ("%s %s->%s" % (preset, top, end), off)

check("every bone keeps the length the rig authored, on every body",
      worst_length[1] < 1e-9, "worst %s off by %.2e" % worst_length)
check("and points the way its two keypoints do",
      worst_aim[1] < 1e-9, "worst %s by %.2e" % worst_aim)

# The mesh must not be torn doing it: a sliding subtree leaves a seam, so
# check the limb is still one connected piece of surface by measuring the
# gap between the shin's vertices and the foot's.
child = bank["Child, about 7"]
skeleton = Skeleton(preset_params("Child, about 7"))
points = {n: skeleton.points[i] for i, n in enumerate(KEYPOINT_NAMES)}
solved = mesh_backend.pose_rig(child, points, child["roles"],
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
