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
from openpose3d_editor import BODY_PRESETS

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
# tapering tubes. Count how much of the picture sits on an edge: the mesh has
# far more of it, and that is the whole difference the eye is reacting to.
def detail(image):
    a = np.asarray(image, float)
    step = np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()
    return float(step)


check("and carries more surface detail than it does",
      detail(rigged) > 1.15 * detail(swept),
      "%.2f against %.2f" % (detail(rigged), detail(swept)))

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

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
