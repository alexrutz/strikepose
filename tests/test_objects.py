"""Objects in the editor: placing, picking, dragging, saving. Needs a display.

The geometry itself is checked by `props.py --selftest` and the placement
vocabulary by `pose_agent.py --selftest`; this is the part only the window can
answer - that an object can be got at with the mouse without the joints
becoming unreachable, and that it survives a save and a reload.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "out")
os.makedirs(OUT, exist_ok=True)

import tkinter as tk
import numpy as np
import props
from openpose3d_editor import (EditorApp, KEYPOINT_NAMES, scene_objects,
                               scene_to_dict, vlen, vsub)

ok = True


def check(label, condition, extra=""):
    global ok
    ok = ok and bool(condition)
    print(("PASS " if condition else "FAIL ") + label
          + ("  " + extra if extra else ""))


class E:
    def __init__(self, x, y):
        self.x, self.y, self.state = x, y, 0


root = tk.Tk()
root.geometry("1180x800")
app = EditorApp(root)
root.update()

check("a new scene has no objects", app.props == [] and app.active_prop is None)
check("and the panel says where they will show up",
      "depth map" in app.prop_label.get(), app.prop_label.get())

# -- placing ---------------------------------------------------------------
app.prop_shape.set("crate")
prop = app.add_prop()
root.update()
check("placing puts one in the scene and selects it",
      len(app.props) == 1 and app.active_prop == 0
      and app.props[0]["shape"] == "crate")
check("at its real size", tuple(prop["size"]) == props.default_size("crate"),
      str(prop["size"]))
low = props.bounds(prop)[0][1]
check("standing on the floor the figure stands on",
      abs(low - app.ground_level()) < 1e-6,
      "%.2f vs %.2f" % (low, app.ground_level()))
check("and the panel names it", "crate" in app.prop_label.get(),
      app.prop_label.get())

# -- picking, and not at the expense of the joints -------------------------
sx, sy, _ = app.camera.project(prop["position"])
sx, sy = sx, sy - 20                        # a little way up the crate's face
check("an object under the cursor is picked",
      app.pick_prop(sx, sy, app.camera) == 0)
wrist = KEYPOINT_NAMES.index("l_wrist")
wx, wy, _ = app.camera.project(app.skeleton.points[wrist])
app.on_press(E(wx, wy))
check("but a joint still wins where the two overlap",
      app.selected == wrist and app.drag_prop is None)
app.on_release(E(wx, wy))

# -- dragging --------------------------------------------------------------
before = list(prop["position"])
bones = dict(app.skeleton.lengths)
joints = list(app.skeleton.points)
app.on_press(E(sx, sy))
check("pressing on an object starts dragging it, not an orbit",
      app.drag_prop == 0 and app.orbit_last is None)
for step in range(1, 6):
    app.on_drag(E(sx + 8 * step, sy))
app.on_release(E(sx + 40, sy))
moved = vlen(vsub(prop["position"], before))
check("dragging moves it", moved > 5.0, "%.1f cm" % moved)
check("and the drag ends", app.drag_prop is None)
check("and the figure was not touched by a drag that was not about it",
      all(vlen(vsub(a, b)) < 1e-9
          for a, b in zip(joints, app.skeleton.points))
      and app.skeleton.lengths == bones)

app.undo()
root.update()
check("undo puts it back",
      vlen(vsub(app.props[0]["position"], before)) < 1e-9)

# -- size, turn, drop ------------------------------------------------------
app.scale_prop(2.0)
check("larger scales it", abs(app.props[0]["size"][0]
                              - props.default_size("crate")[0] * 2.0) < 1e-9)
app.turn_prop(45.0)
check("turn spins it", app.props[0]["yaw"] == 45.0)
app.props[0]["position"][1] += 200.0
app.drop_prop()
check("drop puts it back on the floor",
      abs(props.bounds(app.props[0])[0][1] - app.ground_level()) < 1e-6)

# -- the depth map ---------------------------------------------------------
w, h = app._sizes()
with_crate = np.asarray(app.depth_image(w, h))
app.props, app.active_prop = [], None
without = np.asarray(app.depth_image(w, h))
check("an object shows up in the exported depth map",
      int((with_crate > 0).sum()) > int((without > 0).sum()) + 400,
      "%d px vs %d" % (int((with_crate > 0).sum()), int((without > 0).sum())))
pose_before = app.export_people(w, h)
app.props = [props.make("wall", position=(0.0, app.ground_level(), -40.0))]
check("and never in the pose map", app.export_people(w, h) == pose_before)

# -- several objects, selection, removal -----------------------------------
app.props = []
app.active_prop = None
for shape in ("chair", "table", "pillar"):
    app.prop_shape.set(shape)
    app.add_prop(announce=False)
app.next_prop()
check("select next cycles", app.active_prop == 0, str(app.active_prop))
app.next_prop()
check("and moves on", app.active_prop == 1)
app.delete_prop()
check("remove takes one out",
      [p["shape"] for p in app.props] == ["chair", "pillar"],
      str([p["shape"] for p in app.props]))
app.undo()
check("and undo brings it back", len(app.props) == 3)

# -- round trip ------------------------------------------------------------
data = scene_to_dict(app.figures, app.camera,
                     [app.export_points(w, h, f)
                      for f in range(len(app.figures))], w, h, app.props)
text = json.dumps(data)
back = scene_objects(json.loads(text))
check("objects survive a save and a reload", back == app.props,
      "%d back" % len(back))
check("a scene written before objects existed simply has none",
      scene_objects({"people": []}) == [])

# -- the rasteriser: a box has to come out with corners --------------------
#
# The reason objects needed a primitive of their own. An elliptical cylinder
# through the same code renders a crate with rounded sides, and a depth map of
# a room built out of those reads as a room full of cushions.
from openpose3d_editor import Camera, anatomy_depth_image, frame_rect
crate = props.make("box", (80.0, 80.0, 80.0), (0.0, -40.0, 0.0))
flat = Camera(900, 700)
flat.zoom, flat.target = 3.0, (0.0, 0.0, 0.0)
square = np.asarray(anatomy_depth_image([], flat, frame_rect(900, 700, 1.0),
                                        256, 256, 1.0, [crate]))
rows = [int((row > 0).sum()) for row in square if (row > 0).any()]
check("a box renders as a box, the same width at every row",
      max(rows) - min(rows) <= 1 and len(rows) > 60,
      "%d rows, %d..%d px wide" % (len(rows), min(rows), max(rows)))
# From the front an upright cylinder is a rectangle too, so the two are told
# apart from overhead, where a box is a square and a cylinder a circle - and a
# circle covers pi/4 of the square around it.
above = Camera(900, 700)
above.zoom, above.target = 3.0, (0.0, -40.0, 0.0)
above.pitch = math.radians(89.0)


def footprint(shape):
    prop = props.make(shape, (80.0, 80.0, 80.0), (0.0, -80.0, 0.0))
    image = np.asarray(anatomy_depth_image([], above, frame_rect(900, 700, 1.0),
                                           256, 256, 1.0, [prop]))
    return int((image > 0).sum())


ratio = footprint("cylinder") / float(footprint("box"))
check("and a cylinder still comes out round", abs(ratio - math.pi / 4.0) < 0.03,
      "covers %.3f of the box around it, pi/4 is %.3f"
      % (ratio, math.pi / 4.0))

# -- the viewport ----------------------------------------------------------
app.set_flag("show_body", True)
app.redraw()
root.update()
check("the viewport draws without complaint with objects in the scene",
      len(app.canvas.find_all()) > 0)

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
root.destroy()
sys.exit(0 if ok else 1)
