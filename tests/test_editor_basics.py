import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os as _os, tempfile as _tempfile
_os.environ["STRIKEPOSE_SETTINGS"] = _os.path.join(
    _tempfile.mkdtemp(), "settings.json")
import tkinter as tk
from openpose3d_editor import EditorApp, KEYPOINT_NAMES, LIMB_SEQ, vlen, vsub
from ui_app import TAB_ORDER

root = tk.Tk(); root.geometry("1180x800")
app = EditorApp(root)
root.update(); root.update_idletasks()
print("canvas size", app.canvas.winfo_width(), app.canvas.winfo_height())
print("status bar height", root.pack_slaves()[-1].winfo_height() if root.pack_slaves() else "?")
for w in root.pack_slaves():
    print("  child:", w.winfo_class(), w.winfo_width(), "x", w.winfo_height())
app.redraw(); root.update()
print("canvas items after redraw:", len(app.canvas.find_all()))

# find the right wrist on screen and drag it upward. It is a bone of the
# armature now, not one of eighteen keypoints - the editor poses the rig
# directly, so the thing under the cursor is the thing that comes out.
idx = app.skeleton.pose.bone("wrist.R")
sx, sy, _ = app.camera.project(app.skeleton.points[idx])
print("r_wrist screen", round(sx), round(sy), "picked:", app.pick(sx, sy))

class E:  # minimal synthetic event
    def __init__(s, x, y, state=0): s.x, s.y, s.state = x, y, state

before = dict(app.skeleton.lengths)
app.on_press(E(sx, sy))
print("selected:", app.selected, "sign:", app.drag_sign)
for step in range(1, 11):
    app.on_drag(E(sx + 4*step, sy - 12*step))
app.on_release(E(sx, sy))
after = dict(app.skeleton.lengths)     # all 103 bones, not seventeen limbs
print("max bone length drift:", max(abs(after[c]-before[c]) for c in after))
print("undo entries:", len(app.undo_stack))
print("status:", app.status.get())

# click without moving -> no new undo entry
n = len(app.undo_stack)
app.on_press(E(sx, sy)); app.on_release(E(sx, sy))
print("undo entries after bare click:", len(app.undo_stack), "(expected", n, ")")

# orbit by dragging the background, then export
app.on_press(E(30, 30))
for i in range(10): app.on_drag(E(30 + 8*i, 30 + 2*i))
app.on_release(E(100, 50))
print("yaw/pitch after orbit:", round(math.degrees(app.camera.yaw),1), round(math.degrees(app.camera.pitch),1))

app.undo(); print("after undo, undo entries:", len(app.undo_stack))
for k in "12345gndmrvl":
    app.on_key(type("K", (), {"keysym": k, "state": 0})())
root.update()
print("frame rect:", [round(v) for v in app.frame_rect()])
pts = app.export_points(512, 768)
print("export x range", round(min(p[0] for p in pts)), round(max(p[0] for p in pts)))
print("export y range", round(min(p[1] for p in pts)), round(max(p[1] for p in pts)))
app.canvas.postscript(file="/dev/null")

# ---- the randomizer, through the panel rather than through its own module
#
# The module's selftest proves the maths; this proves the button is wired to
# it - that a seed typed into the box is the seed used, that undo takes the
# whole thing back, and above all that a pose arrived at by dice still cannot
# resize a bone, which is the one guarantee the editor makes about every
# other way of moving a joint.
def bone_lengths():
    # every bone of the armature, which is what the figure is now - 103 of
    # them, not the seventeen limbs eighteen keypoints could describe
    return dict(app.skeleton.lengths)

rest_points, rest_lengths = list(app.skeleton.points), bone_lengths()
app.random_seed.set("4242")
app.randomize_pose()
root.update()
moved = max(vlen(vsub(a, b))
            for a, b in zip(rest_points, app.skeleton.points))
stretched = max(abs(rest_lengths[k] - v) for k, v in bone_lengths().items())
print("randomize moved a joint %.1f cm, stretched a bone %.2e cm"
      % (moved, stretched))
assert moved > 1.0, "the randomize button did nothing"
assert stretched < 1e-9, "randomizing resized a bone by %g cm" % stretched
assert "4242" in app.random_status.get(), app.random_status.get()
scrambled = list(app.skeleton.points)
app.undo()
root.update()
assert max(vlen(vsub(a, b))
           for a, b in zip(rest_points, app.skeleton.points)) < 1e-9, \
    "undo did not take the random pose back"
app.random_seed.set("4242")
app.randomize_pose()
root.update()
assert max(vlen(vsub(a, b))
           for a, b in zip(scrambled, app.skeleton.points)) < 1e-9, \
    "the same seed gave a different pose"
for var in app.random_parts.values():
    var.set(False)
app.randomize_pose()
assert "Nothing ticked" in app.random_status.get(), app.random_status.get()
print("randomizer: seeded, reproducible, undoable, no bone resized")

# ---- the export shape
#
# Changing the shape has to RE-FRAME, not just change two numbers: the export
# rectangle is this ratio, so a figure fitted to a tall frame is not fitted to
# a wide one. Turning a 2:3 portrait on its side without re-framing crops the
# head and the feet off, and the only sign of it is in the PNG.
import exporting
app.set_aspect("2:3 portrait")
root.update()
for shape in ("16:9 wide", "1:1 square", "9:16 tall", "3:2 landscape"):
    app.set_aspect(shape)
    root.update()
    w, h = app._sizes()
    assert exporting.aspect_name(w, h) == shape, (shape, w, h)
    inside = app.export_points(w, h)
    assert all(0 <= x <= w and 0 <= y <= h for x, y in inside), \
        "%s cropped a keypoint: x %.0f..%.0f y %.0f..%.0f of %dx%d" % (
            shape, min(p[0] for p in inside), max(p[0] for p in inside),
            min(p[1] for p in inside), max(p[1] for p in inside), w, h)
    assert app.depth_image(w, h).size == (w, h)
print("export shape: four ratios, each framed and exported without a crop")

# Typed pixels win, and are recognised by RATIO - a control that only knew its
# own defaults would call every scaled-up export custom.
app.out_w.set(1024); app.out_h.set(1536); app.sync_aspect()
assert app.aspect_name.get() == "2:3 portrait", app.aspect_name.get()
app.out_w.set(700); app.out_h.set(513); app.sync_aspect()
assert app.aspect_name.get() == "Custom", app.aspect_name.get()
app.flip_aspect()
assert app._sizes() == (513, 700), app._sizes()
print("export shape: typed pixels name their own ratio, and flip swaps them")

# ---- hands and feet, through the panel
#
# The sliders are the only way to set these - there is no keypoint past the
# wrist or the ankle to drag - so what has to hold is that they write the
# figure, read it back, survive undo, and do not spill onto the wrong target.
# The last one is the real trap: a tk Scale fires its `command` from the event
# loop rather than from the assignment, so a guard cleared at the end of the
# method that set them is already down when the callbacks land.
app.part_target.set("Both feet"); app.show_part(); root.update()
app.part_angles[0].set(20.0); app.part_angles[1].set(30.0)
app.set_part(); root.update()
assert app.skeleton.extremities.get("l_foot") == (20.0, 30.0), \
    app.skeleton.extremities
before_parts = dict(app.skeleton.extremities)
app.part_target.set("Left hand"); app.show_part(); root.update()
assert [v.get() for v in app.part_angles] == [0.0, 0.0], \
    "switching target kept the old angles on the sliders"
assert app.skeleton.extremities == before_parts, \
    "switching target wrote the old angles onto the new one"
app.part_angles[0].set(55.0); app.set_part(); root.update()
assert app.skeleton.extremities.get("l_hand") == (55.0, 0.0)
app.part_target.set("Both feet"); app.show_part(); root.update()
assert [v.get() for v in app.part_angles] == [20.0, 30.0], \
    "the sliders do not read the figure back"
app.reset_part(); root.update()
assert "l_foot" not in app.skeleton.extremities
app.undo(); root.update()
assert app.skeleton.extremities.get("l_foot") == (20.0, 30.0), \
    "undo did not bring the feet back: %s" % (app.skeleton.extremities,)
app.reset_all_parts(); root.update()
assert app.skeleton.extremities == {}
print("hands and feet: sliders write, read back, undo, and stay on target")

# ---- the panel's tabs
for name in TAB_ORDER:
    app.show_tab(name)
    root.update()
assert app.active_tab.get() == TAB_ORDER[-1]
app.next_tab(1)
assert app.active_tab.get() == TAB_ORDER[0], "Ctrl+Tab did not wrap"
print("tabs: all four raise, and cycling wraps")

print("OK no exceptions")
root.destroy()
