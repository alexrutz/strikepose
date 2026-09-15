import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# find the r_wrist on screen and drag it upward
idx = KEYPOINT_NAMES.index("r_wrist")
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
after = {c: vlen(vsub(app.skeleton.points[c], app.skeleton.points[p])) for p, c in LIMB_SEQ}
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
    return {c: vlen(vsub(app.skeleton.points[c], app.skeleton.points[p]))
            for p, c in LIMB_SEQ}

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
