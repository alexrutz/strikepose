import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk
from openpose3d_editor import EditorApp, KEYPOINT_NAMES, LIMB_SEQ, vlen, vsub

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
print("OK no exceptions")
root.destroy()
