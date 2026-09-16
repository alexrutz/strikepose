import os, sys, math; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk
from openpose3d_editor import *
ok=True
def check(l,c,e=""):
    global ok; ok=ok and bool(c); print(("PASS " if c else "FAIL ")+l+("  "+e if e else ""))
root=tk.Tk(); root.geometry("1240x860"); app=EditorApp(root); root.update(); root.update_idletasks()
class E:
    def __init__(s,x,y,state=0): s.x,s.y,s.state=x,y,state

# ---- ortho toggle round trip
#
# The locked views are a column down the LEFT now, not a strip across the
# bottom: the export frame is 2:3 and the window is wide, so fitting that
# frame into the canvas left most of the width black whatever the window
# size. So hiding them gives the main canvas width back, not height.
w0 = app.ortho_frame.winfo_width()
c0 = app.canvas.winfo_width()
app.toggle_ortho(); root.update(); root.update_idletasks()
check("ortho hidden", app.ortho_frame.winfo_ismapped() == 0)
check("main canvas grew", app.canvas.winfo_width() > c0)
app.toggle_ortho(); root.update(); root.update_idletasks()
check("ortho reappears", app.ortho_frame.winfo_ismapped() == 1)
check("ortho back at full width", app.ortho_frame.winfo_width() == w0,
      "%d vs %d" % (app.ortho_frame.winfo_width(), w0))
check("main canvas back to size", app.canvas.winfo_width() == c0)
for v in app.ortho_views:
    check("  %s still draws" % v.name, len(v.canvas.find_all()) > 20)
for i in range(3):      # repeated toggling must stay stable
    app.toggle_ortho(); root.update_idletasks(); app.toggle_ortho(); root.update_idletasks()
check("stable after 4 round trips", app.ortho_frame.winfo_width() == w0
      and app.canvas.winfo_width() == c0)

# ---- anchoring
#
# ONE anchor, not two. The pair used to make a hinge the body swung about,
# built by re-rooting the keypoint tree at those two joints so the chain ran
# outward from them. A rig cannot be re-rooted - every one of its 104 bones
# hangs off a single root - so the same gesture is an IK solve, and it is not
# on this path. What a single anchor means carries over exactly: whatever the
# figure does next, it does without moving that point.
sk = app.skeleton
KNEE = sk.pose.bone("lowerleg01.R")
ELBOW = sk.pose.bone("lowerarm01.R")
WRIST = sk.pose.bone("wrist.R")
app.selected = KNEE; app.set_anchor()
check("one anchor pins a joint", sk.anchors == {KNEE} and not sk.hinged)
app.selected = ELBOW; app.set_anchor()
check("a second replaces it rather than making a hinge", sk.anchors == {ELBOW})
app.selected = ELBOW; app.set_anchor()
check("re-picking clears it", not sk.anchors)
app.selected = KNEE; app.set_anchor()
check("and it can be set again", sk.anchors == {KNEE})

# ---- dragging with a joint pinned
knee0 = sk.points[KNEE]
wrist0 = sk.points[WRIST]
lengths0 = dict(sk.lengths)
sx, sy, _ = app.camera.project(sk.points[ELBOW])
app.on_press(E(sx, sy))
check("the press picked the elbow", app.drag_joint == ELBOW,
      str(app.drag_joint))
for i in range(1, 10): app.on_drag(E(sx + 3*i, sy - 2*i))
app.on_release(E(sx, sy))
check("the pinned knee stayed exactly put",
      vlen(vsub(sk.points[KNEE], knee0)) < 1e-9,
      "%.2e cm" % vlen(vsub(sk.points[KNEE], knee0)))
check("and the dragged arm moved",
      vlen(vsub(sk.points[WRIST], wrist0)) > 1.0,
      "%.1f cm" % vlen(vsub(sk.points[WRIST], wrist0)))
check("with every one of the 103 bones the length it was",
      max(abs(v - lengths0[j]) for j, v in sk.lengths.items()) < 1e-9)

# clearing it through the panel, not just by re-picking
app.clear_anchor()
check("the panel clears it too", not sk.anchors)

# the ortho views must hold still while a figure is merely posed
left = [v for v in app.ortho_views if v.name == "left"][0]
left.fit(app.figures)
before = (left.camera.zoom, left.camera.target)
app.redraw()
check("ortho view does not re-frame after every edit",
      left.camera.zoom == before[0] and left.camera.target == before[1])

# ---- stepped turning, pivoting on the anchor
app.reset_pose(); sk = app.skeleton
KNEE = sk.pose.bone("lowerleg01.R")
NECK = sk.pose.bone("neck01")
sk.anchor(KNEE)
app.turn_step.set("10")
n0, knee0 = sk.points[NECK], sk.points[KNEE]
for _ in range(9): app.rotate_figure(1, "y")
check("9 x 10 deg keeps the neck the same distance from the anchor",
      abs(vlen(vsub(sk.points[NECK], sk.points[KNEE]))
          - vlen(vsub(n0, knee0))) < 1e-9)
check("and moved it", vlen(vsub(sk.points[NECK], n0)) > 10.0,
      "%.1f cm" % vlen(vsub(sk.points[NECK], n0)))
check("the anchor itself did not move",
      vlen(vsub(sk.points[KNEE], knee0)) < 1e-9)
for _ in range(9): app.rotate_figure(-1, "y")
check("turning back returns exactly", vlen(vsub(sk.points[NECK], n0)) < 1e-6,
      "%.2e cm" % vlen(vsub(sk.points[NECK], n0)))

# the hinge is gone, and says so rather than doing nothing quietly
app.rotate_hinge(1)
check("the two-anchor hinge explains itself instead of acting",
      "not on the rig path" in app.status.get(), app.status.get())

# undo carries the anchor
sk.anchor(KNEE); app.push_undo(); sk.anchor(NECK); app.undo()
check("the anchor is restored by undo", app.skeleton.anchors == {KNEE},
      str(app.skeleton.anchors))
print("\n"+("ALL PASS" if ok else "FAILURES"))
root.destroy()
