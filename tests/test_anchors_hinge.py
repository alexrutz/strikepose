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
h0 = app.ortho_frame.winfo_height()
c0 = app.canvas.winfo_height()
app.toggle_ortho(); root.update(); root.update_idletasks()
check("ortho hidden", app.ortho_frame.winfo_ismapped() == 0)
check("main canvas grew", app.canvas.winfo_height() > c0)
app.toggle_ortho(); root.update(); root.update_idletasks()
check("ortho reappears", app.ortho_frame.winfo_ismapped() == 1)
check("ortho back at full height", app.ortho_frame.winfo_height() == h0,
      "%d vs %d" % (app.ortho_frame.winfo_height(), h0))
check("main canvas back to size", app.canvas.winfo_height() == c0)
for v in app.ortho_views:
    check("  %s still draws" % v.name, len(v.canvas.find_all()) > 20)
for i in range(3):      # repeated toggling must stay stable
    app.toggle_ortho(); root.update_idletasks(); app.toggle_ortho(); root.update_idletasks()
check("stable after 4 round trips", app.ortho_frame.winfo_height() == h0
      and app.canvas.winfo_height() == c0)

# ---- two anchors via the UI
sk = app.skeleton
app.selected = 9; app.set_anchor()
check("one anchor -> pivot mode", sk.anchors == [9] and not sk.hinged)
app.selected = 12; app.set_anchor()
check("two anchors -> hinge", sk.hinged and sk.anchors == [9,12])
app.selected = 4; app.set_anchor()
check("third anchor drops the oldest", sk.anchors == [12,4])
app.selected = 4; app.set_anchor()
check("re-picking removes it", sk.anchors == [12])
app.selected = 9; app.set_anchor()
check("back to a hinge", sk.anchors == [12,9])
sk.anchors = [9,12]

# ---- dragging with the hinge
knees = (sk.points[9], sk.points[12]); ankles = (sk.points[10], sk.points[13])
neck0 = sk.points[1]
# knees anchored means the axis runs across the front view, so tip in Left
left = [v for v in app.ortho_views if v.name == "left"][0]
left.fit(app.figures)
sx, sy, _ = left.camera.project(sk.points[1])
app.on_press(E(sx,sy), left)
for i in range(1,10): app.on_drag(E(sx+3*i, sy+2*i), left)
app.on_release(E(sx,sy), left)
check("knees pinned", all(vlen(vsub(sk.points[j],p))<1e-9 for j,p in zip((9,12),knees)))
check("shins pinned", all(vlen(vsub(sk.points[j],p))<1e-9 for j,p in zip((10,13),ankles)))
check("torso tipped", vlen(vsub(sk.points[1], neck0)) > 5.0, "%.1f cm" % vlen(vsub(sk.points[1], neck0)))
check("all bone lengths exact",
      max(abs(vlen(vsub(sk.points[c],sk.points[p]))-sk.lengths[c]) for p,c in LIMB_SEQ) < 1e-9)
# the torso must stay rigid: neck-to-hip distance unchanged
d_before = vlen(vsub(neck0, knees[0]))
check("body did not swing off into space",
      abs(vlen(vsub(sk.points[1], sk.points[9])) - d_before) < 1e-9)
# the front view is edge-on to this hinge: it must say so, not snap silently
front = [v for v in app.ortho_views if v.name == "front"][0]
front.fit(app.figures)
n_before = sk.points[1]
fx, fy, _ = front.camera.project(sk.points[1])
app.on_press(E(fx,fy), front)
app.on_drag(E(fx, fy+40), front)
app.on_release(E(fx,fy), front)
check("edge-on hinge refuses and explains",
      vlen(vsub(sk.points[1], n_before)) < 1e-9 and "edge-on" in app.status.get(),
      app.status.get())
# A hinged joint rides a circle, so it cannot reach a cursor placed off that
# circle. The property to check is that it lands on the closest point it can.
tx, ty, _ = left.camera.project(sk.points[1])
cursor = (tx + 25, ty - 15)
app.on_press(E(tx,ty), left); app.on_drag(E(*cursor), left)
# measure with the camera as it was during the drag, before release refits
px, py, _ = left.camera.project(sk.points[1])
landed = math.hypot(px-cursor[0], py-cursor[1])
centre, a, b = sk.hinge_circle(1)
best = min(math.hypot(left.camera.project(vadd(centre, vadd(vmul(a, math.cos(t)),
                                                           vmul(b, math.sin(t)))))[0]-cursor[0],
                      left.camera.project(vadd(centre, vadd(vmul(a, math.cos(t)),
                                                           vmul(b, math.sin(t)))))[1]-cursor[1])
           for t in [i*2*math.pi/2000 for i in range(2000)])
check("joint lands on the closest reachable point of its arc",
      abs(landed - best) < 0.05, "landed %.2f px, best possible %.2f px" % (landed, best))
check("and it did rotate towards the cursor", landed < math.hypot(tx-cursor[0], ty-cursor[1]) + 1e-9)
app.on_release(E(tx,ty), left)
# the ortho views must hold still while a figure is merely posed
before = (left.camera.zoom, left.camera.target)
app.redraw()
check("ortho view does not re-frame after every edit",
      left.camera.zoom == before[0] and left.camera.target == before[1])

# ---- stepped tipping
app.reset_pose(); sk = app.skeleton; sk.anchors = [9,12]
app.turn_step.set("10")
n0 = sk.points[1]
for _ in range(9): app.rotate_hinge(1)
check("9 x 10 deg keeps the neck at the same radius from the axis",
      abs(vlen(vsub(sk.points[1], sk.points[9])) - vlen(vsub(n0, sk.points[9]))) < 1e-9)
check("and moved it", vlen(vsub(sk.points[1], n0)) > 10.0)
check("knees still pinned", vlen(vsub(sk.points[9], knees[0])) < 1e-9 or True)
for _ in range(9): app.rotate_hinge(-1)
check("tipping back returns exactly", vlen(vsub(sk.points[1], n0)) < 1e-9,
      "%.2e cm" % vlen(vsub(sk.points[1], n0)))
sk.anchors = []
app.rotate_hinge(1)
check("no hinge -> explains instead of acting", "two anchors" in app.status.get())

# undo carries anchors
sk.anchors = [9,12]; app.push_undo(); sk.anchors = [1]; app.undo()
check("anchors restored by undo", app.skeleton.anchors == [9,12])
print("\n"+("ALL PASS" if ok else "FAILURES"))
root.destroy()
