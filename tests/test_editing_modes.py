import os, sys, math, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk, numpy as np
from openpose3d_editor import *
root=tk.Tk(); root.geometry("1180x800"); app=EditorApp(root); root.update(); root.update_idletasks()
class E:
    def __init__(s,x,y,state=0): s.x,s.y,s.state=x,y,state
ok=True
def check(l,c,e=""):
    global ok; ok=ok and bool(c); print(("PASS " if c else "FAIL ")+l+("  "+e if e else ""))

# --- ortho views
check("five ortho views built", len(app.ortho_views)==5,
      str([v.name for v in app.ortho_views]))
app.redraw(); root.update()
for v in app.ortho_views:
    check("  %s view drew" % v.name, len(v.canvas.find_all()) > 20,
          "%d items" % len(v.canvas.find_all()))
front, left, top = app.ortho_views[0], app.ortho_views[1], app.ortho_views[2]
check("front looks down -Z", abs(front.camera.basis()[2][2]+1) < 1e-9)
check("left looks along X", abs(abs(left.camera.basis()[2][0])-1) < 1e-9)
check("top looks down", left.camera.basis()[2][1] > -1 and abs(top.camera.basis()[2][1]+1) < 0.05)
# ortho views auto-fit: figure fills the small canvas
pts = [front.camera.project(p) for p in app.skeleton.points]
check("front view frames the figure",
      all(0 <= x <= front.camera.width and 0 <= y <= front.camera.height for x,y,_ in pts))

# --- dragging inside an ortho view edits the figure, does not orbit it
before_yaw = top.camera.yaw
sx, sy, _ = top.camera.project(app.skeleton.points[4])
app.on_press(E(sx,sy), top)
for i in range(1,8): app.on_drag(E(sx+3*i, sy+4*i), top)
app.on_release(E(sx,sy), top)
check("dragging in the top view moved the wrist",
      vlen(vsub(app.skeleton.points[4], Skeleton().points[4])) > 1.0)
app.on_press(E(5,5), top); app.on_drag(E(60,40), top); app.on_release(E(60,40), top)
check("locked view refuses to orbit", top.camera.yaw == before_yaw)

# --- symmetry
app.reset_pose(); app.symmetry = True
sk = app.skeleton; plane = sk.sagittal_plane()
sx, sy, _ = app.camera.project(sk.points[3])
app.on_press(E(sx,sy))
for i in range(1,9): app.on_drag(E(sx-6*i, sy-5*i))
app.on_release(E(sx,sy))
r, l = sk.points[3], sk.points[6]
check("mirrored elbow follows", vlen(vsub(l, Skeleton.reflect(r, plane))) < 1e-6,
      "%.4f cm apart" % vlen(vsub(l, Skeleton.reflect(r, plane))))
check("wrists carried along too",
      vlen(vsub(sk.points[7], Skeleton.reflect(sk.points[4], plane))) < 1e-6)
check("bone lengths intact",
      max(abs(vlen(vsub(sk.points[c],sk.points[p]))-sk.lengths[c]) for p,c in LIMB_SEQ) < 1e-9)
app.symmetry = False

# --- double click flips hemisphere
app.reset_pose(); sk = app.skeleton
_,_,fwd = app.camera.basis()
# give the elbow real depth first, or the flip is a no-op and proves nothing
L = sk.lengths[3]
sk.move_joint(3, sk.solve_drag(3, app.camera.screen_delta_to_world(-0.4*L*app.camera.zoom, -0.3*L*app.camera.zoom), fwd, 1.0))
d0 = vdot(vsub(sk.points[3], sk.points[2]), fwd)
check("elbow now has depth to flip", abs(d0) > 5.0, "%.2f cm" % d0)
sx, sy, _ = app.camera.project(sk.points[3])
app.on_double(E(sx,sy))
d1 = vdot(vsub(sk.points[3], sk.points[2]), fwd)
check("double click negates depth", abs(d1+d0) < 1e-9, "%.3f -> %.3f" % (d0,d1))
check("double click kept the bone length",
      abs(vlen(vsub(sk.points[3], sk.points[2])) - sk.lengths[3]) < 1e-9)
app.on_double(E(*app.camera.project(sk.points[3])[:2]))
check("double click again flips it back", abs(vdot(vsub(sk.points[3],sk.points[2]),fwd) - d0) < 1e-9)

# --- anchor
app.reset_pose(); sk = app.skeleton
app.selected = 9; app.set_anchor()
check("anchor set to r_knee", sk.anchor == 9)
knee, ankle, neck = sk.points[9], sk.points[10], sk.points[1]
sx, sy, _ = app.camera.project(sk.points[8])
app.on_press(E(sx,sy))
for i in range(1,10): app.on_drag(E(sx+7*i, sy-3*i))
app.on_release(E(sx,sy))
check("anchored knee stayed put", vlen(vsub(sk.points[9], knee)) < 1e-9)
check("shin below it stayed put", vlen(vsub(sk.points[10], ankle)) < 1e-9)
check("upper body rotated around it", vlen(vsub(sk.points[1], neck)) > 1.0)
app.selected = 9; app.set_anchor()
check("pressing again clears the anchor", sk.anchor == ROOT)

# --- axis rotation in steps
app.reset_pose(); sk = app.skeleton
app.turn_step.set("15")
head0 = sk.points[0]
for _ in range(6): app.rotate_figure(1, "y")
check("6 x 15 deg about Y returns a quarter turn",
      abs(vlen(vsub(sk.points[0], sk.points[1])) - vlen(vsub(head0, Skeleton().points[1]))) < 1e-9)
app.reset_pose(); sk = app.skeleton
app.selected = 9; app.set_anchor(); knee = sk.points[9]
app.rotate_figure(1, "x"); app.rotate_figure(1, "z")
check("rotation pivots on the anchor", vlen(vsub(sk.points[9], knee)) < 1e-9)
app.turn_step.set("bad"); check("bad step falls back to 15", app._step() == 15.0)

# --- perf with the extra views
app.reset_pose(); app.redraw(); root.update()
t=time.time()
for i in range(20): app.camera.orbit(4,1); app.redraw()
root.update()
dt=(time.time()-t)/20
print("4 viewports: %.1f ms/frame (%.0f fps)" % (dt*1000, 1/dt))
app.toggle_ortho(); root.update()
t=time.time()
for i in range(20): app.camera.orbit(4,1); app.redraw()
root.update()
print("ortho off:  %.1f ms/frame" % ((time.time()-t)/20*1000))
print("\n"+("ALL PASS" if ok else "FAILURES"))
root.destroy()
