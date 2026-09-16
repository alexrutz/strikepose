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
WRIST_R = app.skeleton.pose.bone("wrist.R")
WRIST_L = app.skeleton.pose.bone("wrist.L")
ELBOW_R = app.skeleton.pose.bone("lowerarm01.R")
ELBOW_L = app.skeleton.pose.bone("lowerarm01.L")
rest_points = list(app.skeleton.points)
sx, sy, _ = top.camera.project(app.skeleton.points[WRIST_R])
app.on_press(E(sx,sy), top)
for i in range(1,8): app.on_drag(E(sx+3*i, sy+4*i), top)
app.on_release(E(sx,sy), top)
check("dragging in the top view moved the wrist",
      vlen(vsub(app.skeleton.points[WRIST_R], rest_points[WRIST_R])) > 1.0)
app.on_press(E(5,5), top); app.on_drag(E(60,40), top); app.on_release(E(60,40), top)
check("locked view refuses to orbit", top.camera.yaw == before_yaw)

# --- symmetry
app.reset_pose(); app.symmetry = True
sk = app.skeleton; plane = sk.sagittal_plane()
sx, sy, _ = app.camera.project(sk.points[ELBOW_R])
app.on_press(E(sx,sy))
for i in range(1,9): app.on_drag(E(sx-6*i, sy-5*i))
app.on_release(E(sx,sy))
# The body rests facing +Z with its left at +X, so mirroring is a sign flip on
# X - and because what is mirrored is the limb's ANGLES rather than one
# joint's position, it holds all the way to the fingertips, which eighteen
# keypoints had nothing to say about.
flip = lambda p: (-p[0], p[1], p[2])
r, l = sk.points[ELBOW_R], sk.points[ELBOW_L]
check("mirrored elbow follows", vlen(vsub(l, flip(r))) < 1e-6,
      "%.4e cm apart" % vlen(vsub(l, flip(r))))
check("wrists carried along too",
      vlen(vsub(sk.points[WRIST_L], flip(sk.points[WRIST_R]))) < 1e-6)
tipR, tipL = sk.pose.bone("finger3-3.R"), sk.pose.bone("finger3-3.L")
check("and so are the fingertips",
      vlen(vsub(sk.points[tipL], flip(sk.points[tipR]))) < 1e-6,
      "%.4e cm apart" % vlen(vsub(sk.points[tipL], flip(sk.points[tipR]))))
check("bone lengths intact",
      max(abs(v - sk.lengths[j]) for j, v in sk.lengths.items()) < 1e-9)
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
#
# One joint, pinned: whatever the figure does next, it does without moving
# that point. Two anchors used to make a hinge by re-rooting the keypoint
# tree; a rig has one root and every bone hangs off it, so the same gesture is
# an IK solve rather than a re-parent, and it is not on this path.
app.reset_pose(); sk = app.skeleton
KNEE = sk.pose.bone("lowerleg01.R")
app.selected = KNEE; app.set_anchor()
check("anchor set to the right knee", sk.anchors == {KNEE})
knee, neck = sk.points[KNEE], sk.points[sk.pose.bone("neck01")]
sx, sy, _ = app.camera.project(sk.points[sk.pose.bone("lowerarm01.R")])
app.on_press(E(sx,sy))
for i in range(1,10): app.on_drag(E(sx+7*i, sy-3*i))
app.on_release(E(sx,sy))
check("the anchored knee stayed put", vlen(vsub(sk.points[KNEE], knee)) < 1e-9,
      "%.2e cm" % vlen(vsub(sk.points[KNEE], knee)))
check("and the arm that was dragged did move",
      vlen(vsub(sk.points[sk.pose.bone("wrist.R")], rest_points[
          sk.pose.bone("wrist.R")])) > 1.0)
app.selected = KNEE; app.set_anchor()
check("pressing again clears the anchor", not sk.anchors)

# --- axis rotation in steps
app.reset_pose(); sk = app.skeleton
app.turn_step.set("15")
before = list(sk.points)
for _ in range(6): app.rotate_figure(1, "y")
# 90 degrees about Y: x and z swap, and no bone changes length because a turn
# is one rotation of the root bone rather than a rewrite of every point
# about the ROOT BONE's own head, which is where the figure stands, not the
# world origin
pivot = before[sk.pose.roots[0]]
turned = [(pivot[0] + (p[2] - pivot[2]), p[1], pivot[2] - (p[0] - pivot[0]))
          for p in before]
drift = max(vlen(vsub(a, b)) for a, b in zip(sk.points, turned))
check("6 x 15 deg about Y is a quarter turn", drift < 1e-6, "%.2e cm" % drift)
check("and it resized nothing",
      max(abs(v - sk.lengths[j]) for j, v in sk.lengths.items()) < 1e-9)
app.reset_pose(); sk = app.skeleton
app.selected = sk.pose.bone("lowerleg01.R"); app.set_anchor()
knee = sk.points[sk.pose.bone("lowerleg01.R")]
app.rotate_figure(1, "x"); app.rotate_figure(1, "z")
check("rotation pivots on the anchor",
      vlen(vsub(sk.points[sk.pose.bone("lowerleg01.R")], knee)) < 1e-9)
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
