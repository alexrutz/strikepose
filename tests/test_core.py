import os
import math, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

from openpose3d_editor import (Skeleton, Camera, LIMB_SEQ, KEYPOINT_NAMES, PARENT,
                               vlen, vsub, vdot, vadd, vmul, rotation_between, matvec,
                               render_openpose, scene_to_dict, scene_from_dict)

ok = True
def check(name, cond, extra=""):
    global ok
    ok = ok and cond
    print(("PASS " if cond else "FAIL ") + name + ("  " + extra if extra else ""))

# rotation_between
for a, b in [((1,0,0),(0,1,0)), ((1,0,0),(-1,0,0)), ((0.3,-2,1),(0.1,0.2,-5)), ((1,2,3),(1,2,3))]:
    R = rotation_between(a, b)
    got = matvec(R, a)
    want = vmul(b, vlen(a)/vlen(b))
    check(f"rotation {a}->{b}", vlen(vsub(got, want)) < 1e-9, f"err={vlen(vsub(got,want)):.2e}")
    # orthonormality
    cols = [tuple(R[i][j] for i in range(3)) for j in range(3)]
    check("  orthonormal", all(abs(vlen(c)-1) < 1e-9 for c in cols) and abs(vdot(cols[0], cols[1])) < 1e-9)

s = Skeleton()
cam = Camera(900, 700)
right, up, fwd = cam.basis()
print("basis right", right, "up", up, "fwd", fwd)
check("front view fwd is -Z", abs(fwd[2] + 1) < 1e-9)

L = s.lengths[3]  # r_elbow bone
print("r_elbow bone length", L)

# --- drag inside the circle: on-screen length foreshortened, 3D length preserved
off = cam.screen_delta_to_world(0.5*L*cam.zoom, 0.0)   # half the reach, to the right
tgt = s.solve_drag(3, off, fwd, +1.0)
d3 = vlen(vsub(tgt, s.points[PARENT[3]]))
check("inside circle keeps 3D length", abs(d3 - L) < 1e-9, f"{d3:.6f} vs {L:.6f}")
check("inside circle uses far hemisphere", vdot(vsub(tgt, s.points[PARENT[3]]), fwd) > 0)
onscreen = vlen(vsub(vsub(tgt, s.points[PARENT[3]]), vmul(fwd, vdot(vsub(tgt, s.points[PARENT[3]]), fwd))))
check("on-screen length is 0.5L", abs(onscreen - 0.5*L) < 1e-6, f"{onscreen:.4f}")

# --- drag outside the circle: clamped to the silhouette, lies in the view plane
off = cam.screen_delta_to_world(3.0*L*cam.zoom, 1.0*L*cam.zoom)
tgt = s.solve_drag(3, off, fwd, +1.0)
v = vsub(tgt, s.points[PARENT[3]])
check("outside circle keeps 3D length", abs(vlen(v) - L) < 1e-9)
check("outside circle lies in view plane", abs(vdot(v, fwd)) < 1e-9, f"depth={vdot(v,fwd):.2e}")

# --- hemisphere sign
near = s.solve_drag(3, cam.screen_delta_to_world(0.3*L*cam.zoom, 0), fwd, -1.0)
check("sign -1 pulls towards camera", vdot(vsub(near, s.points[PARENT[3]]), fwd) < 0)

# --- move_joint preserves every bone length in the chain
before = dict(s.lengths)
s.move_joint(3, tgt)
after = {c: vlen(vsub(s.points[c], s.points[p])) for p, c in LIMB_SEQ}
worst = max(abs(after[c] - before[c]) for c in after)
check("all bone lengths preserved after FK drag", worst < 1e-9, f"worst={worst:.2e}")
check("wrist followed the elbow", vlen(vsub(s.points[4], s.points[3])) > 0)

# --- dragging the root translates everything rigidly
s2 = Skeleton()
old = list(s2.points)
s2.move_joint(1, vadd(s2.points[1], (10, 5, -3)))
deltas = [vsub(n, o) for n, o in zip(s2.points, old)]
check("root drag is a rigid translation", all(vlen(vsub(d, (10,5,-3))) < 1e-9 for d in deltas))

# --- free length drag changes only that bone
s3 = Skeleton()
L3 = s3.lengths[3]
off = cam.screen_delta_to_world(2.0*L3*cam.zoom, 0)
t3 = s3.solve_drag(3, off, fwd, 1.0, free_length=True)
s3.move_joint(3, t3, stretch=True)   # only the free-length drag resizes
# and the default must refuse to resize
s3b = Skeleton(); L3b = s3b.lengths[3]
s3b.move_joint(3, s3b.solve_drag(3, cam.screen_delta_to_world(2.0*L3b*cam.zoom, 0), fwd, 1.0, free_length=True))
check("without stretch a mismatched target only aims the bone",
      abs(s3b.lengths[3] - L3b) < 1e-9 and
      abs(vlen(vsub(s3b.points[3], s3b.points[2])) - L3b) < 1e-9)
check("free length drag extends the bone", abs(s3.lengths[3] - 2.0*L3) < 1e-6,
      f"{s3.lengths[3]:.3f} vs {2*L3:.3f}")
check("forearm length untouched", abs(vlen(vsub(s3.points[4], s3.points[3])) - s3.lengths[4]) < 1e-9)

# --- flip through the view plane
s4 = Skeleton()
s4.move_joint(3, s4.solve_drag(3, cam.screen_delta_to_world(0.5*L*cam.zoom, 0), fwd, 1.0))
d_before = vdot(vsub(s4.points[3], s4.points[2]), fwd)
s4.flip_depth(3, fwd)
d_after = vdot(vsub(s4.points[3], s4.points[2]), fwd)
check("flip negates depth", abs(d_after + d_before) < 1e-9, f"{d_before:.3f} -> {d_after:.3f}")
check("flip keeps length", abs(vlen(vsub(s4.points[3], s4.points[2])) - L) < 1e-9)

# --- mirror
s5 = Skeleton()
s5.mirror_x()
check("mirror swaps sides", abs(s5.points[2][0] - Skeleton().points[5][0] * -1) < 1e-9)

# --- render + json round trip
pts2d = []
W, H = 512, 768
for p in Skeleton().points:
    x, y, _ = cam.project(p)
    pts2d.append(((x - 200) * W / 500.0, (y - 40) * H / 620.0))
img = render_openpose([(pts2d, [True]*18)], W, H)
img.save(os.path.join(OUT, "test_pose.png"))
px = img.load()
check("render produced non-black pixels", any(px[x, y] != (0,0,0) for x in range(0, W, 3) for y in range(0, H, 3)))

sk = Skeleton()
d = scene_to_dict(sk, cam, pts2d, W, H)
check("json has 54 flat values", len(d["people"][0]["pose_keypoints_2d"]) == 54)
sk2 = Skeleton()
sk2.points = [(0,0,0)]*18
scene_from_dict(json.loads(json.dumps(d)), sk2, cam)
check("json round trip restores pose", all(vlen(vsub(a,b)) < 1e-4 for a,b in zip(sk.points, sk2.points)))

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
