import os
import math, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

from openpose3d_editor import (Skeleton, Camera, LIMB_SEQ, KEYPOINT_NAMES, PARENT,
                               vlen, vsub, vdot, vadd, vmul, vnorm, vcross,
                               rotation_between, matvec, body_parts, carry_frame,
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

# --- limb volumes: the swept cross-section must not twist or flip
#
# Every station of a limb carries a half width, a half depth and a forward
# offset, so which way round the cross-section sits is visible in the depth
# map. Deriving that from the body's facing direction per bone - which is what
# this did - left it undefined for a bone pointing along `facing` and flipped
# it a full 180 degrees either side: a reach, a sitting thigh or a kick came
# out with the calf mass on the wrong side of the tibia and the width and depth
# swapped. These are the checks that were missing.

def aim_limb(sk, parent, child, direction):
    """Point the bone parent->child along `direction`, carrying its subtree."""
    ids = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
    origin = sk.points[ids[parent]]
    R = rotation_between(vsub(sk.points[ids[child]], origin), direction)
    moved, stack = set(), [ids[child]]
    while stack:
        j = stack.pop()
        if j in moved:
            continue
        moved.add(j)
        stack.extend(c for pa, c in LIMB_SEQ if pa == j)
    for j in moved:
        sk.points[j] = vadd(origin, matvec(R, vsub(sk.points[j], origin)))


def limb_cross(parent, child, direction):
    """(axis, forward) of the swept cross-section of one posed limb bone."""
    ids = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
    sk = Skeleton()
    aim_limb(sk, parent, child, direction)
    start = sk.points[ids[parent]]
    axis = vnorm(vsub(sk.points[ids[child]], start))
    for part in body_parts(sk):
        if part[0][0] == "slab" and vlen(vsub(part[0][1], start)) < 1.0:
            return axis, part[0][2][1]
    raise AssertionError("no swept part starts at " + parent)


def twist_of(parent, child, direction):
    """Roll the cross-section has picked up that the bone's swing does not
    account for. A limb keeps its shape only while this stays zero."""
    axis, fwd = limb_cross(parent, child, direction)
    rest_axis, rest_fwd = limb_cross(parent, child, (0.0, -1.0, 0.0))
    want = matvec(rotation_between(rest_axis, axis), rest_fwd)
    want = vsub(want, vmul(axis, vdot(want, axis)))
    if vlen(want) < 1e-9:
        return 0.0
    want = vnorm(want)
    return abs(math.degrees(math.atan2(vdot(vcross(want, fwd), axis),
                                       max(-1.0, min(1.0, vdot(want, fwd))))))


for label, direction in (("hanging", (0, -1, 0)),
                         ("reaching straight forward", (0, 0, 1)),
                         ("reaching forward and up", (0, 1, 1)),
                         ("reaching forward and out", (1, 0, 1)),
                         ("out sideways", (1, 0, 0)),
                         ("just short of forward", (0, -0.02, 1)),
                         ("just past forward", (0, 0.02, 1))):
    worst = max(twist_of("l_shoulder", "l_elbow", vnorm(direction)),
                twist_of("l_hip", "l_knee", vnorm(direction)))
    check("a limb %s keeps its cross-section square" % label, worst < 1.0,
          "%.1f deg of twist" % worst)

# and it must not jump: swing a limb right through the body's own forward
worst, prev = 0.0, None
for deg in range(-170, 171, 4):
    a = math.radians(deg)
    axis, fwd = limb_cross("l_hip", "l_knee", vnorm((0.0, -math.cos(a), math.sin(a))))
    if prev is not None:
        worst = max(worst, math.degrees(math.acos(
            max(-1.0, min(1.0, vdot(prev, fwd))))))
    prev = fwd
check("and does not jump swung right round", worst < 12.0,
      "worst step %.1f deg over 4 deg moves" % worst)

# carry_frame is what guarantees it: a swing adds no roll of its own
axis = vnorm((0.3, -0.5, 0.8))
fwd, side = carry_frame(((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)), axis)
check("carry_frame returns a frame square to the bone",
      abs(vdot(fwd, axis)) < 1e-12 and abs(vdot(side, axis)) < 1e-12
      and abs(vdot(fwd, side)) < 1e-12 and abs(vlen(fwd) - 1.0) < 1e-12)
check("and leaves a bone that has not moved alone",
      vlen(vsub(carry_frame(((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
                            (0.0, -1.0, 0.0))[0], (0.0, 0.0, 1.0))) < 1e-12)


print("\nALL PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
