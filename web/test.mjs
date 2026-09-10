import * as P from "./pose3d.mjs";
let ok = true;
const check = (l, c, e="") => { ok = ok && !!c;
  console.log((c ? "PASS " : "FAIL ") + l + (e ? "  " + e : "")); };

const sk = new P.Skeleton(), cam = new P.Camera(360, 640);
const [, , fwd] = cam.basis();
check("front view looks down -Z", Math.abs(fwd[2] + 1) < 1e-9);
check("figure's left is +X", sk.points[5][0] > 0);

// bone lengths survive chained drags, as in the Python build
const before = {...sk.lengths};
for (const [j, dx, dy, s] of [[3,-40,-60,-1],[4,-20,-90,-1],[9,-30,70,1],[6,50,-40,1]]) {
  const p = sk.parentOf(j);
  const off = cam.screenToWorld(dx, dy);
  sk.moveJoint(j, sk.solveDrag(j, off, fwd, s));
}
let worst = 0;
for (const [p, c] of P.LIMBS)
  worst = Math.max(worst, Math.abs(P.len(P.sub(sk.points[c], sk.points[p])) - before[c]));
check("every bone length preserved", worst < 1e-9, worst.toExponential(2));

// drop past the silhouette lands in the view plane
const sk2 = new P.Skeleton();
const L = sk2.boneLength(2, 3);
let t = sk2.solveDrag(3, cam.screenToWorld(3*L*cam.zoom, 0), fwd, 1);
check("outside the circle lies in the view plane",
      Math.abs(P.dot(P.sub(t, sk2.points[2]), fwd)) < 1e-9);
check("and keeps its length",
      Math.abs(P.len(P.sub(t, sk2.points[2])) - L) < 1e-9);

// dropping on a guide arc puts the limb in that world plane
cam.yaw = 0.6; cam.pitch = 0.35;
const [, , f2] = cam.basis();
const sk3 = new P.Skeleton(), parent = sk3.points[2], R = sk3.boneLength(2, 3);
for (const plane of P.GUIDE_PLANES) {
  let bad = 0;
  for (let step = 0; step < 96; step++) {
    const a = 2*Math.PI*step/96;
    const off = P.add(P.mul(plane.a, R*Math.cos(a)), P.mul(plane.b, R*Math.sin(a)));
    if (P.dot(off, f2) < 0) continue;
    const [px, py] = cam.project(P.add(parent, off));
    const pp = cam.project(parent);
    const got = sk3.solveDrag(3, cam.screenToWorld(px - pp[0], py - pp[1]), f2, 1);
    bad = Math.max(bad, P.len(P.sub(P.sub(got, parent), off)));
  }
  check(`dropping on the ${plane.name} arc lands in that plane`, bad < 1e-9,
        bad.toExponential(2));
}
const arcs = P.guideArcs(cam, parent, R, 1);
check("three arcs, each clipped to one hemisphere",
      arcs.length === 3 && arcs.every(a => a.runs.length >= 1));

// anchors re-root the tree
const sk4 = new P.Skeleton(); sk4.anchors = [9];
check("anchoring a knee makes the hip its child", sk4.parentOf(8) === 9);
const knee = sk4.points[9].slice(), ankle = sk4.points[10].slice();
sk4.moveJoint(8, sk4.solveDrag(8, cam.screenToWorld(40, -20), f2, 1));
check("anchored knee and shin stay put",
      P.len(P.sub(sk4.points[9], knee)) < 1e-9 && P.len(P.sub(sk4.points[10], ankle)) < 1e-9);
check("upper body moved", P.len(P.sub(sk4.points[1], new P.Skeleton().points[1])) > 1);

// renderers
const W = 256, H = 384;
const capsules = P.bodyCapsules(new P.Skeleton("Female, curvy")).map(({a,b,ra,rb}) => {
  const c2 = new P.Camera(W, H); c2.zoom = 1.4;
  const pa = c2.project(a), pb = c2.project(b), k = c2.zoom;
  return {a: [pa[0], pa[1], pa[2]*k], b: [pb[0], pb[1], pb[2]*k], ra: ra*k, rb: rb*k};
});
const t0 = Date.now();
const depth = P.renderDepth(capsules, W, H);
const ms = Date.now() - t0;
const covered = depth.reduce((n, v) => n + (v > 0 ? 1 : 0), 0);
check("depth map covers the figure", covered > W*H*0.05 && covered < W*H*0.6,
      `${covered} px in ${ms} ms`);
check("background is black and the nearest surface is white",
      depth[0] === 0 && Math.max(...depth) === 255);

const cam3 = new P.Camera(W, H); cam3.zoom = 1.4;
const pts = new P.Skeleton().points.map(p => cam3.project(p).slice(0, 2));
const rgb = P.renderPose([{points: pts, visible: KEY(true)}], W, H);
function KEY(v) { return P.KEYPOINTS.map(() => v); }
let lit = 0; for (let i = 0; i < rgb.length; i += 3) if (rgb[i]||rgb[i+1]||rgb[i+2]) lit++;
check("pose raster draws something", lit > 500, `${lit} px`);
const exact = new Set();
for (let i = 0; i < rgb.length; i += 3) exact.add(`${rgb[i]},${rgb[i+1]},${rgb[i+2]}`);
check("canonical keypoint colours present",
      P.COLORS.filter(c => exact.has(c.join(","))).length >= 12);

// --- symmetric editing must not resize anything -------------------------
{
  const s = new P.Skeleton(), c = new P.Camera(360, 640);
  c.yaw = 0.4; c.pitch = 0.2;
  const [, , f] = c.basis();
  const base = {...s.lengths};
  // make the pose asymmetric first: that is when reflecting a position breaks
  s.moveJoint(9, s.solveDrag(9, c.screenToWorld(-70, 40), f, 1));
  let worstLen = 0, worstGeo = 0;
  for (let k = 0; k < 40; k++) {
    const j = [3,4,6,7,9,10,12,13][k % 8];
    const p = s.parentOf(j);
    const off = c.screenToWorld(((k*37)%160)-80, ((k*53)%160)-80);
    const target = s.solveDrag(j, off, f, k % 2 ? 1 : -1);
    s.moveJoint(j, target);
    s.mirrorDrag(j, target, s.sagittalPlane());
    for (const [pp, cc] of P.LIMBS) {
      worstGeo = Math.max(worstGeo,
        Math.abs(P.len(P.sub(s.points[cc], s.points[pp])) - s.lengths[cc]));
      worstLen = Math.max(worstLen, Math.abs(s.lengths[cc] - base[cc]));
    }
  }
  check("symmetric editing keeps every bone length", worstLen < 1e-9,
        worstLen.toExponential(2));
  check("and the geometry matches the stored lengths", worstGeo < 1e-9,
        worstGeo.toExponential(2));

  // it must still actually mirror
  const sym = new P.Skeleton(), plane = sym.sagittalPlane();
  const t = sym.solveDrag(3, c.screenToWorld(-60, -50), f, -1);
  sym.moveJoint(3, t);
  sym.mirrorDrag(3, t, plane);
  const right = P.sub(sym.points[3], sym.points[2]);
  const left = P.sub(sym.points[6], sym.points[5]);
  const n = plane[1];
  const want = P.sub(right, P.mul(n, 2*P.dot(right, n)));
  check("mirrored limb points the reflected way",
        P.len(P.sub(P.norm(left), P.norm(want))) < 1e-9);
}
console.log(ok ? "\nALL PASS" : "\nFAILURES");
process.exit(ok ? 0 : 1);
