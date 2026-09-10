// Core of the 3D OpenPose editor, portable to the browser.
// No DOM here, so it can be unit tested under node and reused by any UI.

export const KEYPOINTS = [
  "nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow",
  "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
  "r_eye", "l_eye", "r_ear", "l_ear"];

export const LIMBS = [[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],
                      [1,11],[11,12],[12,13],[1,0],[0,14],[14,16],[0,15],[15,17]];

export const COLORS = [[255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],
  [85,255,0],[0,255,0],[0,255,85],[0,255,170],[0,255,255],[0,170,255],
  [0,85,255],[0,0,255],[85,0,255],[170,0,255],[255,0,255],[255,0,170],[255,0,85]];

export const ROOT = 1;
export const PARENT = {};
for (const [p, c] of LIMBS) PARENT[c] = p;
export const ADJACENCY = {};
for (const [p, c] of LIMBS) {
  (ADJACENCY[p] ||= []).push(c);
  (ADJACENCY[c] ||= []).push(p);
}
export const MIRROR = {};
for (const [a, b] of [[2,5],[3,6],[4,7],[8,11],[9,12],[10,13],[14,15],[16,17]]) {
  MIRROR[a] = b; MIRROR[b] = a;
}

export const PRESETS = {
  "Male, average":   {sw:19, sd:2, hw:10, tl:52, ua:28.5, fa:26, th:43, ca:43, head:1,
                      chest:[17,11.5], waist:[14,10.4], pelvis:[17,12], girth:1},
  "Female, average": {sw:16.5, sd:1.8, hw:11, tl:49, ua:26.5, fa:24, th:40.5, ca:40,
                      head:0.95, chest:[14.6,9.6], waist:[12.2,9], pelvis:[17.6,12],
                      girth:0.88, bust:true},
  "Male, athletic":  {sw:20, sd:2, hw:9.5, tl:52, ua:28.5, fa:26, th:43, ca:43, head:1,
                      chest:[18,12.2], waist:[13.4,10], pelvis:[16.4,11.6], girth:1.15},
  "Female, curvy":   {sw:16.5, sd:1.8, hw:12, tl:49, ua:26.5, fa:24, th:40.5, ca:40,
                      head:0.95, chest:[15.4,10.2], waist:[11.8,9], pelvis:[19.4,13],
                      girth:1.0, bust:true},
  "Child, about 7":  {sw:12.5, sd:1.4, hw:7, tl:33, ua:19, fa:17, th:28, ca:27,
                      head:0.88, chest:[10.8,8], waist:[10,7.6], pelvis:[10.6,8],
                      girth:0.65},
};

// ---------------------------------------------------------------- vectors
export const add = (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
export const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
export const mul = (a, s) => [a[0]*s, a[1]*s, a[2]*s];
export const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
export const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                                a[0]*b[1]-a[1]*b[0]];
export const len = a => Math.hypot(a[0], a[1], a[2]);
export function norm(a) { const n = len(a); return n < 1e-12 ? [0,0,0] : mul(a, 1/n); }

export function rotationBetween(a, b) {
  const ua = norm(a), ub = norm(b);
  if (len(ua) < 1e-9 || len(ub) < 1e-9) return null;
  const v = cross(ua, ub), c = Math.max(-1, Math.min(1, dot(ua, ub))), s = len(v);
  if (s < 1e-9) {
    if (c > 0) return null;
    let axis = norm(cross(ua, [1,0,0]));
    if (len(axis) < 1e-9) axis = norm(cross(ua, [0,1,0]));
    const [x,y,z] = axis;
    return [[2*x*x-1, 2*x*y, 2*x*z], [2*x*y, 2*y*y-1, 2*y*z], [2*x*z, 2*y*z, 2*z*z-1]];
  }
  const [x,y,z] = mul(v, 1/s), t = 1 - c;
  return [[c+x*x*t, x*y*t-z*s, x*z*t+y*s],
          [x*y*t+z*s, c+y*y*t, y*z*t-x*s],
          [x*z*t-y*s, y*z*t+x*s, c+z*z*t]];
}
export const apply = (m, v) => m ? [dot(m[0],v), dot(m[1],v), dot(m[2],v)] : v.slice();

// ---------------------------------------------------------------- camera
export class Camera {
  constructor(width = 360, height = 640) {
    this.yaw = 0; this.pitch = 0; this.target = [0, -60, 0];
    this.zoom = 2.4; this.width = width; this.height = height;
  }
  basis() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const fwd = [-cp*sy, -sp, -cp*cy];
    let right = norm(cross(fwd, [0,1,0]));
    if (len(right) < 1e-9) right = [1,0,0];
    return [right, norm(cross(right, fwd)), fwd];
  }
  project(p) {
    const [r, u, f] = this.basis(), rel = sub(p, this.target);
    return [dot(rel,r)*this.zoom + this.width/2,
            -dot(rel,u)*this.zoom + this.height/2, dot(rel,f)];
  }
  screenToWorld(dx, dy) {
    const [r, u] = this.basis();
    return add(mul(r, dx/this.zoom), mul(u, -dy/this.zoom));
  }
  orbit(dx, dy) {
    this.yaw -= dx * 0.008;
    this.pitch = Math.max(-1.553, Math.min(1.553, this.pitch + dy * 0.008));
  }
  pan(dx, dy) {
    const [r, u] = this.basis();
    this.target = add(this.target, add(mul(r, -dx/this.zoom), mul(u, dy/this.zoom)));
  }
}

// ---------------------------------------------------------------- skeleton
function restPoints(preset) {
  const p = PRESETS[preset] || PRESETS["Male, average"], h = p.head;
  const pts = {neck: [0,0,0], nose: [0, 16*h, 5*h]};
  for (const [side, sx] of [["r", -1], ["l", 1]]) {
    pts[side+"_eye"] = [sx*3*h, 20*h, 7*h];
    pts[side+"_ear"] = [sx*7.5*h, 19*h, 1*h];
    const sh = [sx*p.sw, -p.sd, 0];
    const el = [sh[0] + sx*0.141*p.ua, sh[1] - 0.99*p.ua, 0];
    const wr = [el[0] + sx*0.077*p.fa, el[1] - 0.997*p.fa, 2];
    const hp = [sx*p.hw, -p.tl, 0];
    const kn = [hp[0] + sx*0.023*p.th, hp[1] - 0.9997*p.th, 0];
    const an = [kn[0], kn[1] - 0.9976*p.ca, kn[2] - 0.0697*p.ca];
    pts[side+"_shoulder"] = sh; pts[side+"_elbow"] = el; pts[side+"_wrist"] = wr;
    pts[side+"_hip"] = hp; pts[side+"_knee"] = kn; pts[side+"_ankle"] = an;
  }
  return KEYPOINTS.map(n => pts[n]);
}

const rerootCache = new Map();
export function reroot(anchor) {
  if (rerootCache.has(anchor)) return rerootCache.get(anchor);
  const parents = {[anchor]: -1}, children = {}, queue = [anchor];
  while (queue.length) {
    const j = queue.shift();
    for (const k of ADJACENCY[j] || []) {
      if (k in parents) continue;
      parents[k] = j; (children[j] ||= []).push(k); queue.push(k);
    }
  }
  const out = {parents, children};
  rerootCache.set(anchor, out);
  return out;
}

export class Skeleton {
  constructor(preset = "Male, average") {
    this.preset = preset;
    this.points = restPoints(preset);
    this.visible = KEYPOINTS.map(() => true);
    this.lengths = {};
    for (const [p, c] of LIMBS) this.lengths[c] = len(sub(this.points[c], this.points[p]));
    this.anchors = [];
  }
  clone() {
    const s = new Skeleton(this.preset);
    s.points = this.points.map(p => p.slice());
    s.visible = this.visible.slice();
    s.lengths = {...this.lengths};
    s.anchors = this.anchors.slice();
    return s;
  }
  tree() { return reroot(this.anchors.length ? this.anchors[0] : ROOT); }
  parentOf(i) { const p = this.tree().parents[i]; return p === undefined ? -1 : p; }
  boneLength(a, b) { return PARENT[b] === a ? this.lengths[b] : this.lengths[a]; }
  subtree(i) {
    const kids = this.tree().children, out = [], stack = [i];
    while (stack.length) {
      const j = stack.pop(); out.push(j);
      for (const k of kids[j] || []) stack.push(k);
    }
    return out;
  }
  translate(delta, indices) {
    for (const j of indices || this.points.map((_, i) => i))
      this.points[j] = add(this.points[j], delta);
  }
  /** Map an in-plane drop offset onto the sphere of reach. */
  solveDrag(i, planeOffset, fwd, sign) {
    const p = this.parentOf(i);
    if (p < 0) return add(this.points[i], planeOffset);
    const parent = this.points[p], length = this.boneLength(p, i), d = len(planeOffset);
    if (d >= length) return d < 1e-9 ? this.points[i]
                                     : add(parent, mul(planeOffset, length/d));
    const depth = Math.sqrt(Math.max(0, length*length - d*d));
    return add(parent, add(planeOffset, mul(fwd, sign*depth)));
  }
  /** Aim the bone at target, carrying the chain. The bone keeps its length
   *  unless stretch is asked for, so a target at the wrong distance aims the
   *  limb instead of resizing it. */
  moveJoint(i, target, stretch = false) {
    const pi = this.parentOf(i);
    if (pi < 0) { this.translate(sub(target, this.points[i])); return; }
    const parent = this.points[pi];
    const oldDir = sub(this.points[i], parent), newDir = sub(target, parent);
    if (len(newDir) < 1e-9) return;
    const rot = rotationBetween(oldDir, newDir), chain = this.subtree(i);
    for (const j of chain)
      this.points[j] = add(parent, apply(rot, sub(this.points[j], parent)));
    const fix = sub(target, this.points[i]);
    if (stretch && len(fix) > 1e-9) this.translate(fix, chain);
  }
  sagittalPlane() {
    let n = norm(sub(this.points[5], this.points[2]));
    if (len(n) < 1e-6) n = [1,0,0];
    const o = mul(add(add(this.points[2], this.points[5]),
                      add(this.points[8], this.points[11])), 0.25);
    return [o, n];
  }
  reflect(p, [o, n]) { return sub(p, mul(n, 2*dot(sub(p, o), n))); }
  /** Apply a drag to the opposite limb. Reflecting the target *position* only
   *  lands at the right distance from the twin's parent while the figure is
   *  perfectly symmetric; reflect the direction and use the twin's own length. */
  mirrorDrag(i, target, plane) {
    const twin = MIRROR[i];
    if (twin === undefined) return;
    const p = this.parentOf(i), tp = this.parentOf(twin);
    if (p < 0 || tp < 0) return;
    const n = plane[1];
    const off = sub(target, this.points[p]);
    const mirrored = sub(off, mul(n, 2*dot(off, n)));
    if (len(mirrored) < 1e-9) return;
    this.moveJoint(twin, add(this.points[tp],
                             mul(norm(mirrored), this.boneLength(tp, twin))));
  }
}

// ------------------------------------------------------- guides while dragging
export const GUIDE_PLANES = [
  {name: "XY", a: [1,0,0], b: [0,1,0], colour: "#4f7fff"},
  {name: "YZ", a: [0,1,0], b: [0,0,1], colour: "#ff5a5a"},
  {name: "ZX", a: [0,0,1], b: [1,0,0], colour: "#5ad07a"},
];

/** Arcs of the reach sphere in each world plane, clipped to the live
 *  hemisphere. Dropping on an arc puts the limb exactly in that plane. */
export function guideArcs(camera, parent, radius, sign) {
  const [, , fwd] = camera.basis(), out = [];
  for (const plane of GUIDE_PLANES) {
    const runs = []; let run = [];
    for (let step = 0; step <= 96; step++) {
      const t = 2*Math.PI*step/96;
      const off = add(mul(plane.a, radius*Math.cos(t)), mul(plane.b, radius*Math.sin(t)));
      if (dot(off, fwd)*sign < -1e-9) {
        if (run.length > 2) runs.push(run);
        run = []; continue;
      }
      const [x, y] = camera.project(add(parent, off));
      run.push([x, y]);
    }
    if (run.length > 2) runs.push(run);
    out.push({colour: plane.colour, runs});
  }
  return out;
}

// ---------------------------------------------------------------- body volume
/** Tapered capsules approximating the body. Simpler than the desktop build's
 *  swept elliptical profiles, but the same idea and enough for a phone. */
export function bodyCapsules(skeleton) {
  const P = skeleton.points, g = (PRESETS[skeleton.preset] || {}).girth ?? 1;
  const idx = {}; KEYPOINTS.forEach((n, i) => idx[n] = i);
  const at = n => P[idx[n]];
  const out = [];
  const push = (a, b, ra, rb) => out.push({a, b, ra: ra*g, rb: rb*g});

  for (const [child, ra, rb] of [[3, 6, 4.6], [6, 6, 4.6], [4, 4.6, 3.2],
                                 [7, 4.6, 3.2], [9, 9, 6], [12, 9, 6],
                                 [10, 6, 3.6], [13, 6, 3.6]])
    push(P[PARENT[child]], P[child], ra, rb);

  const shMid = mul(add(at("r_shoulder"), at("l_shoulder")), 0.5);
  const hipMid = mul(add(at("r_hip"), at("l_hip")), 0.5);
  const side = norm(sub(at("l_shoulder"), at("r_shoulder")));
  const up = norm(sub(shMid, hipMid));
  let facing = norm(cross(side, up));
  if (len(facing) < 1e-6) facing = [0,0,1];
  const preset = PRESETS[skeleton.preset] || PRESETS["Male, average"];
  const slabs = 26;
  for (let i = 0; i < slabs; i++) {
    const t = -0.05 + 1.1*i/(slabs-1), tc = Math.max(0, Math.min(1, t));
    const [cw, cd] = preset.chest, [ww, wd] = preset.waist, [pw, pd] = preset.pelvis;
    const u = 1 - tc, ctrlW = 2*ww - 0.5*(cw+pw), ctrlD = 2*wd - 0.5*(cd+pd);
    const halfW = u*u*cw + 2*u*tc*ctrlW + tc*tc*pw;
    const halfD = u*u*cd + 2*u*tc*ctrlD + tc*tc*pd;
    const centre = add(shMid, mul(sub(hipMid, shMid), t));
    const arm = Math.max(0, halfW - halfD);
    push(sub(centre, mul(side, arm)), add(centre, mul(side, arm)), halfD, halfD);
  }
  if (preset.bust)
    for (const s of [-1, 1])
      push(add(add(add(shMid, mul(sub(hipMid, shMid), 0.22)), mul(side, s*6.4)),
               mul(facing, preset.chest[1]*0.74)),
           add(add(add(shMid, mul(sub(hipMid, shMid), 0.22)), mul(side, s*6.4)),
               mul(facing, preset.chest[1]*0.74)), 6.4, 6.4);
  for (const sh of ["r_shoulder", "l_shoulder"]) push(at(sh), at(sh), 6.4, 6.4);
  push(at("neck"), shMid, preset.chest[1]*0.95, preset.chest[1]*0.95);

  const earMid = mul(add(at("r_ear"), at("l_ear")), 0.5);
  let axis = norm(sub(earMid, at("neck")));
  if (len(axis) < 1e-6) axis = up;
  push(sub(at("neck"), mul(axis, 3)), sub(earMid, mul(axis, 7)), 6.2, 5.6);
  push(sub(earMid, mul(axis, 2.5*preset.head)), add(earMid, mul(axis, 3*preset.head)),
       8*preset.head, 7.4*preset.head);
  for (const [wrist, elbow] of [["r_wrist","r_elbow"], ["l_wrist","l_elbow"]]) {
    const d = norm(sub(at(wrist), at(elbow)));
    push(at(wrist), add(at(wrist), mul(d, 7)), 3.8, 3.2);
  }
  for (const ankle of ["r_ankle", "l_ankle"])
    push(at(ankle), add(add(at(ankle), mul(facing, 11)), mul(up, -3)), 4.2, 3);
  return out;
}

// ---------------------------------------------------------------- depth map
/** Nearest-surface depth of a round cone along +Z, the same analytic
 *  intersection the desktop build uses. */
function coneDepth(x, y, a, b, ra, rb) {
  const bax = b[0]-a[0], bay = b[1]-a[1], baz = b[2]-a[2];
  const oax = x-a[0], oay = y-a[1], oaz = -a[2];
  const obx = x-b[0], oby = y-b[1], obz = -b[2];
  const rr = ra - rb;
  const m0 = bax*bax + bay*bay + baz*baz;
  const m1 = bax*oax + bay*oay + baz*oaz;
  const m2 = baz, m3 = oaz;
  const m5 = oax*oax + oay*oay + oaz*oaz;
  const m6 = obz, m7 = obx*obx + oby*oby + obz*obz;
  const d2 = m0 - rr*rr;
  const h1 = m3*m3 - m5 + ra*ra, h2 = m6*m6 - m7 + rb*rb;
  let caps = Infinity;
  if (h1 > 0) caps = -m3 - Math.sqrt(h1);
  if (h2 > 0) caps = Math.min(caps, -m6 - Math.sqrt(h2));
  const k2 = d2 - m2*m2;
  if (d2 <= 1e-9 || Math.abs(k2) < 1e-9) return caps;
  const k1 = d2*m3 - m1*m2 + m2*rr*ra;
  const k0 = d2*m5 - m1*m1 + m1*rr*ra*2 - m0*ra*ra;
  const h = k1*k1 - k0*k2;
  if (h < 0) return caps;
  const t = (-Math.sqrt(h) - k1)/k2;
  const yy = m1 - ra*rr + t*m2;
  return (yy > 0 && yy < d2) ? t : caps;
}

/** capsules already in pixel space. Returns greyscale bytes, white nearest. */
export function renderDepth(capsules, width, height, near = 255, far = 45) {
  const z = new Float64Array(width*height).fill(Infinity);
  for (const {a, b, ra, rb} of capsules) {
    const x0 = Math.max(0, Math.floor(Math.min(a[0]-ra, b[0]-rb)));
    const x1 = Math.min(width, Math.ceil(Math.max(a[0]+ra, b[0]+rb))+1);
    const y0 = Math.max(0, Math.floor(Math.min(a[1]-ra, b[1]-rb)));
    const y1 = Math.min(height, Math.ceil(Math.max(a[1]+ra, b[1]+rb))+1);
    for (let py = y0; py < y1; py++)
      for (let px = x0; px < x1; px++) {
        const d = coneDepth(px+0.5, py+0.5, a, b, ra, rb);
        const k = py*width + px;
        if (d < z[k]) z[k] = d;
      }
  }
  let lo = Infinity, hi = -Infinity;
  for (const v of z) if (isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const out = new Uint8ClampedArray(width*height);
  const span = Math.max(1e-6, hi - lo);
  for (let i = 0; i < z.length; i++)
    out[i] = isFinite(z[i]) ? (hi - lo < 1e-6 ? near
                                              : far + (near-far)*(hi - z[i])/span) : 0;
  return out;
}

/** OpenPose's own drawing: full-brightness dots, limbs blended at 0.6. */
export function renderPose(people, width, height, stick = 4, dotR = 4) {
  const rgb = new Uint8ClampedArray(width*height*3);
  const setPx = (x, y, c, alpha) => {
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    const k = (y*width + x)*3;
    for (let i = 0; i < 3; i++) rgb[k+i] = rgb[k+i]*(1-alpha) + c[i]*alpha;
  };
  for (const {points, visible} of people)
    points.forEach(([x, y], i) => {
      if (!visible[i]) return;
      for (let dy = -dotR; dy <= dotR; dy++)
        for (let dx = -dotR; dx <= dotR; dx++)
          if (dx*dx + dy*dy <= dotR*dotR)
            setPx(Math.round(x+dx), Math.round(y+dy), COLORS[i], 1);
    });
  for (const {points, visible} of people)
    LIMBS.forEach(([a, b], i) => {
      if (!visible[a] || !visible[b]) return;
      const [ax, ay] = points[a], [bx, by] = points[b];
      const cx = (ax+bx)/2, cy = (ay+by)/2;
      const dx = bx-ax, dy = by-ay, length = Math.hypot(dx, dy) || 1e-6;
      const ux = dx/length, uy = dy/length, nx = -uy, ny = ux;
      const halfL = length/2;
      const x0 = Math.max(0, Math.floor(cx - halfL - stick));
      const x1 = Math.min(width, Math.ceil(cx + halfL + stick) + 1);
      const y0 = Math.max(0, Math.floor(cy - halfL - stick));
      const y1 = Math.min(height, Math.ceil(cy + halfL + stick) + 1);
      for (let py = y0; py < y1; py++)
        for (let px = x0; px < x1; px++) {
          const rx = px + 0.5 - cx, ry = py + 0.5 - cy;
          const along = (rx*ux + ry*uy)/halfL, across = (rx*nx + ry*ny)/stick;
          if (along*along + across*across <= 1) setPx(px, py, COLORS[i], 0.6);
        }
    });
  return rgb;
}
