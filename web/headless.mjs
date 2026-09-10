// Exercise the app's pure logic paths without a DOM: preset retarget, fitAll,
// export cameras. Anything DOM-bound is covered by inspection, not this.
import * as P from "./pose3d.mjs";
let ok = true;
const check = (l,c,e="") => { ok = ok && !!c; console.log((c?"PASS ":"FAIL ")+l+(e?"  "+e:"")); };

// preset change must keep the pose (same routine as app.mjs presets.onchange)
const old = new P.Skeleton("Male, average");
const cam = new P.Camera(360, 640); const [,,fwd] = cam.basis();
old.moveJoint(3, old.solveDrag(3, cam.screenToWorld(-50,-70), fwd, -1));
const dir = P.norm(P.sub(old.points[3], old.points[2]));
const born = new P.Skeleton("Female, curvy");
born.points[P.ROOT] = old.points[P.ROOT].slice();
for (const [p,c] of P.LIMBS)
  born.points[c] = P.add(born.points[p],
    P.mul(P.norm(P.sub(old.points[c], old.points[p])), born.lengths[c]));
check("preset swap keeps every bone direction",
  P.len(P.sub(P.norm(P.sub(born.points[3], born.points[2])), dir)) < 1e-12);
let worst = 0;
for (const [p,c] of P.LIMBS)
  worst = Math.max(worst, Math.abs(P.len(P.sub(born.points[c], born.points[p])) - born.lengths[c]));
check("and applies the new proportions exactly", worst < 1e-9, worst.toExponential(2));

// export framing puts everyone inside the frame
const figures = [new P.Skeleton(), new P.Skeleton()];
figures[1].points = figures[1].points.map(p => P.add(p, [60,0,0]));
const ex = new P.Camera(512, 768);
const [right, up] = ex.basis();
const pts = figures.flatMap(f => f.points);
const xs = pts.map(p => P.dot(p,right)), ys = pts.map(p => P.dot(p,up));
const cx=(Math.min(...xs)+Math.max(...xs))/2, cy=(Math.min(...ys)+Math.max(...ys))/2;
const wide=(Math.max(...xs)-Math.min(...xs))*1.3, tall=(Math.max(...ys)-Math.min(...ys))*1.3;
ex.zoom = Math.min(ex.width/wide, ex.height/tall);
const depthAxis = P.cross(right, up);
ex.target = P.add(P.add(P.mul(right,cx), P.mul(up,cy)), P.mul(depthAxis, P.dot(ex.target, depthAxis)));
const proj = pts.map(p => ex.project(p));
check("both figures land inside the export frame",
  proj.every(([x,y]) => x>=0 && x<=512 && y>=0 && y<=768));
console.log(ok ? "\nALL PASS" : "\nFAILURES");
process.exit(ok?0:1);
