// Mobile front end. The interaction model is the part that genuinely differs
// from the desktop build: no hover, no modifier keys, no right button, and a
// fingertip covers the joint it is dragging.
import * as P from "./pose3d.mjs";

const $ = id => document.getElementById(id);
const stage = $("stage"), ctx = stage.getContext("2d");
const pip = $("pipcanvas"), pctx = pip.getContext("2d");

const app = {
  figures: [new P.Skeleton()],
  active: 0,
  camera: new P.Camera(),
  pipCamera: Object.assign(new P.Camera(), {yaw: 0, pitch: 0}),
  pipNames: ["FRONT", "LEFT", "TOP"],
  pipIndex: 0,
  selected: null,
  symmetry: false,
  anchorMode: false,
  showBody: true,
  undo: [],
  dragSign: 1,
};
const figure = () => app.figures[app.active];

// --------------------------------------------------------------- geometry
const TINTS = [[235,235,240],[255,165,80],[105,205,255],[150,240,115],
               [250,140,230],[240,225,110],[175,160,255]];

function pushUndo() {
  app.undo.push({figures: app.figures.map(f => f.clone()), active: app.active});
  if (app.undo.length > 60) app.undo.shift();
}
function popUndo() {
  const state = app.undo.pop();
  if (!state) return;
  app.figures = state.figures; app.active = state.active;
  app.selected = null; refresh();
}

function fitAll(camera, margin = 1.25) {
  const [right, up] = camera.basis();
  const pts = app.figures.flatMap(f => f.points);
  const xs = pts.map(p => P.dot(p, right)), ys = pts.map(p => P.dot(p, up));
  const cx = (Math.min(...xs) + Math.max(...xs))/2;
  const cy = (Math.min(...ys) + Math.max(...ys))/2;
  const wide = Math.max(20, Math.max(...xs) - Math.min(...xs)) * margin;
  const tall = Math.max(20, Math.max(...ys) - Math.min(...ys)) * margin;
  camera.zoom = Math.max(0.1, Math.min(camera.width/wide, camera.height/tall));
  const depthAxis = P.cross(right, up);
  const keep = P.dot(camera.target, depthAxis);
  camera.target = P.add(P.add(P.mul(right, cx), P.mul(up, cy)),
                        P.mul(depthAxis, keep));
}

// ----------------------------------------------------------------- drawing
function drawScene(canvas, context, camera, {compact = false} = {}) {
  const w = camera.width, h = camera.height;
  context.clearRect(0, 0, w, h);
  const screens = app.figures.map(f => f.points.map(p => camera.project(p)));
  const all = screens.flat().map(s => s[2]);
  const lo = Math.min(...all), span = Math.max(1e-6, Math.max(...all) - lo);

  if (app.showBody) {
    const quads = [];
    app.figures.forEach((f, index) => {
      for (const {a, b, ra, rb} of P.bodyCapsules(f)) {
        const pa = camera.project(a), pb = camera.project(b);
        const dx = pb[0]-pa[0], dy = pb[1]-pa[1];
        const l = Math.hypot(dx, dy);
        const [nx, ny] = l < 0.5 ? [1, 0] : [-dy/l, dx/l];
        const RA = ra*camera.zoom, RB = rb*camera.zoom;
        quads.push({depth: (pa[2]+pb[2])/2, index, poly: [
          [pa[0]+nx*RA, pa[1]+ny*RA], [pa[0]-nx*RA, pa[1]-ny*RA],
          [pb[0]-nx*RB, pb[1]-ny*RB], [pb[0]+nx*RB, pb[1]+ny*RB]],
          round: [[pa[0], pa[1], RA], [pb[0], pb[1], RB]]});
      }
    });
    quads.sort((u, v) => v.depth - u.depth);
    for (const q of quads) {
      const t = (Math.max(...all) - q.depth)/span;
      const g = 44 + 150*t, tint = TINTS[q.index % TINTS.length];
      context.fillStyle = `rgb(${g*tint[0]/255|0},${g*tint[1]/255|0},${g*tint[2]/255|0})`;
      context.beginPath();
      q.poly.forEach(([x, y], i) => i ? context.lineTo(x, y) : context.moveTo(x, y));
      context.closePath(); context.fill();
      for (const [x, y, r] of q.round) {
        context.beginPath(); context.arc(x, y, r, 0, 7); context.fill();
      }
    }
  }

  // guide arcs while dragging: the reach sphere cut by the world planes
  if (drag.joint !== null && drag.camera === camera) {
    const f = figure(), p = f.parentOf(drag.joint);
    if (p >= 0) {
      const R = f.boneLength(p, drag.joint);
      const [px, py] = camera.project(f.points[p]);
      context.setLineDash([5, 4]); context.lineWidth = 1;
      context.strokeStyle = "#4a4a58";
      context.beginPath(); context.arc(px, py, R*camera.zoom, 0, 7); context.stroke();
      for (const arc of P.guideArcs(camera, f.points[p], R, app.dragSign)) {
        context.strokeStyle = arc.colour;
        for (const run of arc.runs) {
          context.beginPath();
          run.forEach(([x, y], i) => i ? context.lineTo(x, y) : context.moveTo(x, y));
          context.stroke();
        }
      }
      context.setLineDash([]);
    }
  }

  const limbs = [];
  screens.forEach((screen, f) => P.LIMBS.forEach(([a, b], i) => {
    if (app.figures[f].visible[a] && app.figures[f].visible[b])
      limbs.push({d: (screen[a][2]+screen[b][2])/2, f, i});
  }));
  limbs.sort((u, v) => v.d - u.d);
  context.lineCap = "round";
  for (const {d, f, i} of limbs) {
    const [a, b] = P.LIMBS[i], s = screens[f];
    let k = 1 - 0.5*((d - lo)/span);
    if (f !== app.active) k *= 0.92;
    const c = P.COLORS[i];
    context.strokeStyle = `rgb(${c[0]*k|0},${c[1]*k|0},${c[2]*k|0})`;
    context.lineWidth = compact ? 3 : 7;
    context.beginPath();
    context.moveTo(s[a][0], s[a][1]); context.lineTo(s[b][0], s[b][1]);
    context.stroke();
  }
  const joints = [];
  screens.forEach((screen, f) => screen.forEach(([x, y, d], i) =>
    joints.push({d, f, i, x, y})));
  joints.sort((u, v) => v.d - u.d);
  for (const {d, f, i, x, y} of joints) {
    let k = 1 - 0.5*((d - lo)/span);
    if (f !== app.active) k *= 0.92;
    const c = P.COLORS[i], r = compact ? 2.6 : 6;
    context.fillStyle = app.figures[f].visible[i]
      ? `rgb(${c[0]*k|0},${c[1]*k|0},${c[2]*k|0})` : "#55555f";
    context.beginPath(); context.arc(x, y, r, 0, 7); context.fill();
    if (!compact && f === app.active && app.figures[f].anchors.includes(i)) {
      context.strokeStyle = "#ffd27f"; context.lineWidth = 2;
      context.beginPath(); context.arc(x, y, r + 6, 0, 7); context.stroke();
    }
    if (!compact && f === app.active && i === app.selected) {
      context.strokeStyle = "#7aa2ff"; context.lineWidth = 2.5;
      context.beginPath(); context.arc(x, y, r + 7, 0, 7); context.stroke();
    }
  }
}

function refresh() {
  drawScene(stage, ctx, app.camera);
  if (!$("pip").classList.contains("hidden")) {
    fitAll(app.pipCamera, 1.3);
    drawScene(pip, pctx, app.pipCamera, {compact: true});
  }
  $("person").innerHTML = `<b>${app.active+1}</b> / <span>${app.figures.length}</span>`;
  $("preset").value = figure().preset;
  const info = $("readout");
  if (app.selected === null) info.classList.remove("on");
  else {
    info.classList.add("on");
    const f = figure(), p = f.parentOf(app.selected);
    let text = P.KEYPOINTS[app.selected].replace("_", " ");
    if (p >= 0) {
      const [, , fwd] = app.camera.basis();
      const off = P.sub(f.points[app.selected], f.points[p]);
      text += `   depth ${P.dot(off, fwd) >= 0 ? "+" : ""}${P.dot(off, fwd).toFixed(0)}`;
    }
    $("jointinfo").textContent = text;
  }
}

function resize() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  for (const [canvas, camera] of [[stage, app.camera], [pip, app.pipCamera]]) {
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.round(rect.width*dpr);
    canvas.height = Math.round(rect.height*dpr);
    camera.width = canvas.width; camera.height = canvas.height;
  }
  ctx.setTransform(1,0,0,1,0,0); pctx.setTransform(1,0,0,1,0,0);
  refresh();
}

// ------------------------------------------------------------------ touch
const drag = {joint: null, offset: [0,0], camera: null, plane: null, moved: false};
const pointers = new Map();
let pinch = null, tapTime = 0, tapJoint = null, longPress = null;

function toCanvas(event, canvas, camera) {
  const rect = canvas.getBoundingClientRect();
  return [(event.clientX - rect.left)*camera.width/rect.width,
          (event.clientY - rect.top)*camera.height/rect.height];
}

function pick(x, y, camera) {
  // A fingertip is far bigger than a mouse cursor, so the pick radius scales
  // with the display and ties break towards the camera.
  const radius = 26*(camera.width/stage.getBoundingClientRect().width);
  let best = null;
  app.figures.forEach((f, fi) => f.points.forEach((p, i) => {
    const [sx, sy, d] = camera.project(p);
    const dist = Math.hypot(sx - x, sy - y);
    if (dist < radius && (!best || dist < best.dist - 6 ||
                          (Math.abs(dist - best.dist) <= 6 && d < best.d)))
      best = {f: fi, i, dist, d};
  }));
  return best;
}

stage.addEventListener("pointerdown", event => {
  stage.setPointerCapture(event.pointerId);
  pointers.set(event.pointerId, event);
  if (pointers.size === 2) {                       // second finger: pan + zoom
    drag.joint = null;
    const [a, b] = [...pointers.values()];
    pinch = {dist: Math.hypot(a.clientX-b.clientX, a.clientY-b.clientY),
             cx: (a.clientX+b.clientX)/2, cy: (a.clientY+b.clientY)/2,
             zoom: app.camera.zoom};
    return;
  }
  const [x, y] = toCanvas(event, stage, app.camera);
  const hit = pick(x, y, app.camera);
  drag.moved = false;
  if (!hit) { drag.joint = null; drag.orbit = [event.clientX, event.clientY]; return; }
  if (hit.f !== app.active) { app.active = hit.f; }
  pushUndo();
  app.selected = hit.i;
  const f = figure();
  const [sx, sy] = app.camera.project(f.points[hit.i]);
  drag.joint = hit.i;
  drag.camera = app.camera;
  drag.offset = [x - sx, y - sy];        // keeps the joint out from under the thumb
  drag.plane = f.sagittalPlane();
  drag.orbit = null;
  const p = f.parentOf(hit.i);
  const [, , fwd] = app.camera.basis();
  app.dragSign = p >= 0 && P.dot(P.sub(f.points[hit.i], f.points[p]), fwd) < 0 ? -1 : 1;
  longPress = setTimeout(() => {          // hold a joint to anchor it
    longPress = null;
    drag.joint = null;
    toggleAnchor(hit.i);
    navigator.vibrate?.(18);
  }, 420);
  refresh();
});

stage.addEventListener("pointermove", event => {
  if (!pointers.has(event.pointerId)) return;
  pointers.set(event.pointerId, event);
  if (pinch && pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const dist = Math.hypot(a.clientX-b.clientX, a.clientY-b.clientY);
    const cx = (a.clientX+b.clientX)/2, cy = (a.clientY+b.clientY)/2;
    app.camera.zoom = Math.max(0.3, Math.min(40, pinch.zoom*dist/pinch.dist));
    app.camera.pan((cx - pinch.cx)*2, (cy - pinch.cy)*2);
    pinch.cx = cx; pinch.cy = cy;
    refresh();
    return;
  }
  drag.moved = true;
  if (longPress) { clearTimeout(longPress); longPress = null; }
  if (drag.orbit) {
    app.camera.orbit(event.clientX - drag.orbit[0], event.clientY - drag.orbit[1]);
    drag.orbit = [event.clientX, event.clientY];
    refresh();
    return;
  }
  if (drag.joint === null) return;
  const f = figure();
  const [cx, cy] = toCanvas(event, stage, app.camera);
  const mx = cx - drag.offset[0], my = cy - drag.offset[1];
  const p = f.parentOf(drag.joint);
  const pivot = p >= 0 ? f.points[p] : f.points[drag.joint];
  const [ax, ay] = app.camera.project(pivot);
  const [, , fwd] = app.camera.basis();
  const target = f.solveDrag(drag.joint,
                             app.camera.screenToWorld(mx - ax, my - ay),
                             fwd, app.dragSign);
  f.moveJoint(drag.joint, target);
  if (app.symmetry) f.mirrorDrag(drag.joint, target, drag.plane);
  refresh();
});

function endPointer(event) {
  pointers.delete(event.pointerId);
  if (pointers.size < 2) pinch = null;
  if (longPress) { clearTimeout(longPress); longPress = null; }
  if (drag.joint !== null && !drag.moved) {
    const now = Date.now();
    if (tapJoint === drag.joint && now - tapTime < 320) flipSelected();
    tapTime = now; tapJoint = drag.joint;
  }
  if (!drag.moved && drag.joint === null && !drag.orbit) app.selected = null;
  drag.joint = null; drag.orbit = null;
  refresh();
}
stage.addEventListener("pointerup", endPointer);
stage.addEventListener("pointercancel", endPointer);

// ---------------------------------------------------------------- commands
function flipSelected() {
  const f = figure(), i = app.selected;
  if (i === null) return;
  const p = f.parentOf(i);
  if (p < 0) return;
  pushUndo();
  const [, , fwd] = app.camera.basis();
  const old = P.sub(f.points[i], f.points[p]);
  f.moveJoint(i, P.add(f.points[p], P.sub(old, P.mul(fwd, 2*P.dot(old, fwd)))));
  refresh();
}
function toggleAnchor(i) {
  const f = figure(), list = f.anchors.slice();
  const at = list.indexOf(i);
  if (at >= 0) list.splice(at, 1);
  else { list.push(i); if (list.length > 2) list.shift(); }
  f.anchors = list;
  say(list.length === 2
      ? `Hinge ${P.KEYPOINTS[list[0]]} → ${P.KEYPOINTS[list[1]]}`
      : list.length ? `Anchored at ${P.KEYPOINTS[list[0]]}` : "Anchors cleared");
  refresh();
}
const say = text => { $("status").textContent = text; };

function turn(axis, direction) {
  pushUndo();
  const f = figure(), pivot = f.points[f.anchors.length ? f.anchors[0] : P.ROOT];
  const a = direction*15*Math.PI/180, c = Math.cos(a), s = Math.sin(a);
  f.points = f.points.map(pt => {
    const [dx, dy, dz] = P.sub(pt, pivot);
    const out = axis === "y" ? [dx*c + dz*s, dy, -dx*s + dz*c]
              : axis === "x" ? [dx, dy*c - dz*s, dy*s + dz*c]
                             : [dx*c - dy*s, dx*s + dy*c, dz];
    return P.add(pivot, out);
  });
  refresh();
}

// ------------------------------------------------------------------ export
function exportCanvas(width, height, painter, name) {
  const out = document.createElement("canvas");
  out.width = width; out.height = height;
  painter(out.getContext("2d"));
  out.toBlob(blob => {
    const file = new File([blob], name, {type: "image/png"});
    if (navigator.canShare?.({files: [file]})) navigator.share({files: [file]});
    else {
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url; link.download = name; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
    }
  }, "image/png");
}
function exportCamera(width, height) {
  const camera = new P.Camera(width, height);
  camera.yaw = app.camera.yaw; camera.pitch = app.camera.pitch;
  camera.target = app.camera.target.slice();
  fitAll(camera, 1.3);
  return camera;
}
function exportPose() {
  const w = +$("outw").value || 512, h = +$("outh").value || 768;
  const camera = exportCamera(w, h);
  const people = app.figures.map(f => ({
    points: f.points.map(p => camera.project(p).slice(0, 2)), visible: f.visible}));
  const rgb = P.renderPose(people, w, h);
  exportCanvas(w, h, context => {
    const image = context.createImageData(w, h);
    for (let i = 0, k = 0; i < w*h; i++, k += 3) {
      image.data[i*4] = rgb[k]; image.data[i*4+1] = rgb[k+1];
      image.data[i*4+2] = rgb[k+2]; image.data[i*4+3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, "pose.png");
  say("Pose exported.");
}
function exportDepth() {
  const w = +$("outw").value || 512, h = +$("outh").value || 768;
  const camera = exportCamera(w, h), k = camera.zoom;
  const capsules = app.figures.flatMap(f => P.bodyCapsules(f).map(({a, b, ra, rb}) => {
    const pa = camera.project(a), pb = camera.project(b);
    return {a: [pa[0], pa[1], pa[2]*k], b: [pb[0], pb[1], pb[2]*k],
            ra: ra*k, rb: rb*k};
  }));
  const grey = P.renderDepth(capsules, w, h);
  exportCanvas(w, h, context => {
    const image = context.createImageData(w, h);
    for (let i = 0; i < w*h; i++) {
      image.data[i*4] = image.data[i*4+1] = image.data[i*4+2] = grey[i];
      image.data[i*4+3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, "depth.png");
  say("Depth exported.");
}

// -------------------------------------------------------------------- wiring
const sheet = $("sheet");
$("grip").addEventListener("click", () => sheet.classList.toggle("open"));
const tool = (id, fn) => $(id).addEventListener("click", () => fn($(id)));
tool("t-add", () => {
  pushUndo();
  const born = new P.Skeleton(figure().preset);
  const [right] = app.camera.basis();
  const edge = Math.max(...app.figures.flatMap(f => f.points.map(p => P.dot(p, right))));
  born.translate(P.mul(right, edge + 55 - P.dot(born.points[P.ROOT], right)));
  app.figures.push(born); app.active = app.figures.length - 1;
  fitAll(app.camera); refresh();
});
tool("t-sym", el => { app.symmetry = !app.symmetry; el.classList.toggle("on");
                      say(app.symmetry ? "Mirroring edits" : "Mirror off"); });
tool("t-anchor", el => { el.classList.toggle("on");
  say(app.selected === null ? "Select a joint, or hold one to anchor it"
                            : "Anchor toggled");
  if (app.selected !== null) toggleAnchor(app.selected); });
tool("t-body", el => { app.showBody = !app.showBody; el.classList.toggle("on");
                       refresh(); });
tool("t-depth", () => { sheet.classList.add("open"); exportDepth(); });
tool("t-export", () => { sheet.classList.add("open"); exportPose(); });
$("t-body").classList.add("on");

$("prev").onclick = () => { app.active = (app.active + app.figures.length - 1) % app.figures.length; refresh(); };
$("next").onclick = () => { app.active = (app.active + 1) % app.figures.length; refresh(); };
$("undo").onclick = popUndo;
$("frame").onclick = () => { fitAll(app.camera); refresh(); };
$("flip").onclick = flipSelected;
$("pip").addEventListener("click", () => {
  app.pipIndex = (app.pipIndex + 1) % 3;
  const [yaw, pitch] = [[0,0], [Math.PI/2, 0], [0, 1.553]][app.pipIndex];
  app.pipCamera.yaw = yaw; app.pipCamera.pitch = pitch;
  $("pipname").textContent = app.pipNames[app.pipIndex];
  refresh();
});
for (const [id, axis, dir] of [["b-y-","y",-1],["b-y+","y",1],["b-x-","x",-1],
                               ["b-x+","x",1],["b-z-","z",-1],["b-z+","z",1]])
  $(id).onclick = () => turn(axis, dir);
for (const [id, yaw, pitch] of [["v-front",0,0],["v-left",Math.PI/2,0],
                                ["v-top",0,1.553]])
  $(id).onclick = () => { app.camera.yaw = yaw; app.camera.pitch = pitch; refresh(); };
$("b-copy").onclick = () => {
  pushUndo();
  const twin = figure().clone();
  const [right] = app.camera.basis();
  twin.translate(P.mul(right, 55));
  app.figures.push(twin); app.active = app.figures.length - 1; refresh();
};
$("b-del").onclick = () => {
  if (app.figures.length === 1) return say("A scene needs one person");
  pushUndo();
  app.figures.splice(app.active, 1);
  app.active = Math.min(app.active, app.figures.length - 1); refresh();
};
$("b-reset").onclick = () => { pushUndo();
  app.figures[app.active] = new P.Skeleton(figure().preset); refresh(); };
$("b-mirror").onclick = () => {
  pushUndo();
  const f = figure();
  f.points = f.points.map(([x, y, z]) => [-x, y, z]);
  for (const [a, b] of [[2,5],[3,6],[4,7],[8,11],[9,12],[10,13],[14,15],[16,17]]) {
    const t = f.points[a]; f.points[a] = f.points[b]; f.points[b] = t;
  }
  refresh();
};
$("b-pose").onclick = exportPose;
$("b-depth").onclick = exportDepth;
const presets = $("preset");
for (const name of Object.keys(P.PRESETS)) {
  const option = document.createElement("option");
  option.value = option.textContent = name;
  presets.appendChild(option);
}
presets.onchange = () => {
  pushUndo();
  const old = figure();
  const born = new P.Skeleton(presets.value);
  // keep the pose: every bone keeps its direction and takes the new length
  for (const [p, c] of P.LIMBS) {
    let d = P.sub(old.points[c], old.points[p]);
    if (P.len(d) < 1e-9) d = P.sub(born.points[c], born.points[p]);
    born.points[c] = P.add(born.points[p], P.mul(P.norm(d), born.lengths[c]));
  }
  born.points[P.ROOT] = old.points[P.ROOT].slice();
  for (const [p, c] of P.LIMBS)
    born.points[c] = P.add(born.points[p],
                           P.mul(P.norm(P.sub(old.points[c], old.points[p])),
                                 born.lengths[c]));
  app.figures[app.active] = born;
  refresh();
};

window.addEventListener("resize", resize);
window.addEventListener("orientationchange", () => setTimeout(resize, 250));
resize();
fitAll(app.camera);
refresh();
