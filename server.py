#!/usr/bin/env python3
"""Serve the editor to a phone.

    python3 server.py                       # http://127.0.0.1:8765
    python3 server.py --host 0.0.0.0        # reachable from the phone on the
                                            # same wifi; see the warning below
    python3 server.py --bodies bodies/      # depth from rigged .glb bodies

Why a server rather than more JavaScript
----------------------------------------
`web/` already had a mobile front end that did the maths again in JavaScript,
and CLAUDE.md lists that as the weak spot it is: two implementations of the
same geometry drift, and the one on the phone is the one nobody tests. It had
drifted by a third of the child's torso before anything compared them.

So the phone keeps only the part that has to be local - projecting and
dragging at sixty frames a second, which cannot survive a round trip - and
everything that decides what comes *out* is served from the Python that the
desktop build and the test suite already use: the pose catalogue, the body
presets, objects, clothing, the local-LLM prompt, and above all the export.
The phone used to write its own depth map out of capsules; now it asks for the
same analytic sweep, or the same rigged mesh, that `everyday.py` renders.

Binding
-------
The default binding is loopback, so nothing outside the machine can reach it.
`--host 0.0.0.0` is what puts it on the phone, and it puts it on everything
else on that network too: there is no authentication, and a request can ask
for a render, which costs CPU. Use it on a network you trust.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import socket
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import exporting
import rigpose
from exporting import KEYPOINT_BONES
import bodies_lib
import everyday
import pose_agent
import props as props_module
import wearables
from openpose3d_editor import (
    BODY_PRESETS, DEFAULT_PRESET, KEYPOINT_NAMES, VERSION, Camera, Skeleton,
    frame_rect, preset_params,
)

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")
VIEW_W, VIEW_H = 900, 700
MAX_BODY = 4 << 20            # a plan is small; anything larger is a mistake

# Loaded once at startup when --bodies is given: reading a 13k-vertex rig per
# request would put a second and a half on every export.
BODIES = {}


# ---------------------------------------------------------------------------
# The scene, from either end
#
# Two ways in, because the phone has two kinds of request. A library pose or a
# prompt arrives as *commands* and the server works out the keypoints. A drag
# arrives as the keypoints themselves - the phone has already solved it, the
# same way the desktop editor's own drag does - and the server only has to
# render them.
# ---------------------------------------------------------------------------

def scene_from_plan(plan, aspect):
    return pose_agent.build_scene(plan, VIEW_W, VIEW_H, aspect)


def scene_from_points(payload, aspect):
    """Figures the phone has already posed, plus its objects and camera."""
    warnings = []
    figures = []
    for entry in payload.get("people") or []:
        preset = entry.get("preset")
        if preset not in BODY_PRESETS:
            if preset is not None:
                warnings.append("unknown preset %r" % (preset,))
            preset = DEFAULT_PRESET
        skeleton = rigpose.figure_for(preset)
        # A client that is echoing a scene back sends the joint angles it was
        # given and gets the same pixels; a client that has DRAGGED sends the
        # eighteen keypoints, because that is all the phone has, and those are
        # solved onto the armature. Keypoints alone cannot carry a rig pose -
        # the round trip loses up to 6.6 cm of body, most of it roll and the
        # bones no keypoint names - so the bones travel too rather than the
        # wire format quietly being the lossy one.
        bones = entry.get("bones")
        if isinstance(bones, dict) and bones:
            skeleton.pose.from_dict({"bones": bones,
                                     "offset": entry.get("offset")
                                     or [0.0, 0.0, 0.0]})
        points = entry.get("points")
        if bones:
            points = None
        if isinstance(points, list) and len(points) == len(KEYPOINT_NAMES):
            try:
                # The phone drags eighteen keypoints - it has no rig and no
                # room for one - so a drag arrives as keypoints and is solved
                # onto the armature here. The wire format is unchanged; what
                # changed is that the answer is a rig pose, which is what the
                # depth map is made from.
                skeleton.from_keypoints(dict(zip(
                    KEYPOINT_NAMES,
                    [tuple(float(v) for v in p[:3]) for p in points])))
            except (TypeError, ValueError):
                warnings.append("a figure's points are not numbers; using rest")
        visible = entry.get("visible")
        if isinstance(visible, list) and len(visible) == len(KEYPOINT_NAMES):
            for name, seen in zip(KEYPOINT_NAMES, visible):
                bone = KEYPOINT_BONES.get(name)
                if bone in skeleton.pose.index:
                    for j in skeleton.pose.subtree(skeleton.pose.bone(bone)):
                        if not seen:
                            skeleton.visible[j] = False
        # Only when the entry carries one. `clean({})` fills in every slot
        # with "none", which reads the same to a person and is NOT the same
        # object as the empty dict a freshly built figure has - enough to make
        # an echoed scene's depth map differ from the one it echoed.
        worn = entry.get("outfit")
        if worn:
            skeleton.outfit = wearables.clean(worn)
        figures.append(skeleton)
    if not figures:
        figures = [rigpose.figure_for(DEFAULT_PRESET)]

    objects = []
    for entry in payload.get("props") or []:
        if entry.get("shape") in props_module.SHAPES:
            objects.append(props_module.make(
                entry["shape"], entry.get("size"), entry.get("position"),
                float(entry.get("yaw", 0.0))))

    camera = Camera(VIEW_W, VIEW_H)
    view = (payload.get("camera") or {}).get("view")
    rect = frame_rect(VIEW_W, VIEW_H, aspect)
    if view in pose_agent.CAMERA_VIEWS:
        yaw, pitch = pose_agent.CAMERA_VIEWS[view]
    else:
        chosen = pose_agent.legible_view(figures, props=objects, rect=rect)
        yaw, pitch = pose_agent.CAMERA_VIEWS[chosen]
    import math
    camera.yaw, camera.pitch = math.radians(yaw), math.radians(pitch)
    pose_agent.frame_scene(figures, camera, rect, props=objects)
    return figures, objects, camera, warnings


def describe(figures, objects, camera, warnings):
    """What the phone needs to draw the scene it just asked for."""
    return {
        # the eighteen, read off each posed rig - the phone draws capsules
        # around them and has nothing to do with the 104 bones behind
        # The preset travels too. Without it an echoed scene comes back on
        # whatever body the default is, and since the pose is joint ANGLES it
        # applies cleanly to the wrong figure - a woman's pose on a man's
        # skeleton, 15 cm out at the fingertips and correct everywhere a
        # keypoint would have looked.
        "people": [{"preset": figure.body.get("preset"),
                    "bones": figure.pose.to_dict()["bones"],
                    "offset": figure.pose.to_dict()["offset"],
                    "points": [[float(v) for v in figure.keypoints()[n]]
                               for n in KEYPOINT_NAMES],
                    "visible": [bool(figure.visible[
                        figure.pose.bone(KEYPOINT_BONES[n])])
                        for n in KEYPOINT_NAMES],
                    "outfit": dict(figure.outfit or {})} for figure in figures],
        "props": [{"shape": p["shape"], "size": list(p["size"]),
                   "position": list(p["position"]), "yaw": p["yaw"]}
                  for p in objects],
        "camera": {"yaw": camera.yaw, "pitch": camera.pitch,
                   "zoom": camera.zoom, "target": list(camera.target)},
        "warnings": warnings,
    }


def png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def render(figures, objects, camera, width, height, ground=True):
    """Rigged geometry, always. `render_scene` resolves the body set itself
    when the server was not started with one, and refuses rather than dropping
    to the built-in sweep."""
    meshes = ([BODIES.get(f.body.get("preset")) for f in figures]
              if BODIES else None)
    if meshes is not None and any(m is None for m in meshes):
        meshes = None                 # let the resolver find or refuse
    pose, depth, _rect = pose_agent.render_scene(
        figures, camera, width, height, VIEW_W, VIEW_H, props=objects,
        meshes=meshes, ground=ground)
    return pose, depth


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def api_vocabulary():
    groups = [{"name": group,
               "poses": [{"name": name, "about": everyday.ABOUT[name]}
                         for name in everyday.names_in(group)]}
              for group in everyday.GROUPS]
    return {
        "version": VERSION,
        "presets": list(BODY_PRESETS),
        "keypoints": list(KEYPOINT_NAMES),
        "groups": groups,
        "basic": [n for n in sorted(pose_agent.STANCES)
                  if n not in everyday.POSES],
        "directions": sorted(pose_agent.DIRECTIONS),
        "bend_joints": sorted(pose_agent.BEND_JOINTS),
        "point_targets": list(pose_agent.POINT_TARGETS),
        "cameras": list(pose_agent.VIEW_ORDER),
        "shapes": list(props_module.SHAPE_NAMES),
        # The export shapes, so the phone offers the same ones the desktop
        # does and there is one table rather than two that drift.
        "aspects": [{"name": name, "width": w, "height": h}
                    for name, w, h in exporting.ASPECTS],
        "anchors": sorted(pose_agent.ANCHORS),
        "wearables": {slot: wearables.options(slot)
                      for slot in wearables.SLOT_ORDER},
        "slots": list(wearables.SLOT_ORDER),
        # the looks themselves, not just their names: a phone that has the
        # table can dress a figure without a round trip, and there is still
        # only one copy of it - this one, generated from the Python.
        "outfits": {name: wearables.OUTFITS[name]
                    for name in wearables.OUTFIT_NAMES},
        "bodies": sorted(BODIES),
        "rigged": bool(BODIES),
    }


def api_scene(payload):
    aspect = size_of(payload)[0] / float(size_of(payload)[1])
    if payload.get("plan"):
        scene = scene_from_plan(payload["plan"], aspect)
    else:
        scene = scene_from_points(payload, aspect)
    return describe(*scene)


def api_render(payload):
    width, height = size_of(payload)
    aspect = width / float(height)
    if payload.get("plan"):
        figures, objects, camera, warnings = scene_from_plan(payload["plan"],
                                                             aspect)
    else:
        figures, objects, camera, warnings = scene_from_points(payload, aspect)
    # The floor is on unless the client says otherwise: a depth map with
    # nothing under the feet says the person is floating.
    pose, depth = render(figures, objects, camera, width, height,
                         ground=bool(payload.get("ground", True)))
    out = describe(figures, objects, camera, warnings)
    out["pose"] = png(pose)
    out["depth"] = png(depth)
    # Always true now: `render` refuses rather than returning the sweep, so
    # reaching this line at all means the depth came from real geometry.
    out["rigged"] = True
    return out


def api_prompt(payload):
    """Read a sentence into a plan, with a local model if one is listening.

    `discover` probes the usual ports and returns None when nothing answers,
    and `plan_for` falls back to reading the prompt by keyword - so a phone
    with no model behind it still gets a pose, and is told which route read
    it rather than being left to guess.
    """
    text = str(payload.get("prompt") or "").strip()
    if not text:
        return {"error": "say what the figure should be doing"}
    llm = pose_agent.discover(payload.get("backend") or "auto",
                              payload.get("host") or None)
    plan, source, warnings = pose_agent.plan_for(text, llm)
    return {"plan": plan, "source": source, "warnings": warnings}


def size_of(payload):
    try:
        width = int(payload.get("width", 512))
        height = int(payload.get("height", 768))
    except (TypeError, ValueError):
        return 512, 768
    # bounded both ways: a phone asking for 8000 px would tie the machine up
    # in the analytic rasteriser for minutes
    return max(64, min(1536, width)), max(64, min(1536, height))


ROUTES = {"/api/scene": api_scene, "/api/render": api_render,
          "/api/prompt": api_prompt}

STATIC = {".html": "text/html; charset=utf-8", ".mjs": "text/javascript",
          ".js": "text/javascript", ".css": "text/css",
          ".json": "application/json", ".png": "image/png",
          ".svg": "image/svg+xml", ".ico": "image/x-icon",
          ".webmanifest": "application/manifest+json"}


class Handler(BaseHTTPRequestHandler):
    server_version = "strikepose/" + VERSION
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if self.server.chatty:
            sys.stderr.write("  %s %s\n" % (self.address_string(), fmt % args))

    def reply(self, code, body, kind="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/vocabulary":
            return self.reply(200, api_vocabulary())
        if path == "/":
            path = "/index.html"
        # Serve only what is inside web/. A path that climbs out of it after
        # normalising is a traversal attempt, not a file this is willing to
        # read - the process can see the whole disk and the browser must not.
        target = os.path.normpath(os.path.join(WEB, path.lstrip("/")))
        if not target.startswith(WEB + os.sep) or not os.path.isfile(target):
            return self.reply(404, {"error": "no such thing"})
        kind = STATIC.get(os.path.splitext(target)[1])
        if kind is None:
            return self.reply(404, {"error": "not a servable type"})
        with open(target, "rb") as fh:
            self.reply(200, fh.read(), kind)

    do_HEAD = do_GET

    def do_POST(self):
        route = ROUTES.get(self.path.split("?", 1)[0])
        if route is None:
            return self.reply(404, {"error": "no such endpoint"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self.reply(400, {"error": "bad length"})
        if length > MAX_BODY:
            return self.reply(413, {"error": "that is too much scene"})
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("expected an object")
        except ValueError as problem:
            return self.reply(400, {"error": "not JSON: %s" % problem})
        try:
            self.reply(200, route(payload))
        except bodies_lib.MissingBodies as missing:
            # Not a server fault and not something to paper over: the phone
            # says what is missing and how to build it.
            self.reply(503, {"error": str(missing)})
        except Exception:                      # a bad scene is a 500, not a death
            traceback.print_exc()
            self.reply(500, {"error": "the scene could not be built"})


def local_address(port):
    """The address to type into the phone, best effort."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return "http://%s:%d" % (probe.getsockname()[0], port)
    except OSError:
        return "http://<this machine>:%d" % port
    finally:
        probe.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the editor to a phone.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 to reach it from the phone (no auth)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bodies", metavar="DIR",
                        help="folder of rigged .glb bodies for the depth map")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    found = bodies_lib.load(args.bodies, BODY_PRESETS, required=False)
    BODIES.update(found)
    if found:
        print("rigged bodies: %d loaded from %s"
              % (len(found), args.bodies or ", ".join(bodies_lib.folders())))
    else:
        print("NO RIGGED BODIES - depth export will refuse until there are.")
        print(bodies_lib.HOW)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.chatty = not args.quiet
    print("strikepose %s serving %s" % (VERSION, WEB))
    print("  http://%s:%d" % (args.host, args.port))
    if args.host == "0.0.0.0":
        print("  %s   <- from the phone, on this wifi" % local_address(args.port))
        print("  no authentication: only on a network you trust")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
