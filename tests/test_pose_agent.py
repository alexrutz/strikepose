"""The prompt route end to end, against a stub local model server.

No real model is downloaded or run. A thread serves both wire formats - the
Ollama one and the OpenAI-compatible one - so the client's request shape, its
fallback chain and its handling of a model that answers badly are all
exercised for real rather than mocked out. That matters more than usual here:
the two formats are the only part of this that cannot be checked by reading
the code, and a runtime that rejects one spelling of the schema is a case that
has actually shipped.
"""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "out", "agent")

import everyday
import pose_agent
from openpose3d_editor import LIMB_SEQ, vlen, vsub

ok = True


def check(label, condition, extra=""):
    global ok
    ok = ok and bool(condition)
    print(("PASS " if condition else "FAIL ") + label
          + ("  " + extra if extra else ""))


PLAN = {"figures": [{"preset": "Female, average",
                     "commands": [{"op": "stance", "name": "running"},
                                  {"op": "look", "direction": "forward_up"}]}],
        "camera": "three_quarter_left"}


class Stub(BaseHTTPRequestHandler):
    """Serves both formats. `mode` decides how awkward it is being."""

    mode = "ollama"
    seen = []

    def log_message(self, *_args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags" and Stub.mode.startswith("ollama"):
            return self._send(200, {"models": [{"name": "stub-7b"}]})
        if self.path == "/v1/models" and Stub.mode.startswith("openai"):
            return self._send(200, {"data": [{"id": "stub-7b"}]})
        return self._send(404, {"error": "no"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        Stub.seen.append((self.path, request))
        if self.path == "/api/chat":
            if Stub.mode == "ollama_chatty":     # answers in a markdown fence
                text = "Here you go:\n```json\n%s\n```" % json.dumps(PLAN)
            elif Stub.mode == "ollama_rubbish":
                text = "I'd rather not."
            else:
                text = json.dumps(PLAN)
            return self._send(200, {"message": {"content": text}})
        if self.path == "/v1/chat/completions":
            fmt = (request.get("response_format") or {}).get("type")
            if Stub.mode == "openai_no_schema" and fmt == "json_schema":
                # what llama.cpp has shipped: the OpenAI spelling refused
                return self._send(400, {"error": "json_schema not supported"})
            return self._send(200, {"choices": [
                {"message": {"content": json.dumps(PLAN)}}]})
        return self._send(404, {"error": "no"})


def serve():
    server = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, "http://127.0.0.1:%d" % server.server_address[1]


server, host = serve()

# -- the Ollama wire format ------------------------------------------------
Stub.mode, Stub.seen = "ollama", []
llm = pose_agent.discover(backend="auto", host=host)
check("an Ollama-shaped server is found and its model resolved",
      llm is not None and llm.backend == "ollama" and llm.model == "stub-7b",
      "" if llm is None else "%s %s" % (llm.backend, llm.model))
plan = llm.complete("a runner", pose_agent.response_schema())
path, request = Stub.seen[-1]
check("it is asked on /api/chat with the schema in `format`",
      path == "/api/chat" and isinstance(request.get("format"), dict)
      and request["format"]["properties"]["figures"]["type"] == "array")
check("and not streamed", request.get("stream") is False)
check("the plan comes back", plan == PLAN)

# -- the OpenAI-compatible wire format ------------------------------------
Stub.mode, Stub.seen = "openai", []
llm = pose_agent.discover(backend="auto", host=host)
check("an OpenAI-compatible server is found",
      llm is not None and llm.backend == "openai")
llm.complete("a runner", pose_agent.response_schema())
_path, request = Stub.seen[-1]
fmt = request.get("response_format") or {}
check("it is asked with response_format json_schema",
      fmt.get("type") == "json_schema"
      and fmt["json_schema"]["schema"]["properties"]["figures"]["type"]
      == "array")

# -- a runtime that refuses that spelling ---------------------------------
Stub.mode, Stub.seen = "openai_no_schema", []
llm = pose_agent.LocalLLM(host, "openai", "stub-7b")
plan = llm.complete("a runner", pose_agent.response_schema())
# The reasoning pass has no schema at all, so the first request carries no
# response_format - that is the point of it. Only the extraction calls are
# asked about here.
kinds = [(r.get("response_format") or {}).get("type") for _p, r in Stub.seen
         if r.get("response_format")]
check("a refused json_schema falls back rather than failing the run",
      plan == PLAN and kinds[:2] == ["json_schema", "json_object"], str(kinds))

# -- reason first, then write it down --------------------------------------
#
# A JSON grammar forces the first token to be `{`, so a model constrained from
# the start commits before it has considered anything; and llama.cpp drops the
# grammar entirely when thinking is on (ggml-org #20345), so "both at once" is
# not on offer either. Two calls: think with no schema, extract with one.
Stub.mode, Stub.seen = "openai", []
llm = pose_agent.LocalLLM(host, "openai", "stub-7b")
llm.complete("a runner", pose_agent.response_schema())
schema_calls = [r for _p, r in Stub.seen if r.get("response_format")]
free_calls = [r for _p, r in Stub.seen if not r.get("response_format")]
check("it reasons before it answers", len(free_calls) >= 1 and schema_calls,
      "%d free, %d constrained" % (len(free_calls), len(schema_calls)))
check("the reasoning call asks for thinking",
      (free_calls[0].get("chat_template_kwargs") or {}).get("enable_thinking")
      is True, str(free_calls[0].get("chat_template_kwargs")))
check("and the extraction call asks for none",
      (schema_calls[0].get("chat_template_kwargs") or {}).get("enable_thinking")
      is False, str(schema_calls[0].get("chat_template_kwargs")))
check("each half gets its own sampler settings",
      abs(free_calls[0]["temperature"] - 0.6) < 1e-9
      and abs(schema_calls[0]["temperature"] - 0.7) < 1e-9
      and abs(schema_calls[0]["top_p"] - 0.8) < 1e-9,
      "thinking %.2f, writing %.2f" % (free_calls[0]["temperature"],
                                       schema_calls[0]["temperature"]))
check("top_k and min_p reach the server even though OpenAI has no such field",
      free_calls[0].get("top_k") == 20 and "min_p" in free_calls[0])

Stub.mode, Stub.seen = "openai", []
llm = pose_agent.LocalLLM(host, "openai", "stub-7b",
                          sampling=pose_agent.Sampling(reasoning="off"))
llm.complete("a runner", pose_agent.response_schema())
check("reasoning off makes one call, not two", len(Stub.seen) == 1,
      "%d call(s)" % len(Stub.seen))

# -- the model is shown worked examples ------------------------------------
asked = pose_agent.with_examples("a person typing at a desk")
check("the nearest catalogue poses are shown as examples",
      "typing_at_desk" in asked and '"op": "point"' in asked)
check("and a request nothing matches is sent as it is",
      pose_agent.with_examples("zzzqqq") == "zzzqqq")

# -- what was actually built, measured -------------------------------------
#
# The point of the check pass: this is NOT the path that built the pose, so a
# second opinion from it is new information rather than the same reasoning
# reaching the same place.
import rigpose
floater = rigpose.figure_for("Male, average")
floater.pose.translate((0.0, 40.0, 0.0))
found = pose_agent.critique([floater])
check("a figure off the ground is measured as off the ground",
      any("floating" in f for f in found), str(found[:1]))
bare = rigpose.figure_for("Male, average")
check("a figure with no hands set is told so",
      any("no `hand`" in f for f in pose_agent.critique([bare])))
posed = rigpose.figure_for("Male, average")
pose_agent.apply_commands(posed, [{"op": "hand", "side": "both",
                                   "bend": 30, "turn": 10}], [])
check("and one with them set is not",
      not any("no `hand`" in f for f in pose_agent.critique([posed])))
check("where a hand ended up is reported in centimetres",
      any("cm" in f and "hand is" in f for f in pose_agent.critique([posed])))

# -- the fingers are a separate thing from the wrist ------------------------
import numpy as _np
open_hand = rigpose.figure_for("Male, average")
tip = _np.asarray(open_hand.points[open_hand.pose.bone("finger3-3.R")])
pose_agent.apply_commands(open_hand, [{"op": "grip", "side": "right",
                                       "amount": 1.0}], [])
closed = _np.asarray(open_hand.points[open_hand.pose.bone("finger3-3.R")])
check("grip closes the fingers", _np.linalg.norm(closed - tip) > 5.0,
      "%.1f cm" % _np.linalg.norm(closed - tip))
check("and it is in the vocabulary the model is given",
      "grip" in pose_agent.OPS
      and "grip" in pose_agent.command_schema()["properties"]["op"]["enum"])
wearing = sum(1 for _n, steps in everyday.POSES.items()
              if any(c.get("op") in ("hand", "foot", "grip") for c in steps))
check("and the catalogue actually uses hands and feet",
      wearing > 40, "%d of %d poses" % (wearing, len(everyday.POSES)))

# -- a model that wraps its answer ----------------------------------------
Stub.mode = "ollama_chatty"
llm = pose_agent.LocalLLM(host, "ollama", "stub-7b")
check("a fenced reply is still read",
      llm.complete("a runner", pose_agent.response_schema()) == PLAN)

# -- a model that refuses --------------------------------------------------
Stub.mode = "ollama_rubbish"
llm = pose_agent.LocalLLM(host, "ollama", "stub-7b")
_figures, _props, _camera, report = pose_agent.pose_from_prompt(
    "a runner mid stride", llm)
check("a model that answers nothing usable falls back to keywords",
      report["source"] == "keywords" and len(report["warnings"]) == 1,
      str(report["warnings"])[:90])

# -- nothing listening -----------------------------------------------------
check("no server at all is not an error",
      pose_agent.discover(backend="auto",
                          host="http://127.0.0.1:1") is None)

# -- the model's plan actually poses the figure ---------------------------
Stub.mode = "ollama"
llm = pose_agent.LocalLLM(host, "ollama", "stub-7b")
figures, _props, camera, report = pose_agent.pose_from_prompt(
    "a runner mid stride", llm)
check("a model plan is used when one arrives",
      report["source"].startswith("stub-7b"), report["source"])
check("and the preset it chose is applied",
      figures[0].body.get("preset") == "Female, average")
# every bone of the armature against the same body untouched: 103 of them,
# not the seventeen limbs eighteen keypoints could describe
import rigpose
rest_lengths = list(figures[0].lengths.values())
fresh = rigpose.figure_for("Female, average")
want = list(fresh.lengths.values())
check("and no bone changed length applying it",
      max(abs(a - b) for a, b in zip(rest_lengths, want)) < 1e-9)
server.shutdown()

# -- the CLI ---------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
result = subprocess.run(
    [sys.executable, os.path.join(root, "openpose3d_editor.py"),
     "--prompt", "a person sitting, seen from the side", "--out", OUT,
     "--no-llm", "--width", "384", "--height", "576"],
    capture_output=True, text=True, cwd=root)
check("the editor's own --prompt runs headless", result.returncode == 0,
      (result.stderr or result.stdout)[-200:])
written = {name: os.path.join(OUT, name)
           for name in ("pose.png", "depth.png", "scene.json")}
check("it writes a pose map, a depth map and the scene",
      all(os.path.exists(p) for p in written.values()))

try:
    from PIL import Image
    pose = Image.open(written["pose.png"])
    depth = Image.open(written["depth.png"])
    check("both maps come out at the size asked for and match each other",
          pose.size == depth.size == (384, 576), "%s %s" % (pose.size, depth.size))
except ImportError:                               # pragma: no cover
    check("Pillow present for the image checks", False)

with open(written["scene.json"], encoding="utf-8") as fh:
    scene = json.load(fh)
check("the scene records the prompt that made it",
      scene.get("prompt") == "a person sitting, seen from the side")
check("and the plan, so a run can be replayed or hand-edited",
      isinstance(scene.get("plan"), dict)
      and scene["plan"]["figures"][0]["commands"][0]["name"] == "sitting")
check("and stays loadable as an ordinary scene file",
      len(scene["people"][0]["pose_keypoints_2d"]) == 54)
check("and carries the objects the prompt put in the scene",
      [o["shape"] for o in scene.get("objects", [])] == ["chair"],
      str(scene.get("objects")))

# --require-llm must fail rather than quietly produce a keyword pose
result = subprocess.run(
    [sys.executable, os.path.join(root, "pose_agent.py"),
     "--prompt", "anything", "--out", OUT,
     "--host", "http://127.0.0.1:1", "--require-llm"],
    capture_output=True, text=True, cwd=root)
check("--require-llm fails instead of falling back", result.returncode == 2,
      result.stderr.strip()[:80])

# -- a stance that reaches itself is skipped, not fatal --------------------
#
# The everyday catalogue names a stance for a list of these same commands, and
# a catalogue entry may start from another stance. Merging a catalogue entry
# that repeats a basic name over the top of it pointed "walking" at itself and
# took the process down with a RecursionError - not a skipped command, a dead
# CLI. The merge uses setdefault so the name stays an alias, and the recursion
# has a depth guard so a cycle introduced any other way is still just a
# warning.
from openpose3d_editor import KEYPOINT_NAMES, Skeleton
INDEX = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
import everyday

check("the catalogue never shadows a basic stance",
      all(pose_agent.STANCES[n] is everyday.POSES[n]
          or n in ("walking", "running", "sitting", "t_pose")
          for n in everyday.POSES))
walker = rigpose.figure_for("Male, average")
warnings = pose_agent.apply_commands(walker, [{"op": "stance",
                                               "name": "walking"}])
check("an aliased stance resolves to the real one", not warnings
      and abs(walker.at("l_knee")[2]) > 5.0,
      "knee moved %.1f cm forward" % walker.at("l_knee")[2])

pose_agent.STANCES["ouroboros"] = [{"op": "stance", "name": "ouroboros"}]
try:
    victim = rigpose.figure_for("Male, average")
    warnings = pose_agent.apply_commands(victim, [{"op": "stance",
                                                   "name": "ouroboros"}])
    check("a stance that reaches itself is reported, not a stack overflow",
          warnings and "too deep" in warnings[0], str(warnings))
finally:
    del pose_agent.STANCES["ouroboros"]

# -- an object is anchored against the FINISHED figure ---------------------
#
# "sit down AND put a chair under the hips" reads naturally in either order.
# Placed first, the chair anchors to a standing hip - and since a figure's
# ground is its own lowest foot, it came out 86 cm tall with the desk in front
# of it ending up below the seat.
import props as props_module
seat_first, objects_first = rigpose.figure_for("Male, average"), []
pose_agent.apply_commands(seat_first,
                          [{"op": "place", "shape": "chair",
                            "at": "under_hips"},
                           {"op": "stance", "name": "sitting"}], objects_first)
seat_last, objects_last = rigpose.figure_for("Male, average"), []
pose_agent.apply_commands(seat_last,
                          [{"op": "stance", "name": "sitting"},
                           {"op": "place", "shape": "chair",
                            "at": "under_hips"}], objects_last)
heights = [props_module.bounds(o[0])[1][1] - props_module.bounds(o[0])[0][1]
           for o in (objects_first, objects_last)]
check("a chair is the same chair whichever order it was asked for",
      abs(heights[0] - heights[1]) < 0.5, "%.0f cm vs %.0f cm" % tuple(heights))
check("and it is a chair-sized chair, not one stretched to a standing hip",
      35.0 < heights[0] < 60.0, "%.0f cm tall" % heights[0])

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
