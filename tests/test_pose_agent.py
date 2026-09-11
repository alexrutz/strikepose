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
kinds = [(r.get("response_format") or {}).get("type") for _p, r in Stub.seen]
check("a refused json_schema falls back rather than failing the run",
      plan == PLAN and kinds[:2] == ["json_schema", "json_object"], str(kinds))

# -- a model that wraps its answer ----------------------------------------
Stub.mode = "ollama_chatty"
llm = pose_agent.LocalLLM(host, "ollama", "stub-7b")
check("a fenced reply is still read",
      llm.complete("a runner", pose_agent.response_schema()) == PLAN)

# -- a model that refuses --------------------------------------------------
Stub.mode = "ollama_rubbish"
llm = pose_agent.LocalLLM(host, "ollama", "stub-7b")
_figures, _camera, report = pose_agent.pose_from_prompt("a runner mid stride", llm)
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
figures, camera, report = pose_agent.pose_from_prompt("a runner mid stride", llm)
check("a model plan is used when one arrives",
      report["source"].startswith("stub-7b"), report["source"])
check("and the preset it chose is applied",
      figures[0].body.get("preset") == "Female, average")
rest_lengths = [vlen(vsub(figures[0].points[c], figures[0].points[p]))
                for p, c in LIMB_SEQ]
from openpose3d_editor import Skeleton, preset_params
fresh = Skeleton(preset_params("Female, average"))
want = [vlen(vsub(fresh.points[c], fresh.points[p])) for p, c in LIMB_SEQ]
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

# --require-llm must fail rather than quietly produce a keyword pose
result = subprocess.run(
    [sys.executable, os.path.join(root, "pose_agent.py"),
     "--prompt", "anything", "--out", OUT,
     "--host", "http://127.0.0.1:1", "--require-llm"],
    capture_output=True, text=True, cwd=root)
check("--require-llm fails instead of falling back", result.returncode == 2,
      result.stderr.strip()[:80])

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
