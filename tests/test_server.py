"""The phone's server, over a real socket.

Not a call into the handler functions: the point of this file is the wire -
that the routes exist, that a malformed body is a 400 rather than a traceback,
that static files come back and files outside `web/` do not, and that a scene
sent as keypoints renders to the same pixels as the plan it came from. That
last one is what says the phone and the desktop are the same editor.
"""

import io
import json
import os
import sys
import threading
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server
from http.server import ThreadingHTTPServer

ok = True


def check(label, condition, extra=""):
    global ok
    ok = ok and bool(condition)
    print(("PASS " if condition else "FAIL ") + label
          + ("  " + extra if extra else ""))


httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
httpd.chatty = False
host = "http://127.0.0.1:%d" % httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()


def get(path):
    with urllib.request.urlopen(host + path, timeout=30) as response:
        return response.status, response.read(), response.headers


def post(path, payload):
    request = urllib.request.Request(
        host + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def status_of(path, payload=None, method="GET"):
    try:
        if method == "GET":
            return get(path)[0]
        return post(path, payload or {})[0]
    except urllib.error.HTTPError as problem:
        return problem.code


# -- it serves the app -----------------------------------------------------
code, body, headers = get("/")
check("the root is the app", code == 200 and b"<canvas id=\"stage\"" in body,
      "%d, %d bytes" % (code, len(body)))
check("and says it is HTML", "text/html" in headers.get("Content-Type", ""))
check("the modules it loads are served",
      get("/app.mjs")[0] == 200 and get("/pose3d.mjs")[0] == 200
      and get("/api.mjs")[0] == 200)
check("and are served as JavaScript, not downloaded",
      "javascript" in get("/api.mjs")[2].get("Content-Type", ""))

# Nothing outside web/. The process can read the whole disk and the browser
# must not be able to ask it to.
for escape in ("/../server.py", "/../../etc/passwd", "/..%2fserver.py",
               "/web/../server.py"):
    check("a path climbing out of web/ is refused: %s" % escape,
          status_of(escape) == 404)
check("and so is a file type that is not servable",
      status_of("/../CLAUDE.md") == 404)
check("a missing file is a 404, not a crash", status_of("/nope.mjs") == 404)
check("the icon and the manifest are served, so it installs to a home screen",
      get("/icon.svg")[0] == 200 and get("/app.webmanifest")[0] == 200
      and "manifest" in get("/app.webmanifest")[2].get("Content-Type", ""))

# -- the vocabulary --------------------------------------------------------
code, raw, _ = get("/api/vocabulary")
vocabulary = json.loads(raw.decode("utf-8"))
check("the vocabulary lists every body preset",
      len(vocabulary["presets"]) >= 9, str(len(vocabulary["presets"])))
check("and the whole pose catalogue, grouped",
      sum(len(g["poses"]) for g in vocabulary["groups"]) >= 80
      and len(vocabulary["groups"]) == 10,
      "%d poses in %d groups"
      % (sum(len(g["poses"]) for g in vocabulary["groups"]),
         len(vocabulary["groups"])))
check("and every pose says what it is",
      all(p["about"] for g in vocabulary["groups"] for p in g["poses"]))
check("and the objects and clothing the editor has",
      len(vocabulary["shapes"]) >= 20
      and len(vocabulary["wearables"]) == len(vocabulary["slots"]) == 5,
      "%d shapes, %d slots" % (len(vocabulary["shapes"]),
                               len(vocabulary["wearables"])))
# The phone dresses a figure without a round trip, so it needs the looks
# themselves and not only their names - and every slot and garment one names
# has to be one the same vocabulary offers, or a tap sets a slot to something
# the renderer will drop on the floor.
check("and the named outfits, as the slots they set",
      len(vocabulary["outfits"]) >= 30
      and all(isinstance(v, dict) for v in vocabulary["outfits"].values()),
      "%d outfits" % len(vocabulary["outfits"]))
stray = sorted({"%s/%s" % (slot, name)
                for look in vocabulary["outfits"].values()
                for slot, name in look.items()
                if name not in vocabulary["wearables"].get(slot, [])})
check("every outfit names slots and garments the vocabulary has",
      not stray, str(stray[:3]))
check("and \"bare\" names every one of them",
      set(vocabulary["outfits"]["bare"]) == set(vocabulary["slots"]),
      str(sorted(vocabulary["outfits"]["bare"])))

# -- a named pose ----------------------------------------------------------
_, scene = post("/api/scene", {"plan": {"figures": [
    {"preset": "Female, average",
     "commands": [{"op": "stance", "name": "typing_at_desk"}]}]}})
check("a named pose comes back as keypoints",
      len(scene["people"]) == 1
      and len(scene["people"][0]["points"]) == len(vocabulary["keypoints"]),
      "%d points" % len(scene["people"][0]["points"]))
check("with the objects it leans on",
      sorted(p["shape"] for p in scene["props"]) == ["chair", "desk"],
      str([p["shape"] for p in scene["props"]]))
check("and a camera that is not the default",
      abs(scene["camera"]["yaw"]) > 1e-6)

# -- the render, both ways in ---------------------------------------------
_, planned = post("/api/render", {"plan": {"figures": [
    {"preset": "Female, average",
     "commands": [{"op": "stance", "name": "reading_a_book"}]}]},
    "width": 192, "height": 288})
check("a render returns both conditioning images",
      planned["pose"].startswith("data:image/png;base64,")
      and planned["depth"].startswith("data:image/png;base64,"),
      "%d and %d bytes" % (len(planned["pose"]), len(planned["depth"])))
check("and they are not the same image", planned["pose"] != planned["depth"])

# The phone drags locally and sends the keypoints back. If that does not land
# on the same pixels, the thing on the phone is not this editor.
_, echoed = post("/api/render", {
    "people": [{"preset": "Female, average",
                "points": planned["people"][0]["points"],
                "visible": planned["people"][0]["visible"]}],
    "props": planned["props"], "width": 192, "height": 288})
check("the same scene sent back as keypoints renders identically",
      echoed["pose"] == planned["pose"] and echoed["depth"] == planned["depth"])

# -- it survives what a client gets wrong ---------------------------------
check("a body that is not JSON is a 400",
      status_of("/api/render", None, "POST") in (200, 400))
request = urllib.request.Request(host + "/api/render", data=b"{not json",
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
try:
    urllib.request.urlopen(request, timeout=30)
    code = 200
except urllib.error.HTTPError as problem:
    code = problem.code
check("a malformed body is a 400, not a traceback", code == 400, str(code))
check("an unknown endpoint is a 404",
      status_of("/api/nonsense", {}, "POST") == 404)

_, odd = post("/api/render", {"people": [{"preset": "Nonexistent",
                                          "points": "not points"}],
                              "width": 96, "height": 96})
check("a nonsense figure is reported and still renders something",
      odd["warnings"] and odd["pose"].startswith("data:"), str(odd["warnings"]))

_, huge = post("/api/render", {"people": [{"preset": "Male, average"}],
                               "width": 99999, "height": 99999})
check("an absurd export size is clamped rather than obeyed",
      huge["pose"].startswith("data:"))

# -- the prompt ------------------------------------------------------------
_, answer = post("/api/prompt", {"prompt": "someone kneeling, looking up",
                                 "backend": "none", "host": "http://127.0.0.1:1"})
check("a prompt with no model behind it still returns a plan",
      isinstance(answer.get("plan"), dict) and answer["plan"].get("figures"),
      str(answer.get("source")))
check("and says which route read it", bool(answer.get("source")))
_, empty = post("/api/prompt", {"prompt": "   "})
check("an empty prompt is refused politely", "error" in empty)

httpd.shutdown()
print("\nALL PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
