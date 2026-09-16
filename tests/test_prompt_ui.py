"""The prompt box in the editor window. Needs a display; run_all.sh wraps it."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from openpose3d_editor import EditorApp, LIMB_SEQ, vdot, vlen, vnorm, vsub

ok = True


def check(label, condition, extra=""):
    global ok
    ok = ok and bool(condition)
    print(("PASS " if condition else "FAIL ") + label
          + ("  " + extra if extra else ""))


root = tk.Tk()
root.geometry("1180x800")
app = EditorApp(root)
root.update()

# A real text box, not a one-line Entry: a pose is a sentence and sometimes
# two, and the settings that used to sit in front of it now fold away below.
check("the panel has a prompt box", hasattr(app, "prompt_box"))
check("with room for a sentence", int(app.prompt_box["height"]) >= 3,
      "%s lines" % app.prompt_box["height"])
check("and it says how to send it", "ctrl+enter" in app.prompt_status.get().lower(),
      app.prompt_status.get())
check("the model settings are there too",
      all(hasattr(app, n) for n in ("prompt_host", "prompt_model",
                                    "prompt_key", "prompt_backend")))
check("and an API key box that does not show the key",
      hasattr(app, "prompt_key"))
check("sampling is configurable, reasoning included",
      set(app.sampling_vars) >= {"reasoning", "temperature", "top_p", "top_k"},
      str(sorted(app.sampling_vars)))
check("and reasoning is ON by default",
      app.sampling_vars["reasoning"].get() == "high",
      app.sampling_vars["reasoning"].get())
check("at Qwen3's published thinking numbers",
      abs(app.sampling_vars["temperature"].get() - 0.6) < 1e-9
      and abs(app.sampling_vars["top_p"].get() - 0.95) < 1e-9
      and app.sampling_vars["top_k"].get() == 20)

lengths = dict(app.skeleton.lengths)
before = list(app.skeleton.points)

# no model is running in the test environment, so this goes through the
# keyword route, which is the path that has to stay usable offline
app.prompt_box.insert("1.0", "a person kneeling")
app.pose_from_prompt()
# the call is on a worker thread now, so pump the loop until it lands rather
# than expecting it to have finished by the time the call returns
import time
deadline = time.time() + 30
while app.prompt_busy and time.time() < deadline:
    root.update()
    time.sleep(0.02)
root.update()
check("posing from the prompt reports which route read it",
      "keywords" in app.prompt_status.get(), app.prompt_status.get())
check("and it actually moved the figure",
      any(vlen(vsub(a, b)) > 1.0 for a, b in zip(before, app.skeleton.points)))
thigh = vnorm(vsub(app.skeleton.at("r_knee"), app.skeleton.at("r_hip")))
check("into the pose that was asked for", vdot(thigh, (0.0, -1.0, 0.0)) > 0.9,
      "thigh down %.3f" % vdot(thigh, (0.0, -1.0, 0.0)))
after = dict(app.skeleton.lengths)     # all 103 bones of the armature
check("without resizing a single bone",
      max(abs(after[j] - lengths[j]) for j in lengths) < 1e-6)

app.undo()
root.update()
check("undo puts the old scene back",
      all(vlen(vsub(a, b)) < 1e-9 for a, b in zip(before, app.skeleton.points)))

def ask(text):
    """Type into the box and wait for the worker, the way a person does."""
    import time
    app.prompt_box.delete("1.0", "end")
    app.prompt_box.insert("1.0", text)
    app.pose_from_prompt()
    deadline = time.time() + 30
    while app.prompt_busy and time.time() < deadline:
        root.update()
        time.sleep(0.02)
    root.update()


ask("   ")
check("an empty prompt asks for one instead of posing",
      "Type" in app.prompt_status.get(), app.prompt_status.get())

# the export the prompt route produces has to be the export the window makes
ask("a runner mid stride")
w, h = app._sizes()
pose = app.depth_image(w, h)
check("the depth map still exports after a prompt pose", pose.size == (w, h))
points = app.export_points(w, h)
check("and every keypoint lands inside the frame",
      all(0 <= x <= w and 0 <= y <= h for x, y in points),
      "x %.0f..%.0f y %.0f..%.0f" % (min(p[0] for p in points),
                                     max(p[0] for p in points),
                                     min(p[1] for p in points),
                                     max(p[1] for p in points)))

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
root.destroy()
sys.exit(0 if ok else 1)
