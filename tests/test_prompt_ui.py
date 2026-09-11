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

check("the panel has a prompt box", hasattr(app, "prompt_entry"))
check("and says what it needs before it is used",
      "model" in app.prompt_status.get().lower(), app.prompt_status.get())

lengths = {c: vlen(vsub(app.skeleton.points[c], app.skeleton.points[p]))
           for p, c in LIMB_SEQ}
before = list(app.skeleton.points)

# no model is running in the test environment, so this goes through the
# keyword route, which is the path that has to stay usable offline
app.prompt_text.set("a person kneeling")
app.pose_from_prompt()
root.update()
check("posing from the prompt reports which route read it",
      "keywords" in app.prompt_status.get(), app.prompt_status.get())
check("and it actually moved the figure",
      any(vlen(vsub(a, b)) > 1.0 for a, b in zip(before, app.skeleton.points)))
thigh = vnorm(vsub(app.skeleton.points[9], app.skeleton.points[8]))
check("into the pose that was asked for", vdot(thigh, (0.0, -1.0, 0.0)) > 0.9,
      "thigh down %.3f" % vdot(thigh, (0.0, -1.0, 0.0)))
after = {c: vlen(vsub(app.skeleton.points[c], app.skeleton.points[p]))
         for p, c in LIMB_SEQ}
check("without resizing a single bone",
      max(abs(after[c] - lengths[c]) for c in lengths) < 1e-6)

app.undo()
root.update()
check("undo puts the old scene back",
      all(vlen(vsub(a, b)) < 1e-9 for a, b in zip(before, app.skeleton.points)))

app.prompt_text.set("   ")
app.pose_from_prompt()
check("an empty prompt asks for one instead of posing",
      "Type" in app.prompt_status.get(), app.prompt_status.get())

# the export the prompt route produces has to be the export the window makes
app.prompt_text.set("a runner mid stride")
app.pose_from_prompt()
root.update()
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
