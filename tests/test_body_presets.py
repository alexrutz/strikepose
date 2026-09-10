import os, sys, math; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk
from openpose3d_editor import EditorApp, KEYPOINT_NAMES, BODY_PRESETS, vlen, vsub, LIMB_SEQ
root = tk.Tk(); root.geometry("1180x800")
app = EditorApp(root); root.update()

class E:
    def __init__(s,x,y,state=0): s.x,s.y,s.state=x,y,state

# pose, then switch preset, then confirm the pose survived
i = KEYPOINT_NAMES.index("l_elbow")
sx,sy,_ = app.camera.project(app.skeleton.points[i])
app.on_press(E(sx,sy))
for k in range(1,8): app.on_drag(E(sx+9*k, sy-11*k))
app.on_release(E(sx,sy))
def dirs(sk):
    return [tuple(round(v,6) for v in (lambda d: (d[0]/ (vlen(d) or 1), d[1]/(vlen(d) or 1), d[2]/(vlen(d) or 1)))(vsub(sk.points[c], sk.points[p]))) for p,c in LIMB_SEQ]
before = dirs(app.skeleton)
n0 = len(app.undo_stack)
app.preset_name.set("Female, curvy"); app.apply_preset("Female, curvy"); root.update()
after = dirs(app.skeleton)
print("all bone directions unchanged:", before == after)
print("hip width now:", round(vlen(vsub(app.skeleton.points[8], app.skeleton.points[11])),1))
print("undo pushed:", len(app.undo_stack) == n0+1, "| status:", app.status.get())
app.undo(); root.update()
print("undo restored male hip width:", round(vlen(vsub(app.skeleton.points[8], app.skeleton.points[11])),1))

for n in BODY_PRESETS:
    app.apply_preset(n)
    d = app.depth_image(160, 240)
    import numpy as np
    print(f"  {n:18s} depth px {int((np.asarray(d)>0).sum()):5d}")

# reset keeps the selected preset
app.preset_name.set("Child, about 7"); app.apply_preset("Child, about 7")
app.reset_pose()
print("reset kept preset:", app.skeleton.body["preset"])
# scale keeps volumes in proportion
import numpy as np
a = int((np.asarray(app.depth_image(200,300))>0).sum())
app.scale(1.3)
b = int((np.asarray(app.depth_image(200,300))>0).sum())
print("scaling grows the volume too:", b > a*1.4, a, b)
# save/load round trip with body params
import json, tempfile, os
p = os.path.join(tempfile.gettempdir(), "scene.json")
from openpose3d_editor import scene_to_dict, scene_from_dict
d = scene_to_dict(app.skeleton, app.camera, app.export_points(512,768), 512,768)
json.loads(json.dumps(d))
sk2 = app.skeleton.__class__()
scene_from_dict(json.loads(json.dumps(d)), sk2, app.camera)
print("body params survive json:", sk2.body["preset"], round(sk2.body_scale,3))
print("OK")
root.destroy()
