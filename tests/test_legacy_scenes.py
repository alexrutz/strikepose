import os, sys, json, os, tempfile; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk
from openpose3d_editor import *
root = tk.Tk(); root.geometry("1180x800")
app = EditorApp(root); root.update()

# write a legacy-format scene, load it through the real file path, export depth
old = {"version":1, "pose_3d":{n:list(v) for n,v in REST_POSE.items()},
       "visible":[True]*18,
       "body":{"preset":"Male, heavy","chest":[18.5,13.5],"waist":[18.0,14.5],
               "pelvis":[18.0,13.0],"shoulder_r":7.6,"neck_r":7.0,
               "thigh_r":[11.0,9.6,7.0]},
       "body_scale":1.0,"camera":{"yaw":0.4,"pitch":0.1,"zoom":2.6,"target":[0,-60,0]}}
path = os.path.join(tempfile.gettempdir(), "legacy.json")
open(path,"w").write(json.dumps(old))
with open(path) as fh: data = json.load(fh)
app.push_undo(); scene_from_dict(data, app.skeleton, app.camera); app.redraw()
app.depth_image(256, 384); print("legacy scene -> depth export OK, preset:", app.skeleton.body["preset"])
app.export_points(512,768); print("pose export OK")
app.undo(); app.depth_image(256,384); print("undo then depth OK")
for n in BODY_PRESETS:
    app.apply_preset(n); app.depth_image(200,300)
print("all presets render OK")
app.preview_depth(); root.update(); print("preview OK")
print("DONE")
root.destroy()
