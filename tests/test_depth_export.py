import os, sys, math; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

import tkinter as tk
from openpose3d_editor import EditorApp, KEYPOINT_NAMES
root = tk.Tk(); root.geometry("1180x800")
app = EditorApp(root); root.update()

class E:
    def __init__(s,x,y,state=0): s.x,s.y,s.state=x,y,state

# pose something so the depth map is not the rest pose
i = KEYPOINT_NAMES.index("l_wrist")
sx, sy, _ = app.camera.project(app.skeleton.points[i])
app.on_press(E(sx,sy))
for k in range(1,9): app.on_drag(E(sx+6*k, sy-14*k))
app.on_release(E(sx,sy))

# depth aligns with pose export
W,H = app._sizes()
d = app.depth_image(W,H)
print("depth image", d.size, d.mode)
import numpy as np
a = np.asarray(d)
print("background is black:", a[0,0]==0, " max:", a.max(), " covered px:", int((a>0).sum()))
d.save(os.path.join(OUT, "gui_depth.png"))

# preview window opens without ImageTk
app.preview_depth(); root.update()
tops = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
print("preview windows:", len(tops), "size", tops[0].winfo_children()[0].winfo_reqwidth(), "x", tops[0].winfo_children()[0].winfo_reqheight())
tops[0].destroy()

# thickness entry drives the model
app.body_thickness.set("1.6")
b = np.asarray(app.depth_image(W,H))
print("thicker body covers more px:", int((b>0).sum()), ">", int((a>0).sum()), (b>0).sum() > (a>0).sum())
app.body_thickness.set("garbage"); print("bad input falls back to", app._thickness())

# P key
app.body_thickness.set("1.0")
app.on_key(type("K",(),{"keysym":"p","state":0})()); root.update()
print("P opened preview:", len([w for w in root.winfo_children() if isinstance(w, tk.Toplevel)])>0)
print("status:", app.status.get())
print("OK")
root.destroy()
