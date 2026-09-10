import os, sys, numpy as np; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk, tkinter.messagebox as mb
from openpose3d_editor import *
errors = []
mb.showerror = lambda t,m: errors.append(t)          # capture dialogs
root = tk.Tk(); root.geometry("1180x800")
app = EditorApp(root); root.update()

# toggling SMPL-X with nothing installed must fall back, not crash
app.use_smplx.set(True); app.on_smplx_toggle(); root.update()
print("toggle handled, checkbox now:", app.use_smplx.get(), "| dialog:", errors)
img = app.depth_image(200, 300)
print("depth still renders via built-in fallback:", img.size, img.mode)
print("checkbox auto-cleared:", app.use_smplx.get() is False)

# force the smplx path with a stub backend to exercise the projection glue
import types
stub = types.ModuleType("smplx_backend")
def mesh_from_skeleton(body, points, fit_shape=True):
    # a cube around the figure, in editor centimetres
    c = np.mean([points[n] for n in ("r_hip","l_hip","r_shoulder","l_shoulder")], axis=0)
    v = np.array([[-20,-20,-10],[20,-20,-10],[20,40,-10],[-20,40,-10],
                  [-20,-20,10],[20,-20,10],[20,40,10],[-20,40,10]], float) + c
    f = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],
                  [1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]])
    return v, f
import smplx_backend as real
stub.mesh_from_skeleton = mesh_from_skeleton
stub.rasterize_depth = real.rasterize_depth
stub.depth_to_image = real.depth_to_image
sys.modules["smplx_backend"] = stub
app.smplx_body = lambda: object()
app.use_smplx.set(True)
img = app.depth_image(256, 384)
a = np.asarray(img)
print("stub mesh depth:", img.size, "covered px:", int((a>0).sum()), "max:", a.max())
ys, xs = np.nonzero(a>0)
print("box lands inside the frame:", xs.min()>=0 and xs.max()<256 and ys.min()>=0 and ys.max()<384)
print("box is rectangular:", (a>0).sum() == (xs.max()-xs.min()+1)*(ys.max()-ys.min()+1))
app.preview_depth(); root.update(); print("preview via smplx path OK")
print("DONE")
root.destroy()
