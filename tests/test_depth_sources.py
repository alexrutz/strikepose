import os, sys, os, tempfile; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk, tkinter.messagebox as mb, tkinter.filedialog as fd, numpy as np
fd.askopenfilename = lambda **k: ""
errs=[]; mb.showerror=lambda t,m: errs.append(t)
from openpose3d_editor import *
import mesh_backend
root=tk.Tk(); root.geometry("1180x800"); app=EditorApp(root); root.update()

def px(img): return int((np.asarray(img)>0).sum())
builtin = px(app.depth_image(200,300))
print("1. built-in (default):", builtin, "px")

# SMPL-X untouched: still refuses cleanly when torch/model absent
app.use_smplx.set(True); app.on_smplx_toggle()
print("2. SMPL-X toggle w/o install -> cleared:", app.use_smplx.get() is False, errs)

# stub SMPL-X exactly as the earlier test did, to prove that path is intact
import types
real = __import__("smplx_backend")
stub = types.ModuleType("smplx_backend")
def mesh_from_skeleton(body, points, fit_shape=True):
    c = np.mean([points[n] for n in ("r_hip","l_hip","r_shoulder","l_shoulder")], axis=0)
    v = np.array([[-20,-20,-10],[20,-20,-10],[20,40,-10],[-20,40,-10],
                  [-20,-20,10],[20,-20,10],[20,40,10],[-20,40,10]],float)+c
    f = np.array([[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],
                  [1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]])
    return v, f
stub.mesh_from_skeleton = mesh_from_skeleton
stub.rasterize_depth = real.rasterize_depth
stub.depth_to_image = real.depth_to_image
sys.modules["smplx_backend"] = stub
app.smplx_body = lambda: object()
app.use_smplx.set(True); app.on_smplx_toggle()
smplx_px = px(app.depth_image(200,300))
print("3. SMPL-X path still renders:", smplx_px, "px, distinct from built-in:", smplx_px != builtin)

# load a rigged mesh: must take over AND clear smplx
glb = mesh_backend._write_test_glb(os.path.join(tempfile.gettempdir(),"rig.glb"))
m = mesh_backend.load_rigged_mesh(glb); m["roles"]=mesh_backend.resolve_bones(m["joint_names"])
app._rigged_mesh = m; app.use_mesh.set(True); app.on_mesh_toggle()
print("4. enabling mesh cleared SMPL-X:", app.use_smplx.get() is False)
mesh_px = px(app.depth_image(200,300))
print("   mesh renders:", mesh_px, "px")

# switching back to SMPL-X must clear the mesh, not be shadowed by it
app.use_smplx.set(True); app.on_smplx_toggle()
print("5. enabling SMPL-X cleared mesh:", app.use_mesh.get() is False)
back = px(app.depth_image(200,300))
print("   back to SMPL-X output:", back == smplx_px)

# turn both off -> built-in returns
app.use_smplx.set(False); app.on_smplx_toggle()
print("6. built-in restored:", px(app.depth_image(200,300)) == builtin)
print("DONE"); root.destroy()
