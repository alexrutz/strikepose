import os, sys, os, tempfile; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk, tkinter.messagebox as mb, tkinter.filedialog as fd, numpy as np
fd.askopenfilename = lambda **k: ""
from openpose3d_editor import *
import mesh_backend
errors=[]; mb.showerror=lambda t,m: errors.append(t)
root=tk.Tk(); root.geometry("1180x800"); app=EditorApp(root); root.update()

glb = mesh_backend._write_test_glb(os.path.join(tempfile.gettempdir(),"rig.glb"))
mesh = mesh_backend.load_rigged_mesh(glb)
mesh["roles"] = mesh_backend.resolve_bones(mesh["joint_names"])
app._rigged_mesh = mesh; app.use_mesh.set(True)
img = app.depth_image(240, 360)
a = np.asarray(img)
print("rigged-mesh depth:", img.size, "covered px:", int((a>0).sum()), "max", a.max())

app.add_figure(); app.apply_preset("Female, curvy"); root.update()
img2 = np.asarray(app.depth_image(240,360))
print("two figures from one rig:", int((img2>0).sum()) > int((a>0).sum()))
ys,xs = np.nonzero(img2>0)
print("both drawn, x span:", xs.min(), "->", xs.max())

app.use_mesh.set(False)
print("built-in still works:", app.depth_image(200,300).size)
app.use_mesh.set(True); app._rigged_mesh = None; app.on_mesh_toggle()
print("toggling with no mesh loaded is safe:", app.use_mesh.get() is False)
print("dialogs:", errors)
print("DONE"); root.destroy()
