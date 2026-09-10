import os, sys, math, json, os, tempfile, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk
import numpy as np
from openpose3d_editor import *
root = tk.Tk(); root.geometry("1180x800")
app = EditorApp(root); root.update()
class E:
    def __init__(s,x,y,state=0): s.x,s.y,s.state=x,y,state

print("start:", app.figure_label.get())
app.add_figure(); app.add_figure(copy_active=True); root.update()
print("after add+copy:", app.figure_label.get(), "| active", app.active)
xs = [max(p[0] for p in f.points) for f in app.figures]
mins = [min(p[0] for p in f.points) for f in app.figures]
print("figures separated in x:", all(mins[i+1] > xs[i] for i in range(len(xs)-1)), [round(x) for x in xs])

# preset per figure
app.set_active(1, announce=False); app.apply_preset("Female, curvy")
app.set_active(0, announce=False)
print("per-figure presets:", [f.body["preset"] for f in app.figures])
print("dropdown follows active:", app.preset_name.get())

# picking selects across figures
sx, sy, _ = app.camera.project(app.figures[2].points[4])
hit = app.pick(sx, sy)
print("pick returns (figure, joint):", hit)
app.on_press(E(sx,sy))
print("clicking person 3 made it active:", app.active == 2)
before = list(app.figures[0].points)
for i in range(1,9): app.on_drag(E(sx+5*i, sy-7*i))
app.on_release(E(sx,sy))
print("only that person moved:", app.figures[0].points == before,
      "and person 3 changed:", app.figures[2].points[4] != app.figures[2].points[3])

# turning
p0 = app.figures[2].points[0]
app.rotate_figure(90)
p1 = app.figures[2].points[0]
root_pt = app.figures[2].points[ROOT]
import math as m
d0 = m.dist(p0, root_pt); d1 = m.dist(p1, root_pt)
print("turn preserves shape:", abs(d0-d1) < 1e-9, "| head moved:", p0 != p1)

# undo restores the whole scene, including figure count
n = len(app.figures)
app.delete_figure()
print("delete:", len(app.figures) == n-1, app.figure_label.get())
app.undo()
print("undo restored the deleted person:", len(app.figures) == n)

# exports
w, h = 320, 480
people = app.export_people(w, h)
print("export_people:", len(people), "entries, each", len(people[0][0]), "points")
img = render_openpose(people, w, h)
a = np.asarray(img).sum(axis=2)
cols = np.nonzero(a.sum(axis=0))[0]
print("all three drawn across the frame:", cols.min(), "->", cols.max())
t=time.time(); d = app.depth_image(w, h); dt=time.time()-t
z = np.asarray(d)
print("multi-person depth %dx%d in %.2fs, covered px %d" % (w,h,dt,int((z>0).sum())))

# scene json round trip
data = scene_to_dict(app.figures, app.camera, [app.export_points(w,h,f) for f in range(len(app.figures))], w, h)
print("json people:", len(data["people"]), "figures:", len(data["figures"]))
figs = scene_load(json.loads(json.dumps(data)), app.camera)
same = all(all(abs(x-y)<1e-3 for a,b in zip(f1.points,f2.points) for x,y in zip(a,b))
           for f1,f2 in zip(figs, app.figures))
print("reload restores every figure:", len(figs)==len(app.figures), "positions match:", same)
print("presets survive:", [f.body["preset"] for f in figs])

# legacy single-figure scene still loads
legacy = {"pose_3d":{n:list(v) for n,v in REST_POSE.items()}, "visible":[True]*18,
          "body":{"preset":"Male, heavy"}, "camera":{}}
one = scene_load(legacy, app.camera)
print("legacy scene ->", len(one), "figure,", one[0].body["preset"])

# viewport speed with three people
app.redraw(); root.update()
t=time.time()
for i in range(20): app.camera.orbit(4,1); app.redraw()
root.update()
print("3-person viewport: %.1f ms/frame (%.0f fps), %d items"
      % ((time.time()-t)/20*1000, 20/(time.time()-t), len(app.canvas.find_all())))
print("DONE")
root.destroy()
