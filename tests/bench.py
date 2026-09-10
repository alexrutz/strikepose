import os, sys, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk
from openpose3d_editor import *
root = tk.Tk(); root.geometry("1180x800")
app = EditorApp(root); root.update()
class E:
    def __init__(s,x,y,state=0): s.x,s.y,s.state=x,y,state
for label, flag in (("body preview ON", True), ("body preview OFF", False)):
    app.show_body = flag
    app.redraw(); root.update()
    t=time.time()
    for i in range(30):
        app.camera.orbit(4, 1); app.redraw()
    root.update()
    orbit = (time.time()-t)/30
    idx = KEYPOINT_NAMES.index("l_elbow")
    sx, sy, _ = app.camera.project(app.skeleton.points[idx])
    app.on_press(E(sx,sy))
    t=time.time()
    for i in range(30): app.on_drag(E(sx+2*i, sy-2*i))
    drag = (time.time()-t)/30
    app.on_release(E(sx,sy))
    print("%s: orbit %.1f ms/frame (%.0f fps), drag %.1f ms/frame (%.0f fps), items %d"
          % (label, orbit*1000, 1/orbit, drag*1000, 1/drag, len(app.canvas.find_all())))
root.destroy()
