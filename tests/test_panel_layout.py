import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tkinter as tk
from openpose3d_editor import *
ok=True
def check(l,c,e=""):
    global ok; ok=ok and bool(c); print(("PASS " if c else "FAIL ")+l+("  "+e if e else ""))

for geom in ("1240x860", "1000x600", "900x560"):
    root=tk.Tk(); root.geometry(geom); app=EditorApp(root)
    root.update(); root.update_idletasks()
    bbox = app.panel_canvas.bbox("all")
    visible = app.panel_canvas.winfo_height()
    content = bbox[3] - bbox[1]
    # find every button in the panel and check each can be scrolled into view
    def walk(w, out):
        for c in w.winfo_children():
            if isinstance(c, (tk.Button, tk.Checkbutton, tk.OptionMenu, tk.Entry)):
                out.append(c)
            walk(c, out)
        return out
    widgets = walk(app.panel, [])
    lowest = max(w.winfo_y() + w.winfo_height() for w in widgets)
    print("%s: content %dpx, window %dpx, %d controls, lowest at y=%d"
          % (geom, content, visible, len(widgets), lowest))
    check("  scrollregion covers every control", bbox[3] >= lowest)
    check("  panel scrolls when content overflows",
          content > visible or bbox[3] <= visible)
    # scroll to the bottom and confirm the last control is on screen
    app.panel_canvas.yview_moveto(1.0); root.update_idletasks()
    top = app.panel_canvas.canvasy(0)
    check("  bottom control reachable by scrolling",
          lowest <= top + visible + 2, "lowest %d, view ends %d" % (lowest, top+visible))
    # With every section folded the panel fits, so there is nothing to scroll;
    # open them all to test the wheel where it actually applies.
    for name in list(app._sections):
        state, flip = app._sections[name]
        if not state["open"]:
            flip()
    root.update_idletasks()
    overflows = app.panel_canvas.bbox("all")[3] > app.panel_canvas.winfo_height()
    check("  panel overflows once every section is open", overflows)
    app.panel_canvas.yview_moveto(0.0); root.update_idletasks()
    class W: num=5; delta=-120
    r = app._panel_scroll(W())
    root.update_idletasks()
    check("  wheel scrolls the panel and stops there",
          app.panel_canvas.yview()[0] > 0 and r == "break")
    zoom = app.camera.zoom
    check("  wheel over the panel did not zoom the scene", app.camera.zoom == zoom)
    root.destroy()
print("\n"+("ALL PASS" if ok else "FAILURES"))
