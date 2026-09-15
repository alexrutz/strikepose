"""The side panel: four tabs, and every control on every one of them reachable.

The panel used to be twelve collapsible sections in one column, which fitted
on a tall window and scrolled on a short one. It is four tabs of three or four
sections now, so the thing that has to hold is per TAB: on the smallest window
the editor allows, every control on every tab can be brought into view.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk

from openpose3d_editor import EditorApp
from ui_app import SECTION_TABS, TAB_ORDER

ok = True


def check(label, condition, extra=""):
    global ok
    ok = ok and bool(condition)
    print(("PASS " if condition else "FAIL ") + label
          + ("  " + extra if extra else ""))


def bottom_in_canvas(app, widget):
    """How far down the scrolling canvas a widget's last pixel sits.

    `winfo_y` is relative to the widget's own PARENT, so summing it for a
    button nested three frames deep measures its offset inside that frame and
    nothing else - which is how the previous version of this test reported the
    lowest control in a 31-control tab as 107 pixels down and passed. Screen
    position against the canvas, plus wherever the canvas is scrolled to, is
    the only thing that answers "can this be brought into view".
    """
    return (widget.winfo_rooty() - app.panel_canvas.winfo_rooty()
            + app.panel_canvas.canvasy(0) + widget.winfo_height())


def controls(widget, found=None):
    found = [] if found is None else found
    for child in widget.winfo_children():
        if isinstance(child, (tk.Button, tk.Checkbutton, tk.OptionMenu,
                              tk.Entry, tk.Scale)):
            found.append(child)
        controls(child, found)
    return found


for geometry in ("1240x860", "1000x600", "900x560"):
    root = tk.Tk()
    root.geometry(geometry)
    app = EditorApp(root)
    root.update()
    root.update_idletasks()
    print(geometry)

    # Every section has a home, and every tab has something on it. A section
    # left out of the table would still be built - into whichever tab came
    # first - and would look like it belonged there.
    orphans = [name for name in app._sections if name not in SECTION_TABS]
    check("  every section names the tab it lives on", not orphans, str(orphans))
    check("  and every tab has at least one",
          all(any(SECTION_TABS[s] == tab for s in app._sections)
              for tab in TAB_ORDER))

    for tab in TAB_ORDER:
        app.show_tab(tab)
        for name, (state, flip) in app._sections.items():
            if SECTION_TABS[name] == tab and not state["open"]:
                flip()
        root.update_idletasks()
        app.panel_canvas.yview_moveto(0.0)
        root.update_idletasks()

        here = controls(app._tabs[tab])
        box = app.panel_canvas.bbox("all")
        visible = app.panel_canvas.winfo_height()
        lowest = max(bottom_in_canvas(app, w) for w in here)
        check("  %-6s scrollregion covers its %2d controls"
              % (tab, len(here)), box[3] >= lowest,
              "region %d, lowest %d" % (box[3], lowest))

        # Scroll to the bottom; the last control has to be on screen there,
        # which is the whole promise - a control you cannot reach is a
        # control that is not in the program.
        app.panel_canvas.yview_moveto(1.0)
        root.update_idletasks()
        top = app.panel_canvas.canvasy(0)
        lowest = max(bottom_in_canvas(app, w) for w in here)
        check("  %-6s bottom control reachable" % tab,
              lowest <= top + visible + 2,
              "lowest %d, view ends %d" % (lowest, top + visible))

        # The tabs that are not showing must contribute no height, or the
        # scroll region is sized for all four and the visible one floats in
        # the middle of it.
        for other in TAB_ORDER:
            if other != tab:
                check("  %-6s hides %s" % (tab, other),
                      not app._tabs[other].winfo_ismapped())

    # The wheel belongs to the panel while the pointer is over it: it used to
    # fall through and zoom the scene behind.
    app.show_tab(TAB_ORDER[0])
    app.panel_canvas.yview_moveto(0.0)
    root.update_idletasks()
    zoom = app.camera.zoom

    class Wheel:
        num = 5
        delta = -120

    handled = app._panel_scroll(Wheel())
    root.update_idletasks()
    overflows = app.panel_canvas.bbox("all")[3] > app.panel_canvas.winfo_height()
    check("  the wheel is taken by the panel", handled == "break")
    check("  and scrolls it when there is anything to scroll",
          app.panel_canvas.yview()[0] > 0 or not overflows,
          "overflows %s, at %.2f" % (overflows, app.panel_canvas.yview()[0]))
    check("  and never zooms the scene behind it", app.camera.zoom == zoom)
    root.destroy()

print("\n" + ("ALL PASS" if ok else "FAILURES"))
sys.exit(0 if ok else 1)
