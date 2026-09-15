#!/usr/bin/env python3
"""The tkinter application: the viewport, the panel and everything a key does.

Everything this file draws comes from somewhere else - `skeleton` holds the
pose, `anatomy` the swept preview, `exporting` the two conditioning images -
so a headless run and a windowed one produce the same pixels. What lives here
is only the part that needs a window.
"""

from __future__ import annotations

import base64
import colorsys
import io
import json
import math
import os
import random
import sys
import time
from copy import deepcopy

import props as props_module

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except ImportError:      # the maths and the export work with no display
    tk = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None

from anatomy import (body_parts, body_segments, carry_chain, carry_frame,
                     sample_profile, sweep, torso_profile)
from anthro import (ANSUR, BODY_PRESETS, DEFAULT_PRESET, REST_POSE,
                    build_rest_points, derive_proportions, merge_body,
                    preset_params)
from camera import Camera
from exporting import (anatomy_depth_image, frame_rect, frame_scene,
                       pose_body, pose_image, project_people,
                       prop_groups, rigged_depth_image,
                       rigged_keypoints, to_camera_space)
from posemap import render_openpose, resolution_stickwidth
import randomize
from raster import (depth_to_grey, inside_polygon, render_depth,
                    silhouette_quads, solid_quads)
from scenefile import (scene_from_dict, scene_load, scene_objects,
                       scene_to_dict)
from skeleton import (ADJACENCY, CHILDREN, COLORS, FIGURE_BODY_TINTS,
                      FIGURE_STYLES, GIRDLE, KEYPOINT_NAMES, LIMB_SEQ,
                      MIRROR_OF, MIRROR_PAIRS, PARENT, PROP_TINT, ROOT,
                      Skeleton, reroot)
from vecmath import (any_perpendicular, matvec, rotation_between, vadd, vcross,
                     vdot, vlen, vmul, vnorm, vsub)
from version import VERSION


BG = "#0d0d11"          # viewport
VIEW_BG = "#0a0a0e"     # the small locked views
PANEL = "#16161c"       # side panel
CONTROL = "#22222b"     # buttons and fields
HOVER = "#2f2f3b"
EDGE = "#2a2a34"        # hairline separators
FG = "#e4e4ea"
MUTED = "#82828f"
ACCENT = "#7aa2ff"
PICK_RADIUS = 14.0

# The panel's tabs, and which section lives on each. Grouped by what you are
# doing rather than by what the code calls them: everything that changes the
# POSE is on one tab, everything about the FIGURE on the next, the room it
# stands in on SCENE, and the two conditioning images on EXPORT.
TAB_ORDER = ("Pose", "Figure", "Scene", "Export")

# Wide enough for a standing figure to read at a glance, narrow enough that
# the main view still gets the room the export frame needs. Two columns: at
# one column of five the thumbnails were a third of the height they had room
# for and most of the window was still black.
ORTHO_WIDTH = 296
ORTHO_COLUMNS = 2

SECTION_TABS = {
    "Prompt": "Pose",
    "Randomize": "Pose",
    "Edit": "Pose",
    "Turn figure": "Pose",
    "Body": "Figure",
    "Hair and clothes": "Figure",
    "Scene": "Scene",
    "Objects": "Scene",
    "View": "Scene",
    "Export": "Export",
    "Depth source": "Export",
    "Keys": "Export",
}

# Guide circles drawn while dragging: the sphere of reach sliced by the three
# world planes through the parent joint. Named by the plane, coloured by the
# axis perpendicular to it.
GUIDE_PLANES = (("XY", (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "#4f7fff"),
                ("YZ", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), "#ff5a5a"),
                ("ZX", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), "#5ad07a"))

HELP = [
    "Drag a joint      pose it (length locked)",
    "Drag background   orbit the view",
    "Right-drag        pan     Wheel  zoom",
    "Shift-drag        push joint away from you",
    "Ctrl-drag         pull joint towards you",
    "Alt-drag / L      change the limb's length",
    "F  flip selected bone through the screen",
    "V  toggle keypoint visibility",
    "M  mirror   R  reset   Ctrl+Z  undo",
    "B  body preview   P  full depth map",
    "S  mirror edits   A  anchor (up to 2)",
    "< >  tip about the hinge axis",
    "Double click  flip a limb front/back",
    "O  front/left/top + two 3/4 views",
    "X  randomize   Shift+X  again, new seed",
    "Ctrl+Tab  next panel tab",
    "Tab  next person   [ ] ; ' , .  turn",
    "0  frame everyone in the export",
    "1 2 3 4 5 6  front back right left top bottom",
]


class Viewport:
    """A canvas with its own camera. The main view is free; the three ortho
    views are locked to an axis and refit themselves to the scene."""

    def __init__(self, canvas, camera, name, locked=False):
        self.canvas = canvas
        self.camera = camera
        self.name = name
        self.locked = locked

    def refit_if_needed(self, figures):
        """Refit only when the scene has drifted out of frame or shrunk into a
        corner. Refitting every redraw makes the view jump after each edit."""
        points = [p for f in figures for p in f.points]
        screen = [self.camera.project(p) for p in points]
        xs = [p[0] for p in screen]
        ys = [p[1] for p in screen]
        width = max(1.0, self.camera.width)
        height = max(1.0, self.camera.height)
        inside = (min(xs) > width * 0.04 and max(xs) < width * 0.96
                  and min(ys) > height * 0.04 and max(ys) < height * 0.96)
        filled = max((max(xs) - min(xs)) / width, (max(ys) - min(ys)) / height)
        if inside and filled > 0.45:
            return
        self.fit(figures)

    def fit(self, figures, margin=1.25):
        """Frame every figure. Only used by the locked views, so the main view
        never moves under the user."""
        right, up, _ = self.camera.basis()
        points = [p for f in figures for p in f.points]
        across = [vdot(p, right) for p in points]
        along = [vdot(p, up) for p in points]
        cx = (min(across) + max(across)) / 2.0
        cy = (min(along) + max(along)) / 2.0
        width = max(1.0, self.camera.width)
        height = max(1.0, self.camera.height)
        wide = max(20.0, (max(across) - min(across))) * margin
        tall = max(20.0, (max(along) - min(along))) * margin
        self.camera.zoom = max(0.05, min(width / wide, height / tall))
        depth_axis = vcross(right, up)
        keep = vdot(self.camera.target, depth_axis)
        self.camera.target = vadd(vadd(vmul(right, cx), vmul(up, cy)),
                                  vmul(depth_axis, keep))


class EditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D OpenPose editor %s" % VERSION)
        self.root.configure(bg=BG)

        self.preset_name = tk.StringVar(value=DEFAULT_PRESET)
        self.props = []                 # objects standing in the scene
        self.active_prop = None
        self.prop_shape = tk.StringVar(value="chair")
        import wearables as _wearables
        self.outfit_vars = {slot: tk.StringVar(value="none")
                            for slot in _wearables.SLOT_ORDER}
        self.outfit_name = tk.StringVar(value="bare")
        self.prop_label = tk.StringVar(value="No objects. Objects show in the "
                                              "depth map, not the pose map.")
        self.drag_prop = None
        self.prompt_text = tk.StringVar(value="")
        self.prompt_host = tk.StringVar(
            value=os.environ.get("POSE_AGENT_HOST", ""))
        self.prompt_model = tk.StringVar(
            value=os.environ.get("POSE_AGENT_MODEL", ""))
        self.prompt_status = tk.StringVar(
            value="Describe a pose and press Enter. Needs a local model "
                  "running; falls back to keywords without one.")
        self.random_parts = {
            part: tk.BooleanVar(value=part in randomize.DEFAULT_PARTS)
            for part in randomize.PART_ORDER}
        self.random_amount = tk.DoubleVar(value=0.55)
        self.random_seed = tk.StringVar(value="")
        self.random_status = tk.StringVar(
            value="Edge cases a catalogue never reaches. 1.0 is a "
                  "contortionist. A seed repeats one exactly, so an odd "
                  "figure can be reported.")
        self.figures = [Skeleton(preset_params(DEFAULT_PRESET))]
        self.active = 0
        self.figure_label = tk.StringVar(value="Person 1 of 1")
        self.turn_step = tk.StringVar(value="15")
        self.camera = Camera()
        self.selected = None
        self.hovered = None
        self.undo_stack = []

        self.drag_joint = None
        self.pending_undo = None
        self.drag_view = None
        self.drag_plane = None
        self.symmetry = False
        self.drag_offset = (0.0, 0.0)
        self.drag_sign = 1.0
        self.drag_free_length = False
        self._length_guard = None
        self.orbit_last = None
        self.pan_last = None

        self.show_ortho = True
        self._sections = {}
        self._toggle_vars = {}
        self.ortho_views = []
        self._ortho_sig = None
        self._ortho_time = 0.0
        self._palettes = {}
        # The rig is what the export is made of, so it is what the viewport
        # shows. The swept body is the fast approximation underneath it and
        # defaults off now that there is something better to look at.
        self.show_rig = True
        self.show_body = False
        self._parts_cache = {}
        self._parts_sig = None
        self.show_grid = True
        self.show_labels = False
        self.depth_shading = True
        self.length_mode = False
        self.thick_lines = tk.BooleanVar(value=False)
        self.use_smplx = tk.BooleanVar(value=False)
        self.smplx_dir = os.environ.get("SMPLX_MODEL_DIR", "")
        self._smplx_cache = {}
        self.use_mesh = tk.BooleanVar(value=False)
        self.mesh_path = ""
        self._rigged_mesh = None
        self.mesh_library = {}          # preset name -> .glb path
        self.assets = {}                # asset name -> .glb path
        self.asset_rows = None
        self._mesh_cache = {}           # path -> loaded mesh
        self.body_thickness = tk.StringVar(value="1.0")
        self.out_w = tk.IntVar(value=512)
        self.out_h = tk.IntVar(value=768)
        self.status = tk.StringVar(
            value="3D OpenPose editor %s. Drag a joint to pose it." % VERSION)

        self._build_ui()
        self._bind_events()
        self.root.after(50, self.redraw)

    # the rest of the editor was written against a single figure; keeping
    # `skeleton` as the active one leaves all of that code unchanged
    @property
    def skeleton(self):
        return self.figures[self.active]

    @skeleton.setter
    def skeleton(self, value):
        self.figures[self.active] = value

    # -- figures -----------------------------------------------------------
    def set_active(self, index, announce=True):
        self.active = max(0, min(index, len(self.figures) - 1))
        self.preset_name.set(self.skeleton.body.get("preset", DEFAULT_PRESET))
        self.figure_label.set("Person %d of %d" % (self.active + 1,
                                                   len(self.figures)))
        self.refresh_outfit()
        if self.assets:
            self.rebuild_asset_rows()
        self.selected = None
        self.redraw()
        if announce:
            self.status.set("Editing person %d." % (self.active + 1))

    def add_figure(self, copy_active=False):
        self.push_undo()
        if copy_active:
            new = Skeleton(dict(self.skeleton.body))
            new.points = list(self.skeleton.points)
            new.visible = list(self.skeleton.visible)
            new.lengths = dict(self.skeleton.lengths)
            new.body_scale = self.skeleton.body_scale
        else:
            new = Skeleton(preset_params(self.preset_name.get()))
        # stand the newcomer clear of everyone else, along the view's right
        right, _, _ = self.camera.basis()
        edge = max(vdot(p, right) for f in self.figures for p in f.points)
        shift = edge + 55.0 - vdot(new.points[ROOT], right)
        base = self.figures[self.active].points[ROOT]
        new.translate(vsub(vadd(base, vmul(right, shift)), new.points[ROOT]))
        self.figures.append(new)
        self.set_active(len(self.figures) - 1, announce=False)
        self.status.set("Added person %d." % len(self.figures))

    def delete_figure(self):
        if len(self.figures) == 1:
            self.status.set("A scene needs at least one person.")
            return
        self.push_undo()
        del self.figures[self.active]
        self.set_active(min(self.active, len(self.figures) - 1), announce=False)
        self.status.set("Person removed, %d left." % len(self.figures))

    def next_figure(self):
        self.set_active((self.active + 1) % len(self.figures))

    def frame_all(self):
        """Pan and zoom so every figure sits inside the export frame.

        `exporting.frame_scene`, the very call a headless export makes, rather
        than a second fit of its own. The editor used to have one, framing on
        the eighteen keypoints with a flat 18% margin, and it cropped the
        crown and the feet off a standing figure at rest - a hand reaches
        17 cm past the wrist, a sole 8 cm past the ankle and the crown 14 cm
        past the nose, and none of them is a keypoint. Two framings also meant
        the viewport could disagree with the PNG about what was in shot, which
        is the one thing this rectangle exists to promise.
        """
        frame_scene(self.figures, self.camera, self.frame_rect(),
                    props=self.props)
        self.redraw()
        self.status.set("Framed %d %s." % (
            len(self.figures), "person" if len(self.figures) == 1 else "people"))

    def _step(self):
        try:
            return max(0.1, min(180.0, float(self.turn_step.get())))
        except (ValueError, tk.TclError):
            return 15.0

    def rotate_figure(self, direction, axis="y"):
        """Turn the active person about a world axis, pivoting on the anchor
        joint so an anchored knee or hip stays put."""
        self.push_undo()
        sk = self.skeleton
        pivot = sk.points[sk.anchor]
        angle = math.radians(direction * self._step())
        ca, sa = math.cos(angle), math.sin(angle)
        turned = []
        for point in sk.points:
            dx, dy, dz = vsub(point, pivot)
            if axis == "y":
                out = (dx * ca + dz * sa, dy, -dx * sa + dz * ca)
            elif axis == "x":
                out = (dx, dy * ca - dz * sa, dy * sa + dz * ca)
            else:
                out = (dx * ca - dy * sa, dx * sa + dy * ca, dz)
            turned.append(vadd(pivot, out))
        sk.points = turned
        self.redraw()
        self.status.set("Person %d turned %.1f degrees about %s."
                        % (self.active + 1, direction * self._step(),
                           axis.upper()))

    # -- layout ------------------------------------------------------------
    def _build_ui(self):
        self.root.configure(bg=BG)
        self._sections = {}
        self._toggle_vars = {}

        column = tk.Frame(self.root, bg=PANEL, width=268)
        column.pack(side="right", fill="y")
        column.pack_propagate(False)
        self.tab_strip = tk.Frame(column, bg=PANEL)
        self.tab_strip.pack(side="top", fill="x")
        self.panel_canvas = tk.Canvas(column, bg=PANEL, highlightthickness=0, bd=0)
        bar = tk.Scrollbar(column, orient="vertical", bg=PANEL,
                           troughcolor=PANEL, activebackground=EDGE,
                           relief="flat", bd=0, width=10,
                           command=self.panel_canvas.yview)
        self.panel_canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.panel_canvas.pack(side="left", fill="both", expand=True)

        panel = tk.Frame(self.panel_canvas, bg=PANEL)
        window = self.panel_canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda _e: self._refresh_scroll())
        self.panel_canvas.bind(
            "<Configure>",
            lambda e: self.panel_canvas.itemconfigure(window, width=e.width))
        self.panel = panel

        # One tab's worth of sections at a time. Twelve collapsible groups in
        # one column is a list to scroll past and hunt through even when every
        # one of them is folded, and folding is not free either: the thing you
        # want is three clicks away and you have to remember which heading it
        # lives under. Four tabs of three or four sections each fit a laptop
        # screen with nothing hidden and nothing to scroll.
        self._tabs = {}
        self._tab_buttons = {}
        self.active_tab = tk.StringVar(value=TAB_ORDER[0])
        tabs = tk.Frame(self.tab_strip, bg=PANEL)
        tabs.pack(fill="x")
        for name in TAB_ORDER:
            holder = tk.Frame(panel, bg=PANEL)
            self._tabs[name] = holder
            button = tk.Label(tabs, text=name.upper(), bg=PANEL, fg=MUTED,
                              cursor="hand2", pady=6,
                              font=("TkDefaultFont", 8, "bold"))
            button.pack(side="left", fill="x", expand=True)
            button.bind("<Button-1>", lambda _e, n=name: self.show_tab(n))
            self._tab_buttons[name] = button
        tk.Frame(self.tab_strip, bg=EDGE, height=1).pack(fill="x")

        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0,
                                width=900, height=760)

        # ---- small widget vocabulary -------------------------------------
        def section(title, opened=True, tab=None):
            """A collapsible group inside one tab.

            Everything still folds - a tab holds three or four of these and the
            ones you are not using stay shut - but which tab it lives on is
            what keeps the column short enough to see all at once.
            """
            home = self._tabs[tab or SECTION_TABS.get(title, TAB_ORDER[0])]
            head = tk.Frame(home, bg=PANEL, cursor="hand2")
            head.pack(fill="x", pady=(9, 0))
            tk.Frame(home, bg=EDGE, height=1).pack(fill="x", padx=12)
            chevron = tk.Label(head, text="", bg=PANEL, fg=MUTED,
                               font=("TkDefaultFont", 7))
            chevron.pack(side="right", padx=(0, 14))
            tk.Label(head, text=title.upper(), bg=PANEL, fg=MUTED, anchor="w",
                     font=("TkDefaultFont", 8, "bold")).pack(side="left", padx=14)
            body = tk.Frame(home, bg=PANEL)
            state = {"open": True}

            def flip(_event=None):
                state["open"] = not state["open"]
                if state["open"]:
                    body.pack(fill="x", after=head, pady=(4, 6))
                else:
                    body.pack_forget()
                chevron.configure(text="\u25be" if state["open"] else "\u25b8")
                self._refresh_scroll()

            for widget in (head, chevron) + tuple(head.winfo_children()):
                widget.bind("<Button-1>", flip)
            body.pack(fill="x", after=head, pady=(4, 6))
            chevron.configure(text="\u25be")
            self._sections[title] = (state, flip)
            if not opened:
                flip()
            return body

        def button(parent, text, command, small=False):
            widget = tk.Button(parent, text=text, command=command, bg=CONTROL,
                               fg=FG, relief="flat", bd=0, highlightthickness=0,
                               activebackground=HOVER, activeforeground=FG,
                               cursor="hand2", pady=4 if small else 5,
                               font=("TkDefaultFont", 8 if small else 9))
            widget.bind("<Enter>", lambda _e: widget.configure(bg=HOVER))
            widget.bind("<Leave>", lambda _e: widget.configure(bg=CONTROL))
            return widget

        def buttons(parent, items, cols=2, small=False):
            grid = tk.Frame(parent, bg=PANEL)
            grid.pack(fill="x", padx=12, pady=1)
            for i, (label, command) in enumerate(items):
                button(grid, label, command, small).grid(
                    row=i // cols, column=i % cols, sticky="ew", padx=1, pady=1)
            for c in range(cols):
                grid.columnconfigure(c, weight=1, uniform="cell")
            return grid

        def switch(parent, text, attr, apply=None):
            """Checkbutton bound to one of the editor's flags, so the panel
            shows the current state instead of a button labelled 'Toggle'."""
            var = tk.BooleanVar(value=getattr(self, attr))
            self._toggle_vars[attr] = var
            tk.Checkbutton(
                parent, text=text, variable=var,
                command=lambda: self.set_flag(attr, var.get()),
                bg=PANEL, fg=FG, anchor="w", selectcolor=CONTROL,
                activebackground=PANEL, activeforeground=FG, relief="flat",
                bd=0, highlightthickness=0, cursor="hand2", pady=1,
                font=("TkDefaultFont", 9)).pack(fill="x", padx=10)

        def field(parent, label, variable, width=6):
            line = tk.Frame(parent, bg=PANEL)
            line.pack(fill="x", padx=12, pady=2)
            tk.Label(line, text=label, bg=PANEL, fg=MUTED, anchor="w",
                     font=("TkDefaultFont", 9)).pack(side="left")
            entry = tk.Entry(line, textvariable=variable, width=width, bg=CONTROL,
                             fg=FG, relief="flat", insertbackground=FG,
                             justify="center", highlightthickness=1,
                             highlightbackground=EDGE, highlightcolor=ACCENT)
            entry.pack(side="right", ipady=3)
            entry.bind("<Return>", lambda _e: self.redraw())
            entry.bind("<FocusOut>", lambda _e: self.redraw())
            return entry

        # ---- prompt -------------------------------------------------------
        body = section("Prompt")
        entry = tk.Entry(body, textvariable=self.prompt_text, bg=CONTROL,
                         fg=FG, relief="flat", insertbackground=FG,
                         highlightthickness=1, highlightbackground=EDGE,
                         highlightcolor=ACCENT)
        entry.pack(fill="x", padx=12, pady=(0, 3), ipady=4)
        entry.bind("<Return>", lambda _e: self.pose_from_prompt())
        self.prompt_entry = entry
        buttons(body, [("Pose it", self.pose_from_prompt)], cols=1)
        field(body, "Model host", self.prompt_host, width=18)
        field(body, "Model", self.prompt_model, width=18)
        tk.Label(body, textvariable=self.prompt_status, bg=PANEL, fg=MUTED,
                 anchor="w", justify="left", wraplength=210,
                 font=("TkDefaultFont", 8)).pack(fill="x", padx=13, pady=(1, 2))

        # ---- randomize ----------------------------------------------------
        #
        # Selectable because the whole point is isolating a case. "The depth
        # map goes wrong when an arm comes round the back" is a hypothesis you
        # test by randomizing arms and nothing else, forty times; a button that
        # scrambles everything at once gives you forty pictures and no answer.
        body = section("Randomize", opened=True)
        grid = tk.Frame(body, bg=PANEL)
        grid.pack(fill="x", padx=10, pady=(0, 2))
        for i, part in enumerate(randomize.PART_ORDER):
            var = self.random_parts[part]
            tk.Checkbutton(
                grid, text=part.title(), variable=var, bg=PANEL, fg=FG,
                anchor="w", selectcolor=CONTROL, activebackground=PANEL,
                activeforeground=FG, relief="flat", bd=0, highlightthickness=0,
                cursor="hand2", pady=0, font=("TkDefaultFont", 8)
            ).grid(row=i // 2, column=i % 2, sticky="ew")
        for c in (0, 1):
            grid.columnconfigure(c, weight=1, uniform="rnd")
        tk.Label(body, text="How far from rest", bg=PANEL, fg=MUTED,
                 anchor="w",
                 font=("TkDefaultFont", 8)).pack(fill="x", padx=13, pady=(3, 0))
        # A tk Scale paints its handle in `bg` and its groove in `troughcolor`,
        # so a slider given the panel's own background has an invisible
        # handle: all you see is the number floating over an empty strip.
        tk.Scale(body, from_=0.0, to=1.2, resolution=0.05, orient="horizontal",
                 variable=self.random_amount, bg=CONTROL, fg=FG,
                 troughcolor=BG, activebackground=ACCENT,
                 highlightthickness=0, bd=0, relief="flat",
                 sliderrelief="flat", showvalue=True, sliderlength=20,
                 width=10, font=("TkDefaultFont", 7)).pack(fill="x", padx=12,
                                                           pady=(0, 2))
        field(body, "Seed", self.random_seed, width=10)
        buttons(body, [("Randomize (X)", self.randomize_pose),
                       ("Again (Shift+X)", self.randomize_again)])
        tk.Label(body, textvariable=self.random_status, bg=PANEL, fg=MUTED,
                 anchor="w", justify="left", wraplength=210,
                 font=("TkDefaultFont", 8)).pack(fill="x", padx=13, pady=(1, 2))

        # ---- scene --------------------------------------------------------
        body = section("Scene")
        tk.Label(body, textvariable=self.figure_label, bg=PANEL, fg=FG,
                 anchor="w", font=("TkDefaultFont", 10)).pack(fill="x", padx=13,
                                                              pady=(0, 3))
        buttons(body, [("Add", lambda: self.add_figure()),
                       ("Copy", lambda: self.add_figure(True)),
                       ("Delete", self.delete_figure),
                       ("Next \u21e5", self.next_figure)])
        buttons(body, [("Frame all (0)", self.frame_all)], cols=1)

        # ---- worn ---------------------------------------------------------
        body = section("Hair and clothes", opened=False)
        import wearables
        # A named look first: six dropdowns is the slow way to say "chef".
        line = tk.Frame(body, bg=PANEL)
        line.pack(fill="x", padx=12, pady=(2, 4))
        tk.Label(line, text="Outfit", bg=PANEL, fg=MUTED, anchor="w", width=8,
                 font=("TkDefaultFont", 9)).pack(side="left")
        looks = tk.OptionMenu(line, self.outfit_name, *wearables.OUTFIT_NAMES,
                              command=lambda _v: self.set_outfit())
        looks.configure(bg=CONTROL, fg=FG, relief="flat", bd=0, anchor="w",
                        highlightthickness=0, activebackground=HOVER,
                        activeforeground=FG, padx=8, pady=2, cursor="hand2",
                        font=("TkDefaultFont", 8))
        looks["menu"].configure(bg=PANEL, fg=FG, relief="flat", bd=0,
                                activebackground=HOVER, activeforeground=FG)
        looks.pack(side="right", fill="x", expand=True)
        for slot, label in (("hair", "Hair"), ("headgear", "Headgear"),
                            ("top", "Top"), ("bottom", "Bottom"),
                            ("shoes", "Feet")):
            line = tk.Frame(body, bg=PANEL)
            line.pack(fill="x", padx=12, pady=1)
            tk.Label(line, text=label, bg=PANEL, fg=MUTED, anchor="w", width=8,
                     font=("TkDefaultFont", 9)).pack(side="left")
            var = self.outfit_vars[slot]
            menu = tk.OptionMenu(line, var, *wearables.options(slot),
                                 command=lambda _v, s=slot: self.set_worn(s))
            menu.configure(bg=CONTROL, fg=FG, relief="flat", bd=0, anchor="w",
                           highlightthickness=0, activebackground=HOVER,
                           activeforeground=FG, padx=8, pady=2, cursor="hand2",
                           font=("TkDefaultFont", 8))
            menu["menu"].configure(bg=PANEL, fg=FG, relief="flat", bd=0,
                                   activebackground=HOVER, activeforeground=FG)
            menu.pack(side="right", fill="x", expand=True)
        buttons(body, [("Take it all off", self.strip)], cols=1)

        # ---- objects ------------------------------------------------------
        body = section("Objects", opened=False)
        shapes = tk.OptionMenu(body, self.prop_shape, *props_module.SHAPE_NAMES)
        shapes.configure(bg=CONTROL, fg=FG, relief="flat", bd=0, anchor="w",
                         highlightthickness=0, activebackground=HOVER,
                         activeforeground=FG, padx=10, pady=4, cursor="hand2",
                         font=("TkDefaultFont", 9))
        shapes["menu"].configure(bg=PANEL, fg=FG, relief="flat", bd=0,
                                 activebackground=HOVER, activeforeground=FG)
        shapes.pack(fill="x", padx=12, pady=1)
        buttons(body, [("Place", self.add_prop),
                       ("Remove", self.delete_prop),
                       ("Select next", self.next_prop),
                       ("Drop to floor", self.drop_prop)])
        buttons(body, [("Smaller", lambda: self.scale_prop(0.9)),
                       ("Larger", lambda: self.scale_prop(1.1)),
                       ("Turn \u2212", lambda: self.turn_prop(-15.0)),
                       ("Turn +", lambda: self.turn_prop(15.0))], small=True)
        tk.Label(body, textvariable=self.prop_label, bg=PANEL, fg=MUTED,
                 anchor="w", font=("TkDefaultFont", 8)).pack(fill="x", padx=13,
                                                             pady=(1, 2))

        # ---- body ---------------------------------------------------------
        body = section("Body")
        menu = tk.OptionMenu(body, self.preset_name, *BODY_PRESETS,
                             command=self.apply_preset)
        menu.configure(bg=CONTROL, fg=FG, relief="flat", bd=0, anchor="w",
                       highlightthickness=0, activebackground=HOVER,
                       activeforeground=FG, padx=10, pady=4, cursor="hand2",
                       font=("TkDefaultFont", 9))
        menu["menu"].configure(bg=PANEL, fg=FG, relief="flat", bd=0,
                               activebackground=HOVER, activeforeground=FG)
        menu.pack(fill="x", padx=12, pady=1)
        buttons(body, [("Smaller", lambda: self.scale(0.95)),
                       ("Larger", lambda: self.scale(1.05)),
                       ("Reset (R)", self.reset_pose),
                       ("Mirror (M)", self.mirror)])
        buttons(body, [("Restore proportions", self.restore_proportions)], cols=1)

        # ---- edit ---------------------------------------------------------
        body = section("Edit")
        switch(body, "Symmetric editing (S)", "symmetry")
        buttons(body, [("Anchor (A)", self.set_anchor),
                       ("Clear anchors", self.clear_anchor),
                       ("Flip bone (F)", self.flip_selected),
                       ("Hide point (V)", self.toggle_visibility),
                       ("Tip \u2212 (<)", lambda: self.rotate_hinge(-1)),
                       ("Tip + (>)", lambda: self.rotate_hinge(1))])
        buttons(body, [("Undo (Ctrl+Z)", self.undo)], cols=1)

        # ---- turn ---------------------------------------------------------
        body = section("Turn figure", opened=False)
        field(body, "Step (degrees)", self.turn_step, width=5)
        buttons(body, [("\u2212 Y  [", lambda: self.rotate_figure(-1, "y")),
                       ("+ Y  ]", lambda: self.rotate_figure(1, "y")),
                       ("\u2212 X  ;", lambda: self.rotate_figure(-1, "x")),
                       ("+ X  '", lambda: self.rotate_figure(1, "x")),
                       ("\u2212 Z  ,", lambda: self.rotate_figure(-1, "z")),
                       ("+ Z  .", lambda: self.rotate_figure(1, "z"))],
                small=True)

        # ---- view ---------------------------------------------------------
        body = section("View")
        buttons(body, [("Front", lambda: self.set_view("front")),
                       ("Back", lambda: self.set_view("back")),
                       ("Top", lambda: self.set_view("top")),
                       ("Right", lambda: self.set_view("right")),
                       ("Left", lambda: self.set_view("left")),
                       ("Bottom", lambda: self.set_view("bottom"))],
                cols=3, small=True)
        switch(body, "Extra views (O)", "show_ortho")
        switch(body, "Native rig (K)", "show_rig")
        switch(body, "Swept preview (B)", "show_body")
        switch(body, "Floor grid (G)", "show_grid")
        switch(body, "Joint names (N)", "show_labels")
        switch(body, "Depth shading (D)", "depth_shading")

        # ---- export -------------------------------------------------------
        body = section("Export", opened=False)
        size = tk.Frame(body, bg=PANEL)
        size.pack(fill="x", padx=12, pady=2)
        tk.Label(size, text="Size", bg=PANEL, fg=MUTED, anchor="w",
                 font=("TkDefaultFont", 9)).pack(side="left")
        for variable in (self.out_h, self.out_w):
            entry = tk.Entry(size, textvariable=variable, width=5, bg=CONTROL,
                             fg=FG, relief="flat", insertbackground=FG,
                             justify="center", highlightthickness=1,
                             highlightbackground=EDGE, highlightcolor=ACCENT)
            entry.pack(side="right", padx=2, ipady=3)
            entry.bind("<Return>", lambda _e: self.redraw())
            entry.bind("<FocusOut>", lambda _e: self.redraw())
        tk.Checkbutton(body, text="Thicker lines when large",
                       variable=self.thick_lines, bg=PANEL, fg=FG, anchor="w",
                       selectcolor=CONTROL, activebackground=PANEL,
                       activeforeground=FG, relief="flat", bd=0,
                       highlightthickness=0, cursor="hand2",
                       font=("TkDefaultFont", 9)).pack(fill="x", padx=10)
        buttons(body, [("Pose PNG\u2026", self.export_png),
                       ("Depth PNG\u2026", self.export_depth),
                       ("Save scene\u2026", self.save_json),
                       ("Load scene\u2026", self.load_json)])

        # ---- depth --------------------------------------------------------
        body = section("Depth source", opened=False)
        field(body, "Body thickness", self.body_thickness, width=5)
        tk.Checkbutton(body, text="SMPL-X mesh", variable=self.use_smplx,
                       command=self.on_smplx_toggle, bg=PANEL, fg=FG, anchor="w",
                       selectcolor=CONTROL, activebackground=PANEL,
                       activeforeground=FG, relief="flat", bd=0,
                       highlightthickness=0, cursor="hand2",
                       font=("TkDefaultFont", 9)).pack(fill="x", padx=10)
        tk.Checkbutton(body, text="Rigged mesh (.glb)", variable=self.use_mesh,
                       command=self.on_mesh_toggle, bg=PANEL, fg=FG, anchor="w",
                       selectcolor=CONTROL, activebackground=PANEL,
                       activeforeground=FG, relief="flat", bd=0,
                       highlightthickness=0, cursor="hand2",
                       font=("TkDefaultFont", 9)).pack(fill="x", padx=10)
        buttons(body, [("SMPL-X folder\u2026", self.choose_smplx_dir),
                       ("Preview (P)", self.preview_depth),
                       ("Load mesh\u2026", self.choose_mesh),
                       ("Mesh library\u2026", self.choose_mesh_library),
                       ("Assets folder\u2026", self.choose_assets),
                       ("Clear assets", self.clear_assets)], small=True)
        self.asset_rows = tk.Frame(body, bg=PANEL)
        self.asset_rows.pack(fill="x", pady=(4, 0))

        # ---- keys ---------------------------------------------------------
        body = section("Keys", opened=False)
        for line in HELP:
            tk.Label(body, text=line, bg=PANEL, fg=MUTED, anchor="w",
                     justify="left", font=("TkFixedFont", 8)).pack(fill="x",
                                                                   padx=13)
        tk.Frame(panel, bg=PANEL, height=10).pack(fill="x")
        self._bind_panel_wheel(panel)

        # ---- status bar and viewports -------------------------------------
        strip = tk.Frame(self.root, bg=PANEL)
        strip.pack(side="bottom", fill="x")
        tk.Frame(strip, bg=EDGE, height=1).pack(fill="x")
        tk.Label(strip, textvariable=self.status, bg=PANEL, fg=FG, anchor="w",
                 padx=12, pady=5, font=("TkDefaultFont", 9)).pack(fill="x")

        # The locked views used to be a strip across the bottom, which cost the
        # main view 196px of height it needed and left the room either side of
        # the export frame empty. The export frame is 2:3 and the window is
        # wide, so fitting that frame into the canvas leaves most of the width
        # black however big the window gets: at 1600x1000 the frame was 460px
        # of a 1340px canvas and the other 880 were nothing at all. Stacked
        # down the left they fill exactly that space, and the main view gets
        # the height back.
        self.ortho_frame = tk.Frame(self.root, bg=BG, width=ORTHO_WIDTH)
        self.ortho_frame.pack(side="left", fill="y")
        self.ortho_frame.pack_propagate(False)
        tk.Frame(self.ortho_frame, bg=EDGE, width=1).pack(side="right", fill="y")
        holders = tk.Frame(self.ortho_frame, bg=BG)
        holders.pack(fill="both", expand=True)
        # The two 3/4 views look from the +X side, the same side the Left view
        # uses, 30 degrees above the horizon and 45 round: one from the front
        # quarter, one from the back quarter.
        for label, yaw, pitch in (("Front", 0.0, 0.0),
                                  ("Left", math.pi / 2, 0.0),
                                  ("Top", 0.0, math.radians(89.0)),
                                  ("Front R", math.radians(45.0),
                                   math.radians(30.0)),
                                  ("Back R", math.radians(135.0),
                                   math.radians(30.0))):
            holder = tk.Frame(holders, bg=BG)
            row, col = divmod(len(self.ortho_views), ORTHO_COLUMNS)
            holder.grid(row=row, column=col, sticky="nsew", padx=1, pady=1)
            holders.columnconfigure(col, weight=1, uniform="ortho")
            holders.rowconfigure(row, weight=1, uniform="ortho")
            tk.Label(holder, text=label.upper(), bg=BG, fg=MUTED, anchor="w",
                     font=("TkDefaultFont", 7, "bold")).pack(fill="x", padx=6,
                                                             pady=(3, 1))
            # width=10 and height=10: a tk Canvas defaults to 378x188, and
            # five of those stacked overflow the column and collapse the last
            # ones to a pixel. Ask for nothing and let `expand` share.
            small = tk.Canvas(holder, bg=VIEW_BG, highlightthickness=0,
                              width=10, height=10)
            camera = Camera(ORTHO_WIDTH // ORTHO_COLUMNS, 200)
            small.pack(fill="both", expand=True)
            camera.yaw, camera.pitch = yaw, pitch
            view = Viewport(small, camera, label.lower().replace(" ", "_"),
                            locked=True)
            self.ortho_views.append(view)
            small.bind("<Configure>", lambda e, v=view: self._resize_view(v, e))
            small.bind("<Button-1>", lambda e, v=view: self.on_press(e, v))
            small.bind("<Alt-ButtonPress-1>",
                       lambda e, v=view: self.on_press(e, v, free_length=True))
            small.bind("<Shift-ButtonPress-1>",
                       lambda e, v=view: self.on_press(e, v, hemisphere=1))
            small.bind("<Control-ButtonPress-1>",
                       lambda e, v=view: self.on_press(e, v, hemisphere=-1))
            small.bind("<B1-Motion>", lambda e, v=view: self.on_drag(e, v))
            small.bind("<ButtonRelease-1>",
                       lambda e, v=view: self.on_release(e, v))
            small.bind("<Motion>", lambda e, v=view: self.on_hover(e, v))
            small.bind("<Double-Button-1>", lambda e, v=view: self.on_double(e, v))
        self.canvas.pack(side="left", fill="both", expand=True)
        self.main_view = Viewport(self.canvas, self.camera, "main")
        self.show_tab(TAB_ORDER[0])
        # Frame the figure once the window has a real size. Without this the
        # editor opens on whatever zoom the Camera defaults to, which left the
        # figure at about 60% of the height of its own export frame - a small
        # figure in a large empty rectangle, which is what the frame is there
        # to stop. `after_idle` rather than now: the canvas is still 1x1 until
        # tk has laid the window out, so framing to it here frames to nothing.
        self.root.after_idle(self._initial_frame)

    def _initial_frame(self):
        if self.canvas.winfo_width() > 1:
            self.frame_all()
            self.status.set("Drag a joint to pose it. X randomizes.")
        else:
            self.root.after(40, self._initial_frame)

    def _refresh_scroll(self):
        self.panel_canvas.update_idletasks()
        self.panel_canvas.configure(scrollregion=self.panel_canvas.bbox("all"))

    def show_tab(self, name):
        """Raise one tab of the panel. Ctrl+Tab cycles; the tabs are also keys.

        Packed and unpacked rather than stacked with `lift`, because the panel
        scrolls: a hidden tab that still takes height would leave the scroll
        region tall enough for all four and the visible one floating in the
        middle of it.
        """
        if name not in self._tabs:
            return
        self.active_tab.set(name)
        for tab, holder in self._tabs.items():
            chosen = tab == name
            if chosen:
                holder.pack(fill="x", expand=False)
            else:
                holder.pack_forget()
            self._tab_buttons[tab].configure(
                fg=FG if chosen else MUTED,
                bg=CONTROL if chosen else PANEL)
        self.panel_canvas.yview_moveto(0.0)
        self._refresh_scroll()

    def next_tab(self, step=1):
        order = list(TAB_ORDER)
        here = order.index(self.active_tab.get()) if \
            self.active_tab.get() in order else 0
        self.show_tab(order[(here + step) % len(order)])
        return "break"

    def set_flag(self, attr, value):
        """Single path for every display toggle, so the panel checkboxes and
        the keyboard shortcuts can never disagree about the state."""
        setattr(self, attr, bool(value))
        var = self._toggle_vars.get(attr)
        if var is not None and var.get() != bool(value):
            var.set(bool(value))
        if attr == "show_ortho":
            if value:
                # pack_forget drops it from the packing order, so re-packing
                # puts it last and the expanding canvas has taken the space
                self.ortho_frame.pack(side="left", fill="y",
                                      before=self.canvas)
            else:
                self.ortho_frame.pack_forget()
        self.redraw(force_ortho=(attr == "show_ortho"))

    def _bind_panel_wheel(self, widget):
        """Bind the wheel on each panel widget rather than globally, so it does
        not also fire on the viewport where the wheel means zoom."""
        for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(event, self._panel_scroll)
        for child in widget.winfo_children():
            self._bind_panel_wheel(child)

    def _panel_scroll(self, event):
        number = getattr(event, "num", None)
        if number == 4:
            step = -1
        elif number == 5:
            step = 1
        else:
            step = -1 if getattr(event, "delta", 0) > 0 else 1
        self.panel_canvas.yview_scroll(step * 2, "units")
        return "break"

    def _resize_view(self, view, event):
        view.camera.width, view.camera.height = event.width, event.height
        view.fit(self.figures)
        self.redraw(force_ortho=True)

    def _bind_events(self):
        c = self.canvas
        c.bind("<Configure>", self.on_resize)
        c.bind("<Button-1>", self.on_press)
        c.bind("<Alt-ButtonPress-1>",
               lambda e: self.on_press(e, free_length=True))
        c.bind("<Shift-ButtonPress-1>", lambda e: self.on_press(e, hemisphere=1))
        c.bind("<Control-ButtonPress-1>",
               lambda e: self.on_press(e, hemisphere=-1))
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Motion>", self.on_hover)
        c.bind("<Double-Button-1>", self.on_double)
        c.bind("<Button-3>", self.on_pan_start)
        c.bind("<B3-Motion>", self.on_pan_move)
        c.bind("<Button-2>", self.on_pan_start)
        c.bind("<B2-Motion>", self.on_pan_move)
        c.bind("<MouseWheel>", self.on_wheel)
        c.bind("<Button-4>", lambda e: self.on_wheel(e, 1))
        c.bind("<Button-5>", lambda e: self.on_wheel(e, -1))
        self.root.bind("<Key>", self.on_key)
        self.root.bind("<Control-z>", lambda e: self.undo())
        # Tab would otherwise move focus between the panel's widgets
        self.root.bind("<Tab>", lambda e: (self.next_figure(), "break")[1])
        self.root.bind("<Control-Tab>", lambda e: self.next_tab(1))

    # -- helpers -----------------------------------------------------------
    def scene_snapshot(self):
        return ([f.snapshot() for f in self.figures], self.active,
                deepcopy(self.props), self.active_prop)

    def restore_scene(self, snap):
        states, active = snap[0], snap[1]
        if len(snap) > 2:               # snapshots taken before objects existed
            self.props = deepcopy(snap[2])
            self.active_prop = snap[3]
        while len(self.figures) < len(states):
            self.figures.append(Skeleton())
        del self.figures[len(states):]
        for figure, state in zip(self.figures, states):
            figure.restore(state)
        self.active = max(0, min(active, len(self.figures) - 1))
        self.figure_label.set("Person %d of %d" % (self.active + 1,
                                                   len(self.figures)))
        self.preset_name.set(self.skeleton.body.get("preset", DEFAULT_PRESET))
        self.refresh_outfit()

    def pose_from_prompt(self):
        """Pose the scene from the prompt box with a local model.

        Replaces the figures rather than editing them: a prompt describes a
        whole pose, and undo puts the old scene back. The import is deferred
        because pose_agent imports this module.
        """
        prompt = self.prompt_text.get().strip()
        if not prompt:
            self.prompt_status.set("Type what the figure should be doing.")
            return
        import pose_agent
        self.prompt_status.set("Asking the model\u2026")
        self.root.update_idletasks()
        llm = pose_agent.discover(host=self.prompt_host.get().strip() or None,
                                  model=self.prompt_model.get().strip() or None)
        width, height = self._sizes()
        figures, props, camera, report = pose_agent.pose_from_prompt(
            prompt, llm, self.camera.width, self.camera.height, width / height)
        self.push_undo()
        self.figures = figures
        self.props = props
        self.active_prop = None
        self.refresh_props()
        self.refresh_outfit()
        # Take the VIEW the agent chose - that is a real decision about which
        # way the pose reads - but frame it here, against this window's own
        # export rectangle, rather than adopting a zoom fitted to a rectangle
        # of a different size.
        self.camera.yaw, self.camera.pitch = camera.yaw, camera.pitch
        frame_scene(self.figures, self.camera, self.frame_rect(),
                    props=self.props)
        self.set_active(0, announce=False)
        self.redraw()
        note = "read by %s" % report["source"]
        if report["warnings"]:
            note += "; %d command(s) skipped" % len(report["warnings"])
        self.prompt_status.set(note)
        self.status.set("Posed from prompt (%s). Ctrl+Z puts it back." % note)

    # -- what the figure is wearing ----------------------------------------
    def set_worn(self, slot):
        """Put one slot on the active figure. Each slot is independent, so a
        hat does not take the coat off."""
        self.push_undo()
        outfit = dict(self.skeleton.outfit or {})
        outfit[slot] = self.outfit_vars[slot].get()
        self.skeleton.outfit = outfit
        self.redraw()
        self.status.set("%s: %s." % (slot.title(), outfit[slot]))

    def set_outfit(self):
        """Put a named look on the figure and show it in the slot menus."""
        import wearables
        dressed = wearables.dress(self.skeleton.outfit, self.outfit_name.get())
        if dressed is None:
            return
        self.push_undo()
        self.skeleton.outfit = dressed
        self.refresh_outfit()
        self.redraw()
        self.status.set("Outfit: %s." % self.outfit_name.get())

    def strip(self):
        self.push_undo()
        self.skeleton.outfit = {}
        self.refresh_outfit()
        self.redraw()
        self.status.set("Bare again.")

    def refresh_outfit(self):
        """Point the panel at the active figure's outfit."""
        import wearables
        outfit = wearables.clean(getattr(self.skeleton, "outfit", None))
        for slot, var in self.outfit_vars.items():
            var.set(outfit.get(slot, "none"))

    # -- objects -----------------------------------------------------------
    def ground_level(self):
        """The y the scene stands on: the lowest point of the lowest figure.

        Read off the figures rather than kept as a number, so a shorter preset
        or a figure that has been dragged downwards still has objects land on
        the floor it is actually standing on.
        """
        lows = []
        for figure in self.figures:
            ankles = [figure.points[i] for i in (10, 13) if figure.visible[i]]
            lows.extend(p[1] for p in (ankles or figure.points))
        return (min(lows) - 8.0) if lows else -150.0

    def add_prop(self, shape=None, announce=True):
        shape = shape or self.prop_shape.get()
        self.push_undo()
        size = props_module.default_size(shape)
        centre = self.skeleton.points[1]
        prop = props_module.make(
            shape, size,
            (centre[0], self.ground_level(), centre[2] + size[2] / 2.0 + 55.0))
        self.props.append(prop)
        self.active_prop = len(self.props) - 1
        self.refresh_props()
        self.redraw()
        if announce:
            self.status.set("Added a %s. Drag it in the viewport; Delete "
                            "removes it." % shape)
        return prop

    def delete_prop(self):
        if self.active_prop is None:
            self.status.set("No object selected.")
            return
        self.push_undo()
        shape = self.props.pop(self.active_prop)["shape"]
        self.active_prop = (len(self.props) - 1) if self.props else None
        self.refresh_props()
        self.redraw()
        self.status.set("Removed the %s." % shape)

    def next_prop(self):
        if not self.props:
            self.status.set("No objects in the scene yet.")
            return
        self.active_prop = 0 if self.active_prop is None \
            else (self.active_prop + 1) % len(self.props)
        self.prop_shape.set(self.props[self.active_prop]["shape"])
        self.refresh_props()
        self.redraw()
        self.status.set("Selected object %d of %d (%s)."
                        % (self.active_prop + 1, len(self.props),
                           self.props[self.active_prop]["shape"]))

    def refresh_props(self):
        if not self.props:
            self.prop_label.set("No objects. Objects show in the depth map, "
                                "not the pose map.")
        elif self.active_prop is None:
            self.prop_label.set("%d object(s); none selected."
                                % len(self.props))
        else:
            prop = self.props[self.active_prop]
            self.prop_label.set("%d of %d: %s, %.0f x %.0f x %.0f cm, %.0f deg"
                                % ((self.active_prop + 1, len(self.props),
                                    prop["shape"]) + tuple(prop["size"])
                                   + (prop["yaw"],)))

    def _with_prop(self, change):
        if self.active_prop is None:
            self.status.set("No object selected.")
            return
        self.push_undo()
        change(self.props[self.active_prop])
        self.refresh_props()
        self.redraw()

    def scale_prop(self, factor):
        def apply(prop):
            prop["size"] = [max(2.0, v * factor) for v in prop["size"]]
            self.status.set("%s is now %.0f x %.0f x %.0f cm."
                            % ((prop["shape"],) + tuple(prop["size"])))
        self._with_prop(apply)

    def turn_prop(self, degrees):
        def apply(prop):
            prop["yaw"] = (prop["yaw"] + degrees) % 360.0
            self.status.set("%s turned to %.0f degrees."
                            % (prop["shape"], prop["yaw"]))
        self._with_prop(apply)

    def drop_prop(self):
        """Put the selected object back on the floor, keeping where it stands."""
        def apply(prop):
            prop["position"][1] = self.ground_level()
            self.status.set("%s dropped to the floor." % prop["shape"])
        self._with_prop(apply)

    def pick_prop(self, x, y, camera):
        """Which object is under the cursor, nearest to the camera first."""
        hits = []
        for poly, depth, index, _picked in solid_quads(self.props, camera):
            if inside_polygon(poly, x, y):
                hits.append((depth, index))
        return min(hits)[1] if hits else None

    def push_undo(self):
        self.undo_stack.append(self.scene_snapshot())
        del self.undo_stack[:-120]

    def undo(self):
        if self.undo_stack:
            self.restore_scene(self.undo_stack.pop())
            self.redraw()
            self.status.set("Undone.")

    def toggle_ortho(self):
        self.set_flag("show_ortho", not self.show_ortho)
        self.status.set("Front, left, top and two 3/4 views on."
                        if self.show_ortho else "Extra views off.")

    def toggle_body(self):
        self.set_flag("show_body", not self.show_body)
        self.status.set("Swept preview on: the fast approximation, not the "
                        "export." if self.show_body else "Swept preview off.")

    def toggle_rig(self):
        self.set_flag("show_rig", not self.show_rig)
        self.status.set("Native rig on: the armature the depth map is made "
                        "from." if self.show_rig else "Native rig off.")

    def toggle(self, attr):
        self.set_flag(attr, not getattr(self, attr))

    def body_quads(self, camera=None, coarse=False):
        """Silhouette polygons for the viewport. The swept stations only change
        when the figure does, so orbiting reprojects a cached body instead of
        rebuilding it."""
        sig = tuple((tuple(f.points), tuple(f.visible), f.body_scale,
                     f.body.get("preset")) for f in self.figures) \
            + (self._thickness(),)
        if self._parts_sig != sig:
            self._parts_cache = {}
            self._parts_sig = sig
        level = 15.0 if coarse else 5.0
        if level not in self._parts_cache:
            self._parts_cache[level] = [
                body_parts(figure, self._thickness(), coarsen=level)
                for figure in self.figures]
        camera = camera or self.camera
        quads = []
        for index, parts in enumerate(self._parts_cache[level]):
            for poly, depth in silhouette_quads(parts, camera):
                quads.append((poly, depth, index))
        quads.sort(key=lambda q: -q[1])   # one global sort across figures
        return quads

    def draw_guides(self, canvas, camera, parent_screen, parent_world, length):
        """The reach sphere sliced by the world XY, YZ and ZX planes.

        Only the half of each circle lying in the hemisphere the drag is locked
        to is drawn, because the other half is unreachable without flipping.
        Dropping the joint on an arc puts the limb exactly in that plane: the
        projection is one-to-one over a hemisphere, so a cursor on the drawn
        arc solves back to the very point that drew it.
        """
        _, _, fwd = camera.basis()
        sign = self.drag_sign
        for _name, axis_a, axis_b, colour in GUIDE_PLANES:
            run = []
            for step in range(97):
                angle = 2.0 * math.pi * step / 96.0
                offset = vadd(vmul(axis_a, length * math.cos(angle)),
                              vmul(axis_b, length * math.sin(angle)))
                if vdot(offset, fwd) * sign < -1e-9:
                    if len(run) >= 4:
                        canvas.create_line(*run, fill=colour, dash=(5, 4))
                    run = []
                    continue
                sx, sy, _ = camera.project(vadd(parent_world, offset))
                run.extend((sx, sy))
            if len(run) >= 4:
                canvas.create_line(*run, fill=colour, dash=(5, 4))

    def rig_bones(self, figure):
        """The posed rig's bones for one figure, as world segments.

        This is the armature the depth map is actually made from - Anny's own
        104 bones, posed by joint angle - rather than the eighteen keypoints
        that say which way to point them. They are two different things and
        the editor used to draw only the second, so what you dragged and what
        came out of the export were never the same picture.

        Cached on the pose, because a drag redraws several times a second and
        solving is numpy over a hundred bones.
        """
        import bodies_lib
        key = (figure, tuple(tuple(p) for p in self.figures[figure].points))
        if getattr(self, "_rig_key", None) == key:
            return self._rig_cache
        self._rig_key = key
        self._rig_cache = []
        try:
            mesh = bodies_lib.for_figures([self.figures[figure]])[0]
            if mesh is not None:
                _solution, _points = pose_body(self.figures[figure], mesh)
                names = mesh["joint_names"]
                parents = mesh["parents"]
                at = lambda j: _solution["bones"][names[j]][1]
                self._rig_cache = [(at(int(parents[j])), at(j))
                                   for j in range(len(names))
                                   if int(parents[j]) >= 0]
        except Exception:
            self._rig_cache = []            # no body set, or a rig that will
        return self._rig_cache              # not map: fall back to keypoints

    def draw_rig(self, camera, canvas, compact=False):
        """The native armature, drawn back to front with the figure."""
        segments = []
        for f in range(len(self.figures)):
            for a, b in self.rig_bones(f):
                pa, pb = camera.project(a), camera.project(b)
                segments.append(((pa[2] + pb[2]) / 2.0, pa, pb, f))
        if not segments:
            return False
        depths = [d for d, _a, _b, _f in segments]
        lo, hi = min(depths), max(depths)
        span = max(1e-6, hi - lo)
        for mid, pa, pb, f in sorted(segments, key=lambda t: -t[0]):
            near = 1.0 - 0.55 * ((mid - lo) / span)
            grey = int(70 + 150 * near)
            if f != self.active:
                grey = int(grey * 0.75)
            canvas.create_line(pa[0], pa[1], pb[0], pb[1],
                               fill="#%02x%02x%02x" % (grey, grey,
                                                       min(255, grey + 12)),
                               width=1 if compact else 2, capstyle="round")
        return True

    def draw_body(self, camera=None, canvas=None, coarse=False):
        """Figures and objects, one depth sort across the lot.

        Sorted together rather than one after the other: a crate in front of a
        knee has to cover the knee, and drawing all the people and then all the
        objects puts every object in front of every person.
        """
        camera = camera or self.camera
        canvas = canvas or self.canvas
        quads = [(poly, depth, index, None)
                 for poly, depth, index in self.body_quads(camera, coarse)]
        quads += [(poly, depth, None, picked)
                  for poly, depth, _i, picked
                  in solid_quads(self.props, camera, self.active_prop)]
        if not quads:
            return
        quads.sort(key=lambda q: -q[1])
        depths = [z for _poly, z, _f, _s in quads]
        lo, hi = min(depths), max(depths)
        span = max(1e-6, hi - lo)
        for poly, depth, index, picked in quads:   # already sorted far to near
            t = (hi - depth) / span
            g = 44.0 + 150.0 * t
            tint = (PROP_TINT if index is None
                    else FIGURE_BODY_TINTS[index % len(FIGURE_BODY_TINTS)])
            shade = "#%02x%02x%02x" % tuple(
                max(0, min(255, int(g * c / 255.0))) for c in tint)
            flat = [v for point in poly for v in point]
            # outline in the fill colour closes the hairline cracks tkinter
            # leaves between adjacent unantialiased polygons
            canvas.create_polygon(*flat, fill=shade,
                                  outline=ACCENT if picked else shade)

    def projected(self, figure=None, camera=None):
        skeleton = self.figures[figure] if figure is not None else self.skeleton
        camera = camera or self.camera
        return [camera.project(p) for p in skeleton.points]

    def pick(self, x, y, camera=None):
        """Nearest joint across every figure. Among near-equal hits the one
        closest to the camera wins, so overlapping people stay reachable.
        Returns (figure, joint)."""
        hits = []
        for f in range(len(self.figures)):
            for i, (sx, sy, depth) in enumerate(self.projected(f, camera)):
                d = math.hypot(sx - x, sy - y)
                if d < PICK_RADIUS:
                    hits.append((d, depth, f, i))
        if not hits:
            return None
        best = min(hits, key=lambda h: (round(h[0] / 6.0), h[1]))
        return best[2], best[3]

    def frame_rect(self):
        """Safe frame matching the export aspect ratio.

        Measured against the CAMERA's size, not the canvas widget's. They are
        the same thing in a live window - `on_resize` sets one from the other
        - but `camera.project` puts the origin at the middle of the camera, so
        a rectangle centred on anything else is not centred on the projection.
        Where the two drifted apart, the figure was framed about a point the
        export did not share and keypoints landed outside the PNG. Widening
        the panel was enough to expose it; the bug was there all along.
        """
        w, h = int(self.camera.width), int(self.camera.height)
        if w < 50 or h < 50:        # asked for before the window was laid out
            w, h = self.canvas.winfo_reqwidth(), self.canvas.winfo_reqheight()
        try:
            aspect = max(1, self.out_w.get()) / max(1, self.out_h.get())
        except tk.TclError:
            aspect = 512 / 768
        return frame_rect(w, h, aspect)

    def export_points(self, out_w, out_h, figure=None):
        which = self.figures if figure is None else [self.figures[figure]]
        return project_people(which, self.camera, self.frame_rect(),
                              out_w, out_h)[0][0]

    def export_people(self, out_w, out_h):
        return project_people(self.figures, self.camera, self.frame_rect(),
                              out_w, out_h)

    # -- mouse -------------------------------------------------------------
    def on_resize(self, event):
        self.camera.width, self.camera.height = event.width, event.height
        self.redraw()


    def on_press(self, event, view=None, free_length=False, hemisphere=0):
        """hemisphere: +1 forces the joint away from the camera, -1 towards it.

        Modifiers arrive as separate bindings rather than being read out of
        event.state, whose bit values are not the same on Windows, X11 and
        macOS. Reading them directly made every drag on Windows look like an
        Alt drag, which silently stretched the bone being dragged.
        """
        view = view or self.main_view
        self.drag_view = view
        view.canvas.focus_set()
        hit = self.pick(event.x, event.y, view.camera)
        if hit is None:
            # joints win over objects: a keypoint inside a crate must stay
            # reachable, and an object is far easier to hit by accident
            prop = self.pick_prop(event.x, event.y, view.camera)
            if prop is not None:
                self.push_undo()
                self.active_prop = prop
                self.prop_shape.set(self.props[prop]["shape"])
                self.drag_prop = prop
                self.drag_last = (event.x, event.y)
                self.selected = None
                self.refresh_props()
                self.redraw()
                return
            # the locked views must not orbit, or they stop being front/top/left
            self.orbit_last = None if view.locked else (event.x, event.y)
            self.selected = None
            self.active_prop = None
            self.redraw()
            return
        figure, idx = hit
        if figure != self.active:
            self.set_active(figure, announce=False)
        self.pending_undo = self.scene_snapshot()
        self._length_guard = dict(self.skeleton.lengths)
        self.selected = self.drag_joint = idx
        sx, sy, _ = view.camera.project(self.skeleton.points[idx])
        self.drag_offset = (event.x - sx, event.y - sy)
        self.drag_free_length = free_length or self.length_mode
        self.drag_plane = self.skeleton.sagittal_plane()
        _, _, fwd = view.camera.basis()
        parent_idx = self.skeleton.parent_of(idx)
        if parent_idx >= 0:
            offset = vsub(self.skeleton.points[idx], self.skeleton.points[parent_idx])
            sign = vdot(offset, fwd)
            self.drag_sign = 1.0 if sign >= 0 else -1.0
        if hemisphere > 0:
            self.drag_sign = 1.0
        elif hemisphere < 0:
            self.drag_sign = -1.0
        self.redraw()

    def on_drag(self, event, view=None):
        view = view or self.main_view
        if self.drag_prop is not None:
            # objects slide in the view plane: an orthographic camera gives no
            # depth from a cursor, and orbiting to push something back is both
            # obvious and exact
            dx = event.x - self.drag_last[0]
            dy = event.y - self.drag_last[1]
            self.drag_last = (event.x, event.y)
            move = view.camera.screen_delta_to_world(dx, dy)
            prop = self.props[self.drag_prop]
            prop["position"] = [a + b for a, b in zip(prop["position"], move)]
            self.redraw()
            return
        if self.orbit_last is not None:
            dx, dy = event.x - self.orbit_last[0], event.y - self.orbit_last[1]
            self.camera.orbit(dx, dy)
            self.orbit_last = (event.x, event.y)
            self.redraw()
            return
        if self.drag_joint is None:
            return
        if self.pending_undo is not None:
            self.undo_stack.append(self.pending_undo)
            del self.undo_stack[:-120]
            self.pending_undo = None
        idx = self.drag_joint
        camera = view.camera
        mx = event.x - self.drag_offset[0]
        my = event.y - self.drag_offset[1]
        _, _, fwd = camera.basis()
        parent_idx = self.skeleton.parent_of(idx)
        pivot = self.skeleton.points[parent_idx] if parent_idx >= 0 \
            else self.skeleton.points[idx]
        if self.skeleton.hinged:
            # two anchors: the joint can only swing about the axis joining
            # them, so the cursor picks an angle rather than a position
            if idx in self.skeleton.anchors:
                here = camera.project(self.skeleton.points[idx])
                self.skeleton.translate(camera.screen_delta_to_world(
                    mx - here[0], my - here[1]))
                self.redraw()
                return
            if self.skeleton.hinge_screen_extent(idx, camera.project) < 8.0:
                self.status.set(
                    "The hinge is edge-on here: swing %s in the Left or Top "
                    "view, or use < and >." % KEYPOINT_NAMES[idx])
                return
            angle = self.skeleton.hinge_angle(idx, (mx, my), camera.project)
            if angle is None:
                self.status.set("%s lies on the hinge axis, nothing to swing."
                                % KEYPOINT_NAMES[idx])
                return
            self.skeleton.hinge_spin(idx, angle)
            self.redraw()
            self.report_joint(idx)
            return
        ax, ay, _ = camera.project(pivot)
        offset = camera.screen_delta_to_world(mx - ax, my - ay)
        target = self.skeleton.solve_drag(idx, offset, fwd, self.drag_sign,
                                          self.drag_free_length)
        self.skeleton.move_joint(idx, target, stretch=self.drag_free_length)
        if self.symmetry and idx in MIRROR_OF:
            self.mirror_drag(idx, target)
        self.redraw()
        self.report_joint(idx)

    def check_lengths(self):
        """Undo a drag that changed a bone length without being asked to.

        A backstop rather than a fix: the drag maths cannot resize a bone
        unless told to, but the flag that tells it depends on reading a
        modifier key, and that reading is platform-specific. If a length moves
        anyway, put the figure back rather than let the error accumulate.
        """
        guard = getattr(self, "_length_guard", None)
        self._length_guard = None
        if guard is None or self.drag_free_length or self.length_mode:
            return
        moved = [c for c in guard
                 if abs(self.skeleton.lengths.get(c, guard[c]) - guard[c]) > 1e-6]
        if not moved:
            return
        if self.undo_stack:
            self.restore_scene(self.undo_stack.pop())
        self.status.set("That drag would have resized %s, so it was undone. "
                        "Hold Alt or turn on length mode to resize a limb."
                        % KEYPOINT_NAMES[moved[0]])

    def mirror_drag(self, idx, target):
        """Apply the drag to the opposite limb.

        The reflected *position* is the wrong thing to aim at: it only sits at
        the right distance from the twin's own parent while the figure is
        perfectly symmetric. Reflect the bone's direction instead and give it
        the twin's own length, which holds however asymmetric the pose is.
        """
        twin = MIRROR_OF[idx]
        skeleton = self.skeleton
        parent = skeleton.parent_of(idx)
        twin_parent = skeleton.parent_of(twin)
        if parent < 0 or twin_parent < 0:
            return
        _origin, normal = self.drag_plane
        offset = vsub(target, skeleton.points[parent])
        mirrored = vsub(offset, vmul(normal, 2.0 * vdot(offset, normal)))
        if vlen(mirrored) < 1e-9:
            return
        length = skeleton.bone_length(twin_parent, twin)
        skeleton.move_joint(twin, vadd(skeleton.points[twin_parent],
                                       vmul(vnorm(mirrored), length)))

    def on_release(self, _event, view=None):
        was_dragging = self.drag_joint is not None or self.drag_prop is not None
        self.check_lengths()
        self.drag_prop = None
        self.drag_joint = None
        self.drag_view = None
        self.pending_undo = None
        self.drag_free_length = False
        self._length_guard = None
        self.orbit_last = None
        self.redraw(force_ortho=was_dragging)

    def on_hover(self, event, view=None):
        view = view or self.main_view
        hit = self.pick(event.x, event.y, view.camera)
        if hit != self.hovered:
            self.hovered = hit
            self.redraw()

    def on_double(self, event, view=None):
        """Double click sends a limb to the other side of the view plane."""
        view = view or self.main_view
        hit = self.pick(event.x, event.y, view.camera)
        if hit is None:
            return
        figure, idx = hit
        if figure != self.active:
            self.set_active(figure, announce=False)
        self.selected = idx
        if self.skeleton.parent_of(idx) < 0:
            self.status.set("%s has nothing above it to swing."
                            % KEYPOINT_NAMES[idx])
            return
        self.push_undo()
        _, _, fwd = view.camera.basis()
        self.skeleton.flip_depth(idx, fwd)
        if self.symmetry and idx in MIRROR_OF:
            self.skeleton.flip_depth(MIRROR_OF[idx], fwd)
        self.redraw()
        self.report_joint(idx)

    def toggle_symmetry(self):
        self.set_flag("symmetry", not self.symmetry)
        self.status.set("Symmetric editing on: the opposite limb mirrors."
                        if self.symmetry else "Symmetric editing off.")

    def set_anchor(self):
        """Toggle the selected joint as an anchor. One anchor re-hangs the
        tree on it; two define a hinge axis."""
        if self.selected is None:
            self.status.set("Select a joint to anchor it.")
            return
        self.push_undo()
        anchors = list(self.skeleton.anchors)
        if self.selected in anchors:
            anchors.remove(self.selected)
        else:
            anchors.append(self.selected)
            if len(anchors) > 2:
                anchors.pop(0)          # oldest gives way
        self.skeleton.anchors = anchors
        self.redraw()
        if not anchors:
            self.status.set("Anchors cleared, back to the neck.")
        elif len(anchors) == 1:
            self.status.set("Anchored at %s: the body pivots around it. "
                            "Anchor a second joint to make a hinge."
                            % KEYPOINT_NAMES[anchors[0]])
        else:
            self.status.set("Hinge %s to %s: dragging now swings about that "
                            "axis only." % (KEYPOINT_NAMES[anchors[0]],
                                            KEYPOINT_NAMES[anchors[1]]))

    def clear_anchor(self):
        self.push_undo()
        self.skeleton.anchors = []
        self.redraw()
        self.status.set("Anchors cleared, back to the neck.")

    def rotate_hinge(self, direction):
        """Step the torso about the hinge axis, so it tips cleanly rather than
        being swung by hand."""
        if not self.skeleton.hinged:
            self.status.set("Set two anchors first, then this tips the body "
                            "about the axis between them.")
            return
        self.push_undo()
        origin, axis = self.skeleton.hinge_axis()
        moving = self.skeleton.hinge_set(ROOT)
        self.skeleton.rotate_about_axis(
            moving, origin, axis, math.radians(direction * self._step()))
        self.redraw()
        self.status.set("Tipped %.1f degrees about %s-%s."
                        % (direction * self._step(),
                           KEYPOINT_NAMES[self.skeleton.anchors[0]],
                           KEYPOINT_NAMES[self.skeleton.anchors[1]]))

    def on_pan_start(self, event):
        self.pan_last = (event.x, event.y)

    def on_pan_move(self, event):
        if self.pan_last is None:
            return
        self.camera.pan(event.x - self.pan_last[0], event.y - self.pan_last[1])
        self.pan_last = (event.x, event.y)
        self.redraw()

    def on_wheel(self, event, direction=None):
        if direction is None:
            direction = 1 if getattr(event, "delta", 0) > 0 else -1
        self.camera.zoom_by(1.12 ** direction)
        self.redraw()

    def on_key(self, event):
        if isinstance(self.root.focus_get(), (tk.Entry, tk.Spinbox)):
            return
        key = event.keysym.lower()
        actions = {
            "1": lambda: self.set_view("front"), "2": lambda: self.set_view("back"),
            "3": lambda: self.set_view("right"), "4": lambda: self.set_view("left"),
            "5": lambda: self.set_view("top"), "6": lambda: self.set_view("bottom"),
            "b": self.toggle_body,
            "k": self.toggle_rig,       # r is taken by reset_pose
            "tab": self.next_figure, "0": self.frame_all,
            "bracketleft": lambda: self.rotate_figure(-1, "y"),
            "bracketright": lambda: self.rotate_figure(1, "y"),
            "semicolon": lambda: self.rotate_figure(-1, "x"),
            "apostrophe": lambda: self.rotate_figure(1, "x"),
            "comma": lambda: self.rotate_figure(-1, "z"),
            "period": lambda: self.rotate_figure(1, "z"),
            "s": self.toggle_symmetry, "a": self.set_anchor,
            "less": lambda: self.rotate_hinge(-1),
            "greater": lambda: self.rotate_hinge(1),
            "o": self.toggle_ortho,
            "g": lambda: self.toggle("show_grid"),
            "n": lambda: self.toggle("show_labels"),
            "d": lambda: self.toggle("depth_shading"),
            "r": self.reset_pose, "m": self.mirror,
            "f": self.flip_selected, "v": self.toggle_visibility,
            "l": self.toggle_length_mode, "p": self.preview_depth,
            "x": self.randomize_pose,
        }
        if event.keysym == "X":          # shift: a fresh seed, same settings
            return self.randomize_again()
        if key in actions:
            actions[key]()

    # -- commands ----------------------------------------------------------
    def set_view(self, name):
        self.camera.set_view(name)
        self.redraw()

    def apply_preset(self, name):
        """Change proportions without losing the pose."""
        self.push_undo()
        self.skeleton.apply_body(preset_params(name))
        self.redraw()
        self.status.set(f"{name}: proportions applied, pose kept.")

    def restore_proportions(self):
        """Put the bone lengths back to the preset without losing the pose."""
        self.push_undo()
        name = self.skeleton.body.get("preset", DEFAULT_PRESET)
        self.skeleton.apply_body(preset_params(name))
        self.redraw()
        self.status.set("Proportions restored from %s; pose kept." % name)

    def reset_pose(self):
        self.push_undo()
        self.skeleton = Skeleton(preset_params(self.preset_name.get()))
        self.redraw()
        self.status.set("Rest pose restored.")

    def mirror(self):
        self.push_undo()
        self.skeleton.mirror_x()
        self.redraw()
        self.status.set("Pose mirrored.")

    def chosen_random_parts(self):
        return tuple(part for part in randomize.PART_ORDER
                     if self.random_parts[part].get())

    def randomize_pose(self, seed=None):
        """Scramble the chosen parts of the active figure.

        Goes through `randomize`, which goes through `move_joint` and
        `rotate_about_axis` - the same calls a drag makes - so the result is a
        pose the editor could have been dragged into and `check_lengths` has
        nothing to complain about. An edge case it finds is therefore a real
        one rather than an artefact of how it was generated.
        """
        parts = self.chosen_random_parts()
        if not parts:
            self.random_status.set("Nothing ticked: choose what to randomize.")
            return
        if seed is None:
            typed = self.random_seed.get().strip()
            try:
                seed = int(typed) if typed else random.randrange(1, 10 ** 9)
            except ValueError:
                seed = random.randrange(1, 10 ** 9)
        self.push_undo()
        amount = float(self.random_amount.get())
        if "preset" in parts:
            name = random.Random(seed).choice(list(BODY_PRESETS))
            self.preset_name.set(name)
            self.apply_preset(name)
        if "outfit" in parts:
            import wearables
            look = random.Random(seed + 1).choice(
                [n for n in wearables.OUTFIT_NAMES if n != "bare"])
            self.outfit_name.set(look)
            self.set_outfit()
        randomize.randomize_figure(self.skeleton, parts, amount, seed=seed)
        if "camera" in parts:
            self.frame_all()
        self.random_seed.set(str(seed))
        self.random_status.set("Seed %d, %s at %.2f. The seed reproduces it "
                               "exactly." % (seed, "+".join(parts), amount))
        self.redraw()
        self.status.set("Randomized: seed %d." % seed)

    def randomize_again(self):
        """Another draw from the same settings, and a new seed to name it."""
        self.random_seed.set("")
        self.randomize_pose()

    def scale(self, factor):
        self.push_undo()
        self.skeleton.scale(factor)
        self.redraw()

    def flip_selected(self):
        if self.selected is None or self.selected not in PARENT:
            self.status.set("Select a joint below the neck first.")
            return
        self.push_undo()
        _, _, fwd = self.camera.basis()
        self.skeleton.flip_depth(self.selected, fwd)
        self.redraw()
        self.status.set(f"Flipped {KEYPOINT_NAMES[self.selected]} through the view plane.")

    def toggle_visibility(self):
        if self.selected is None:
            return
        self.push_undo()
        self.skeleton.visible[self.selected] = not self.skeleton.visible[self.selected]
        state = "shown" if self.skeleton.visible[self.selected] else "hidden"
        self.redraw()
        self.status.set(f"{KEYPOINT_NAMES[self.selected]} {state}.")

    def toggle_length_mode(self):
        self.length_mode = not self.length_mode
        self.status.set("Length mode on: dragging changes limb length."
                        if self.length_mode else "Length mode off.")
        self.redraw()

    def report_joint(self, idx):
        if idx not in PARENT:
            self.status.set("Moving the whole figure.")
            return
        _, _, fwd = self.camera.basis()
        offset = vsub(self.skeleton.points[idx], self.skeleton.points[PARENT[idx]])
        length = vlen(offset)
        depth = vdot(offset, fwd)
        visible_len = math.sqrt(max(0.0, length * length - depth * depth))
        self.status.set(
            f"{KEYPOINT_NAMES[idx]}   length {length:.1f}   "
            f"on screen {visible_len:.1f}   depth {depth:+.1f}")

    # -- file I/O ----------------------------------------------------------
    def _sizes(self):
        try:
            return max(16, self.out_w.get()), max(16, self.out_h.get())
        except tk.TclError:
            return 512, 768

    def _thickness(self):
        try:
            return max(0.2, min(3.0, float(self.body_thickness.get())))
        except ValueError:
            return 1.0

    # -- SMPL-X ------------------------------------------------------------
    def choose_smplx_dir(self):
        path = filedialog.askdirectory(
            title="Folder containing smplx/SMPLX_NEUTRAL.npz")
        if not path:
            return
        self.smplx_dir = path
        self._smplx_cache.clear()
        self.status.set(f"SMPL-X model folder: {path}")

    def on_smplx_toggle(self):
        if self.use_smplx.get() and not self.smplx_body():
            self.use_smplx.set(False)
            return
        if self.use_smplx.get():
            self.use_mesh.set(False)     # the two mesh sources are exclusive
            self.status.set("Depth source: SMPL-X mesh.")
        else:
            self.status.set("Depth source: built-in anatomy.")

    def smplx_body(self):
        """Load and cache the model for the current preset's gender."""
        preset = self.skeleton.body.get("preset", DEFAULT_PRESET)
        gender = ("female" if preset.startswith("Female")
                  else "male" if preset.startswith("Male") else "neutral")
        if gender in self._smplx_cache:
            return self._smplx_cache[gender]
        try:
            import smplx_backend
            body = smplx_backend.SmplxBody(self.smplx_dir or None, gender=gender)
        except ImportError as exc:
            messagebox.showerror(
                "SMPL-X not installed",
                f"{exc}\n\npip install smplx torch\n\n"
                "smplx_backend.py must sit next to this script.")
            return None
        except Exception as exc:
            messagebox.showerror(
                "SMPL-X model not found",
                f"{exc}\n\nDownload SMPL-X v1.1 from smpl-x.is.tue.mpg.de and "
                "arrange it as\n\n  <folder>/smplx/SMPLX_NEUTRAL.npz\n\n"
                "then pick <folder> with the model folder button.")
            return None
        self._smplx_cache[gender] = body
        return body

    # -- rigged mesh (MPFB2, Mixamo, Daz, VRoid) ---------------------------
    def choose_mesh(self):
        path = filedialog.askopenfilename(
            title="Rigged humanoid exported with its armature",
            filetypes=[("glTF binary", "*.glb"), ("glTF", "*.gltf")])
        if not path:
            return
        try:
            import mesh_backend
            mesh = mesh_backend.load_rigged_mesh(path)
            roles = mesh_backend.resolve_bones(mesh["joint_names"])
            problems = mesh_backend.validate_roles(mesh, roles)
            if problems:
                messagebox.showerror(
                    "This rig does not map cleanly",
                    "\n".join(problems) + "\n\nRun\n  python3 mesh_backend.py "
                    "--inspect %s\nto see the bone names."
                    % os.path.basename(path))
                return
            mesh["roles"] = roles
        except ImportError:
            messagebox.showerror("mesh_backend.py missing",
                                 "Put mesh_backend.py next to this script.")
            return
        except Exception as exc:
            messagebox.showerror("Could not read the mesh", str(exc))
            return
        self._rigged_mesh = mesh
        self.mesh_path = path
        self.use_mesh.set(True)
        self.use_smplx.set(False)
        dropped = mesh.get("dropped_vertices", 0)
        note = ""
        if dropped:
            note = (", dropped %d vertices of loose geometry (MakeHuman "
                    "helpers)" % dropped)
        self.status.set("Loaded %s: %d vertices, %d bones%s."
                        % (os.path.basename(path), len(mesh["vertices"]),
                           len(mesh["joint_names"]), note))

    def choose_mesh_library(self):
        """A folder of .glb bodies, one per body type.

        Files are matched to presets by name, so "female_curvy.glb" or
        "Female, curvy.glb" both bind to that preset. Each figure then renders
        with the body matching its own preset.
        """
        folder = filedialog.askdirectory(title="Folder of .glb bodies")
        if not folder:
            return
        def key(text):
            return "".join(ch for ch in text.lower() if ch.isalnum())
        found, unmatched = {}, []
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith((".glb", ".gltf")):
                continue
            stem = key(os.path.splitext(name)[0])
            match = None
            for preset in BODY_PRESETS:
                if key(preset) == stem:
                    match = preset
                    break
                if match is None and (key(preset) in stem or stem in key(preset)):
                    match = preset
            if match:
                found[match] = os.path.join(folder, name)
            else:
                unmatched.append(name)
        if not found:
            messagebox.showerror(
                "No bodies matched",
                "None of the files matched a body type.\n\nName them after the "
                "presets, for example:\n  male_average.glb\n  female_curvy.glb"
                "\n\nPresets: " + ", ".join(BODY_PRESETS))
            return
        self.mesh_library = found
        self._mesh_cache.clear()
        self.use_mesh.set(True)
        self.use_smplx.set(False)
        note = (", ignored %d file(s)" % len(unmatched)) if unmatched else ""
        self.status.set("Mesh library: %d of %d body types matched%s."
                        % (len(found), len(BODY_PRESETS), note))
        self.redraw()

    def choose_assets(self):
        """A folder of hair and clothing exported on the same armature."""
        folder = filedialog.askdirectory(title="Folder of hair / clothing .glb")
        if not folder:
            return
        import mesh_backend
        self.assets = mesh_backend.load_assets(folder)
        if not self.assets:
            messagebox.showerror("No assets", "No .glb or .gltf files there.")
            return
        self._mesh_cache.clear()
        self.rebuild_asset_rows()
        self.status.set("%d assets found. Tick one to put it on this person."
                        % len(self.assets))

    def clear_assets(self):
        self.skeleton.assets = []
        self.rebuild_asset_rows()
        self.status.set("Person %d is wearing nothing." % (self.active + 1))

    def rebuild_asset_rows(self):
        if self.asset_rows is None:
            return
        """Assets are discovered at runtime, so this part of the panel is
        rebuilt rather than laid out up front."""
        for child in self.asset_rows.winfo_children():
            child.destroy()
        for name in self.assets:
            var = tk.BooleanVar(value=name in getattr(self.skeleton, "assets", []))
            tk.Checkbutton(
                self.asset_rows, text=name, variable=var,
                command=lambda n=name, v=var: self.set_asset(n, v.get()),
                bg=PANEL, fg=FG, anchor="w", selectcolor=CONTROL,
                activebackground=PANEL, activeforeground=FG, relief="flat",
                bd=0, highlightthickness=0, cursor="hand2",
                font=("TkDefaultFont", 9)).pack(fill="x", padx=10)
        self._bind_panel_wheel(self.asset_rows)
        self._refresh_scroll()

    def set_asset(self, name, wanted):
        worn = list(getattr(self.skeleton, "assets", []))
        if wanted and name not in worn:
            worn.append(name)
        elif not wanted and name in worn:
            worn.remove(name)
        self.skeleton.assets = worn
        self.use_mesh.set(True)
        self.status.set("%s: %s" % (name, "on" if wanted else "off"))

    def asset_meshes(self, figure):
        """Loaded assets this figure is wearing. Loose shells are kept: hair is
        many separate strands, not helper geometry."""
        import mesh_backend
        out = []
        for name in getattr(figure, "assets", []):
            path = self.assets.get(name)
            if path is None:
                continue
            key = ("asset", path)
            if key not in self._mesh_cache:
                self._mesh_cache[key] = mesh_backend.load_rigged_mesh(
                    path, drop_loose=False)
            out.append(self._mesh_cache[key])
        return out

    def mesh_for(self, figure):
        """The rigged body this figure should use, by its preset."""
        import mesh_backend
        path = self.mesh_library.get(figure.body.get("preset"))
        if path is None:
            return self._rigged_mesh
        if path not in self._mesh_cache:
            mesh = mesh_backend.load_rigged_mesh(path)
            mesh["roles"] = mesh_backend.resolve_bones(mesh["joint_names"])
            problems = mesh_backend.validate_roles(mesh, mesh["roles"])
            if problems:
                raise RuntimeError("%s: %s" % (os.path.basename(path),
                                               problems[0]))
            self._mesh_cache[path] = mesh
        return self._mesh_cache[path]

    def on_mesh_toggle(self):
        if self.use_mesh.get():
            if self._rigged_mesh is None and not self.mesh_library:
                self.use_mesh.set(False)
                self.choose_mesh()
            else:
                self.use_smplx.set(False)
                self.status.set("Depth source: rigged mesh.")
        else:
            self.status.set("Depth source: built-in anatomy.")

    def mesh_depth_image(self, width, height):
        jobs = [(figure, self.mesh_for(figure), self.asset_meshes(figure))
                for figure in self.figures]
        return rigged_depth_image(jobs, self.camera, self.frame_rect(),
                                  width, height, self.props)

    def smplx_depth_image(self, width, height):
        import smplx_backend
        body = self.smplx_body()
        if body is None:
            return None
        x0, y0, x1, y1 = self.frame_rect()
        s = width / (x1 - x0)
        k = self.camera.zoom * s
        right, up, fwd = self.camera.basis()
        zbuf = np.full((height, width), np.inf)
        for figure in self.figures:
            points = {name: figure.points[i]
                      for i, name in enumerate(KEYPOINT_NAMES)}
            verts, faces = smplx_backend.mesh_from_skeleton(body, points)
            rel = verts - np.asarray(self.camera.target, dtype=float)
            px = np.column_stack([
                (rel @ np.asarray(right) * self.camera.zoom
                 + self.camera.width / 2.0 - x0) * s,
                (-(rel @ np.asarray(up)) * self.camera.zoom
                 + self.camera.height / 2.0 - y0) * s,
                rel @ np.asarray(fwd) * k])
            np.minimum(zbuf, smplx_backend.rasterize_depth(px, faces, width,
                                                           height), out=zbuf)
        return smplx_backend.depth_to_image(zbuf)

    def depth_image(self, width, height, anatomy=False):
        """Depth map framed identically to the pose export, so the two line up
        pixel for pixel.

        Rigged geometry, from a mesh loaded by hand, from SMPL-X, or from the
        body set on disk - in that order, because each is a deliberate choice
        over the one after it. What it will *not* do is fall through to the
        built-in anatomy: that sweep is a stack of tapering cross-sections, it
        is there to draw the viewport and to cut garments out of, and an export
        of it is a picture of a mannequin. Falling back silently is the trap,
        because the file still appears.

        `anatomy=True` asks for the sweep deliberately; the low-resolution
        viewport preview does.
        """
        if anatomy:
            return anatomy_depth_image(self.figures, self.camera,
                                       self.frame_rect(), width, height,
                                       self._thickness(), self.props)
        if self.use_mesh.get() and (self._rigged_mesh is not None
                                    or self.mesh_library):
            return self.mesh_depth_image(width, height)
        if self.use_smplx.get():
            image = self.smplx_depth_image(width, height)
            if image is not None:
                return image
        import bodies_lib
        jobs = [(figure, mesh, self.asset_meshes(figure))
                for figure, mesh in zip(self.figures,
                                        bodies_lib.for_figures(self.figures))]
        return rigged_depth_image(jobs, self.camera, self.frame_rect(),
                                  width, height, self.props)

    def _depth_or_complain(self, width, height):
        """The depth map, or a dialog saying what is missing and None.

        There is no cheaper thing to return. An export that quietly drops to
        the built-in sweep is worse than no export, because the file is there
        and it is a picture of a mannequin.
        """
        import bodies_lib
        try:
            return self.depth_image(width, height)
        except bodies_lib.MissingBodies as missing:
            messagebox.showerror("No rigged body", str(missing))
        except Exception as problem:
            messagebox.showerror("Depth map failed", str(problem))
        return None

    def preview_depth(self):
        if np is None or Image is None:
            messagebox.showerror("Missing dependency",
                                 "Depth maps need NumPy and Pillow.\n\n"
                                 "pip install numpy pillow")
            return
        w, h = self._sizes()
        f = min(1.0, 560.0 / max(w, h))
        # The preview shows what the export will be, rigged geometry included,
        # rather than a cheaper stand-in that flatters it.
        img = self._depth_or_complain(max(16, int(w * f)), max(16, int(h * f)))
        if img is None:
            return
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        win = tk.Toplevel(self.root)
        win.title("Depth preview")
        win.configure(bg=BG)
        photo = tk.PhotoImage(master=win, data=base64.b64encode(buf.getvalue()))
        label = tk.Label(win, image=photo, bg=BG, bd=0)
        label.image = photo
        label.pack()
        win.bind("<Escape>", lambda _e: win.destroy())
        self.status.set("Depth preview: white is nearest. Esc closes it.")

    def export_depth(self):
        if np is None or Image is None:
            messagebox.showerror("Missing dependency",
                                 "Depth maps need NumPy and Pillow.\n\n"
                                 "pip install numpy pillow")
            return
        w, h = self._sizes()
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG image", "*.png")],
                                            initialfile="depth.png")
        if not path:
            return
        image = self._depth_or_complain(w, h)
        if image is None:
            return
        image.save(path)
        self.status.set(f"Saved depth map {os.path.basename(path)} ({w}x{h}).")

    def export_png(self):
        if Image is None:
            messagebox.showerror("Pillow missing",
                                 "PNG export needs Pillow.\n\npip install pillow")
            return
        w, h = self._sizes()
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG image", "*.png")],
                                            initialfile="pose.png")
        if not path:
            return
        stick = resolution_stickwidth(w, h) if self.thick_lines.get() else 4
        img = render_openpose(self.export_people(w, h), w, h,
                              stickwidth=stick, dot_radius=stick)
        img.save(path)
        self.status.set(f"Saved {os.path.basename(path)} ({w}x{h}).")

    def save_json(self):
        w, h = self._sizes()
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON", "*.json")],
                                            initialfile="pose.json")
        if not path:
            return
        data = scene_to_dict(self.figures, self.camera,
                             [self.export_points(w, h, f)
                              for f in range(len(self.figures))], w, h,
                             self.props)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        self.status.set(f"Saved {os.path.basename(path)}.")

    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.push_undo()
        self.figures = scene_load(data, self.camera)
        self.props = scene_objects(data)
        self.active_prop = None
        self.refresh_props()
        self.set_active(0, announce=False)
        self.status.set("Loaded %s, %d %s." % (
            os.path.basename(path), len(self.figures),
            "person" if len(self.figures) == 1 else "people"))

    # -- drawing -----------------------------------------------------------
    def figure_palette(self, index):
        """Viewport-only tint so people can be told apart at a glance.

        The keypoint colours are part of the OpenPose format, so this must
        never reach an export: hues are rotated a little per figure, the first
        figure keeping the canonical palette exactly.
        """
        if index in self._palettes:
            return self._palettes[index]
        # A fixed, bounded set of hue offsets rather than a growing one: the
        # ramp must still read as the OpenPose palette however many people the
        # scene holds, so past seven figures the tints repeat instead of
        # drifting into a different colour scheme.
        # Rotating hue alone was too subtle, since depth shading and the
        # dimming of inactive figures scale the colours down and shrink the
        # difference with them. Pairing it with saturation and value changes
        # keeps the figures apart at any brightness.
        hue, sat_scale, val_scale = FIGURE_STYLES[index % len(FIGURE_STYLES)]
        palette = []
        for r, g, b in COLORS:
            h, sat, val = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            r2, g2, b2 = colorsys.hsv_to_rgb((h + hue) % 1.0,
                                             min(1.0, sat * sat_scale),
                                             min(1.0, val * val_scale))
            palette.append((int(round(r2 * 255)), int(round(g2 * 255)),
                            int(round(b2 * 255))))
        self._palettes[index] = palette
        return palette

    @staticmethod
    def shade(color, factor):
        return "#%02x%02x%02x" % tuple(max(0, min(255, int(c * factor))) for c in color)

    def scene_signature(self):
        """What the locked views actually depend on. Deliberately excludes the
        main camera and the hover highlight: orbiting cannot change a fixed
        view, and a stale hover ring in a thumbnail is harmless."""
        return (tuple(tuple(f.points) for f in self.figures),
                tuple(tuple(f.visible) for f in self.figures),
                tuple(tuple(f.anchors) for f in self.figures),
                self.active, self.selected, self.show_body, self.show_rig,
                self._thickness())

    def redraw(self, force_ortho=False):
        self.draw_scene(self.main_view)
        if not self.show_ortho:
            for view in self.ortho_views:
                view.canvas.delete("all")
            return
        signature = self.scene_signature()
        if not force_ortho and signature == self._ortho_sig:
            return
        if not force_ortho and self.drag_joint is not None:
            # Mid-drag the main view must stay responsive, so the thumbnails
            # repaint at a limited rate and get a full one on release. The
            # interval is measured from when the last repaint *finished*:
            # timing it from the start feeds back on itself, since a slow
            # repaint then makes the next one due sooner.
            if time.monotonic() - self._ortho_time < 0.12:
                return
        self._ortho_sig = signature
        for view in self.ortho_views:
            if self.drag_joint is None:
                view.refit_if_needed(self.figures)
            self.draw_scene(view)
        self._ortho_time = time.monotonic()

    def draw_scene(self, view):
        c = view.canvas
        camera = view.camera
        compact = view.locked
        c.delete("all")
        screens = [self.projected(f, camera) for f in range(len(self.figures))]
        pts = screens[self.active]
        every = [p[2] for pf in screens for p in pf]
        lo, hi = min(every), max(every)
        span = max(1e-6, hi - lo)

        if self.show_grid and not compact:
            self.draw_grid(camera, c)
        if not compact:
            self.draw_frame()
        if self.show_body:
            self.draw_body(camera, c, coarse=compact)
        rig_drawn = self.show_rig and self.draw_rig(camera, c, compact)

        if self.skeleton.hinged:
            a, b = self.skeleton.anchors
            ax, ay, _ = pts[a]
            bx, by, _ = pts[b]
            ex, ey = bx - ax, by - ay
            c.create_line(ax - ex * 0.6, ay - ey * 0.6,
                          bx + ex * 0.6, by + ey * 0.6,
                          fill="#ffd27f", dash=(6, 4))

        drag_parent = (self.skeleton.parent_of(self.drag_joint)
                       if self.drag_joint is not None else -1)
        if drag_parent >= 0 and not self.skeleton.hinged \
                and view is (self.drag_view or self.main_view):
            self.draw_guides(c, camera, pts[drag_parent],
                             self.skeleton.points[drag_parent],
                             self.skeleton.bone_length(drag_parent,
                                                       self.drag_joint))
            px, py, _ = pts[drag_parent]
            r = self.skeleton.bone_length(drag_parent,
                                          self.drag_joint) * camera.zoom
            c.create_oval(px - r, py - r, px + r, py + r, outline="#4a4a58",
                          dash=(4, 4))
            c.create_line(px, py, pts[self.drag_joint][0], pts[self.drag_joint][1],
                          fill="#4a4a58", dash=(2, 4))

        # limbs from every figure in one back-to-front pass, so people in
        # front of each other overlap correctly
        limbs = []
        for f, screen in enumerate(screens):
            visible = self.figures[f].visible
            for i, (a, b) in enumerate(LIMB_SEQ):
                if visible[a] and visible[b]:
                    limbs.append(((screen[a][2] + screen[b][2]) / 2.0, f, i))
        for mid, f, i in sorted(limbs, key=lambda t: -t[0]):
            a, b = LIMB_SEQ[i]
            screen = screens[f]
            factor = 1.0 - 0.5 * ((mid - lo) / span) if self.depth_shading else 1.0
            if f != self.active:
                factor *= 0.92          # dim, but not enough to hide the tint
            # With the rig on screen the OpenPose limbs are the handles you
            # drag, not the subject, so they thin down out of its way.
            thick = ((2 if compact else 3) if rig_drawn
                     else (3 if compact else 7) if self.show_body
                     else (5 if compact else 9))
            c.create_line(screen[a][0], screen[a][1], screen[b][0], screen[b][1],
                          fill=self.shade(self.figure_palette(f)[i], factor),
                          width=thick, capstyle="round")

        joints = [(screen[i][2], f, i) for f, screen in enumerate(screens)
                  for i in range(len(KEYPOINT_NAMES))]
        for depth, f, i in sorted(joints, key=lambda t: -t[0]):
            x, y, _ = screens[f][i]
            factor = 1.0 - 0.5 * ((depth - lo) / span) if self.depth_shading else 1.0
            if f != self.active:
                factor *= 0.92
            r = (2.5 if compact else 4.0) if self.show_body \
                else (3.5 if compact else 5.5)
            if self.figures[f].visible[i]:
                c.create_oval(x - r, y - r, x + r, y + r,
                              fill=self.shade(self.figure_palette(f)[i], factor),
                              outline="")
            else:
                c.create_oval(x - r, y - r, x + r, y + r, outline="#55555f",
                              dash=(2, 2))
            if f == self.active and i == self.selected:
                c.create_oval(x - r - 4, y - r - 4, x + r + 4, y + r + 4,
                              outline=ACCENT, width=2)
            elif self.hovered == (f, i):
                c.create_oval(x - r - 3, y - r - 3, x + r + 3, y + r + 3,
                              outline="#8a8a99", width=1)
            if f == self.active and i in self.figures[f].anchors:
                c.create_line(x - r - 7, y, x + r + 7, y, fill="#ffd27f")
                c.create_line(x, y - r - 7, x, y + r + 7, fill="#ffd27f")
                c.create_oval(x - r - 5, y - r - 5, x + r + 5, y + r + 5,
                              outline="#ffd27f", width=1)
            if self.show_labels and f == self.active and not compact:
                c.create_text(x + 10, y - 10, text=KEYPOINT_NAMES[i], fill=MUTED,
                              anchor="w", font=("TkFixedFont", 8))

        if compact:
            return
        mode = "length" if (self.length_mode or self.drag_free_length) else "rotate"
        if self.symmetry:
            mode += " +mirror"
        if self.skeleton.hinged:
            mode += " hinge:%s-%s" % (KEYPOINT_NAMES[self.skeleton.anchors[0]],
                                      KEYPOINT_NAMES[self.skeleton.anchors[1]])
        elif self.skeleton.anchors:
            mode += " anchor:" + KEYPOINT_NAMES[self.skeleton.anchors[0]]
        c.create_text(12, 12, anchor="nw", fill=MUTED, font=("TkFixedFont", 9),
                      text=f"yaw {math.degrees(self.camera.yaw):+.0f}"
                           f"   pitch {math.degrees(self.camera.pitch):+.0f}"
                           f"   zoom {self.camera.zoom:.2f}   mode {mode}"
                           f"   person {self.active + 1}/{len(self.figures)}"
                           f"   v{VERSION}")

    def draw_frame(self):
        x0, y0, x1, y1 = self.frame_rect()
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        veil = "#0a0a0d"
        for box in ((0, 0, w, y0), (0, y1, w, h), (0, y0, x0, y1), (x1, y0, w, y1)):
            self.canvas.create_rectangle(*box, fill=veil, outline="", stipple="gray50")
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="#3a3a46")

    def draw_grid(self, camera=None, canvas=None):
        camera = camera or self.camera
        canvas = canvas or self.canvas
        ground = min(p[1] for f in self.figures for p in f.points) - 2.0
        extent, step = 120, 30
        for i in range(-extent, extent + 1, step):
            for a, b in (((i, ground, -extent), (i, ground, extent)),
                         ((-extent, ground, i), (extent, ground, i))):
                x0, y0, _ = camera.project(a)
                x1, y1, _ = camera.project(b)
                canvas.create_line(x0, y0, x1, y1, fill="#22222a")
