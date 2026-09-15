#!/usr/bin/env python3
"""
3D OpenPose editor - drag-and-drop pose authoring with fixed limb lengths.

Interaction model
-----------------
Every limb has a fixed length L. Dragging a joint does not move it freely: the
drop point is projected onto the sphere of radius L centred on the upstream
(parent) joint. Because the view is orthographic, that sphere projects to a
circle of radius L on screen, so:

  * cursor inside the circle -> the missing length becomes depth, and the limb
    appears foreshortened,
  * cursor on or outside the circle -> the limb lies exactly in the view plane
    and shows its maximum possible on-screen length,
  * the hemisphere (towards / away from the camera) is locked for the duration
    of a drag, so the limb never pops through the screen plane by accident.

Rotating a joint carries its whole downstream chain with it (forward kinematics),
so moving a shoulder swings the elbow and wrist too.

Requires: Python 3.8+, tkinter (python3-tk on Debian/Ubuntu).
Optional: Pillow, only for PNG export.

Run:  python3 openpose3d_editor.py

Where everything lives
----------------------
This file was one ~4300-line module grown by successive edits. It is now the
entry point and the public surface, and the parts live in modules of their
own, roughly in dependency order:

    version.py      the build number, so nothing has to import the app for it
    vecmath.py      three-vectors and 3x3 matrices, as plain tuples
    anthro.py       ANSUR II, the presets, and the rest pose they derive
    camera.py       the orthographic turntable
    skeleton.py     the eighteen keypoints and the pose they hold
    posemap.py      the OpenPose PNG and its canonical palette
    anatomy.py      the swept body: viewport preview and garment profiles
    raster.py       silhouettes for the canvas, analytic depth for the export
    exporting.py    framing, and the two conditioning images that must agree
    scenefile.py    reading and writing a scene, old ones included
    randomize.py    random poses, for finding the edge cases
    ui_app.py       the window: Viewport, EditorApp, every binding

Everything those modules define is re-exported here, so
`from openpose3d_editor import Skeleton` and
`import openpose3d_editor as editor; editor.render_openpose(...)` both keep
working exactly as before. New code is welcome to import the module it
actually wants instead.
"""

from __future__ import annotations

import sys

from version import VERSION

from vecmath import (IDENTITY, any_perpendicular, matvec, rotation_between,
                     vadd, vcross, vdot, vlen, vmul, vnorm, vsub)
from anthro import (ANSUR, BASE_BODY, BODY_PRESETS, DEFAULT_PRESET,
                    HEAD_EXPONENT, REST_POSE, build_rest_points,
                    derive_proportions, merge_body, preset_params)
from camera import Camera
from skeleton import (ADJACENCY, CHILDREN, COLORS, EXTREMITIES,
                      EXTREMITY_ANGLES, FIGURE_BODY_TINTS,
                      FIGURE_STYLES, GIRDLE, KEYPOINT_NAMES, LIMB_SEQ,
                      MIRROR_OF, MIRROR_PAIRS, PARENT, PROP_TINT, ROOT,
                      Skeleton, clean_extremities, reroot)
from posemap import (ellipse_polygon, render_openpose, resolution_stickwidth)
from anatomy import (P_CALF, P_FOOT, P_FOREARM, P_HAND, P_HEAD, P_NECK,
                     P_THIGH, P_UPPER_ARM, _blob, _tube, body_parts,
                     body_segments, carry_chain, carry_frame, sample_profile,
                     sweep, torso_profile)
from raster import (depth_buffer, depth_to_grey, inside_polygon, render_depth,
                    silhouette_quads, solid_quads)
from exporting import (ASPECTS, ASPECT_NAMES, GROUND_EXTENT,
                       anatomy_depth_image, aspect_name,
                       body_frame, frame_rect, frame_scene,
                       ground_level, ground_part, parse_size,
                       pose_body, pose_image, project_people,
                       prop_groups, rigged_depth_image,
                       rigged_keypoints, silhouette_points,
                       size_for, to_camera_space)
from scenefile import (scene_from_dict, scene_load, scene_objects,
                       scene_to_dict)
from randomize import (RANDOM_PARTS, randomize_figure, random_scene)

# The window. Importing it must not need a display - the suites import this
# module headless - and it does not: `ui_app` falls back to tk = None.
from ui_app import (ACCENT, BG, CONTROL, EDGE, FG, GUIDE_PLANES, HELP, HOVER,
                    MUTED, PANEL, PICK_RADIUS, VIEW_BG, EditorApp, Viewport,
                    tk)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        # A prompt run needs no display at all, which is the point of it: the
        # same posing and the same framing, driven by a local model instead of
        # the mouse. pose_agent imports this module, so it is imported here.
        import pose_agent
        return pose_agent.main(argv)
    if tk is None:
        raise SystemExit(
            "tkinter is not available.\n"
            "Debian/Ubuntu: sudo apt install python3-tk\n"
            "macOS (Homebrew): brew install python-tk\n"
            "Windows: reinstall Python with the tcl/tk option enabled.")
    print("3D OpenPose editor %s" % VERSION)
    root = tk.Tk()
    root.geometry("1240x860")
    root.minsize(900, 560)
    EditorApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
