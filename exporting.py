#!/usr/bin/env python3
"""Framing, and the two conditioning images that have to line up.

The pose PNG and the depth map are the same frame or they are useless as a
pair, so the framing lives here rather than on the editor window: the window
passes its canvas size, a headless run passes the camera's, and both get the
identical rectangle.

A depth export is rigged geometry or it does not happen. `rigged_depth_image`
resolves its bodies through `bodies_lib` and raises `MissingBodies` rather
than quietly falling back to the swept anatomy - the silent fallback being the
whole problem, since the PNG still appears and nothing says it is a picture of
a mannequin. Asking for the sweep deliberately is `anatomy_depth_image`.
"""

from __future__ import annotations

import props as props_module

try:
    import numpy as np
except ImportError:
    np = None

from anatomy import body_parts
from anthro import build_rest_points
from posemap import render_openpose, resolution_stickwidth
from raster import (depth_buffer, depth_to_grey, grade_scene,
                    rasterize_depth, render_depth)
from skeleton import KEYPOINT_NAMES, Skeleton, clean_extremities
from vecmath import vadd, vcross, vdot, vlen, vmul, vnorm, vsub

INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


# The aspect ratios worth naming, longest-edge 768 so every one of them is a
# comparable amount of pixels rather than a comparable width. A conditioning
# image is used at whatever size the generator wants, so what matters here is
# the SHAPE; the sizes are a sensible default to go with each shape and
# anything can still be typed in.
#
# 2:3 is first because it is what SD 1.5 and SDXL portrait checkpoints are
# trained near, and a standing figure is a tall thin subject.
ASPECTS = (
    ("2:3 portrait", 512, 768),
    ("3:4 portrait", 576, 768),
    ("9:16 tall", 432, 768),
    ("1:1 square", 768, 768),
    ("4:3 landscape", 768, 576),
    ("3:2 landscape", 768, 512),
    ("16:9 wide", 768, 432),
)

ASPECT_NAMES = tuple(name for name, _w, _h in ASPECTS)


def size_for(name):
    """(width, height) for a named ratio, or None if it is not one of them."""
    for entry, w, h in ASPECTS:
        if entry == name:
            return w, h
    return None


def aspect_name(width, height, tolerance=0.01):
    """Which named ratio these pixels are, or None for anything else.

    By ratio and not by the exact numbers: 1024x1536 is 2:3 as much as 512x768
    is, and a control that only recognised its own defaults would call every
    scaled-up export "custom".
    """
    if width <= 0 or height <= 0:
        return None
    ratio = float(width) / height
    for name, w, h in ASPECTS:
        if abs(ratio - float(w) / h) <= tolerance * ratio:
            return name
    return None


def parse_size(text, long_edge=768):
    """"768x512", "16:9" or "2:3 portrait" -> (width, height).

    A ratio is as useful as a size on the command line and easier to get
    right - nobody remembers that 9:16 at this long edge is 432 - so both
    spellings are accepted wherever a size is.
    """
    text = str(text).strip().lower()
    named = [n for n in ASPECT_NAMES if text == n.lower()]
    if named:
        return size_for(named[0])
    for entry, w, h in ASPECTS:          # "16:9", the bare ratio of a name
        if text == entry.split()[0]:
            return w, h
    if "x" in text:
        wide, tall = text.split("x", 1)
        return max(16, int(wide)), max(16, int(tall))
    if ":" in text:
        wide, tall = (float(v) for v in text.split(":", 1))
        if wide <= 0 or tall <= 0:
            raise ValueError("a ratio needs two positive numbers: %r" % text)
        if wide >= tall:
            return long_edge, max(16, int(round(long_edge * tall / wide)))
        return max(16, int(round(long_edge * wide / tall))), long_edge
    raise ValueError("size should be WxH or a ratio like 16:9, not %r" % text)


def frame_rect(view_w, view_h, aspect):
    """Safe frame of the given aspect ratio, centred in a view that size."""
    fh = view_h * 0.92
    fw = fh * aspect
    if fw > view_w * 0.92:
        fw = view_w * 0.92
        fh = fw / aspect
    return ((view_w - fw) / 2.0, (view_h - fh) / 2.0,
            (view_w + fw) / 2.0, (view_h + fh) / 2.0)


# Which rig bone hides which keypoint, so "hide the left arm" still takes the
# arm out of the pose PNG. The five face keypoints ride the skull and the
# neck is the shoulder midpoint, so those follow the head and the collars.
KEYPOINT_BONES = {
    "nose": "head", "neck": "spine01", "l_eye": "head", "r_eye": "head",
    "l_ear": "head", "r_ear": "head",
    "l_shoulder": "clavicle.L", "r_shoulder": "clavicle.R",
    "l_elbow": "lowerarm01.L", "r_elbow": "lowerarm01.R",
    "l_wrist": "wrist.L", "r_wrist": "wrist.R",
    "l_hip": "upperleg01.L", "r_hip": "upperleg01.R",
    "l_knee": "lowerleg01.L", "r_knee": "lowerleg01.R",
    "l_ankle": "foot.L", "r_ankle": "foot.R",
}


def project_people(figures, camera, rect, out_w, out_h):
    """Every figure's EIGHTEEN keypoints in export pixels, with visibility.

    Read off the posed rig, not taken from the figure's joints: a figure holds
    104 of those and the OpenPose format has eighteen, in its own order, with
    its own idea of where a shoulder and a neck are. Handing it the armature
    instead runs off the end of the palette - which is what it did.
    """
    x0, y0, x1, y1 = rect
    sx, sy = out_w / (x1 - x0), out_h / (y1 - y0)
    out = []
    for figure in figures:
        drawn = figure.as_skeleton() if hasattr(figure, "as_skeleton") \
            else figure
        seen = ([figure.visible[figure.pose.bone(KEYPOINT_BONES[n])]
                 if KEYPOINT_BONES.get(n) in figure.pose.index else True
                 for n in KEYPOINT_NAMES]
                if hasattr(figure, "pose") else list(figure.visible))
        out.append(([((px - x0) * sx, (py - y0) * sy)
                     for px, py, _ in (camera.project(pt)
                                       for pt in drawn.points)], seen))
    return out


def pose_image(figures, camera, rect, out_w, out_h, thick_lines=True):
    """The OpenPose conditioning image for a scene."""
    stick = resolution_stickwidth(out_w, out_h) if thick_lines else 4
    return render_openpose(project_people(figures, camera, rect, out_w, out_h),
                           out_w, out_h, stickwidth=stick, dot_radius=stick)


def to_camera_space(part, camera, rect, out_w):
    """One part's primitives, mapped from world centimetres into the pixel
    space the rasteriser solves in. Depth is scaled by the same factor as x
    and y, so a z-buffer written here is comparable with any other."""
    x0, y0 = rect[0], rect[1]
    s = out_w / (rect[2] - rect[0])
    k = camera.zoom * s
    right, up, fwd = camera.basis()
    out = []
    for kind, centre, axes, radii in part:
        pc = camera.project(centre)
        out.append((kind, ((pc[0] - x0) * s, (pc[1] - y0) * s, pc[2] * k),
                    tuple((vdot(u, right), -vdot(u, up), vdot(u, fwd))
                          for u in axes),
                    tuple(r * k for r in radii)))
    return out


def prop_groups(props, camera, rect, out_w):
    """Objects as render groups: one each, so they meet with an edge."""
    return [[to_camera_space(props_module.parts(prop), camera, rect, out_w)]
            for prop in props]


# How far the ground reaches, in centimetres, and how thick the slab is. Big
# enough that its edges are never in shot at any framing the editor allows -
# an edge reads as a platform the figure is standing on, which is a different
# picture - and no bigger, because every centimetre of it is solved per pixel.
GROUND_EXTENT = 2400.0

# Thin on purpose. An orthographic camera at pitch zero sees a horizontal
# plane exactly edge-on, so all that is left of the floor there is its front
# face - and at twelve centimetres that is a bright bar across the picture
# that reads as a step the figure is standing on. At three it is a ground
# line, which is the most a level view can honestly say: where the floor is,
# not how it recedes. Pitch is what makes a floor carry depth, and the views
# that have some - high_three_quarter, bird, over_shoulder - are where this
# earns its keep.
GROUND_THICK = 3.0

# How the floor falls away: its brightness halves every GROUND_FADE
# centimetres behind the figure, from GROUND_BRIGHT of full white at the feet.
#
# These are what make it a pool of ground around the figure rather than a room
# floor crossing the whole frame. Seventy centimetres puts it at a third of
# its starting brightness a metre back and into the background within three,
# so the slab's far edge is never a visible line - a floor that stops at a
# fixed grey ends in a horizontal band across the picture and reads as a
# platform, which is the version this replaced.
#
# GROUND_BRIGHT keeps it under the figure. At 1.0 the floor at the feet is the
# brightest thing in the picture, which is true - it is nearest - and wrong
# for a conditioning image, where the subject should lead.
GROUND_FADE = 22.0
GROUND_BRIGHT = 0.60


def ground_level(figures):
    """The y the figures are standing on: the lowest point any of them has.

    A figure's floor is its own lowest foot rather than a plane in the scene -
    nothing moves the pelvis, so a squat is a figure with its feet closer to
    its hips and not a figure lower down. Measured on the silhouette, because
    the sole is 8 cm past the ankle keypoint and a floor drawn at the ankle
    cuts through both feet.
    """
    points = [p for figure in figures for p in silhouette_points(figure)]
    if not points:
        return None
    return min(p[1] for p in points)


def ground_part(figures, camera, rect, out_w, level=None):
    """The ground, as one camera-space part ready for the depth rasteriser.

    Not a prop. A prop is something somebody placed, and it takes part in the
    framing and in the buried-figure test; the ground is the room the figure
    is in, and it must do neither. Framing to hold a 24-metre slab would
    shrink the figure to nothing, and a floor covers the whole lower frame by
    design, so counting it as something in the way would reject every camera
    with any pitch at all - which is every camera the floor is any use to.
    """
    level = ground_level(figures) if level is None else level
    if level is None:
        return None
    points = [p for figure in figures for p in silhouette_points(figure)]
    _right, _up, fwd = camera.basis()

    # The floor starts at the figure and runs AWAY from the camera. It is not
    # a plane centred under the scene, and the difference is the whole thing
    # working or not.
    #
    # A real floor does of course carry on towards the lens, but in a
    # photograph that stretch is below the bottom of the frame - the frame is
    # fitted to the person, and the person's feet are its lower edge. This
    # camera is orthographic, so there is no "below the frame" to hide it in:
    # a slab centred on the scene puts its near edge twelve metres in front of
    # the figure, which becomes the nearest thing in the buffer and takes the
    # whole bright end of the range. The figure came out a black silhouette on
    # every level view, and from a low angle the slab covered it completely.
    back = (fwd[0], 0.0, fwd[2])
    if vlen(back) < 1e-6:        # straight down or straight up: any heading
        back = (0.0, 0.0, 1.0)   # will do, the floor fills the frame anyway
    back = vnorm(back)
    side = vnorm(vcross(back, (0.0, 1.0, 0.0)))
    near = min(vdot(p, back) for p in points)
    across = sum(vdot(p, side) for p in points) / len(points)

    half = GROUND_EXTENT / 2.0
    centre = vadd(vmul(side, across),
                  vadd(vmul(back, near + half),
                       (0.0, level - GROUND_THICK / 2.0, 0.0)))
    part = [("box", centre, (side, (0.0, 1.0, 0.0), back),
             (half, GROUND_THICK / 2.0, half))]
    return to_camera_space(part, camera, rect, out_w)


def ground_fade(camera, rect, out_w):
    """`GROUND_FADE` in the buffer's own units, which are pixels."""
    return GROUND_FADE * camera.zoom * out_w / (rect[2] - rect[0])


def anatomy_depth_image(figures, camera, rect, out_w, out_h, thickness=1.0,
                        props=(), ground=True):
    """Depth map from the built-in anatomy, framed to match `pose_image`.

    The approximation, and never an export: it draws the viewport and gives
    `wearables` a profile to cut a garment out of. A depth map that reaches a
    conditioning set comes from the rigged bodies, through `rigged_depth_image`.

    Objects go in as one group each. Groups meet with a hard minimum, so a
    chair occludes the figure on it with an edge instead of melting into it,
    while the parts inside one figure still blend.
    """
    k = camera.zoom * out_w / (rect[2] - rect[0])   # world -> pixels, z too
    groups = [[to_camera_space(part, camera, rect, out_w)
               for part in body_parts(figure, thickness)]
              for figure in figures]
    groups += prop_groups(props, camera, rect, out_w)
    if not groups:
        groups = [[]]
    subject = depth_buffer(groups, out_w, out_h, blend=2.0 * k)
    # The ground is buffered on its own, never blended into the figure: it is
    # graded by a curve of its own, and a floor smooth-minimumed into the feet
    # would fillet a person into the ground they stand on.
    floor = ground_part(figures, camera, rect, out_w) if ground else None
    return grade_scene(subject,
                       depth_buffer([[floor]], out_w, out_h)
                       if floor is not None else None,
                       fade=ground_fade(camera, rect, out_w),
                       brightest=GROUND_BRIGHT)


def pose_body(figure, mesh):
    """(solution, keypoints) for one figure - both read off its own rig.

    There is nothing to solve any more. The figure IS the armature: what the
    editor dragged is a local rotation per bone, which is exactly what the
    skinning wants, so this hands it over and reads the eighteen keypoints
    back off the result for whoever still wants a pose PNG.

    What stood here was a solver that aimed each of the rig's bones at one of
    eighteen keypoints. It worked, and it was still a translation between two
    descriptions of one pose - so it could only ever lose what the poorer of
    the two could not say, which was the whole hand, the collarbones, the roll
    of every limb and 85 of the rig's 104 bones.
    """
    return figure.solution(), figure.keypoints()


def rigged_keypoints(jobs):
    """Every figure's keypoints as its own rig lays them out.

    A figure with no body behind it keeps the ones it was drawn with; there is
    nothing better to say about it.
    """
    out = []
    for figure, mesh, _assets in jobs:
        if mesh is None:
            out.append(list(figure.points))
            continue
        try:
            _solution, points = pose_body(figure, mesh)
        except Exception:                   # a rig that will not map
            out.append(list(figure.points))
            continue
        out.append([list(points.get(name, figure.points[i]))
                    for i, name in enumerate(KEYPOINT_NAMES)])
    return out


def rigged_depth_image(jobs, camera, rect, out_w, out_h, props=(),
                       ground=True):
    """Depth map from posed rigged meshes, framed to match `pose_image`.

    `jobs` is a list of (figure, mesh, assets): the editor supplies what it has
    loaded, a headless check supplies what it was handed on the command line.
    Framing and the camera mapping are shared with the anatomy path, so the
    three depth sources line up with the pose PNG and with each other.

    Objects come along, rasterised the analytic way while the mesh goes through
    the triangle z-buffer, into the same buffer. So does anything the figure is
    wearing: `wearables` carries no meshes, it clips and pads the body's own
    swept profile, and that profile is built from the same eighteen keypoints
    the rig is - so a coat cut for the sweep lands on the rigged body too.
    Without this the clothes were simply absent from a rigged export, which is
    the sort of thing nobody notices until they look for a coat.
    """
    import mesh_backend
    x0, y0, _x1, _y1 = rect
    s = out_w / (rect[2] - rect[0])
    k = camera.zoom * s
    right, up, fwd = camera.basis()
    zbuf = np.full((out_h, out_w), np.inf)
    for figure, mesh, assets in jobs:
        if mesh is None:
            continue
        # `pose_body` reads this body's own rest keypoints for itself, so the
        # rig's head is turned by how far the head has *moved* rather than
        # aimed at an absolute direction - there is no keypoint on a skull.
        solution = pose_body(figure, mesh)[0]
        # Real garments, where the library has one. `wearables` builds a
        # garment out of the body's own swept profile, which is right for the
        # viewport and an approximation everywhere else; these are the CC0
        # MakeHuman meshes, fitted to this very body by
        # tools/make_wearables.py and skinned to its armature, so they ride
        # the solution above rather than being solved again.
        import garments_lib
        preset = (figure.body or {}).get("preset")
        worn, mesh = (garments_lib.dress(figure, preset, mesh) if preset
                      else ([], mesh))
        pieces = [(mesh_backend.skin_with(mesh, solution), mesh["faces"])]
        for cloth in worn:
            pieces.append((mesh_backend.skin_with(cloth, solution),
                           cloth["faces"]))
        for asset in assets:
            # assets ride the body's own solution, so they cannot drift
            pieces.append((mesh_backend.skin_with(asset, solution),
                           asset["faces"]))
        for verts, faces in pieces:
            rel = verts - np.asarray(camera.target, dtype=float)
            px = np.column_stack([
                (rel @ np.asarray(right) * camera.zoom
                 + camera.width / 2.0 - x0) * s,
                (-(rel @ np.asarray(up)) * camera.zoom
                 + camera.height / 2.0 - y0) * s,
                rel @ np.asarray(fwd) * k])
            np.minimum(zbuf, rasterize_depth(px, faces, out_w, out_h), out=zbuf)
    # Clothes and hair are real meshes on this path, added above with the
    # body. `wearables`' swept garments do not come along: they are the body's
    # own profile clipped and padded, which is the right thing for a viewport
    # at sixty frames a second and an approximation next to a fitted mesh -
    # and mixing the two is worse than either, because the eye reads the join.
    # A slot the garment library has nothing for is simply not worn here;
    # `garments_lib.describe()` says which those are.
    if len(props):
        # the triangle rasteriser and the analytic one write the same units, so
        # the objects simply join the buffer and the nearer surface wins
        np.minimum(zbuf, depth_buffer(prop_groups(props, camera, rect, out_w),
                                      out_w, out_h), out=zbuf)
    floor = ground_part([f for f, _m, _a in jobs], camera, rect, out_w) \
        if ground else None
    return grade_scene(zbuf,
                       depth_buffer([[floor]], out_w, out_h)
                       if floor is not None else None,
                       fade=ground_fade(camera, rect, out_w),
                       brightest=GROUND_BRIGHT)


def body_frame(figure):
    """(side, up, facing) of the figure as it stands now, side to its left.

    Read off the rig's own hips and spine rather than off a keypoint torso:
    the pelvis is a bone with an orientation, which is a better witness to
    which way someone is facing than three points that happen to lie near it.
    """
    return figure.body_frame()


def silhouette_points(figure):
    """The figure's own surface - which is what has to fit in the frame.

    This used to be the eighteen keypoints plus a hand-written allowance for
    every part that reaches past the last keypoint on its chain: 14 cm above
    the nose for the crown, 19 past the wrist for a hand, 21 in front of the
    ankle for a toe. Every one of those was a guess at a body it could not
    see, and they were all wrong by a centimetre or two on any preset but the
    average adult male.

    The figure carries a rigged mesh now, so the silhouette is the mesh. It is
    exact for every preset, and a body that is wearing something is measured
    with it on rather than being allowed for.
    """
    return [tuple(v) for v in figure.surface()]


def frame_scene(figures, camera, rect, margin=1.06, props=(), grow=1.9):
    """Point the camera at the scene and zoom so the whole of it fits `rect`.

    A posed figure is not the same size on screen as the rest pose - arms up
    adds a head's height, lying down turns it on its side - so a fixed zoom
    crops exactly the poses a prompt is most likely to ask for. The fit is to
    the export rectangle rather than the whole view, because that is the part
    that becomes the PNG.

    A skeleton has no thickness and a body does, so the fit is padded by the
    widest cross-section the figure carries. A 6% margin is ample on a
    standing figure, where the 175 cm of height dwarfs it, and not nearly
    enough on a deep crouch: folded up, the figure is 90 cm across and a 19 cm
    chest half-width is a fifth of that, which is a head and two hands over
    the edge of the frame.
    """
    points = [p for figure in figures for p in silhouette_points(figure)]
    if not points:
        return
    girth = max([0.0] + [max(figure.body[part][i] for part in
                             ("chest", "waist", "pelvis") for i in (0, 1))
                         for figure in figures])
    right, up, _fwd = camera.basis()
    xs = [vdot(p, right) for p in points]
    ys = [vdot(p, up) for p in points]
    # The people set the scale; objects may widen the frame but only so far.
    # A 6 m floor or a 4 m wall is a backdrop, and framing to hold all of one
    # shrinks the figure the whole image is about to a few dozen pixels.
    xs = [x - girth for x in xs] + [x + girth for x in xs]
    ys = [y - girth for y in ys] + [y + girth for y in ys]
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    room_x = (hi_x - lo_x) * (grow - 1.0) / 2.0
    room_y = (hi_y - lo_y) * (grow - 1.0) / 2.0
    for prop in props:
        lo, hi = props_module.bounds(prop)
        # the eight corners, not the two: a wall is in frame only if its far
        # top corner is, and that is neither of them
        for corner in ((x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                       for z in (lo[2], hi[2])):
            cx, cy = vdot(corner, right), vdot(corner, up)
            xs.append(min(max(cx, lo_x - room_x), hi_x + room_x))
            ys.append(min(max(cy, lo_y - room_y), hi_y + room_y))
    mid_x, mid_y = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    # put the bounding box centre on the camera target, in the camera's own
    # plane: the mean of the keypoints would pull the frame towards the head,
    # which carries five of the eighteen
    zero = vmul(right, 0.0)
    camera.target = vadd(zero, vadd(vmul(right, mid_x), vmul(up, mid_y)))
    half_w = (max(xs) - min(xs)) / 2.0
    half_h = (max(ys) - min(ys)) / 2.0
    rect_w, rect_h = rect[2] - rect[0], rect[3] - rect[1]
    camera.zoom = max(0.3, min(40.0,
                               min(rect_w / (2.0 * max(1.0, half_w) * margin),
                                   rect_h / (2.0 * max(1.0, half_h) * margin))))
