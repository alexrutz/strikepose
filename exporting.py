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
from raster import depth_buffer, depth_to_grey, render_depth
from skeleton import KEYPOINT_NAMES
from vecmath import vdot


def frame_rect(view_w, view_h, aspect):
    """Safe frame of the given aspect ratio, centred in a view that size."""
    fh = view_h * 0.92
    fw = fh * aspect
    if fw > view_w * 0.92:
        fw = view_w * 0.92
        fh = fw / aspect
    return ((view_w - fw) / 2.0, (view_h - fh) / 2.0,
            (view_w + fw) / 2.0, (view_h + fh) / 2.0)


def project_people(figures, camera, rect, out_w, out_h):
    """Every figure's keypoints in export pixels, with their visibility."""
    x0, y0, x1, y1 = rect
    sx, sy = out_w / (x1 - x0), out_h / (y1 - y0)
    return [([((px - x0) * sx, (py - y0) * sy)
              for px, py, _ in (camera.project(pt) for pt in figure.points)],
             list(figure.visible))
            for figure in figures]


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


def anatomy_depth_image(figures, camera, rect, out_w, out_h, thickness=1.0,
                        props=()):
    """Depth map from the built-in anatomy, framed to match `pose_image`.

    The always-available depth source: no model files, no torch, numpy and
    Pillow only. The rigged-mesh and SMPL-X sources live on the editor because
    they need files the user has to supply.

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
    return render_depth(groups, out_w, out_h, blend=2.0 * k)


def pose_body(figure, mesh):
    """(solution, keypoints) for one figure on its own rig.

    The rig is POSED, not fitted. The older path aimed each bone at a keypoint
    and then slid and scaled it until it landed there, so the body came out
    wearing the keypoint skeleton's proportions - stretched by up to a tenth
    per segment, and every disagreement between the two conventions had to be
    reconciled by hand somewhere in `mesh_backend`. A depth map does not need
    any of that: eighteen keypoints are a good witness to which way a limb
    points and a poor one to how long it is, so only the directions are taken
    and the body keeps every length it was measured with.

    The keypoints handed back are read off the posed rig rather than the ones
    that went in, so the OpenPose PNG describes the same body the depth map
    shows. It is the only order that cannot disagree with itself.
    """
    import mesh_backend
    rest = build_rest_points(figure.body)
    stature = figure.body.get("stature")
    solution = mesh_backend.pose_rig(mesh, {name: figure.points[i]
                                            for i, name
                                            in enumerate(KEYPOINT_NAMES)},
                                     mesh.get("roles"), rest_points=rest,
                                     stature=stature)
    riders = mesh_backend.keypoint_riders(mesh, rest, mesh.get("roles"),
                                          stature=stature)
    return solution, mesh_backend.keypoints_of(solution, riders, mesh)


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


def rigged_depth_image(jobs, camera, rect, out_w, out_h, props=()):
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
    from smplx_backend import rasterize_depth
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
    return depth_to_grey(zbuf)
