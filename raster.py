#!/usr/bin/env python3
"""Turning oriented primitives into pixels, and into viewport polygons.

Two paths over the same solids. `silhouette_quads` and `solid_quads` are the
cheap screen-space outline the canvas draws while a joint is being dragged;
`depth_buffer` and `render_depth` are the analytic rasteriser the export runs,
which solves every pixel as a ray along +Z against each primitive and keeps
the nearest.

`_screen_extent` must be exact per kind, because it sets the pixel window a
primitive is solved in and anything outside that window is simply not drawn.
The ellipsoid formula under-measures a cylinder by its end caps and a box by
most of a corner: invisible on a limb station a quarter of a centimetre thick,
a shaved edge on a table.
"""

from __future__ import annotations

import math

import props as props_module

from vecmath import vadd, vdot, vlen, vmul, vnorm, vsub

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None


def silhouette_quads(parts, camera):
    """Screen-space outline of the body, as depth-sorted polygons.

    Cheap enough to redraw while dragging: each swept station contributes two
    silhouette points (the extremes of its projected cross-section ellipse
    measured across the tube), and consecutive stations form a quad. No
    rasterising, no z-buffer, just a painter's-order list.
    """
    right, up, fwd = camera.basis()
    zoom = camera.zoom
    quads = []

    def ellipse_poly(x, y, a, b, n=14):
        pts = []
        for i in range(n):
            t = 2.0 * math.pi * i / n
            ct, st = math.cos(t), math.sin(t)
            pts.append((x + a[0] * ct + b[0] * st, y + a[1] * ct + b[1] * st))
        return pts

    for part in parts:
        stations = []
        for kind, centre, axes, radii in part:
            sx, sy, depth = camera.project(centre)
            if kind == "slab":
                s_ax, f_ax, _ = axes
                a = (vdot(s_ax, right) * radii[0] * zoom,
                     -vdot(s_ax, up) * radii[0] * zoom)
                b = (vdot(f_ax, right) * radii[1] * zoom,
                     -vdot(f_ax, up) * radii[1] * zoom)
                stations.append((sx, sy, depth, a, b))
            else:
                ex = math.sqrt(sum((r * vdot(u, right)) ** 2
                                   for u, r in zip(axes, radii))) * zoom
                ey = math.sqrt(sum((r * vdot(u, up)) ** 2
                                   for u, r in zip(axes, radii))) * zoom
                r = math.sqrt(max(1e-6, ex * ey))
                quads.append((ellipse_poly(sx, sy, (r, 0.0), (0.0, r)), depth))
        for i in range(len(stations) - 1):
            x0, y0, z0, a0, b0 = stations[i]
            x1, y1, z1, a1, b1 = stations[i + 1]
            tx, ty = x1 - x0, y1 - y0
            span = math.hypot(tx, ty)
            if span < 0.5:
                # tube pointing at the camera: no meaningful across direction,
                # so draw the cross-section itself
                quads.append((ellipse_poly(x0, y0, a0, b0), z0))
                continue
            nx, ny = -ty / span, tx / span
            e0 = math.hypot(a0[0] * nx + a0[1] * ny, b0[0] * nx + b0[1] * ny)
            e1 = math.hypot(a1[0] * nx + a1[1] * ny, b1[0] * nx + b1[1] * ny)
            quads.append(([(x0 + nx * e0, y0 + ny * e0),
                           (x0 - nx * e0, y0 - ny * e0),
                           (x1 - nx * e1, y1 - ny * e1),
                           (x1 + nx * e1, y1 + ny * e1)],
                          (z0 + z1) * 0.5))
    quads.sort(key=lambda q: -q[1])
    return quads


def _hull(points):
    """2D convex hull, monotone chain. Small inputs, so the sort dominates."""
    points = sorted(set(points))
    if len(points) < 3:
        return list(points)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) > 0:
                    break
                out.pop()
            out.append(p)
        return out[:-1]

    return half(points) + half(reversed(points))


def inside_polygon(poly, x, y):
    """Crossing-number point-in-polygon, for picking an object in the viewport."""
    inside = False
    for i in range(len(poly)):
        (x0, y0), (x1, y1) = poly[i - 1], poly[i]
        if (y0 > y) != (y1 > y):
            cut = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < cut:
                inside = not inside
    return inside


def solid_quads(props, camera, selected=None):
    """Screen outlines of the scene's objects, depth sorted, for the viewport.

    One polygon per primitive rather than the swept quads the body uses: an
    object is a handful of standalone solids, not a chain of stations. A box
    is the hull of its eight projected corners, a cylinder the hull of its two
    end rings, an ellipsoid its projected ellipse - which is exact under an
    orthographic camera. Returns (polygon, depth, prop index, selected).
    """
    out = []
    for index, prop in enumerate(props):
        for kind, centre, axes, radii in props_module.parts(prop):
            sx, sy, depth = camera.project(centre)
            if kind == "ball":
                right, up, _fwd = camera.basis()
                ex = math.sqrt(sum((r * vdot(u, right)) ** 2
                                   for u, r in zip(axes, radii))) * camera.zoom
                ey = math.sqrt(sum((r * vdot(u, up)) ** 2
                                   for u, r in zip(axes, radii))) * camera.zoom
                poly = [(sx + ex * math.cos(t), sy - ey * math.sin(t))
                        for t in (i * math.pi / 8.0 for i in range(16))]
                out.append((poly, depth, index, index == selected))
                continue
            corners = []
            if kind == "box":
                for a in (-1.0, 1.0):
                    for b in (-1.0, 1.0):
                        for c in (-1.0, 1.0):
                            corners.append(vadd(centre, vadd(
                                vmul(axes[0], a * radii[0]),
                                vadd(vmul(axes[1], b * radii[1]),
                                     vmul(axes[2], c * radii[2])))))
            else:                          # elliptical cylinder about axes[2]
                for end in (-1.0, 1.0):
                    base = vadd(centre, vmul(axes[2], end * radii[2]))
                    for i in range(12):
                        t = 2.0 * math.pi * i / 12.0
                        corners.append(vadd(base, vadd(
                            vmul(axes[0], radii[0] * math.cos(t)),
                            vmul(axes[1], radii[1] * math.sin(t)))))
            flat = [camera.project(c)[:2] for c in corners]
            poly = _hull([(round(x, 3), round(y, 3)) for x, y in flat])
            if len(poly) >= 3:
                out.append((poly, depth, index, index == selected))
    out.sort(key=lambda q: -q[1])
    return out


def _ellipsoid_z(X, Y, centre, axes, radii):
    """Nearest surface z for rays along +Z through (x, y, 0). Exact: the ray
    becomes a quadratic once the ellipsoid is mapped to a unit sphere."""
    qa = 0.0
    qb = 0.0
    qc = -1.0
    for u, r in zip(axes, radii):
        A = u[2] / r
        Bc = ((X - centre[0]) * u[0] + (Y - centre[1]) * u[1]
              - centre[2] * u[2]) / r
        qa = qa + A * A
        qb = qb + 2.0 * A * Bc
        qc = qc + Bc * Bc
    disc = qb * qb - 4.0 * qa * qc
    hit = disc > 0.0
    return np.where(hit, (-qb - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * qa),
                    np.inf)


def _slab_z(X, Y, centre, axes, radii):
    """Nearest surface z for a finite elliptical cylinder. The ray is clipped
    against the elliptical side wall and the two end planes; the first point
    inside both is the hit."""
    (s_ax, f_ax, ax), (w, d, hz) = axes, radii
    ox, oy, oz = X - centre[0], Y - centre[1], -centre[2]
    a1, a2 = s_ax[2] / w, f_ax[2] / d
    b1 = (ox * s_ax[0] + oy * s_ax[1] + oz * s_ax[2]) / w
    b2 = (ox * f_ax[0] + oy * f_ax[1] + oz * f_ax[2]) / d
    qa = a1 * a1 + a2 * a2
    qb = 2.0 * (a1 * b1 + a2 * b2)
    qc = b1 * b1 + b2 * b2 - 1.0
    big = 1e18
    if qa > 1e-12:
        disc = qb * qb - 4.0 * qa * qc
        ok = disc > 0.0
        root = np.sqrt(np.maximum(disc, 0.0))
        lo = np.where(ok, (-qb - root) / (2.0 * qa), np.inf)
        hi = np.where(ok, (-qb + root) / (2.0 * qa), -np.inf)
    else:                                   # ray parallel to the axis
        inside = qc <= 0.0
        lo = np.where(inside, -big, np.inf)
        hi = np.where(inside, big, -np.inf)
    aa = ax[2]
    ba = ox * ax[0] + oy * ax[1] + oz * ax[2]
    if abs(aa) > 1e-12:
        u0, u1 = (-hz - ba) / aa, (hz - ba) / aa
        lo = np.maximum(lo, np.minimum(u0, u1))
        hi = np.minimum(hi, np.maximum(u0, u1))
    else:
        inside = np.abs(ba) <= hz
        lo = np.where(inside, lo, np.inf)
        hi = np.where(inside, hi, -np.inf)
    return np.where(lo <= hi, lo, np.inf)


def _box_z(X, Y, centre, axes, radii):
    """Nearest surface z for an oriented rectangular block.

    The same slab clip `_slab_z` does against its end planes, three times over
    - once per axis - which is all a box is. Square corners are why objects
    need it: a crate or a table top swept as an elliptical cylinder has
    rounded sides, and a depth map of a room full of those reads as a room
    full of cushions.
    """
    ox, oy, oz = X - centre[0], Y - centre[1], -centre[2]
    lo = np.full(np.shape(ox), -1e18, dtype=float)
    hi = np.full(np.shape(ox), 1e18, dtype=float)
    for axis, r in zip(axes, radii):
        along = axis[2]
        offset = ox * axis[0] + oy * axis[1] + oz * axis[2]
        if abs(along) > 1e-12:
            a, b = (-r - offset) / along, (r - offset) / along
            lo = np.maximum(lo, np.minimum(a, b))
            hi = np.minimum(hi, np.maximum(a, b))
        else:                              # ray parallel to this pair of faces
            inside = np.abs(offset) <= r
            lo = np.where(inside, lo, np.inf)
            hi = np.where(inside, hi, -np.inf)
    return np.where(lo <= hi, lo, np.inf)


def _primitive_z(kind, X, Y, centre, axes, radii):
    if kind == "slab":
        return _slab_z(X, Y, centre, axes, radii)
    if kind == "box":
        return _box_z(X, Y, centre, axes, radii)
    return _ellipsoid_z(X, Y, centre, axes, radii)


def _screen_extent(kind, axes, radii):
    """How far a primitive reaches from its centre on screen, exactly.

    Exactly, per kind, because this sets the pixel window the primitive is
    solved in and anything outside it is simply not drawn. The ellipsoid
    formula used for all three under-measures a cylinder by its end caps and a
    box by most of a corner, which trimmed the ends off anything but a thin
    slab - invisible on a limb station a quarter of a centimetre thick, a
    shaved edge on a table.
    """
    def reach(i):
        if kind == "box":
            return sum(abs(r * u[i]) for u, r in zip(axes, radii))
        if kind == "slab":                 # elliptical cylinder about axes[2]
            return (abs(radii[2] * axes[2][i])
                    + math.sqrt((radii[0] * axes[0][i]) ** 2
                                + (radii[1] * axes[1][i]) ** 2))
        return math.sqrt(sum((r * u[i]) ** 2 for u, r in zip(axes, radii)))
    return reach(0), reach(1)


def _part_bounds(part, width, height):
    x0 = y0 = 1e18
    x1 = y1 = -1e18
    for kind, centre, axes, radii in part:
        ex, ey = _screen_extent(kind, axes, radii)
        x0 = min(x0, centre[0] - ex)
        x1 = max(x1, centre[0] + ex)
        y0 = min(y0, centre[1] - ey)
        y1 = max(y1, centre[1] + ey)
    x0 = max(0, int(math.floor(x0)))
    y0 = max(0, int(math.floor(y0)))
    x1 = min(width, int(math.ceil(x1)) + 1)
    y1 = min(height, int(math.ceil(y1)) + 1)
    return x0, y0, x1, y1


def _smooth_min(dst, src, k):
    """Quadratic smooth minimum. Only surfaces within k of each other blend, so
    an arm held clear of the chest still occludes it cleanly, while an arm
    resting against it gets a fillet instead of a seam."""
    out = np.minimum(dst, src)
    if k <= 0.0:
        return out
    both = np.isfinite(dst) & np.isfinite(src)
    if not both.any():
        return out
    a, b = src[both], dst[both]
    hh = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    out[both] = b * (1.0 - hh) + a * hh - k * hh * (1.0 - hh)
    return out


def depth_buffer(groups, width, height, blend=0.0):
    """Rasterise one or more figures to a shared z-buffer, in pixel depth.

    groups: a list of figures, each a list of parts. Parts inside a figure
    blend into each other with a smooth minimum; separate figures use a hard
    minimum, so two people standing close occlude cleanly instead of fusing
    into one mass.

    Returns the raw buffer rather than an image so a caller mixing sources -
    a rigged mesh through the triangle rasteriser plus objects through this one
    - can take the nearer of the two before anything is normalised. Both are in
    the same units, world centimetres times the pixels-per-centimetre the
    camera mapping uses, so a plain minimum is the right composite.
    """
    if np is None:
        raise RuntimeError("Depth export needs NumPy: pip install numpy")
    if Image is None:
        raise RuntimeError("Depth export needs Pillow: pip install pillow")

    if groups and groups[0] and isinstance(groups[0][0], tuple):
        groups = [groups]                      # a bare parts list is one figure

    zbuf = np.full((height, width), np.inf, dtype=np.float64)
    for parts in groups:
      figure = np.full((height, width), np.inf, dtype=np.float64)
      for part in parts:
        if not part:
            continue
        x0, y0, x1, y1 = _part_bounds(part, width, height)
        if x0 >= x1 or y0 >= y1:
            continue
        local = np.full((y1 - y0, x1 - x0), np.inf)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        xs = xs + 0.5
        ys = ys + 0.5
        for kind, centre, axes, radii in part:
            ex, ey = _screen_extent(kind, axes, radii)
            a0 = max(x0, int(math.floor(centre[0] - ex))) - x0
            a1 = min(x1, int(math.ceil(centre[0] + ex)) + 1) - x0
            b0 = max(y0, int(math.floor(centre[1] - ey))) - y0
            b1 = min(y1, int(math.ceil(centre[1] + ey)) + 1) - y0
            if a0 >= a1 or b0 >= b1:
                continue
            sub = (slice(b0, b1), slice(a0, a1))
            z = _primitive_z(kind, xs[sub], ys[sub], centre, axes, radii)
            np.minimum(local[sub], z, out=local[sub])   # hard union within part
        window = figure[y0:y1, x0:x1]
        figure[y0:y1, x0:x1] = _smooth_min(window, local, blend)
      np.minimum(zbuf, figure, out=zbuf)

    return zbuf


def render_depth(groups, width, height, blend=0.0, near=255, far=45,
                 background=0):
    """The same, as a ControlNet depth image. See `depth_buffer`."""
    return depth_to_grey(depth_buffer(groups, width, height, blend),
                         near, far, background)


def depth_to_grey(zbuf, near=255, far=45, background=0):
    """Normalise a z-buffer to the ControlNet convention: nearest brightest,
    background black. Shared by every depth source, so a scene rendered from
    the anatomy, a rigged mesh or a mix of the two is graded the same way."""
    covered = np.isfinite(zbuf)
    height, width = zbuf.shape
    img = np.full((height, width), float(background))
    if covered.any():
        z = zbuf[covered]
        lo, hi = z.min(), z.max()
        span = hi - lo
        if span < 1e-6:                 # flat surface: it is all "nearest"
            img[covered] = near
        else:
            img[covered] = far + (near - far) * (hi - zbuf[covered]) / span
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"), mode="L")


def grade_scene(subject, ground=None, near=255, far=45, background=0,
                fade=None, brightest=0.62):
    """The scene graded to grey, with the ground on a falloff of its own.

    Two curves, because the ground is not the subject. The subject - the
    figures and anything placed with them - is normalised over its OWN depth
    range exactly as `depth_to_grey` does, so a figure is graded identically
    whether or not there is a floor under it. The ground then falls away from
    `brightest` with an inverse-depth curve, halving every `fade` units, and
    reaches the background before it runs out.

    One curve over both does not work, and the reason is worth keeping. A
    linear grade spends its range on the room: with a floor three metres back
    the body gets 61 grey levels of 210. Grading everything in inverse depth
    instead - MiDaS's own convention - buys some of that back, but the floor
    then crosses the whole frame as a slow gradient and stops at `far`, so it
    ends in a hard grey line against the black background and reads as a
    platform the figure is standing on. A floor should fade out, and fading
    out means reaching the background, which is below `far` by definition.

    So the ground gets to go all the way to black, which is also the truthful
    answer: disparity really does go to zero at the horizon.
    """
    height, width = subject.shape
    img = np.full((height, width), float(background))
    lit = np.isfinite(subject)
    if lit.any():
        z = subject[lit]
        lo, hi = z.min(), z.max()
        span = hi - lo
        if span < 1e-6:
            img[lit] = near
        else:
            img[lit] = far + (near - far) * (hi - subject[lit]) / span
    if ground is not None and fade:
        here = np.isfinite(ground)
        # whichever surface is nearer wins the pixel, as everywhere else
        show = here & (~lit | (ground < subject))
        if show.any():
            z0 = ground[show].min()
            img[show] = (near * brightest * fade
                         / (fade + (ground[show] - z0)))
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"), mode="L")
