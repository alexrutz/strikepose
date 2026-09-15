#!/usr/bin/env python3
"""The OpenPose PNG: the canonical palette and nothing else.

Viewport tinting for telling figures apart must never reach `render_openpose`,
and the suites assert that no tinted colour appears in an exported PNG.
"""

from __future__ import annotations

import math

from skeleton import COLORS, LIMB_SEQ

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None


def ellipse_polygon(p1, p2, half_thickness, samples=48):
    """Ellipse spanning p1..p2, matching cv2.ellipse2Poly as used by OpenPose."""
    cx, cy = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    a, b = length / 2.0, half_thickness
    pts = []
    for i in range(samples):
        t = 2.0 * math.pi * i / samples
        ca, sa = math.cos(t) * a, math.sin(t) * b
        pts.append((cx + ux * ca + nx * sa, cy + uy * ca + ny * sa))
    return pts


def render_openpose(people, width, height, stickwidth=4,
                    dot_radius=4, background=(0, 0, 0)):
    """Reproduce OpenPose's draw_bodypose: full-brightness dots, then limbs
    composited at alpha 0.6 (cv2.addWeighted(canvas, 0.4, layer, 0.6, 0)).

    people: list of (points2d, visible). Every person is drawn onto the one
    canvas with the same colours, exactly as the annotator does.
    """
    if Image is None:
        raise RuntimeError("Pillow is required for PNG export: pip install pillow")
    if people and people[0] and not isinstance(people[0][0], (list, tuple)):
        people = [(people, [True] * len(people))]   # a bare list of points
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas)
    for points2d, visible in people:
        for i, (x, y) in enumerate(points2d):
            if not visible[i]:
                continue
            draw.ellipse([x - dot_radius, y - dot_radius,
                          x + dot_radius, y + dot_radius], fill=COLORS[i])
    for points2d, visible in people:
        for i, (a, b) in enumerate(LIMB_SEQ):
            if not (visible[a] and visible[b]):
                continue
            layer = canvas.copy()
            ImageDraw.Draw(layer).polygon(
                ellipse_polygon(points2d[a], points2d[b], stickwidth),
                fill=COLORS[i])
            canvas = Image.blend(canvas, layer, 0.6)
    return canvas


def resolution_stickwidth(width, height, base=4):
    """Optional xinsir-style thickness scaling for large canvases."""
    m = max(width, height)
    for limit, ratio in ((500, 1.0), (1000, 2.0), (2000, 3.0), (3000, 4.0),
                         (4000, 5.0), (5000, 6.0)):
        if m < limit:
            break
    else:
        ratio = 7.0
    return max(1, int(round(base * ratio)))
