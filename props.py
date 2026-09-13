#!/usr/bin/env python3
"""Objects in the scene: chairs, crates, walls, the floor.

    python3 props.py --list        # every shape and its default size
    python3 props.py --selftest    # no display, no dependencies beyond numpy

Why objects at all
------------------
A depth map conditions everything in frame, not just the person. A figure
sitting with nothing under it reads as a figure floating, and the generator
obliges. Giving the scene a chair, a wall to stand against or a crate to lean
on is what makes the depth map say what the pose already meant.

They are built out of the same primitives the body is - an oriented box,
ellipsoid or elliptical cylinder - so they go through the existing analytic
rasteriser with a z-buffer and occlude the figure, and each other, correctly.
Nothing new is rendered; there is just more of it.

The frame a shape is authored in
--------------------------------
Origin at the *base centre*, +Y up, +Z the front of the object, +X its left,
sized in centimetres. Base-centred rather than centred so standing something
on the floor is a translation with nothing to work out, which is most of what
placing an object is. `place` in `pose_agent` turns "a chair under the hips"
into the translation; what gets stored in a scene is the result, in world
centimetres, so an object can be dragged afterwards without the anchor it came
from meaning anything any more.
"""

from __future__ import annotations

import math

AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# primitives, in the shape's own frame
# ---------------------------------------------------------------------------

def box(x, y, z, w, h, d):
    """A rectangular block, given its centre and its full size."""
    return ("box", (x, y, z), AXES, (w / 2.0, h / 2.0, d / 2.0))


def ball(x, y, z, w, h, d):
    return ("ball", (x, y, z), AXES, (w / 2.0, h / 2.0, d / 2.0))


def column(x, y, z, w, h, d):
    """An elliptical cylinder standing on end.

    The rasteriser's "slab" is a cylinder about its third axis, so the axes are
    handed over rotated: across, front, up.
    """
    return ("slab", (x, y, z), (AXES[0], AXES[2], AXES[1]),
            (w / 2.0, d / 2.0, h / 2.0))


def taper(x, y, z, w, h, d, top=0.0, steps=14):
    """A cone or truncated cone, swept as a stack of cylinders.

    `top` is the fraction of the base width left at the apex. Stacked rather
    than given a primitive of its own: a cone is the only shape here that
    needs one, the stack is exact to within its step height, and a new
    primitive is a new case in four places in the rasteriser.
    """
    out = []
    step = h / steps
    thick = step * 1.15                  # neighbours overlap, so no scalloping
    for i in range(steps):
        # the end discs are pulled inside the span rather than allowed to
        # stick out of it: every shape here has to stand exactly on y = 0 or
        # placing it on a floor leaves it hovering
        level = min(max(step * (i + 0.5), thick / 2.0), h - thick / 2.0)
        scale = 1.0 - (1.0 - top) * (level / h if h else 0.0)
        out.append(column(x, y + level, z, w * scale, thick, d * scale))
    return out


def capsule(x, y, z, w, h, d):
    """A cylinder with domed ends, upright."""
    barrel = max(0.0, h - min(w, d))
    return [column(x, y + h / 2.0, z, w, barrel, d),
            ball(x, y + min(w, d) / 2.0, z, w, min(w, d), d),
            ball(x, y + h - min(w, d) / 2.0, z, w, min(w, d), d)]


# ---------------------------------------------------------------------------
# the shapes
#
# Each builder takes the full (width, height, depth) it is to fill and returns
# primitives inside it. Default sizes are the real thing in centimetres, so
# "chair" next to a 175 cm figure comes out the height of a chair without
# anyone having to say so.
# ---------------------------------------------------------------------------

def _plain(maker):
    return lambda w, h, d: [maker(0.0, h / 2.0, 0.0, w, h, d)]


def _chair(w, h, d):
    seat = h * 0.5                       # seat height; the rest is backrest
    leg, thick = w * 0.09, h * 0.07
    out = [box(0.0, seat - thick / 2.0, 0.0, w, thick, d)]
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            out.append(box(sx * (w - leg) / 2.0, (seat - thick) / 2.0,
                           sz * (d - leg) / 2.0, leg, seat - thick, leg))
    out.append(box(0.0, (seat + h) / 2.0, -(d - leg) / 2.0,
                   w, h - seat, leg))     # back, at the rear of the seat
    return out


def _stool(w, h, d):
    thick = h * 0.12
    out = [column(0.0, h - thick / 2.0, 0.0, w, thick, d)]
    for i in range(3):
        angle = 2.0 * math.pi * i / 3.0 + math.pi / 6.0
        out.append(column(math.cos(angle) * w * 0.36, (h - thick) / 2.0,
                          math.sin(angle) * d * 0.36,
                          w * 0.12, h - thick, d * 0.12))
    return out


def _table(w, h, d):
    thick = h * 0.08
    leg = min(w, d) * 0.09
    out = [box(0.0, h - thick / 2.0, 0.0, w, thick, d)]
    for sx in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            out.append(box(sx * (w - leg) / 2.0 * 0.92, (h - thick) / 2.0,
                           sz * (d - leg) / 2.0 * 0.92,
                           leg, h - thick, leg))
    return out


def _desk(w, h, d):
    thick = h * 0.08
    out = [box(0.0, h - thick / 2.0, 0.0, w, thick, d)]
    for sx in (-1.0, 1.0):               # solid side panels, not legs
        out.append(box(sx * (w - d * 0.08) / 2.0, (h - thick) / 2.0, 0.0,
                       d * 0.08, h - thick, d * 0.9))
    out.append(box(0.0, h * 0.55, -d * 0.45, w * 0.96, h * 0.5, d * 0.06))
    return out


def _bench(w, h, d):
    thick = h * 0.16
    out = [box(0.0, h - thick / 2.0, 0.0, w, thick, d)]
    for sx in (-1.0, 1.0):
        out.append(box(sx * w * 0.38, (h - thick) / 2.0, 0.0,
                       w * 0.06, h - thick, d * 0.85))
    return out


def _steps(w, h, d):
    """Three treads, the top one at `h`, running back into the scene."""
    out = []
    for i in range(3):
        rise = h * (i + 1) / 3.0
        out.append(box(0.0, rise / 2.0, d * (0.5 - (i + 0.5) / 3.0),
                       w, rise, d / 3.0))
    return out


def _bed(w, h, d):
    """Base, mattress and pillow. `h` is the whole thing, pillow included, so
    a bed placed on the floor reaches the height it was asked for."""
    top = h * 0.78                       # mattress surface
    frame = top * 0.5
    return [box(0.0, frame / 2.0, 0.0, w, frame, d),
            box(0.0, (frame + top) / 2.0, 0.0, w * 0.98, top - frame, d * 0.98),
            box(0.0, (top + h) / 2.0, -d * 0.36, w * 0.56, h - top, d * 0.16)]


def _barrel(w, h, d):
    """Staves plus the bulge round the middle, inside the width given."""
    return [column(0.0, h / 2.0, 0.0, w * 0.94, h, d * 0.94),
            column(0.0, h / 2.0, 0.0, w, h * 0.55, d)]


def _archway(w, h, d):
    """A doorway: two jambs and a lintel, so a figure can stand in it."""
    jamb = w * 0.16
    out = [box(0.0, h - h * 0.08 / 2.0, 0.0, w, h * 0.08, d)]
    for sx in (-1.0, 1.0):
        out.append(box(sx * (w - jamb) / 2.0, (h - h * 0.08) / 2.0, 0.0,
                       jamb, h - h * 0.08, d))
    return out


SHAPES = {
    # raw geometry, for anything with no name of its own
    "box": (_plain(box), (60.0, 60.0, 60.0)),
    "sphere": (_plain(ball), (40.0, 40.0, 40.0)),
    "cylinder": (_plain(column), (40.0, 80.0, 40.0)),
    "cone": (lambda w, h, d: taper(0.0, 0.0, 0.0, w, h, d, 0.05),
             (50.0, 70.0, 50.0)),
    "capsule": (lambda w, h, d: capsule(0.0, 0.0, 0.0, w, h, d),
                (30.0, 90.0, 30.0)),
    "panel": (_plain(box), (120.0, 200.0, 6.0)),
    # things with a size everyone already knows
    "chair": (_chair, (45.0, 92.0, 48.0)),
    "stool": (_stool, (34.0, 62.0, 34.0)),
    "table": (_table, (140.0, 74.0, 80.0)),
    "desk": (_desk, (140.0, 74.0, 70.0)),
    "bench": (_bench, (150.0, 45.0, 40.0)),
    "bed": (_bed, (140.0, 45.0, 200.0)),
    "crate": (_plain(box), (50.0, 50.0, 50.0)),
    "barrel": (_barrel, (56.0, 88.0, 56.0)),
    "ball": (_plain(ball), (22.0, 22.0, 22.0)),
    "pillar": (_plain(column), (36.0, 260.0, 36.0)),
    "pole": (_plain(column), (7.0, 220.0, 7.0)),
    "platform": (_plain(box), (120.0, 30.0, 120.0)),
    "steps": (_steps, (140.0, 54.0, 90.0)),
    "wall": (_plain(box), (400.0, 260.0, 14.0)),
    "floor": (_plain(box), (600.0, 10.0, 600.0)),
    "archway": (_archway, (120.0, 210.0, 24.0)),
}

SHAPE_NAMES = sorted(SHAPES)


def default_size(shape):
    return SHAPES[shape][1] if shape in SHAPES else SHAPES["box"][1]


def make(shape, size=None, position=(0.0, 0.0, 0.0), yaw=0.0):
    """A prop as it is stored in a scene: plain data, world centimetres.

    `position` is the base centre, `yaw` degrees about vertical. Anchors live
    in `pose_agent`; by the time an object is in the scene it has a place, and
    the anchor it came from has stopped being true the moment anything moves.
    """
    if shape not in SHAPES:
        shape = "box"
    w, h, d = size or default_size(shape)
    return {"shape": shape, "size": [float(w), float(h), float(d)],
            "position": [float(v) for v in position], "yaw": float(yaw)}


def parts(prop):
    """One prop as a list of world-space primitives.

    Returned as a single part, so `render_depth` hard-unions the pieces: a
    chair's legs must meet its seat with an edge, not the fillet that suits a
    shoulder meeting a torso.
    """
    shape = prop.get("shape", "box")
    builder = SHAPES.get(shape, SHAPES["box"])[0]
    w, h, d = prop.get("size") or default_size(shape)
    px, py, pz = prop.get("position", (0.0, 0.0, 0.0))
    angle = math.radians(float(prop.get("yaw", 0.0)))
    cos, sin = math.cos(angle), math.sin(angle)

    def turn(v):
        return (v[0] * cos + v[2] * sin, v[1], -v[0] * sin + v[2] * cos)

    out = []
    for kind, centre, axes, radii in builder(float(w), float(h), float(d)):
        spun = turn(centre)
        out.append((kind, (spun[0] + px, spun[1] + py, spun[2] + pz),
                    tuple(turn(a) for a in axes), radii))
    return out


def bounds(prop):
    """(low corner, high corner) of a prop in world space.

    Every primitive here is symmetric about its centre along its own axes, so
    the box that holds it is the centre plus the axes scaled by the radii,
    summed - which is exact for a box and for a cylinder, and covers an
    ellipsoid.
    """
    lo = [1e18, 1e18, 1e18]
    hi = [-1e18, -1e18, -1e18]
    for _kind, centre, axes, radii in parts(prop):
        for i in range(3):
            reach = sum(abs(r * a[i]) for a, r in zip(axes, radii))
            lo[i] = min(lo[i], centre[i] - reach)
            hi[i] = max(hi[i], centre[i] + reach)
    return tuple(lo), tuple(hi)


def describe():
    return "\n".join(
        "  %-10s %4.0f x %4.0f x %4.0f cm  (w x h x d)" % ((name,) + size)
        for name, (_builder, size) in sorted(SHAPES.items()))


def _selftest():
    ok = True

    def check(label, condition, extra=""):
        nonlocal ok
        ok = ok and bool(condition)
        print(("PASS " if condition else "FAIL ") + label
              + ("  " + extra if extra else ""))

    print("props self-test")
    check("every shape builds something", all(parts(make(n)) for n in SHAPE_NAMES),
          str([n for n in SHAPE_NAMES if not parts(make(n))]))

    # the frame every placement relies on: base centred, sized as advertised
    worst_base, worst_size = (None, 0.0), (None, 0.0)
    for name in SHAPE_NAMES:
        prop = make(name)
        lo, hi = bounds(prop)
        w, h, d = prop["size"]
        if abs(lo[1]) > worst_base[1]:
            worst_base = (name, abs(lo[1]))
        for got, want in ((hi[0] - lo[0], w), (hi[1] - lo[1], h),
                          (hi[2] - lo[2], d)):
            slack = abs(got - want) / want
            if slack > worst_size[1]:
                worst_size = (name, slack)
    check("every shape stands on y = 0", worst_base[1] < 1e-6,
          "worst: %s at %.3g" % worst_base)
    check("and fills the size it was given, give or take a trim",
          worst_size[1] < 0.16, "worst: %s off by %.0f%%"
          % (worst_size[0], 100.0 * worst_size[1]))

    # centred across, so "in front of the figure" lands in front of the figure
    worst = (None, 0.0)
    for name in SHAPE_NAMES:
        lo, hi = bounds(make(name))
        for i in (0, 2):
            off = abs(lo[i] + hi[i]) / 2.0
            if off > worst[1]:
                worst = (name, off)
    check("and is centred on its own origin across and front to back",
          worst[1] < 0.6, "worst: %s off by %.2f cm" % worst)

    # placement is a rigid motion: turning must not resize anything
    for name in ("chair", "table", "crate", "wall", "steps"):
        square = make(name, position=(10.0, 0.0, -20.0))
        turned = make(name, position=(10.0, 0.0, -20.0), yaw=90.0)
        a_lo, a_hi = bounds(square)
        b_lo, b_hi = bounds(turned)
        check("%s turned 90 degrees swaps its footprint and keeps its height"
              % name,
              abs((a_hi[0] - a_lo[0]) - (b_hi[2] - b_lo[2])) < 1e-6
              and abs((a_hi[1] - a_lo[1]) - (b_hi[1] - b_lo[1])) < 1e-6)

    # axes have to stay orthonormal or the rasteriser's quadrics are not solid
    worst = 0.0
    for name in SHAPE_NAMES:
        for _kind, _centre, axes, _radii in parts(make(name, yaw=37.0)):
            for i in range(3):
                worst = max(worst, abs(math.sqrt(sum(v * v for v in axes[i]))
                                       - 1.0))
                for j in range(i + 1, 3):
                    worst = max(worst, abs(sum(a * b for a, b
                                               in zip(axes[i], axes[j]))))
    check("every primitive's axes stay orthonormal under a turn", worst < 1e-12,
          "%.2e" % worst)

    check("an unknown shape falls back to a box rather than failing",
          make("spaceship")["shape"] == "box")
    check("a prop is plain JSON-shaped data",
          set(make("chair")) == {"shape", "size", "position", "yaw"})

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    if "--list" in sys.argv:
        print(describe())
        raise SystemExit(0)
    print(__doc__)
