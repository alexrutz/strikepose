#!/usr/bin/env python3
"""Three-component vectors and 3x3 matrices, as plain tuples.

Small enough that a class would cost more than it saves: a drag redraws
several times a second and every one of these is on that path.
"""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------

def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vmul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def vlen(a):
    return math.sqrt(vdot(a, a))


def vnorm(a):
    n = vlen(a)
    return (0.0, 0.0, 0.0) if n < 1e-12 else (a[0] / n, a[1] / n, a[2] / n)


def any_perpendicular(a):
    ref = (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0)
    return vnorm(vcross(a, ref))


IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def matvec(m, v):
    return (vdot(m[0], v), vdot(m[1], v), vdot(m[2], v))


def rotation_between(a, b):
    """Rotation matrix taking direction a onto direction b (Rodrigues)."""
    ua, ub = vnorm(a), vnorm(b)
    if vlen(ua) < 1e-9 or vlen(ub) < 1e-9:
        return IDENTITY
    c = max(-1.0, min(1.0, vdot(ua, ub)))
    axis = vcross(ua, ub)
    s = vlen(axis)
    if s < 1e-9:
        if c > 0:
            return IDENTITY
        axis, s = any_perpendicular(ua), 0.0  # 180 degree flip
    else:
        axis = vmul(axis, 1.0 / s)
    x, y, z = axis
    k = ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))
    kk = tuple(tuple(vdot(k[i], (k[0][j], k[1][j], k[2][j])) for j in range(3))
               for i in range(3))
    return tuple(
        tuple(IDENTITY[i][j] + k[i][j] * s + kk[i][j] * (1.0 - c) for j in range(3))
        for i in range(3)
    )
