#!/usr/bin/env python3
"""The orthographic turntable the whole editor projects through.

Orthographic on purpose: it is what makes the reach sphere project to a circle
and the guide arcs solve exactly. Replacing it with a perspective camera means
replacing the drag maths with ray-sphere intersection.
"""

from __future__ import annotations

import math

from vecmath import (vadd, vcross, vdot, vlen, vmul, vnorm, vsub)


# ---------------------------------------------------------------------------

class Camera:
    def __init__(self, width=900, height=700):
        self.yaw = 0.0
        self.pitch = 0.0
        self.target = (0.0, -60.0, 0.0)
        self.zoom = 2.6           # pixels per world unit
        self.width = width
        self.height = height

    def basis(self):
        """Returns (right, up, forward); forward points away from the viewer."""
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        eye_dir = (cp * sy, sp, cp * cy)          # target -> camera
        fwd = vmul(eye_dir, -1.0)
        right = vnorm(vcross(fwd, (0.0, 1.0, 0.0)))
        if vlen(right) < 1e-9:
            right = (1.0, 0.0, 0.0)
        up = vnorm(vcross(right, fwd))
        return right, up, fwd

    def project(self, p):
        """World point -> (screen x, screen y, depth). Larger depth = further."""
        right, up, fwd = self.basis()
        rel = vsub(p, self.target)
        return (vdot(rel, right) * self.zoom + self.width / 2.0,
                -vdot(rel, up) * self.zoom + self.height / 2.0,
                vdot(rel, fwd))

    def screen_delta_to_world(self, dx, dy):
        """Screen-space offset -> world vector lying in the view plane."""
        right, up, _ = self.basis()
        return vadd(vmul(right, dx / self.zoom), vmul(up, -dy / self.zoom))

    def orbit(self, dx, dy, speed=0.008):
        self.yaw -= dx * speed
        self.pitch += dy * speed
        limit = math.radians(89.0)
        self.pitch = max(-limit, min(limit, self.pitch))

    def pan(self, dx, dy):
        right, up, _ = self.basis()
        self.target = vadd(self.target, vmul(right, -dx / self.zoom))
        self.target = vadd(self.target, vmul(up, dy / self.zoom))

    def zoom_by(self, factor):
        self.zoom = max(0.3, min(40.0, self.zoom * factor))

    def set_view(self, name):
        views = {
            "front": (0.0, 0.0), "back": (math.pi, 0.0),
            "right": (-math.pi / 2, 0.0), "left": (math.pi / 2, 0.0),
            "top": (0.0, math.radians(89.0)), "bottom": (0.0, math.radians(-89.0)),
        }
        if name in views:
            self.yaw, self.pitch = views[name]
