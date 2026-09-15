#!/usr/bin/env python3
"""Print the adult-to-child shape change measured off the body set's own rigs.

    python3 tools/child_ratios.py

There is no child in ANSUR II - it is a survey of soldiers - so the child row
in `openpose3d_editor.ANSUR` cannot be measured the same way as the other two.
What it can be is the adult male row times the shape change from an adult to a
child, and that change *is* measurable: `bodies/` carries an Anny adult and an
Anny child built from the same WHO-calibrated model, so the ratio of their rest
rigs says how a seven-year-old differs from a man.

What this replaced was the male row at 122 cm with `leg_ratio` 0.9 - a 70%
scale soldier with its legs cut, which a child is not. Its shoulders came out
14.4 cm half-width against the 11.8 its own rig has, 22% too broad, and the
leg_ratio dropped the hip to 56.3 cm where the rig puts it at 63.0. The fitted
mesh wore the difference: shoulders shrugged up around the ears and a torso
8 cm too long.

The numbers this prints are the ones written into the table by hand, and
`tests/test_proportions.py` reads them back and fails when the two part
company - the same arrangement the web preset table has.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Landmarks the rig can speak for. Hip, knee and ankle keypoints sit close to
# the rig's own joint centres; the shoulder does not - OpenPose's is the
# acromion and the rig's is the glenohumeral joint - which is exactly why only
# the RATIO between two bodies is taken from here and never the absolute. The
# ratio is the part that survives the landmarks disagreeing, because the same
# disagreement sits on both sides of it.
LANDMARKS = ("shoulder_height", "hip_height", "knee_height", "ankle_height",
             "shoulder_w", "upper_arm", "forearm")


def rig_ratios(mesh):
    import numpy as np
    rest, roles = mesh["rest_position"], mesh["roles"]
    verts = mesh["vertices"]
    sole = float(verts[:, 1].min())
    height = float(verts[:, 1].max() - sole)
    at = lambda role: rest[roles[role]]
    hip = 0.5 * (at("l_hip") + at("r_hip"))
    return {
        "shoulder_height": (at("l_shoulder")[1] - sole) / height,
        "hip_height": (hip[1] - sole) / height,
        "knee_height": (at("l_knee")[1] - sole) / height,
        "ankle_height": (at("l_ankle")[1] - sole) / height,
        "shoulder_w": abs(at("l_shoulder")[0] - hip[0]) / height,
        "upper_arm": float(np.linalg.norm(at("l_elbow") - at("l_shoulder")))
                     / height,
        "forearm": float(np.linalg.norm(at("l_wrist") - at("l_elbow")))
                   / height,
    }


def child_row(adult="Male, average", child="Child, about 7", folder=None):
    """The ANSUR child row: the male row times the measured shape change."""
    import bodies_lib
    from openpose3d_editor import ANSUR
    bank = bodies_lib.load(folder, [adult, child])
    grown, small = rig_ratios(bank[adult]), rig_ratios(bank[child])
    out = dict(ANSUR["male"])
    for key in LANDMARKS:
        if key in ("upper_arm", "forearm"):
            continue          # the split, not the lengths; see the table
        out[key] = ANSUR["male"][key] * (small[key] / grown[key])
    return out, grown, small


def main():
    row, grown, small = child_row()
    print("adult-to-child shape change, from the body set's own rigs\n")
    print("  %-16s %9s %9s %9s %12s %10s"
          % ("landmark", "adult", "child", "child/ad", "male ANSUR", "child row"))
    from openpose3d_editor import ANSUR
    for key in LANDMARKS:
        rel = small[key] / grown[key]
        print("  %-16s %9.4f %9.4f %9.3f %12.4f %10s"
              % (key, grown[key], small[key], rel, ANSUR["male"][key],
                 ("%.4f" % row[key]) if key in row
                 and key not in ("upper_arm", "forearm") else "(kept)"))
    print("\nthe child row as it belongs in ANSUR:")
    for key in LANDMARKS:
        if key in ("upper_arm", "forearm"):
            continue
        print('        "%s": %.4f,' % (key, row[key]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
