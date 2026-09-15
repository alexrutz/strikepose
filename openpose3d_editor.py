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
"""

from __future__ import annotations

VERSION = "1.39.0"          # shown in the title bar, the HUD and on startup

import base64
import colorsys
import io
import json
import time
import math
import os
import sys
from copy import deepcopy

import props as props_module

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except ImportError:  # allows importing the math/export core without a display
    tk = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None


# ---------------------------------------------------------------------------
# OpenPose COCO-18 definition
# Verified against lllyasviel/ControlNet annotator/openpose/util.py
# ---------------------------------------------------------------------------

KEYPOINT_NAMES = [
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]

# (parent, child) pairs in canonical OpenPose limb order; index -> COLORS index
LIMB_SEQ = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

# Viewport-only styling so people can be told apart. Blending towards a
# signature colour was not enough: a tint that already appears in the OpenPose
# ramp leaves that keypoint unchanged, so two figures could share a colour.
# A hue rotation combined with saturation and value scaling moves every
# keypoint instead. (hue turns, saturation scale, value scale); the first
# figure is the untouched palette.
# Signature colour per figure, applied to the body preview. The bones span the
# whole hue circle whatever you do to them, so recolouring them alone never
# reads at a glance; the body is a large flat area where a tint is obvious.
# Objects are one flat neutral, a little cooler than any figure tint, so a
# prop never reads as another person in the viewport.
PROP_TINT = (196, 202, 214)

FIGURE_BODY_TINTS = ((235, 235, 240),     # 1: neutral
                     (255, 165, 80),      # 2: orange
                     (105, 205, 255),     # 3: cyan
                     (150, 240, 115),     # 4: green
                     (250, 140, 230),     # 5: pink
                     (240, 225, 110),     # 6: yellow
                     (175, 160, 255))     # 7: periwinkle

# Ordered most-distinct-first, so a scene with two or three people gets the
# biggest separation: pale and dark read apart at a glance far better than a
# hue shift does.
# Kept gentle: the body tint above already identifies each figure, so the bones
# only need a secondary nudge, which matters when the body preview is off (B).
# Pushing them harder makes the keypoint colours hard to read.
FIGURE_STYLES = ((0.00, 1.00, 1.00),      # 1: the untouched OpenPose palette
                 (0.00, 0.70, 1.00),      # 2: softer
                 (-0.07, 1.00, 0.84),     # 3: cool, deeper
                 (0.09, 1.00, 1.00),      # 4: warm
                 (0.16, 0.72, 1.00),      # 5: soft warm
                 (-0.16, 1.00, 0.92),     # 6: cool
                 (0.24, 0.85, 0.80))      # 7: muted

COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85),
]

ROOT = 1  # neck
PARENT = {child: parent for parent, child in LIMB_SEQ}
CHILDREN = {}
for _p, _c in LIMB_SEQ:
    CHILDREN.setdefault(_p, []).append(_c)

ADJACENCY = {}
for _p, _c in LIMB_SEQ:
    ADJACENCY.setdefault(_p, []).append(_c)
    ADJACENCY.setdefault(_c, []).append(_p)

MIRROR_OF = {}

_REROOT_CACHE = {}


def reroot(anchor):
    """Kinematic tree re-hung from an arbitrary joint.

    The skeleton is a tree, so any joint can serve as its root: breadth-first
    from the anchor reverses the parent links along the path back to the neck
    and leaves everything else alone. Anchoring a knee therefore makes the hip
    a child of the knee, and dragging the hip swings the whole upper body about
    that knee instead of the other way round.
    """
    if anchor in _REROOT_CACHE:
        return _REROOT_CACHE[anchor]
    parents = {anchor: -1}
    order, queue = [anchor], [anchor]
    while queue:
        joint = queue.pop(0)
        for neighbour in ADJACENCY.get(joint, ()):
            if neighbour not in parents:
                parents[neighbour] = joint
                order.append(neighbour)
                queue.append(neighbour)
    children = {}
    for child, parent in parents.items():
        if parent >= 0:
            children.setdefault(parent, []).append(child)
    _REROOT_CACHE[anchor] = (parents, children, order)
    return _REROOT_CACHE[anchor]


GIRDLE = (2, 5, 8, 11)      # shoulders and hips, re-seated on preset change
MIRROR_PAIRS = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (14, 15), (16, 17)]
for _a, _b in MIRROR_PAIRS:
    MIRROR_OF[_a] = _b
    MIRROR_OF[_b] = _a

# ---------------------------------------------------------------------------
# Body presets
#
# A preset holds real measurements in centimetres: skeleton proportions (bone
# lengths, shoulder and hip width) and the volumes used by the depth pass.
# Radii are half-thicknesses. Torso cross-sections are elliptical, given as
# (half width, half depth), so a chest can be broad without becoming round.
# Limb radii are (top, middle, bottom) so muscles can bulge instead of tapering
# in a straight line.
# ---------------------------------------------------------------------------

# Proportions as fractions of stature, from ANSUR II - the 2012 US Army
# anthropometric survey, 4082 men and 1986 women, 93 measurements each, public
# since 2017. Means computed from the released records rather than quoted, so
# every number here can be reproduced from the CSVs.
#
# It replaced the Drillis & Contini (1966) constants reproduced in Winter's
# Biomechanics, which this used before. Those are a century-old convenience
# table; the one that mattered was trochanter height at 0.530 of stature, which
# is 0.513 measured. Three centimetres of leg on a 175 cm figure, taken off the
# torso, and it showed in every export.
#
# The landmarks matter as much as the numbers. Trochanterion is the standard
# proxy for the hip joint centre, the lateral femoral epicondyle for the knee
# and the lateral malleolus for the ankle, so thigh and shank are differences
# of measured heights and close on the floor exactly. Acromion is the shoulder
# keypoint. Tragion is the ear.
ANSUR = {
    "male": {
        "shoulder_height": 0.8203,      # acromial height
        "hip_height": 0.5128,           # trochanterion height
        "knee_height": 0.2799,          # lateral femoral epicondyle height
        "ankle_height": 0.0415,         # lateral malleolus height
        "shoulder_w": 0.1184,           # biacromial breadth / 2
        "hip_w": 0.0500,                # femoral heads; see below
        "neck_rise": 0.0377,          # the neck bone starts this far
                                    # above the shoulder midpoint
        "gh_w": 0.1094,               # glenohumeral half-width
        "gh_drop": 0.0222,            # acromion down to that joint
        "arm_split": 0.5124,          # humerus / (humerus + forearm)
        "hand": 0.1101,                 # hand length
        "span": 1.0330,                 # fingertip to fingertip
        "upper_arm": 0.1909,            # acromion-radiale
        "forearm": 0.1525,              # radiale-stylion
        "ear_drop": 13.11 / 175.62,     # tragion to top of head
        "head": 1.0,                    # head size, male mean is the reference
        "jaw": 1.0,                     # bizygomatic breadth
        "waist_t": 0.593,               # tenth rib, along shoulder to hip
        "stature_ref": 175.62,
    },
    "female": {
        "shoulder_height": 0.8198,
        "hip_height": 0.5190,
        "knee_height": 0.2860,
        "ankle_height": 0.0385,
        "shoulder_w": 0.1122,
        "hip_w": 0.0535,
        "neck_rise": 0.0265,          # the neck bone starts this far
                                    # above the shoulder midpoint
        "gh_w": 0.0982,               # glenohumeral half-width
        "gh_drop": 0.0279,            # acromion down to that joint
        "arm_split": 0.5451,          # humerus / (humerus + forearm)
        "hand": 0.1112,
        "span": 1.0195,
        "upper_arm": 0.1911,
        "forearm": 0.1482,
        "ear_drop": 12.65 / 162.85,
        "head": 0.9545,                 # head length and breadth against his
        "jaw": 0.938,                   # bizygomatic breadth against his
        "waist_t": 0.569,               # her narrowest point sits higher
        "stature_ref": 162.85,
    },
    # There is no child in ANSUR - it is a survey of soldiers - so this one
    # cannot be measured the same way. What it can be is the adult male row
    # times the shape change from an adult to a child, and that change is
    # measurable: the body set carries an Anny adult and an Anny child built
    # from the same WHO-calibrated model, so the ratio of their rest rigs says
    # how a seven-year-old differs from a man. tools/child_ratios.py prints
    # them and tests/test_proportions.py checks these against it.
    #
    # What it replaces was the male row at 122 cm with leg_ratio 0.9, which is
    # a 70% scale soldier with its legs cut - and a child is not that. Its
    # shoulders came out 14.4 cm half-width against the 11.8 its own rig has,
    # 22% too broad, and leg_ratio dropped the hip to 56.3 cm where the rig
    # puts it at 63.0. The fitted body wore the difference.
    "child": {
        "shoulder_height": 0.7943,      # male 0.8203 x 0.968 measured
        "hip_height": 0.5022,           # x 0.979
        "knee_height": 0.2616,          # x 0.934
        "ankle_height": 0.0416,         # x 1.002
        "shoulder_w": 0.1043,           # x 0.881 - the big one
        "hip_w": 0.0500,
        "neck_rise": 0.0386,          # the neck bone starts this far
                                    # above the shoulder midpoint
        "gh_w": 0.0964,               # glenohumeral half-width
        "gh_drop": 0.0215,            # acromion down to that joint
        "arm_split": 0.5231,          # humerus / (humerus + forearm)
        "hand": 0.1101,
        "span": 1.0330,
        "upper_arm": 0.1909,
        "forearm": 0.1525,
        "ear_drop": 13.11 / 175.62,
        "head": 1.0,
        "jaw": 1.0,
        "waist_t": 0.593,
        "stature_ref": 175.62,
    },
}

# A head is not a scaled copy of the body it sits on. Fitting log head size on
# log stature across the survey gives an exponent of 0.07 for head breadth,
# 0.24 for the tragion-to-crown height and 0.34 for head length - call it a
# quarter - against 0.89 for the hand and 0.63 for the shoulders. A tall person
# has a head barely bigger than a short one.
#
# This used stature / 175, an exponent of one, and then had a special case
# forcing a child's head back up because the result was absurd. The exponent
# does that on its own: a 122 cm child comes out at 0.91 where the special case
# said 0.88, and a 163 cm woman at 0.95 where straight scaling said 0.93.
HEAD_EXPONENT = 0.25

# Hip width is the one skeletal number ANSUR cannot give: it measures the iliac
# crests and the soft-tissue hip, not the femoral heads the leg actually swings
# from. The values above are the long-standing ones, kept because the measured
# bicristal ratio between the sexes - 0.1679 / 0.1569 = 1.070 - matches the
# ratio they already had, 0.0535 / 0.0500 = 1.070, so the relationship is right
# even where the absolute is inherited.

# The arm hangs from the JOINT, not from the keypoint. This closed it on the
# span from the acromion and split it by the published acromion-radiale to
# radiale-stylion ratio, and both halves of that are wrong in the same way:
# acromion-radiale is measured from the bony corner on top of the shoulder,
# which is 0.022 of stature above and 0.009 inboard of the glenohumeral joint
# the arm actually swings from. Hanging the arm from the acromion put the
# elbow keypoint 4 cm high, and splitting by a surface ratio of 1.25 where the
# real bones are 1.05 to 1.20 put it further out still. A rig fitted to those
# keypoints then had to choose between the shoulder line and the humerus: hold
# the shoulder where the body's own rig has it and the humerus is crushed by a
# fifth, or let the shoulder ride up to the acromion and every figure stands
# there with its shoulders round its ears. Both shipped, one after the other.
#
# Measured from the joint, it closes: gh_w + humerus + forearm + hand is 0.5172
# of stature against the measured half-span of 0.5165, a tenth of a percent, on
# the body set's own adult male rig. So the span still closes the arm - it is
# the one number measured directly along a straight arm - but from gh_w, and
# the split comes from the rigs. tools/child_ratios.py prints all of them.
def derive_proportions(stature, sex="male", leg_ratio=1.0):
    """Skeleton measurements for a given height.

    Build does not change bone length: a heavy and a lean person of the same
    height have the same skeleton and differ in girth, so the body types share
    these numbers and override only the cross-sections.
    """
    m = ANSUR.get(sex) or ANSUR["male"]
    shoulder_half = stature * m["shoulder_w"]
    hand = m["hand"] * stature
    half_span = 0.5 * stature * m["span"]
    # from the glenohumeral joint, not the acromion keypoint: see above
    arm = max(10.0, half_span - stature * m["gh_w"] - hand)
    forearm = arm * (1.0 - m["arm_split"])
    # thigh and shank are differences of measured heights, so they close on the
    # floor: thigh + shank + ankle height is hip height by construction
    thigh = (m["hip_height"] - m["knee_height"]) * stature * leg_ratio
    shank = (m["knee_height"] - m["ankle_height"]) * stature * leg_ratio
    ear_up = (1.0 - m["ear_drop"] - m["shoulder_height"]) * stature
    return {
        "shoulder_w": shoulder_half,
        # OpenPose's neck, keypoint 1, is defined as the midpoint of the two
        # shoulders - it is inferred that way from the COCO annotations, which
        # have no neck of their own. So the shoulders sit at the neck's own
        # height and this is zero. It was 0.011 of stature, which put every
        # exported neck keypoint two centimetres above where the format says
        # it goes, and dropped the shoulder line the same distance below the
        # measured acromial height. The keys stays so scenes saved with a drop
        # still load.
        "shoulder_drop": 0.0,
        # where the arm actually swings from, so the rest pose can hang it
        # there and a rig fitted to it keeps both its shoulder line and its
        # humerus
        "gh_w": stature * m["gh_w"],
        "gh_drop": stature * m["gh_drop"],
        "neck_rise": stature * m["neck_rise"],
        "hip_w": stature * m["hip_w"],                          # femoral heads
        "torso_len": (m["shoulder_height"]
                      - m["hip_height"] * leg_ratio) * stature,
        "upper_arm": arm - forearm,
        "forearm": forearm,
        "thigh": thigh,
        "calf": shank,
        "head": m["head"] * (stature / m["stature_ref"]) ** HEAD_EXPONENT,
        "jaw": m["jaw"],
        "waist_t": m["waist_t"],
        # the nose tip sits a little below the ear canal, which ANSUR has no
        # landmark pair for; the offset is the one this has always carried
        "nose_up": ear_up - 0.005 * stature,
        "ear_up": ear_up,
    }


BASE_BODY = {
    # skeleton
    "shoulder_w": 19.0, "shoulder_drop": 0.0, "hip_w": 10.0, "torso_len": 52.0,
    "upper_arm": 28.5, "forearm": 26.0, "thigh": 43.0, "calf": 43.0,
    "head": 1.0, "waist_t": 0.593,
    # measured cross-sections: (half width, half depth) in centimetres
    "chest": (14.47, 12.69), "waist": (14.0, 10.4), "pelvis": (17.0, 12.0),
    "belly": 0.0,
    # girth multipliers on the anatomical limb profiles
    "arm_girth": 1.0, "forearm_girth": 1.0,
    "thigh_girth": 1.0, "calf_girth": 1.0,
    "deltoid": (6.0, 6.4, 7.2), "neck_girth": 1.0,
    "hand": 1.0, "foot": 1.0, "jaw": 1.0,
    "bust": None,                       # (half w, projection, half h, t, sep)
    "glute": (8.4, 6.2, 7.0, 1.02, 7.5),
}

BODY_PRESETS = {
    "Male, average": {
        "stature": 175, "sex": "male", "leg_ratio": 1.0,},
    "Male, athletic": {
        "stature": 178, "sex": "male", "leg_ratio": 1.0,
        "chest": (15.32, 13.46), "waist": (13.4, 10.0), "pelvis": (16.4, 11.6),
        "arm_girth": 1.16, "forearm_girth": 1.12,
        "thigh_girth": 1.12, "calf_girth": 1.12,
        "deltoid": (6.8, 7.2, 7.8), "neck_girth": 1.12,
    },
    "Male, heavy": {
        "stature": 175, "sex": "male", "leg_ratio": 1.0,
        "chest": (16.17, 16.00), "waist": (19.0, 16.0), "pelvis": (19.0, 14.0),
        "belly": 3.4,
        "arm_girth": 1.24, "forearm_girth": 1.16,
        "thigh_girth": 1.22, "calf_girth": 1.14,
        "deltoid": (6.8, 7.4, 7.6), "neck_girth": 1.22,
        "glute": (9.4, 7.4, 7.6, 1.02, 8.2),
    },
    "Male, slim": {
        "stature": 176, "sex": "male", "leg_ratio": 1.0,
        "chest": (12.94, 10.81), "waist": (12.2, 8.6), "pelvis": (15.0, 10.4),
        "arm_girth": 0.84, "forearm_girth": 0.86,
        "thigh_girth": 0.86, "calf_girth": 0.88,
        "deltoid": (5.2, 5.6, 6.4), "neck_girth": 0.88,
        "glute": (7.4, 5.0, 6.4, 1.02, 7.0),
    },
    "Female, average": {
        "stature": 162, "sex": "female", "leg_ratio": 1.0,
        "chest": (13.46, 12.37), "waist": (12.2, 9.0), "pelvis": (17.6, 12.0),
        "arm_girth": 0.86, "forearm_girth": 0.86,
        "thigh_girth": 1.00, "calf_girth": 0.92,
        "deltoid": (5.4, 5.8, 6.4), "neck_girth": 0.86,
        "hand": 0.9, "foot": 0.92,
        "bust": (6.8, 5.4, 6.4, 0.22, 6.4),
        "glute": (9.0, 7.0, 7.4, 1.02, 8.0),
    },
    "Female, athletic": {
        "stature": 166, "sex": "female", "leg_ratio": 1.0,
        "chest": (13.83, 12.63), "waist": (11.4, 8.4), "pelvis": (16.6, 11.2),
        "arm_girth": 0.94, "forearm_girth": 0.92,
        "thigh_girth": 1.04, "calf_girth": 0.98,
        "deltoid": (5.9, 6.2, 6.8), "neck_girth": 0.90,
        "hand": 0.9, "foot": 0.92,
        "bust": (6.2, 4.2, 5.8, 0.22, 6.2),
        "glute": (8.8, 6.8, 7.2, 1.02, 7.8),
    },
    "Female, curvy": {
        "stature": 162, "sex": "female", "leg_ratio": 1.0,
        "chest": (14.20, 13.14), "waist": (11.8, 9.0), "pelvis": (19.4, 13.0),
        "arm_girth": 0.92, "forearm_girth": 0.90,
        "thigh_girth": 1.12, "calf_girth": 0.98,
        "deltoid": (5.4, 5.8, 6.4), "neck_girth": 0.88,
        "hand": 0.9, "foot": 0.92,
        "bust": (7.8, 6.6, 7.2, 0.22, 6.8),
        "glute": (10.2, 8.4, 8.2, 1.02, 8.6),
    },
    "Female, slim": {
        "stature": 165, "sex": "female", "leg_ratio": 1.0,
        "chest": (12.36, 11.34), "waist": (10.6, 7.8), "pelvis": (15.4, 10.4),
        "arm_girth": 0.76, "forearm_girth": 0.78,
        "thigh_girth": 0.88, "calf_girth": 0.84,
        "deltoid": (4.8, 5.2, 5.8), "neck_girth": 0.80,
        "hand": 0.88, "foot": 0.9,
        "bust": (5.8, 3.8, 5.4, 0.22, 5.8),
        "glute": (7.8, 5.6, 6.6, 1.02, 7.2),
    },
    "Child, about 7": {
        "stature": 122, "sex": "child", "leg_ratio": 1.0,
        "chest": (9.19, 8.83), "waist": (10.0, 7.6), "pelvis": (10.6, 8.0),
        "arm_girth": 0.62, "forearm_girth": 0.64,
        "thigh_girth": 0.66, "calf_girth": 0.66,
        "deltoid": (4.0, 4.2, 4.6), "neck_girth": 0.64,
        "hand": 0.66, "foot": 0.68,
        "glute": (5.6, 4.2, 4.8, 1.02, 5.0),
    },
}

DEFAULT_PRESET = "Male, average"


def preset_params(name):
    body = dict(BASE_BODY)
    overrides = dict(BODY_PRESETS.get(name, {}))
    stature = overrides.pop("stature", 175)
    sex = overrides.pop("sex", "male")
    leg_ratio = overrides.pop("leg_ratio", 1.0)
    body.update(derive_proportions(stature, sex, leg_ratio))
    if stature < 140:                       # a child's face is narrower
        body["jaw"] = min(body["jaw"], 0.94)
    body.update(overrides)
    body["stature"] = stature
    body["sex"] = sex
    body["preset"] = name
    return body


def merge_body(raw):
    """Reconcile a body dict with the current schema. Scenes saved by earlier
    versions carry keys that no longer exist and lack ones that now do, so
    start from the base body, re-apply the named preset, then keep whatever
    saved values are still meaningful. Unknown keys are dropped rather than
    passed through."""
    raw = raw or {}
    name = raw.get("preset")
    # start from the preset's full parameters, derived proportions included:
    # starting from BASE_BODY alone silently reverted a loaded scene to the
    # default stature's bone lengths
    body = preset_params(name if name in BODY_PRESETS else DEFAULT_PRESET)
    for key, value in raw.items():
        if key in body:
            body[key] = tuple(value) if isinstance(value, list) else value
    return body


def build_rest_points(body):
    """Rest pose from a preset, centimetres, neck at the origin. +X world right,
    +Y up, +Z towards the default camera. The figure faces +Z, so its right side
    sits at -X and appears on the left of a front view, as it should."""
    h = body["head"]
    sw, sd = body["shoulder_w"], body["shoulder_drop"]
    hw, tl = body["hip_w"], body["torso_len"]
    ua, fa = body["upper_arm"], body["forearm"]
    th, ca = body["thigh"], body["calf"]

    nose_up = body.get("nose_up", 16.0 * h)
    ear_up = body.get("ear_up", 19.0 * h)
    pts = {"neck": (0.0, 0.0, 0.0),
           "nose": (0.0, nose_up, 5.0 * h),
           # Not a keypoint either. OpenPose's neck is the midpoint of the
           # shoulders, out in the middle of the upper chest; a neck bone
           # starts at the top of the thorax, 4 to 7 cm above it. Without
           # this the rig's neck is dragged down onto the chest and takes the
           # head with it, which reads as a figure with no neck at all.
           "neck_joint": (0.0, body.get("neck_rise", 0.0), 0.0)}
    for side, sx in (("r", -1.0), ("l", 1.0)):
        pts[side + "_eye"] = (sx * 3.0 * h, ear_up + 1.0 * h, 7.0 * h)
        pts[side + "_ear"] = (sx * 7.5 * h, ear_up, 1.0 * h)
        # The shoulder KEYPOINT is the acromion, which is what OpenPose wants.
        # The arm hangs from the glenohumeral joint below and inboard of it, so
        # the elbow lands where a rig's elbow is rather than 4 cm above it.
        sh = (sx * sw, -sd, 0.0)
        gh = (sx * body.get("gh_w", sw), -sd - body.get("gh_drop", 0.0), 0.0)
        el = (gh[0] + sx * 0.141 * ua, gh[1] - 0.990 * ua, 0.0)
        wr = (el[0] + sx * 0.077 * fa, el[1] - 0.997 * fa, 2.0)
        hp = (sx * hw, -tl, 0.0)
        kn = (hp[0] + sx * 0.023 * th, hp[1] - 0.9997 * th, 0.0)
        an = (kn[0], kn[1] - 0.9976 * ca, kn[2] - 0.0697 * ca)
        pts[side + "_shoulder"], pts[side + "_elbow"], pts[side + "_wrist"] = sh, el, wr
        # Not a keypoint - OpenPose has no such thing - but the rig fitter
        # needs it, and this is the only place that knows it. Carried as an
        # extra entry rather than derived over there from a rig whose torso
        # may not be this figure's: measured that way the offset came out
        # 6.5 cm where the anthropometry says 3.9, and the extra 2.6 was a
        # disagreement about torso length being charged to the humerus.
        pts[side + "_gh"] = gh
        pts[side + "_hip"], pts[side + "_knee"], pts[side + "_ankle"] = hp, kn, an
    return pts


REST_POSE = build_rest_points(preset_params(DEFAULT_PRESET))


# ---------------------------------------------------------------------------
# Small vector helpers (tuples of three floats)
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


# ---------------------------------------------------------------------------
# Camera: orthographic turntable
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


# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------

class Skeleton:
    def __init__(self, body=None):
        self.body = merge_body(body or preset_params(DEFAULT_PRESET))
        self.body_scale = 1.0
        rest = build_rest_points(self.body)
        self.points = [rest[n] for n in KEYPOINT_NAMES]
        self.visible = [True] * len(KEYPOINT_NAMES)
        self.lengths = {c: vlen(vsub(self.points[c], self.points[p]))
                        for p, c in LIMB_SEQ}
        self.anchors = []           # 0: neck-rooted, 1: pivot, 2: hinge axis
        self.assets = []            # rigged-mesh add-ons on the same armature
        # {slot: preset} of hair and clothing; empty is bare. Normalised by
        # wearables.clean on the way in rather than validated here, so this
        # module needs nothing from wearables at import time - wearables
        # imports the geometry helpers from this one.
        self.outfit = {}

    @staticmethod
    def torso_frame(pts):
        """Orthonormal (side, up, facing) built from the posed torso."""
        side = vnorm(vsub(pts[5], pts[2]))
        sh_mid = vmul(vadd(pts[2], pts[5]), 0.5)
        hip_mid = vmul(vadd(pts[8], pts[11]), 0.5)
        up_raw = vnorm(vsub(sh_mid, hip_mid))
        if vlen(side) < 1e-6 or vlen(up_raw) < 1e-6:
            return None
        facing = vnorm(vcross(side, up_raw))
        if vlen(facing) < 1e-6:
            return None
        return side, vnorm(vcross(facing, side)), facing

    def apply_body(self, body):
        """Swap in new proportions while keeping the current pose: every bone
        keeps its direction and takes the new length, walking down the tree
        from the root."""
        old = list(self.points)
        rest = build_rest_points(body)
        lengths = {c: vlen(vsub(rest[KEYPOINT_NAMES[c]], rest[KEYPOINT_NAMES[p]]))
                   for p, c in LIMB_SEQ}
        frame = self.torso_frame(old)
        for parent, child in LIMB_SEQ:      # LIMB_SEQ is ordered root-first
            if child in GIRDLE and frame is not None:
                # shoulders and hips are re-seated in the torso's own frame,
                # otherwise a wider preset would only lengthen the bone and
                # leave the width unchanged
                o = vsub(rest[KEYPOINT_NAMES[child]], rest["neck"])
                self.points[child] = vadd(
                    self.points[ROOT],
                    vadd(vadd(vmul(frame[0], o[0]), vmul(frame[1], o[1])),
                         vmul(frame[2], o[2])))
                continue
            d = vsub(old[child], old[parent])
            if vlen(d) < 1e-9:
                d = vsub(rest[KEYPOINT_NAMES[child]], rest[KEYPOINT_NAMES[parent]])
            self.points[child] = vadd(self.points[parent],
                                      vmul(vnorm(d), lengths[child]))
        self.lengths = lengths
        self.body = merge_body(body)
        self.body_scale = 1.0

    # -- anchors -----------------------------------------------------------
    @property
    def anchor(self):
        """The joint the tree hangs from. With two anchors the first one holds
        the tree together while the pair defines the hinge axis."""
        return self.anchors[0] if self.anchors else ROOT

    @anchor.setter
    def anchor(self, value):
        self.anchors = [] if value == ROOT else [value]

    @property
    def hinged(self):
        return len(self.anchors) == 2

    def hinge_axis(self):
        a, b = self.points[self.anchors[0]], self.points[self.anchors[1]]
        return a, vnorm(vsub(b, a))

    def hinge_set(self, joint):
        """Joints that swing with `joint`: everything reachable from it without
        passing through an anchor. Anchoring both knees and dragging the neck
        therefore carries the torso and thighs, while the shins and feet on the
        far side of the anchors stay put."""
        blocked = set(self.anchors)
        seen, stack = {joint}, [joint]
        while stack:
            current = stack.pop()
            for neighbour in ADJACENCY.get(current, ()):
                if neighbour in blocked or neighbour in seen:
                    continue
                seen.add(neighbour)
                stack.append(neighbour)
        return seen

    def rotate_about_axis(self, joints, origin, axis, angle):
        c, s_ = math.cos(angle), math.sin(angle)
        x, y, z = axis
        K = ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))
        KK = tuple(tuple(vdot(K[i], (K[0][j], K[1][j], K[2][j]))
                         for j in range(3)) for i in range(3))
        R = tuple(tuple(IDENTITY[i][j] + K[i][j] * s_ + KK[i][j] * (1.0 - c)
                        for j in range(3)) for i in range(3))
        for j in joints:
            self.points[j] = vadd(origin, matvec(R, vsub(self.points[j], origin)))

    def hinge_circle(self, joint):
        """Centre and the two radius vectors of the circle `joint` travels on.

        P(t) = centre + a*cos(t) + b*sin(t), with P(0) the joint's position now.
        """
        origin, axis = self.hinge_axis()
        if vlen(axis) < 1e-9:
            return None
        offset = vsub(self.points[joint], origin)
        along = vmul(axis, vdot(offset, axis))
        a = vsub(offset, along)
        if vlen(a) < 1e-6:
            return None
        return vadd(origin, along), a, vcross(axis, a)

    def hinge_angle(self, joint, cursor, project, samples=120):
        """Angle about the hinge that brings `joint` nearest the cursor.

        Solved in screen space rather than in the plane perpendicular to the
        axis: that plane is edge-on whenever the axis runs across the view (two
        knees seen from the front, say), and mapping the cursor into it then
        yields no angle at all. The joint's path projects to an ellipse, so
        sample it coarsely and refine.
        """
        circle = self.hinge_circle(joint)
        if circle is None:
            return None
        centre, a, b = circle

        def at(t):
            ct, st = math.cos(t), math.sin(t)
            return vadd(centre, vadd(vmul(a, ct), vmul(b, st)))

        def distance(t):
            sx, sy = project(at(t))[:2]
            return (sx - cursor[0]) ** 2 + (sy - cursor[1]) ** 2

        best, best_d = 0.0, distance(0.0)
        for i in range(1, samples):
            t = 2.0 * math.pi * i / samples
            d = distance(t)
            if d < best_d:
                best, best_d = t, d
        step = 2.0 * math.pi / samples
        for _ in range(20):
            step *= 0.5
            for candidate in (best - step, best + step):
                d = distance(candidate)
                if d < best_d:
                    best, best_d = candidate, d
        return best

    def hinge_screen_extent(self, joint, project):
        """How wide the joint's circle appears, so an edge-on hinge can be
        reported instead of silently snapping between two positions."""
        circle = self.hinge_circle(joint)
        if circle is None:
            return 0.0
        centre, a, b = circle
        points = [project(vadd(centre, vadd(vmul(a, math.cos(t)),
                                            vmul(b, math.sin(t)))))[:2]
                  for t in (i * math.pi / 6.0 for i in range(12))]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(max(xs) - min(xs), max(ys) - min(ys))

    def hinge_spin(self, joint, angle):
        origin, axis = self.hinge_axis()
        self.rotate_about_axis(self.hinge_set(joint), origin, axis, angle)

    def hinge_drag(self, joint, target_point):
        """Swing `joint` about the axis joining the two anchors.

        A rotation about a fixed axis is an isometry and both anchors lie on
        it, so every bone length survives exactly, including the ones crossing
        from the moving side to the fixed side.
        """
        if joint in self.anchors:
            self.translate(vsub(target_point, self.points[joint]))
            return True
        origin, axis = self.hinge_axis()
        if vlen(axis) < 1e-9:
            return False
        here = vsub(self.points[joint], origin)
        there = vsub(target_point, origin)
        here = vsub(here, vmul(axis, vdot(here, axis)))
        there = vsub(there, vmul(axis, vdot(there, axis)))
        if vlen(here) < 1e-6 or vlen(there) < 1e-6:
            return False        # the joint sits on the axis; nothing to swing
        u, v = vnorm(here), vnorm(there)
        angle = math.atan2(vdot(vcross(u, v), axis),
                           max(-1.0, min(1.0, vdot(u, v))))
        self.rotate_about_axis(self.hinge_set(joint), origin, axis, angle)
        return True

    # -- topology ----------------------------------------------------------
    def tree(self, anchor=None):
        return reroot(self.anchor if anchor is None else anchor)

    def parent_of(self, idx, anchor=None):
        return self.tree(anchor)[0].get(idx, -1)

    def bone_length(self, a, b):
        """Bones are stored by their child in the neck-rooted tree; look them
        up by unordered pair so re-rooting does not lose them."""
        return self.lengths[b] if PARENT.get(b) == a else self.lengths[a]

    def set_bone_length(self, a, b, length):
        key = b if PARENT.get(b) == a else a
        self.lengths[key] = length

    def subtree(self, idx, anchor=None):
        children = self.tree(anchor)[1]
        out, stack = [], [idx]
        while stack:
            j = stack.pop()
            out.append(j)
            stack.extend(children.get(j, ()))
        return out

    # -- symmetry ----------------------------------------------------------
    def sagittal_plane(self):
        """Origin and normal of the body's own plane of symmetry."""
        normal = vnorm(vsub(self.points[5], self.points[2]))
        if vlen(normal) < 1e-6:
            normal = vnorm(vsub(self.points[11], self.points[8]))
        if vlen(normal) < 1e-6:
            normal = (1.0, 0.0, 0.0)
        origin = vmul(vadd(vadd(self.points[2], self.points[5]),
                           vadd(self.points[8], self.points[11])), 0.25)
        return origin, normal

    @staticmethod
    def reflect(point, plane):
        origin, normal = plane
        return vsub(point, vmul(normal, 2.0 * vdot(vsub(point, origin), normal)))

    # -- editing -----------------------------------------------------------
    def translate(self, delta, indices=None):
        for j in (indices if indices is not None else range(len(self.points))):
            self.points[j] = vadd(self.points[j], delta)

    def move_joint(self, idx, target_point, anchor=None, stretch=False):
        """Aim the bone at target_point, carrying the downstream chain along.

        By default the bone keeps its length and only rotates, so a target at
        the wrong distance aims the limb rather than resizing it. Only the
        explicit free-length drag passes stretch=True. Silently resizing was
        how symmetric editing used to stretch a figure a little on every drag.
        """
        parent_idx = self.parent_of(idx, anchor)
        if parent_idx < 0:
            self.translate(vsub(target_point, self.points[idx]))
            return
        parent = self.points[parent_idx]
        old_dir = vsub(self.points[idx], parent)
        new_dir = vsub(target_point, parent)
        if vlen(new_dir) < 1e-9:
            return
        rot = rotation_between(old_dir, new_dir)
        chain = self.subtree(idx, anchor)
        for j in chain:
            self.points[j] = vadd(parent, matvec(rot, vsub(self.points[j], parent)))
        # rotation preserves length; translate the chain for any length change
        fix = vsub(target_point, self.points[idx])
        if stretch and vlen(fix) > 1e-9:
            self.translate(fix, chain)
            self.set_bone_length(parent_idx, idx,
                                 vlen(vsub(target_point, parent)))

    def solve_drag(self, idx, plane_offset, fwd, sign, free_length=False,
                   anchor=None):
        """Map an in-plane drop offset onto the constraint sphere.

        plane_offset : world vector from the parent joint, lying in the view
                       plane, taken from the cursor position.
        fwd          : unit view direction (away from the camera).
        sign         : +1 keeps the joint behind the view plane, -1 in front.
        """
        parent_idx = self.parent_of(idx, anchor)
        if parent_idx < 0:
            return vadd(self.points[idx], plane_offset)
        parent = self.points[parent_idx]
        if free_length:
            return vadd(parent, plane_offset)
        length = self.bone_length(parent_idx, idx)
        d = vlen(plane_offset)
        if d >= length:
            if d < 1e-9:
                return self.points[idx]
            return vadd(parent, vmul(plane_offset, length / d))
        depth = math.sqrt(max(0.0, length * length - d * d))
        return vadd(parent, vadd(plane_offset, vmul(fwd, sign * depth)))

    def flip_depth(self, idx, fwd, anchor=None):
        """Mirror one bone through the view plane, chain included."""
        parent_idx = self.parent_of(idx, anchor)
        if parent_idx < 0:
            return
        parent = self.points[parent_idx]
        old = vsub(self.points[idx], parent)
        new = vsub(old, vmul(fwd, 2.0 * vdot(old, fwd)))
        self.move_joint(idx, vadd(parent, new), anchor)

    def mirror_x(self):
        pts = [(-x, y, z) for (x, y, z) in self.points]
        vis = list(self.visible)
        for a, b in MIRROR_PAIRS:
            pts[a], pts[b] = pts[b], pts[a]
            vis[a], vis[b] = vis[b], vis[a]
        self.points, self.visible = pts, vis
        self.lengths = {c: vlen(vsub(self.points[c], self.points[p]))
                        for p, c in LIMB_SEQ}

    def scale(self, factor):
        root = self.points[ROOT]
        self.points = [vadd(root, vmul(vsub(p, root), factor)) for p in self.points]
        self.lengths = {k: v * factor for k, v in self.lengths.items()}
        self.body_scale *= factor       # keep the depth volumes in proportion

    # -- state -------------------------------------------------------------
    def snapshot(self):
        return (list(self.points), list(self.visible), dict(self.lengths),
                dict(self.body), self.body_scale, list(self.anchors),
                list(self.assets), dict(self.outfit))

    def restore(self, snap):
        self.points, self.visible = list(snap[0]), list(snap[1])
        self.lengths = dict(snap[2])
        if len(snap) > 4:
            self.body, self.body_scale = merge_body(snap[3]), snap[4]
        if len(snap) > 5:
            self.anchors = list(snap[5]) if isinstance(snap[5], list) \
                else ([] if snap[5] == ROOT else [snap[5]])
        if len(snap) > 6:
            self.assets = list(snap[6])
        if len(snap) > 7:
            self.outfit = dict(snap[7])


# ---------------------------------------------------------------------------
# OpenPose-compatible raster export
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Depth map export
#
# The skeleton carries no volume, so the depth pass wraps each bone in a
# tapered capsule (a "round cone": two spheres of different radii and their
# convex hull) and adds a head, torso, hands and feet. Every pixel is then an
# analytic ray/round-cone intersection along +Z with a z-buffer, so limbs
# occlude each other correctly instead of being painted back-to-front.
#
# Output follows the ControlNet depth convention: near = white, far = dark,
# background = black.
# ---------------------------------------------------------------------------

def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


# ---------------------------------------------------------------------------
# Anatomical surface model
#
# Real human models (SMPL, MakeHuman, MHR) deform a scanned template mesh with
# shape parameters. Without a licensed scan template the next best thing is to
# sweep measured cross-sections along the bones: at every station the body gets
# a half width and a half depth, so nothing is ever circular, plus a forward
# offset so masses sit where anatomy puts them (calf behind the tibia, chin in
# front of the neck, buttocks behind the pelvis).
#
# Each station becomes an oriented ellipsoid. Stations are spaced far closer
# than their radii, so their union is the swept surface rather than a string of
# beads. Parts are unioned hard internally and blended with a smooth minimum
# against each other, which fillets shoulder into torso and thigh into pelvis
# the way flesh actually does.
#
# Profiles are (t, half width, half depth, forward offset), t running 0..1 along
# the bone, measurements in centimetres for an average adult male; presets scale
# them by girth factors.
# ---------------------------------------------------------------------------

P_UPPER_ARM = [(0.00, 5.3, 5.5, 0.0), (0.22, 5.2, 5.7, 0.3),
               (0.58, 4.6, 4.9, 0.0), (1.00, 3.7, 4.1, -0.3)]
P_FOREARM = [(0.00, 4.0, 4.4, 0.0), (0.18, 4.3, 4.8, 0.1),
             (0.58, 3.4, 3.7, 0.0), (1.00, 2.6, 3.0, 0.0)]
P_THIGH = [(0.00, 8.4, 8.2, 0.2), (0.18, 8.2, 8.6, -0.4),
           (0.60, 6.6, 6.9, -0.3), (1.00, 5.3, 5.6, 0.0)]
P_CALF = [(0.00, 5.2, 5.6, 0.0), (0.22, 5.1, 6.2, -1.7),
          (0.62, 3.6, 4.1, -0.9), (1.00, 2.7, 3.2, 0.2)]
P_NECK = [(0.00, 6.6, 7.0, -0.6), (0.55, 6.0, 6.2, -0.2), (1.00, 5.5, 5.7, 0.0)]
# head runs chin (0) to crown (1); the chin sits forward of the ear axis
P_HEAD = [(0.00, 2.0, 2.8, 3.6), (0.05, 3.6, 4.8, 3.3), (0.13, 5.2, 6.8, 2.6),
          (0.27, 6.6, 8.4, 1.4), (0.42, 7.4, 9.2, 0.4), (0.57, 7.6, 9.1, 0.0),
          (0.74, 7.0, 7.9, -0.4), (0.86, 6.1, 6.8, -0.8),
          (0.94, 4.8, 5.3, -0.9), (0.98, 3.4, 3.8, -1.0),
          (1.00, 1.8, 2.0, -1.0)]
# hand: thin across the palm, wide front to back, as it hangs beside the thigh
P_HAND = [(0.00, 2.4, 4.2, 0.0), (0.35, 2.3, 4.7, 0.0),
          (0.75, 2.0, 4.1, 0.0), (1.00, 1.5, 2.6, 0.0)]
# foot: swept heel to toe, "depth" is thickness above the sole
P_FOOT = [(0.00, 3.6, 4.2, 0.0), (0.30, 4.2, 3.8, 0.0),
          (0.72, 3.8, 2.7, 0.0), (1.00, 2.6, 1.9, 0.0)]


def torso_profile(body):
    """Shoulder line to hip line, extended past both. Anchored on the preset's
    three measured cross-sections and shaped by the ribcage/waist/pelvis
    relationship that is common to every human torso."""
    cw, cd = body["chest"]
    ww, wd = body["waist"]
    pw, pd = body["pelvis"]
    belly = body.get("belly", 0.0)
    return [
        (-0.085, 0.46 * cw, 0.56 * cd, -0.7),    # base of the neck
        (-0.035, 0.84 * cw, 0.84 * cd, -0.2),    # trapezius sloping outwards
        (0.075, 0.99 * cw, 0.97 * cd, 0.3),      # armpit line
        (0.210, 1.00 * cw, 1.00 * cd, 0.5),      # chest / bust line
        (0.420, 0.88 * cw, 0.92 * cd, 0.3 + 0.35 * belly),
        # the narrowest station is the tenth rib, and it sits higher on a woman:
        # 0.569 of the way from shoulder to hip against 0.593 on a man
        (body.get("waist_t", 0.593), ww, wd, belly),
        (0.780, 0.90 * pw, 0.94 * pd, 0.3 * belly),
        (0.930, pw, pd, -0.5),                   # hips
        (1.090, 0.95 * pw, 1.01 * pd, -1.9),     # seat
        (1.190, 0.78 * pw, 0.88 * pd, -2.6),     # crotch
    ]


def _catmull(p0, p1, p2, p3, s):
    return 0.5 * ((2.0 * p1) + (-p0 + p2) * s
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * s * s
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * s * s * s)


def sample_profile(profile, t):
    """Catmull-Rom through the control stations, so the silhouette curves
    instead of running between straight taper segments."""
    n = len(profile)
    if t <= profile[0][0]:
        return profile[0][1:]
    if t >= profile[-1][0]:
        return profile[-1][1:]
    i = 0
    while i < n - 2 and t > profile[i + 1][0]:
        i += 1
    t0, t1 = profile[i][0], profile[i + 1][0]
    s = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    p0, p1 = profile[max(0, i - 1)], profile[i]
    p2, p3 = profile[i + 1], profile[min(n - 1, i + 2)]
    return tuple(_catmull(p0[j], p1[j], p2[j], p3[j], s) for j in (1, 2, 3))


def carry_frame(ref, axis):
    """Carry a cross-section frame from one bone direction onto another.

    `ref` is (reference axis, reference forward) - the frame the bone inherits
    its roll from, forward perpendicular to the reference axis. The frame is
    swung onto `axis` by the *minimal* rotation between the two directions, so
    the bone gains no twist about itself that the reference did not already
    have. Returns (forward, side), both perpendicular to `axis`.

    Projecting a fixed reference onto the bone instead - which is what this
    used to do - is discontinuous. `facing - axis * (facing . axis)` vanishes
    the moment a bone points along the body's forward direction (a reach, a
    sitting thigh, a kick), so the cross-section snapped 90 degrees there and
    flipped a full 180 either side of it: the calf mass jumped from behind the
    tibia to in front of it and the profile's width and depth swapped over. In
    plain diagonal poses it was 45 degrees out. That is what wrung the limbs in
    the depth map. A swing has no such singularity: it only degenerates when
    the bone points exactly opposite its reference, which down a limb chain
    means folded back on itself.
    """
    ref_axis, ref_fwd = ref
    fwd = matvec(rotation_between(ref_axis, axis), ref_fwd)
    fwd = vsub(fwd, vmul(axis, vdot(fwd, axis)))
    if vlen(fwd) < 1e-6:
        fwd = any_perpendicular(axis)
    fwd = vnorm(fwd)
    return fwd, vnorm(vcross(axis, fwd))


def carry_chain(ref, *joints):
    """One (axis, forward) frame per bone along a chain of joints.

    Each bone inherits the roll of the one above it, so a bent elbow carries
    the forearm round with it instead of letting it pick its own orientation.
    A zero-length bone passes its parent's frame straight through rather than
    restarting the chain from the torso.
    """
    out = []
    for a, b in zip(joints, joints[1:]):
        axis = vnorm(vsub(b, a))
        if vlen(axis) < 1e-9:
            out.append(ref)
            continue
        ref = (axis, carry_frame(ref, axis)[0])
        out.append(ref)
    return out


def _tube(a, b, ref, profile, scale_w=1.0, scale_d=1.0,
          spacing=0.45, round_start=True, round_end=True, coarsen=1.0):
    """Sweep a profile from a to b as a chain of oriented ellipsoids.

    `ref` is the (axis, forward) frame this bone takes its roll from; see
    `carry_frame`. Passing the bone's own axis as the reference axis means
    "use this forward as it stands".
    """
    span = vsub(b, a)
    length = vlen(span)
    if length < 1e-6:
        return []
    axis = vmul(span, 1.0 / length)
    fwd, side = carry_frame(ref, axis)

    t0, t1 = profile[0][0], profile[-1][0]
    steps = max(5, int(abs(t1 - t0) * length / (spacing * coarsen)) + 1)
    half = 0.55 * abs(t1 - t0) * length / (steps - 1.0)
    out = []
    for i in range(steps):
        t = t0 + (t1 - t0) * i / (steps - 1.0)
        w, d, off = sample_profile(profile, t)
        w, d = max(0.2, w * scale_w), max(0.2, d * scale_d)
        centre = vadd(vadd(a, vmul(axis, t * length)), vmul(fwd, off))
        # a stack of touching elliptical slabs sweeps the profile exactly;
        # ellipsoid stations would lose width between stations and scallop
        out.append(("slab", centre, (side, fwd, axis), (w, d, half)))
    for at_end, rounded in ((False, round_start), (True, round_end)):
        if not rounded:
            continue
        t = t1 if at_end else t0
        w, d, off = sample_profile(profile, t)
        w, d = max(0.2, w * scale_w), max(0.2, d * scale_d)
        centre = vadd(vadd(a, vmul(axis, t * length)), vmul(fwd, off))
        out.append(("ball", centre, (side, fwd, axis), (w, d, min(w, d))))
    return out


def _blob(centre, axes, radii):
    return [("ball", centre, axes, radii)]


def body_segments(skeleton, respect_visibility=True):
    """Every swept part of the figure, before any of it is turned into solids.

    Returns (segments, frame). A segment is a dict carrying what `_tube` needs:
    where it runs from and to, the frame it takes its roll from, the profile it
    sweeps and how much to scale that profile by.

    Pulled out of `body_parts` because a garment is the body's own sweep,
    clipped to a stretch of it and padded outward - which is what makes a
    sleeve fit whatever preset and whatever pose it finds, with nothing to fit
    and nothing to drift. `wearables` reads this table; so does the body.
    """
    P = skeleton.points
    ids = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
    B = getattr(skeleton, "body", None)
    if B is None or "deltoid" not in B:      # scene from an older version
        B = merge_body(B)
        skeleton.body = B

    def pt(n):
        return P[ids[n]]

    def vis(*names):
        return True if not respect_visibility else all(
            skeleton.visible[ids[n]] for n in names)

    r_sh, l_sh = pt("r_shoulder"), pt("l_shoulder")
    r_hip, l_hip = pt("r_hip"), pt("l_hip")
    sh_mid = vmul(vadd(r_sh, l_sh), 0.5)
    hip_mid = vmul(vadd(r_hip, l_hip), 0.5)
    side = vnorm(vsub(l_sh, r_sh))
    up_t = vnorm(vsub(sh_mid, hip_mid))
    if vlen(side) < 1e-6:
        side = (1.0, 0.0, 0.0)
    if vlen(up_t) < 1e-6:
        up_t = (0.0, 1.0, 0.0)
    facing = vnorm(vcross(side, up_t))
    if vlen(facing) < 1e-6:
        facing = (0.0, 0.0, 1.0)

    # Frames. Every swept part takes its roll from the one above it, ending
    # at the torso, so a limb keeps the orientation it has in the rest pose
    # however it is posed. See `carry_frame` for why deriving each bone's
    # frame from `facing` on its own twisted them instead.
    down = vmul(up_t, -1.0)
    trunk_ref = (vnorm(vsub(hip_mid, sh_mid)), facing)

    segments = {}

    def add(name, a, b, ref, profile, scale_w=1.0, scale_d=1.0, **kw):
        segments[name] = dict(a=a, b=b, ref=ref, profile=profile,
                              scale_w=scale_w, scale_d=scale_d, **kw)

    if vis("r_shoulder", "l_shoulder", "r_hip", "l_hip"):
        add("torso", sh_mid, hip_mid, trunk_ref, torso_profile(B),
            round_start=False, round_end=False)

    girth = (B["arm_girth"], B["forearm_girth"], B["thigh_girth"], B["calf_girth"])
    for sd in ("r", "l"):
        sh, el, wr = sd + "_shoulder", sd + "_elbow", sd + "_wrist"
        hp, kn, an = sd + "_hip", sd + "_knee", sd + "_ankle"
        # The chains are built whole, before anything is emitted: a hidden
        # upper arm must not change how the forearm below it is rolled.
        # The rest pose hangs both limbs straight down, so both start from the
        # torso's own forward with `down` as the reference axis.
        upper, fore = carry_chain((down, facing), pt(sh), pt(el), pt(wr))
        thigh, calf = carry_chain((down, facing), pt(hp), pt(kn), pt(an))
        if vis(sh, el):
            add(sd + "_upper_arm", pt(sh), pt(el), upper, P_UPPER_ARM,
                girth[0], girth[0])
        if vis(el, wr):
            add(sd + "_forearm", pt(el), pt(wr), fore, P_FOREARM,
                girth[1], girth[1])
        if vis(wr, el):
            # the hand runs on along the forearm, so it shares its frame: the
            # palm keeps facing the way it does with the arm hanging
            d = vnorm(vsub(pt(wr), pt(el)))
            add(sd + "_hand", pt(wr), vadd(pt(wr), vmul(d, 17.0 * B["hand"])),
                fore, P_HAND, B["hand"], B["hand"])
        if vis(hp, kn):
            add(sd + "_thigh", pt(hp), pt(kn), thigh, P_THIGH,
                girth[2], girth[2])
        if vis(kn, an):
            add(sd + "_calf", pt(kn), pt(an), calf, P_CALF, girth[3], girth[3])
        if vis(an):
            drop = vnorm(vsub(pt(an), pt(kn))) if vis(kn) else down
            sole = vadd(pt(an), vmul(drop, 3.2 * B["foot"]))
            heel = vadd(sole, vmul(facing, -6.0 * B["foot"]))
            toe = vadd(sole, vmul(facing, 19.0 * B["foot"]))
            # the foot turns off the shin, so its thickness stays across the
            # sole however the leg is posed
            add(sd + "_foot", heel, toe, calf, P_FOOT, B["foot"], B["foot"])

    neck = pt("neck")
    if vis("r_ear", "l_ear"):
        ear_mid = _lerp(pt("r_ear"), pt("l_ear"), 0.5)
    else:
        ear_mid = vadd(vsub(pt("nose"), vmul(facing, 5.0)), vmul(up_t, 3.0))
    head_axis = vnorm(vsub(ear_mid, neck))
    if vlen(head_axis) < 1e-6:
        head_axis = up_t
    face = vnorm(vsub(pt("nose"), ear_mid)) if vis("nose") else facing
    if vlen(face) < 1e-6:
        face = facing
    face = vnorm(vsub(face, vmul(head_axis, vdot(face, head_axis))))
    if vlen(face) < 1e-6:
        face = facing

    h = B["head"]
    nk = B["neck_girth"]
    add("neck", vsub(neck, vmul(head_axis, 3.0)),
        vsub(ear_mid, vmul(head_axis, 8.0 * h)), (up_t, facing), P_NECK,
        nk, nk, round_start=False)
    # the skull has a forward of its own - the face - so it is its own
    # reference rather than inheriting the neck's
    add("head", vsub(ear_mid, vmul(head_axis, 10.6 * h)),
        vadd(ear_mid, vmul(head_axis, 10.4 * h)), (head_axis, face), P_HEAD,
        h * B.get("jaw", 1.0), h, round_start=False, round_end=False)

    frame = {"side": side, "up": up_t, "facing": facing, "down": down,
             "sh_mid": sh_mid, "hip_mid": hip_mid, "neck": neck,
             "ear_mid": ear_mid, "head_axis": head_axis, "face": face,
             "head_side": vnorm(vcross(head_axis, face)), "body": B}
    return segments, frame


def sweep(segment, coarsen=1.0):
    """One segment as solids."""
    return _tube(segment["a"], segment["b"], segment["ref"],
                 segment["profile"], segment["scale_w"], segment["scale_d"],
                 coarsen=coarsen,
                 round_start=segment.get("round_start", True),
                 round_end=segment.get("round_end", True))


def body_parts(skeleton, thickness=1.0, respect_visibility=True, coarsen=1.0):
    """Posed body as a list of parts; each part is a list of oriented
    ellipsoids (centre, orthonormal axes, radii) in world space.

    Anything the figure is wearing comes with it, so the depth export, the
    viewport preview and the silhouette all get clothes for free rather than
    each having to remember to ask.
    """
    P = skeleton.points
    ids = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
    segments, frame = body_segments(skeleton, respect_visibility)
    B = frame["body"]
    side, up_t, facing = frame["side"], frame["up"], frame["facing"]
    sh_mid, hip_mid = frame["sh_mid"], frame["hip_mid"]
    k = thickness * getattr(skeleton, "body_scale", 1.0)

    def pt(n):
        return P[ids[n]]

    def vis(*names):
        return True if not respect_visibility else all(
            skeleton.visible[ids[n]] for n in names)

    parts = []

    def emit(part):
        if part:
            # radii are in centimetres; k scales the whole figure
            parts.append([(kind, c, ax, (r[0] * k, r[1] * k, r[2] * k))
                          for kind, c, ax, r in part])

    if "torso" in segments:
        emit(sweep(segments["torso"], coarsen))

    # bust: two masses on the front of the ribcage
    bust = B.get("bust")
    if bust and vis("r_shoulder", "l_shoulder"):
        bw, bd, bh, bt, sep = bust
        centre = _lerp(sh_mid, hip_mid, bt)
        depth = B["chest"][1]
        for sgn in (-1.0, 1.0):
            c = vadd(vadd(centre, vmul(side, sgn * sep)),
                     vmul(facing, depth * 0.74))
            emit(_blob(c, (side, facing, up_t), (bw, bd, bh)))

    # buttocks
    glute = B.get("glute")
    if glute and vis("r_hip", "l_hip"):
        gw, gd, gh, gt, gsep = glute
        centre = _lerp(sh_mid, hip_mid, gt)
        for sgn in (-1.0, 1.0):
            c = vadd(vadd(centre, vmul(side, sgn * gsep)),
                     vmul(facing, -B["pelvis"][1] * 0.45))
            emit(_blob(c, (side, facing, up_t), (gw, gd, gh)))

    # deltoids
    dw, dd, dh = B["deltoid"]
    for shoulder in ("r_shoulder", "l_shoulder"):
        if vis("neck", shoulder):
            emit(_blob(vadd(pt(shoulder), vmul(up_t, -1.6)),
                       (side, facing, up_t), (dw, dd, dh)))

    for name, segment in segments.items():
        if name in ("torso", "head"):
            continue
        emit(sweep(segment, coarsen))

    if "head" in segments:
        head = sweep(segments["head"], coarsen)
        h = B["head"]
        head_side, face = frame["head_side"], frame["face"]
        head_axis = frame["head_axis"]
        if vis("nose"):
            head += _blob(vadd(pt("nose"), vmul(face, -1.4 * h)),
                          (head_side, face, head_axis),
                          (1.9 * h, 3.0 * h, 2.4 * h))
        for ear in ("r_ear", "l_ear"):
            if vis(ear):
                head += _blob(pt(ear), (head_side, face, head_axis),
                              (1.3 * h, 2.6 * h, 3.2 * h))
        emit(head)

    import wearables                  # deferred: it imports from this module
    for part in wearables.parts(skeleton, segments, frame, coarsen):
        emit(part)

    return parts


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
            span = None
        if span is not None:
            img[covered] = far + (near - far) * (hi - zbuf[covered]) / span
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"), mode="L")


# ---------------------------------------------------------------------------
# Framing and export
#
# The pose PNG and the depth map have to line up pixel for pixel, which means
# both have to be framed the same way. That framing lives here rather than on
# the editor window so a headless run produces the identical image: the window
# passes its canvas size, a headless run passes the camera's.
# ---------------------------------------------------------------------------

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
        points = {name: figure.points[i]
                  for i, name in enumerate(KEYPOINT_NAMES)}
        # The rest keypoints of this same body, so the rig's head is turned
        # by how far the head has *moved* rather than aimed at an absolute
        # direction - there is no keypoint on the skull to aim at.
        rest = build_rest_points(figure.body)
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


# ---------------------------------------------------------------------------
# Scene serialisation
# ---------------------------------------------------------------------------

def scene_to_dict(figures, camera, points_list, out_w, out_h, props=()):
    """Scene with any number of people. people[] is OpenPose's own format;
    figures[] carries what the editor needs to reload the scene, and objects[]
    the props standing in it."""
    if isinstance(figures, Skeleton):
        figures, points_list = [figures], [points_list]

    people, saved = [], []
    for skeleton, points2d in zip(figures, points_list):
        flat = []
        for i in range(len(KEYPOINT_NAMES)):
            if skeleton.visible[i]:
                flat.extend([round(points2d[i][0], 3),
                             round(points2d[i][1], 3), 1.0])
            else:
                flat.extend([0.0, 0.0, 0.0])
        people.append({"person_id": [len(people)], "pose_keypoints_2d": flat})
        saved.append({
            "pose_3d": {n: [round(v, 6) for v in skeleton.points[i]]
                        for i, n in enumerate(KEYPOINT_NAMES)},
            "visible": list(skeleton.visible),
            "assets": list(getattr(skeleton, "assets", [])),
            "outfit": dict(getattr(skeleton, "outfit", {}) or {}),
            "body": {k: v for k, v in skeleton.body.items()},
            "body_scale": skeleton.body_scale,
        })
    skeleton = figures[0]
    points2d = points_list[0]
    flat = people[0]["pose_keypoints_2d"]
    return {
        "version": 1,
        "producer": "openpose3d_editor",
        "keypoint_format": "COCO-18",
        "canvas_width": out_w,
        "canvas_height": out_h,
        "people": people,
        "figures": saved,
        "pose_3d": {n: [round(v, 4) for v in skeleton.points[i]]
                    for i, n in enumerate(KEYPOINT_NAMES)},
        "visible": list(skeleton.visible),
        "bone_lengths": {KEYPOINT_NAMES[c]: round(v, 4)
                         for c, v in skeleton.lengths.items()},
        "body": {k: v for k, v in skeleton.body.items()},
        "body_scale": skeleton.body_scale,
        "objects": [dict(prop) for prop in props],
        "camera": {"yaw": camera.yaw, "pitch": camera.pitch,
                   "zoom": camera.zoom, "target": list(camera.target)},
    }


def scene_load(data, camera):
    """Rebuild every figure in a scene file. Understands single-figure scenes
    written by earlier versions."""
    cam = data.get("camera") or {}
    camera.yaw = float(cam.get("yaw", camera.yaw))
    camera.pitch = float(cam.get("pitch", camera.pitch))
    camera.zoom = float(cam.get("zoom", camera.zoom))
    camera.target = tuple(cam.get("target", camera.target))
    entries = data.get("figures")
    if not entries:                      # one figure, stored at the top level
        entries = [data]
    figures = []
    for entry in entries:
        skeleton = Skeleton()
        scene_from_dict({"pose_3d": entry.get("pose_3d"),
                         "visible": entry.get("visible"),
                         "body": entry.get("body"),
                         "body_scale": entry.get("body_scale", 1.0)},
                        skeleton, camera)
        skeleton.assets = list(entry.get("assets", []))
        skeleton.outfit = dict(entry.get("outfit") or {})
        figures.append(skeleton)
    return figures or [Skeleton()]


def scene_objects(data):
    """The props in a scene file, rebuilt through `props.make`.

    Separate from `scene_load` rather than returned alongside the figures: a
    scene written before objects existed simply has none, and every caller
    that only wants the people keeps working unchanged.
    """
    out = []
    for entry in data.get("objects") or ():
        if not isinstance(entry, dict):
            continue
        out.append(props_module.make(entry.get("shape", "box"),
                                     entry.get("size"),
                                     entry.get("position", (0.0, 0.0, 0.0)),
                                     entry.get("yaw", 0.0)))
    return out


def scene_from_dict(data, skeleton, camera):
    pose = data.get("pose_3d") or {}
    if pose:
        skeleton.points = [tuple(pose.get(n, REST_POSE[n])) for n in KEYPOINT_NAMES]
    vis = data.get("visible")
    if vis and len(vis) == len(KEYPOINT_NAMES):
        skeleton.visible = [bool(v) for v in vis]
    skeleton.lengths = {c: vlen(vsub(skeleton.points[c], skeleton.points[p]))
                        for p, c in LIMB_SEQ}
    body = data.get("body")
    if body:
        skeleton.body = merge_body(body)
        skeleton.body_scale = float(data.get("body_scale", 1.0))
    cam = data.get("camera") or {}
    camera.yaw = float(cam.get("yaw", camera.yaw))
    camera.pitch = float(cam.get("pitch", camera.pitch))
    camera.zoom = float(cam.get("zoom", camera.zoom))
    camera.target = tuple(cam.get("target", camera.target))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

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
        self.show_body = True
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
        """Pan and zoom so every figure sits inside the export frame."""
        right, up, _ = self.camera.basis()
        points = [p for f in self.figures for p in f.points]
        across = [vdot(vsub(p, self.camera.target), right) for p in points]
        along = [vdot(vsub(p, self.camera.target), up) for p in points]
        cx = (min(across) + max(across)) / 2.0
        cy = (min(along) + max(along)) / 2.0
        self.camera.target = vadd(self.camera.target,
                                  vadd(vmul(right, cx), vmul(up, cy)))
        x0, y0, x1, y1 = self.frame_rect()
        wide = max(1.0, max(across) - min(across)) * 1.18   # margin for flesh
        tall = max(1.0, max(along) - min(along)) * 1.12
        self.camera.zoom = max(0.3, min(40.0, min((x1 - x0) / wide,
                                                  (y1 - y0) / tall)))
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

        column = tk.Frame(self.root, bg=PANEL, width=262)
        column.pack(side="right", fill="y")
        column.pack_propagate(False)
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

        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0,
                                width=900, height=760)

        # ---- small widget vocabulary -------------------------------------
        def section(title, opened=True):
            """A collapsible group. Everything folds, so the panel is a list of
            headings until you open what you need instead of one long column
            that has to be scrolled past."""
            head = tk.Frame(panel, bg=PANEL, cursor="hand2")
            head.pack(fill="x", pady=(9, 0))
            tk.Frame(panel, bg=EDGE, height=1).pack(fill="x", padx=12)
            chevron = tk.Label(head, text="", bg=PANEL, fg=MUTED,
                               font=("TkDefaultFont", 7))
            chevron.pack(side="right", padx=(0, 14))
            tk.Label(head, text=title.upper(), bg=PANEL, fg=MUTED, anchor="w",
                     font=("TkDefaultFont", 8, "bold")).pack(side="left", padx=14)
            body = tk.Frame(panel, bg=PANEL)
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
                            ("top", "Top"), ("over", "Over"),
                            ("bottom", "Bottom"), ("shoes", "Feet")):
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
        switch(body, "Body preview (B)", "show_body")
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

        self.ortho_frame = tk.Frame(self.root, bg=BG, height=196)
        self.ortho_frame.pack(side="bottom", fill="x")
        self.ortho_frame.pack_propagate(False)
        tk.Frame(self.ortho_frame, bg=EDGE, height=1).pack(fill="x")
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
            holder.pack(side="left", fill="both", expand=True, padx=1)
            tk.Label(holder, text=label.upper(), bg=BG, fg=MUTED, anchor="w",
                     font=("TkDefaultFont", 7, "bold")).pack(fill="x", padx=6,
                                                             pady=(3, 1))
            small = tk.Canvas(holder, bg=VIEW_BG, highlightthickness=0,
                              width=10, height=168)   # expand shares the row
            small.pack(fill="both", expand=True)
            camera = Camera(300, 168)
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

    def _refresh_scroll(self):
        self.panel_canvas.update_idletasks()
        self.panel_canvas.configure(scrollregion=self.panel_canvas.bbox("all"))

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
                self.ortho_frame.pack(side="bottom", fill="x",
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
        self.camera.yaw, self.camera.pitch = camera.yaw, camera.pitch
        self.camera.target, self.camera.zoom = camera.target, camera.zoom
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
        self.status.set("Body preview on: shading approximates the depth map."
                        if self.show_body else "Body preview off.")

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
        """Safe frame matching the export aspect ratio."""
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
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
        }
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
                self.active, self.selected, self.show_body, self._thickness())

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
            c.create_line(screen[a][0], screen[a][1], screen[b][0], screen[b][1],
                          fill=self.shade(self.figure_palette(f)[i], factor),
                          width=(3 if compact else 7) if self.show_body
                          else (5 if compact else 9), capstyle="round")

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
