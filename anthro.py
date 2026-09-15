#!/usr/bin/env python3
"""Where a figure's measurements come from.

ANSUR II - the 2012 US Army survey, 4082 men and 1986 women, public since
2017 - as fractions of stature, and the derivation from those to the bone
lengths and rest pose the editor uses. `tests/test_proportions.py` repeats the
reference values and fails when the table and the figure part company.
"""

from __future__ import annotations

import math


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
           "nose": (0.0, nose_up, 5.0 * h)}
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
        pts[side + "_hip"], pts[side + "_knee"], pts[side + "_ankle"] = hp, kn, an
    return pts


REST_POSE = build_rest_points(preset_params(DEFAULT_PRESET))
