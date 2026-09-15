"""The figure's proportions, against the survey they claim to come from.

ANSUR II - the 2012 US Army anthropometric survey, 4082 men and 1986 women,
93 measurements each, public since 2017. The reference values below are means
computed from the released CSVs; they are repeated here so the check needs no
download, and the docstring of each says which measurement it is so a wrong
landmark cannot hide behind a right-looking number.

What this is for: every constant in `ANSUR` is a claim about a real body, and
a claim is worth nothing without something that fails when it stops being
true. The set this replaced - Drillis & Contini's 1966 constants - put the
trochanter at 0.530 of stature against 0.513 measured, three centimetres of
leg taken off the torso of a 175 cm figure, and nothing caught it because
nothing compared it with anything.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openpose3d_editor import (ANSUR, BODY_PRESETS, KEYPOINT_NAMES, Skeleton,
                               build_rest_points, derive_proportions,
                               preset_params, vlen, vsub)

ok = True


def check(label, condition, extra=""):
    global ok
    ok = ok and bool(condition)
    print(("PASS " if condition else "FAIL ") + label
          + ("  " + extra if extra else ""))


def close(got, want, tol, label, units="of stature"):
    check(label, abs(got - want) <= tol,
          "%.4f vs %.4f %s (tolerance %.4f)" % (got, want, units, tol))


# Means over the released records, as fractions of stature unless marked cm.
REFERENCE = {
    "male": {
        "n": 4082, "stature_cm": 175.62,
        "acromial_height": 0.8203,            # shoulder keypoint height
        "trochanterion_height": 0.5128,       # hip joint centre proxy
        "femoral_epicondyle_height": 0.2799,  # knee joint centre proxy
        "lateral_malleolus_height": 0.0415,   # ankle joint centre proxy
        "biacromial_breadth": 0.2368,         # shoulder to shoulder
        "span": 1.0330,                       # fingertip to fingertip
        "hand_length": 0.1101,
        "chest_breadth_cm": 28.94, "chest_depth_cm": 25.38,
        "hip_breadth_cm": 34.57, "buttock_depth_cm": 24.58,
    },
    "female": {
        "n": 1986, "stature_cm": 162.85,
        "acromial_height": 0.8198,
        "trochanterion_height": 0.5190,
        "femoral_epicondyle_height": 0.2860,
        "lateral_malleolus_height": 0.0385,
        "biacromial_breadth": 0.2244,
        "span": 1.0195,
        "hand_length": 0.1112,
        "chest_breadth_cm": 26.93, "chest_depth_cm": 24.74,
        "hip_breadth_cm": 35.38, "buttock_depth_cm": 23.29,
    },
}

ids = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

for sex, ref in REFERENCE.items():
    stature = ref["stature_cm"]
    print("\n-- %s, %d subjects, mean stature %.1f cm" % (sex, ref["n"], stature))
    table = ANSUR[sex]

    close(table["shoulder_height"], ref["acromial_height"], 0.002,
          "the shoulder sits at the acromion")
    close(table["hip_height"], ref["trochanterion_height"], 0.002,
          "the hip at the trochanter")
    close(table["knee_height"], ref["femoral_epicondyle_height"], 0.002,
          "the knee at the femoral epicondyle")
    close(table["ankle_height"], ref["lateral_malleolus_height"], 0.002,
          "the ankle at the lateral malleolus")
    close(2.0 * table["shoulder_w"], ref["biacromial_breadth"], 0.002,
          "the shoulders are a biacromial breadth apart")
    close(table["hand"], ref["hand_length"], 0.002, "the hand is a hand long")

    body = derive_proportions(stature, sex)
    points = build_rest_points(body)

    # the figure that comes out, not just the table that went in
    def at(name):
        return points[name]

    # Measured on the figure that comes out, not by adding table entries: the
    # arm hangs from the glenohumeral joint, not from the acromion keypoint,
    # and adding shoulder_w to the segments counts the 0.9 cm between them as
    # arm. That is how a sum can close while the elbow sits 4 cm high.
    span = 2.0 * (body["gh_w"] + body["upper_arm"] + body["forearm"]
                  + table["hand"] * stature)
    close(span / stature, ref["span"], 0.004,
          "the arm span closes on the measured span")
    # and the shoulder keypoint is still the acromion, outboard of that joint
    close(2.0 * abs(at("l_shoulder")[0]) / stature, ref["biacromial_breadth"],
          0.002, "while the shoulder keypoint stays at the acromion")

    floor = at("l_ankle")[1] - table["ankle_height"] * stature
    hip = at("l_hip")[1] - floor
    close(hip / stature, ref["trochanterion_height"], 0.002,
          "the posed hip stands at the measured height")
    knee = at("l_knee")[1] - floor
    close(knee / stature, ref["femoral_epicondyle_height"], 0.003,
          "and the knee at its own")

    # thigh + shank + ankle must land on the floor, or a standing figure floats
    leg = (vlen(vsub(at("l_hip"), at("l_knee")))
           + vlen(vsub(at("l_knee"), at("l_ankle")))
           + table["ankle_height"] * stature)
    close(leg / stature, ref["trochanterion_height"], 0.002,
          "the leg segments close on the hip height")

    shoulder = at("l_shoulder")[1] - floor
    close(shoulder / stature, ref["acromial_height"], 0.006,
          "the shoulder line stands at acromial height")

    # cross-sections. Chest is the one the survey can settle: chest height
    # lands at t = 0.28 (male) / 0.33 (female) along shoulder-to-hip and the
    # profile's chest station is t = 0.21, the same part of the ribcage.
    preset = "Male, average" if sex == "male" else "Female, average"
    full = preset_params(preset)
    scale = full["stature"] / stature       # the preset is not the survey mean
    close(2.0 * full["chest"][0] / scale, ref["chest_breadth_cm"], 1.2,
          "%s is a measured chest wide" % preset, "cm")
    close(2.0 * full["chest"][1] / scale, ref["chest_depth_cm"], 1.2,
          "and a measured chest deep", "cm")
    close(2.0 * full["pelvis"][0] / scale, ref["hip_breadth_cm"], 1.5,
          "and a measured hip wide", "cm")
    close(2.0 * full["pelvis"][1] / scale, ref["buttock_depth_cm"], 1.5,
          "and a measured seat deep", "cm")

# The waist deliberately has no check. ANSUR measures breadth and depth at
# omphalion, the navel, which lands at t = 0.71 along shoulder-to-hip; the
# profile's waist station is t = 0.59, which is the tenth rib - the narrowest
# point, and a different circumference. Checking one against the other would
# assert that a body is the same width in two places it is not.
print("\n-- the waist is not checked: ANSUR measures it at the navel, the "
      "profile\n   uses the tenth rib, and they are not the same girth")

# every preset has to stay a plausible human, not only the two measured ones
print()
for name in BODY_PRESETS:
    body = preset_params(name)
    skeleton = Skeleton(body)
    stature = body["stature"]
    height = (skeleton.points[ids["l_ankle"]][1]
              - ANSUR[body["sex"]]["ankle_height"] * stature)
    top = skeleton.points[ids["l_ear"]][1]
    built = (top - height) + ANSUR[body["sex"]]["ear_drop"] * stature
    check("%-16s stands %.0f cm tall, asked for %.0f" % (name, built, stature),
          abs(built - stature) < 0.05 * stature)
    chest, waist = body["chest"], body["waist"]
    check("%-16s is deeper than it is half wide" % name,
          0.55 < chest[1] / chest[0] < 1.35,
          "chest %.0f x %.0f cm" % (2 * chest[0], 2 * chest[1]))
    check("%-16s has a waist no wider than its hips" % name,
          waist[0] <= body["pelvis"][0] * 1.05,
          "waist %.0f, hips %.0f cm" % (2 * waist[0], 2 * body["pelvis"][0]))


# The web prototype keeps its own copy of these numbers in JavaScript, which
# CLAUDE.md lists as a known weak spot: it drifts. Read it back and say so.
# It had drifted by a third of the child's torso - 33 cm against 44 - and by
# every arm length, because nothing compared them.
import re

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "web", "pose3d.mjs")
KEYS = {"sw": "shoulder_w", "sd": "shoulder_drop", "hw": "hip_w",
        "tl": "torso_len", "ua": "upper_arm", "fa": "forearm",
        "th": "thigh", "ca": "calf", "head": "head"}
PAIRS = {"chest": "chest", "waist": "waist", "pelvis": "pelvis"}

print()
text = io.open(WEB, encoding="utf-8").read()
table = text[text.index("export const PRESETS = {"):]
table = table[:table.index("\n};")]
found = 0
for name, body in re.findall(r'"([^"]+)":\s*\{(.*?)\},', table, re.S):
    if name not in BODY_PRESETS:
        check("web preset %r exists in the editor too" % name, False)
        continue
    found += 1
    want = preset_params(name)
    worst = (None, 0.0)
    for short, key in KEYS.items():
        m = re.search(r"\b%s:\s*([-\d.]+)" % short, body)
        if not m:
            continue
        gap = abs(float(m.group(1)) - want[key])
        if gap > worst[1]:
            worst = ("%s %s (%.2f vs %.2f)" % (name, key, float(m.group(1)),
                                               want[key]), gap)
    for short, key in PAIRS.items():
        m = re.search(r"\b%s:\s*\[([-\d.]+)\s*,\s*([-\d.]+)\]" % short, body)
        if not m:
            continue
        for i in (0, 1):
            gap = abs(float(m.group(i + 1)) - want[key][i])
            if gap > worst[1]:
                worst = ("%s %s[%d] (%.2f vs %.2f)"
                         % (name, key, i, float(m.group(i + 1)), want[key][i]),
                         gap)
    check("web %-16s matches the Python" % name, worst[1] < 0.06,
          "" if worst[0] is None else "worst: %s" % worst[0])
check("the web prototype carries presets at all", found >= 5,
      "%d found" % found)

print("\nALL PASS" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)


# -- the child row is measured, not inherited ------------------------------
#
# ANSUR is a survey of soldiers, so the child cannot be measured the same way.
# It is the male row times the adult-to-child shape change taken off the body
# set's own rigs, which tools/child_ratios.py computes. This reads it back, so
# the table and the measurement cannot drift apart - the same arrangement the
# web preset table has, and for the same reason: the last version of this
# preset was the male row at 122 cm with its legs cut by a tenth, and nothing
# compared it with anything.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tools"))
    import bodies_lib
    import child_ratios
    measured, _adult, _child = child_ratios.child_row()
except Exception as problem:                       # no body set on this box
    print("SKIP the child row against the rigs it came from: %s" % problem)
else:
    row = ANSUR["child"]
    worst = ("none", 0.0)
    for key in ("shoulder_height", "hip_height", "knee_height",
                "ankle_height", "shoulder_w"):
        gap = abs(row[key] - measured[key])
        if gap > worst[1]:
            worst = ("%s (%.4f vs %.4f)" % (key, row[key], measured[key]), gap)
    check("the child row still matches the rigs it was measured from",
          worst[1] < 0.0005, "worst %s" % worst[0])
    # and it has to actually differ from the male row, or it is the old bug
    check("and it is not just the male row again",
          abs(row["shoulder_w"] - ANSUR["male"]["shoulder_w"]) > 0.01
          and abs(row["hip_height"] - ANSUR["male"]["hip_height"]) > 0.005,
          "shoulder_w %.4f vs %.4f" % (row["shoulder_w"],
                                       ANSUR["male"]["shoulder_w"]))
