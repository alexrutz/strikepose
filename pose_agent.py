#!/usr/bin/env python3
"""Drive the editor from a text prompt with a local LLM.

    python3 pose_agent.py --prompt "boxer throwing a right cross, seen from
                                    three quarters left" --out out/
    python3 pose_agent.py --list                # the whole command vocabulary
    python3 pose_agent.py --selftest            # no model and no network needed

What the model actually returns
-------------------------------
Not coordinates. A 7B model asked for eighteen 3D keypoints returns a figure
with one arm twice the length of the other, and writing those points straight
into the skeleton would walk through every bone-length invariant the editor
has. It returns a short list of *commands* instead - point a limb somewhere,
bend a joint so many degrees, turn the figure, pick a named stance - which are
applied through `Skeleton.move_joint` and `rotate_about_axis` exactly as a drag
in the editor would be. Rotations are isometries, so a bone cannot change
length whatever the model asks for, and a command the model invents is
reported and skipped rather than corrupting the pose.

Everything is named in the figure's own frame: "left" is the figure's left,
not the viewer's, and "forward" is the way the figure faces. That survives a
`turn`, which is what makes the commands composable.

Talking to the model
--------------------
Two wire formats cover every local runtime worth naming:

    ollama   POST {host}/api/chat           format: <json schema>
    openai   POST {host}/v1/chat/completions
             response_format: {"type": "json_schema", ...}

Ollama speaks the first, LM Studio, llama.cpp's server and vLLM the second.
Both constrain generation to the schema, which is what makes a small model
usable here at all. `--backend auto` probes the usual ports. Only the standard
library is used, so this adds no dependency.

With no model reachable the prompt still gets read, by keyword, and the run
says so. That keeps the CLI honest offline and is what the self-test exercises.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import math
import os
import re
import sys
import urllib.error
import urllib.request

import everyday
import props as props_module
import rigpose
import wearables
from openpose3d_editor import (
    BODY_PRESETS, DEFAULT_PRESET, EXTREMITY_ANGLES, KEYPOINT_NAMES, VERSION,
    Camera, Skeleton, clean_extremities,
    anatomy_depth_image, body_frame, carry_chain, frame_rect, frame_scene,
    inside_polygon, parse_size, pose_image,
    rigged_depth_image, silhouette_points,
    preset_params, project_people, scene_to_dict, solid_quads, vadd, vcross,
    vdot, vlen, vmul, vnorm, vsub,
)

INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# ---------------------------------------------------------------------------
# Vocabulary
#
# Kept small on purpose. Every name here goes into the system prompt and into
# the JSON schema, and a small model's accuracy falls off a cliff once the
# enums get long. Anything expressible as a combination of these is left to be
# combined rather than given a name of its own.
# ---------------------------------------------------------------------------

# Unit directions in the figure's own frame: (side, up, facing) coefficients,
# where side points to the figure's left and facing the way it looks.
DIRECTIONS = {
    "up": (0.0, 1.0, 0.0),
    "down": (0.0, -1.0, 0.0),
    "forward": (0.0, 0.0, 1.0),
    "back": (0.0, 0.0, -1.0),
    "left": (1.0, 0.0, 0.0),
    "right": (-1.0, 0.0, 0.0),
    "forward_up": (0.0, 1.0, 1.0),
    "forward_down": (0.0, -1.0, 1.0),
    "back_up": (0.0, 1.0, -1.0),
    "back_down": (0.0, -1.0, -1.0),
    "left_up": (1.0, 1.0, 0.0),
    "left_down": (1.0, -1.0, 0.0),
    "right_up": (-1.0, 1.0, 0.0),
    "right_down": (-1.0, -1.0, 0.0),
    "forward_left": (1.0, 0.0, 1.0),
    "forward_right": (-1.0, 0.0, 1.0),
}

# bone name -> (joint it swings from, joint it ends at)
BONES = {
    "l_upper_arm": ("l_shoulder", "l_elbow"),
    "l_forearm": ("l_elbow", "l_wrist"),
    "r_upper_arm": ("r_shoulder", "r_elbow"),
    "r_forearm": ("r_elbow", "r_wrist"),
    "l_thigh": ("l_hip", "l_knee"),
    "l_shin": ("l_knee", "l_ankle"),
    "r_thigh": ("r_hip", "r_knee"),
    "r_shin": ("r_knee", "r_ankle"),
}

# Whole limbs, so "point the left arm forward" is one command rather than two.
LIMBS = {
    "l_arm": ("l_upper_arm", "l_forearm"),
    "r_arm": ("r_upper_arm", "r_forearm"),
    "l_leg": ("l_thigh", "l_shin"),
    "r_leg": ("r_thigh", "r_shin"),
}

# Joints `bend` can flex, and which way positive degrees carries the far end.
# "facing" and "up" are resolved against the figure's own frame at the time.
BEND_JOINTS = {
    "l_elbow": ("l_shoulder", "l_elbow", "l_wrist", "facing"),
    "r_elbow": ("r_shoulder", "r_elbow", "r_wrist", "facing"),
    "l_knee": ("l_hip", "l_knee", "l_ankle", "-facing"),
    "r_knee": ("r_hip", "r_knee", "r_ankle", "-facing"),
    "l_shoulder": (None, "l_shoulder", "l_elbow", "facing"),
    "r_shoulder": (None, "r_shoulder", "r_elbow", "facing"),
    "l_hip": (None, "l_hip", "l_knee", "facing"),
    "r_hip": (None, "r_hip", "r_knee", "facing"),
    "neck": (None, "neck", "nose", "facing"),
}

# Everything the waist carries: the whole figure above the hip line.
ABOVE_WAIST = ["neck", "nose", "r_eye", "l_eye", "r_ear", "l_ear",
               "r_shoulder", "r_elbow", "r_wrist",
               "l_shoulder", "l_elbow", "l_wrist"]

CAMERA_VIEWS = {
    # the plain ones: flat on, or level three-quarters
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "left": (90.0, 0.0),                  # looking at the figure's left side
    "right": (-90.0, 0.0),
    "three_quarter_left": (40.0, 0.0),
    "three_quarter_right": (-40.0, 0.0),
    "high": (25.0, 35.0),
    "overhead": (0.0, 70.0),
    "low": (25.0, -20.0),
    # and the ones with a lens in them. Six of the nine above sit at pitch
    # zero, which is why so much of the catalogue came out looking like a
    # catalogue. These combine a yaw with a pitch, which is what a camera
    # someone is holding actually does.
    "high_three_quarter": (42.0, 32.0),   # the ordinary flattering portrait
    "low_three_quarter": (35.0, -28.0),   # looking up at them, heroic
    "worm": (18.0, -55.0),                # steeply from below
    "over_shoulder": (152.0, 22.0),       # behind and above, past the head
    "bird": (58.0, 58.0),                 # steep oblique, not straight down
    "profile_high": (88.0, 38.0),         # the side, from a step-ladder
}

# Tried in this order when nothing named a view, so a pose that reads equally
# well from several gets the plainest of them. The cinematic ones are last on
# purpose: a figure that constrains nothing should still come out on a plain
# view, and these are there to be *asked* for - by a prompt, by the library,
# by a caller that wants a set to look like photographs rather than a chart.
VIEW_ORDER = ["front", "three_quarter_left", "three_quarter_right", "left",
              "right", "high", "overhead", "back", "low",
              "high_three_quarter", "low_three_quarter", "profile_high",
              "over_shoulder", "bird", "worm"]

# Named stances, written in the same command vocabulary the model uses, so
# there is one code path and a stance can be refined by further commands.
STANCES = {
    "standing": [],
    "t_pose": [{"op": "point", "target": "l_arm", "direction": "left"},
               {"op": "point", "target": "r_arm", "direction": "right"}],
    "arms_up": [{"op": "point", "target": "l_arm", "direction": "left_up"},
                {"op": "point", "target": "r_arm", "direction": "right_up"}],
    "arms_forward": [{"op": "point", "target": "l_arm", "direction": "forward"},
                     {"op": "point", "target": "r_arm", "direction": "forward"}],
    "sitting": [{"op": "point", "target": "l_thigh", "direction": "forward"},
                {"op": "point", "target": "r_thigh", "direction": "forward"},
                {"op": "point", "target": "l_shin", "direction": "down"},
                {"op": "point", "target": "r_shin", "direction": "down"}],
    "kneeling": [{"op": "point", "target": "l_thigh", "direction": "down"},
                 {"op": "point", "target": "r_thigh", "direction": "down"},
                 {"op": "point", "target": "l_shin", "direction": "back"},
                 {"op": "point", "target": "r_shin", "direction": "back"}],
    "crouching": [{"op": "point", "target": "l_thigh", "direction": "forward_down"},
                  {"op": "point", "target": "r_thigh", "direction": "forward_down"},
                  {"op": "point", "target": "l_shin", "direction": "down"},
                  {"op": "point", "target": "r_shin", "direction": "down"},
                  {"op": "lean", "direction": "forward", "degrees": 25}],
    "walking": [{"op": "bend", "target": "l_hip", "degrees": 22},
                {"op": "bend", "target": "r_hip", "degrees": -22},
                {"op": "bend", "target": "l_knee", "degrees": 12},
                {"op": "bend", "target": "r_knee", "degrees": 22},
                {"op": "bend", "target": "l_shoulder", "degrees": -20},
                {"op": "bend", "target": "r_shoulder", "degrees": 20},
                {"op": "bend", "target": "l_elbow", "degrees": 20},
                {"op": "bend", "target": "r_elbow", "degrees": 20}],
    "running": [{"op": "bend", "target": "l_hip", "degrees": 45},
                {"op": "bend", "target": "r_hip", "degrees": -30},
                {"op": "bend", "target": "l_knee", "degrees": 75},
                {"op": "bend", "target": "r_knee", "degrees": 40},
                {"op": "bend", "target": "l_shoulder", "degrees": -45},
                {"op": "bend", "target": "r_shoulder", "degrees": 45},
                {"op": "bend", "target": "l_elbow", "degrees": 80},
                {"op": "bend", "target": "r_elbow", "degrees": 80},
                {"op": "lean", "direction": "forward", "degrees": 12}],
    "fighting_stance": [{"op": "bend", "target": "l_shoulder", "degrees": 35},
                        {"op": "bend", "target": "r_shoulder", "degrees": 30},
                        {"op": "bend", "target": "l_elbow", "degrees": 105},
                        {"op": "bend", "target": "r_elbow", "degrees": 110},
                        {"op": "bend", "target": "l_knee", "degrees": 18},
                        {"op": "bend", "target": "r_knee", "degrees": 18},
                        {"op": "turn", "direction": "left", "degrees": 25}],
    "lying_down": [{"op": "turn", "direction": "back", "degrees": 90}],
    "sitting_on_floor": [{"op": "point", "target": "l_thigh", "direction": "forward"},
                         {"op": "point", "target": "r_thigh", "direction": "forward"},
                         {"op": "point", "target": "l_shin", "direction": "forward"},
                         {"op": "point", "target": "r_shin", "direction": "forward"}],
}

# The everyday catalogue is the same kind of thing: a name for a list of these
# commands. It is kept in its own module because it is long and because it is
# data, not vocabulary - the ops, directions, joints and limbs above are what
# has to stay short, since a model picks from those on *every* command, while a
# stance name is picked at most once and saves it ten guesses. Merged rather
# than looked up separately so there is one table, one enum and one code path;
# two tables where one is "also accepted" is a name the schema rejects at
# decode time and the prompt promises, which is worse than either.
for _name, _commands in everyday.POSES.items():
    # setdefault, not update: a catalogue entry that repeats a basic name is
    # an *alias* for it - "walking" there is one `stance walking` command - so
    # letting it overwrite points the name at itself and the first figure to
    # walk takes the process down with it.
    STANCES.setdefault(_name, _commands)

# Where an object goes, relative to the figure it is placed against.
# (which point of the figure, how the object lines up with it). "seat" means
# the object's *top* meets that height and its base still reaches the floor,
# which is what a chair under the hips has to do and the only alignment that
# cannot be written as an offset.
ANCHORS = {
    "ground": ("feet", "bottom"),
    "in_front": ("feet_forward", "bottom"),
    "behind": ("feet_back", "bottom"),
    "left_of": ("feet_left", "bottom"),
    "right_of": ("feet_right", "bottom"),
    "under_hips": ("hips", "seat"),
    "under_feet": ("feet", "seat"),
    "at_hands": ("hands", "centre"),
    "overhead": ("head", "bottom"),
}

OPS = ("stance", "point", "bend", "turn", "lean", "look", "hand", "foot",
       "grip", "hide", "place", "wear", "outfit")

SIDES = ("left", "right", "both")

# Every wearable, flattened: a model does far better picking one name out of a
# list than picking a slot and then a name that has to belong to it. The slot
# is looked up from the name, which is unambiguous because no two slots share
# one.
WEARABLES = {name: slot for slot in wearables.SLOTS
             for name in wearables.options(slot) if name != "none"}
WEARABLE_NAMES = sorted(WEARABLES)

# Named outfits get an op of their own rather than joining `wears`. The same
# split `stance` makes against the pose commands: a garment name is picked on
# every `wear`, so that enum stays short, while an outfit is picked at most
# once per figure and saves five guesses, so the catalogue can be long.
OUTFIT_NAMES = list(wearables.OUTFIT_NAMES)

POINT_TARGETS = sorted(set(BONES) | set(LIMBS))
SHAPE_NAMES = list(props_module.SHAPE_NAMES)
HIDE_TARGETS = sorted(set(LIMBS) | set(KEYPOINT_NAMES))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def command_schema():
    """One flat object per command rather than a oneOf over seven shapes.

    Constrained decoders handle a flat object with an `op` enum and optional
    fields far more reliably than a union, and a small model writes it more
    often too. What each field means for each op is spelled out in the system
    prompt; the validator below rejects the combinations that make no sense.
    """
    return {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": list(OPS)},
            "target": {"type": "string"},
            "direction": {"type": "string", "enum": sorted(DIRECTIONS)},
            "degrees": {"type": "number"},
            "name": {"type": "string", "enum": sorted(STANCES)},
            "shape": {"type": "string", "enum": SHAPE_NAMES},
            "at": {"type": "string", "enum": sorted(ANCHORS)},
            "distance": {"type": "number"},
            "size": {"type": "number"},
            "wears": {"type": "string", "enum": WEARABLE_NAMES},
            "outfit": {"type": "string", "enum": OUTFIT_NAMES},
            # hand and foot: the two angles each that the keypoints cannot say
            "side": {"type": "string", "enum": list(SIDES)},
            "bend": {"type": "number"},
            "lift": {"type": "number"},
            "turn": {"type": "number"},
            # grip: 0 open, 1 a closed fist. The `hand` op turns the WRIST;
            # this closes the fingers, which are fifteen bones of their own.
            "amount": {"type": "number"},
        },
        "required": ["op"],
    }


def response_schema():
    return {
        "type": "object",
        "properties": {
            "figures": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "preset": {"type": "string", "enum": sorted(BODY_PRESETS)},
                        "commands": {"type": "array", "items": command_schema()},
                    },
                    "required": ["commands"],
                },
            },
            "camera": {"type": "string", "enum": sorted(CAMERA_VIEWS)},
            "notes": {"type": "string"},
        },
        "required": ["figures"],
    }


SYSTEM_PROMPT = """You pose a 3D human figure for an OpenPose/ControlNet export.

Answer with JSON only. No prose, no markdown fence.

Everything is named in the FIGURE's own frame, never the viewer's:
  "left"/"right" are the figure's own left and right,
  "forward" is the way the figure faces, "up" is above its head.

Commands, applied in order:
  {"op":"stance","name":NAME}                 start from a named pose
  {"op":"point","target":BONE,"direction":DIR} aim a bone or a whole limb
  {"op":"bend","target":JOINT,"degrees":N}     flex a joint, N may be negative
  {"op":"turn","direction":DIR,"degrees":N}    rotate the WHOLE figure
  {"op":"lean","direction":DIR,"degrees":N}    tip the upper body at the waist
  {"op":"look","direction":DIR}                where the head looks
  {"op":"hand","side":SIDE,"bend":N,"turn":N}  + curls the palm in, + rolls
                                              it to face backward
  {"op":"foot","side":SIDE,"lift":N,"turn":N}  + lifts the toes, + points them
                                              outward
  {"op":"grip","side":SIDE,"amount":N}         close the fingers, 0 open to
                                              1 a fist. Anything the figure
                                              HOLDS wants one: a mug 0.8, a
                                              handle 0.9, a phone 0.5, a
                                              box carried flat 0.2. `hand`
                                              turns the wrist; this is the
                                              fingers, and they are different
                                              things.
  {"op":"hide","target":JOINT_OR_LIMB}         mark it off-frame or occluded
  {"op":"place","shape":SHAPE,"at":ANCHOR,"distance":CM,"size":N,"degrees":N}
                                              put an object in the scene
  {"op":"outfit","outfit":NAME}                dress the figure in a whole look
  {"op":"wear","wears":GARMENT}                change one garment

stance NAME, grouped - start from the closest one, then correct it:
%(stances)s
point target: %(points)s
bend target:  %(bends)s
  positive degrees flexes the joint the natural way: an elbow closes, a knee
  folds back, a hip or shoulder swings the limb forward. Negative reverses it.
  A straight limb is 0. A hard bend is about 120.
direction:    %(directions)s
camera:       %(cameras)s
preset:       %(presets)s

outfit NAME:  %(outfits)s
  a whole look in one command - use it when one fits, then correct a slot with
  `wear`. An outfit only sets the slots it names, so a haircut chosen before or
  after it survives.
wear:         %(wearables)s
  one from each of hair, headgear, top, bottom and shoes at most; they
  stack, so a coat does not remove the trousers. Leave a slot out to leave it
  bare.

place shape:  %(shapes)s
place at:     %(anchors)s
  distance is centimetres from the figure, size multiplies the object's real
  size (1 is life size), degrees turns it. Every shape already has the size of
  the real thing, so leave size out unless it should be bigger or smaller than
  usual. A figure that sits needs something under it: sit it down AND place a
  chair "under_hips". Objects appear in the depth map, never in the pose map.

Shape:
{"figures":[{"preset":"Male, average","commands":[ ... ]}],"camera":"front"}

MORE THAN ONE PERSON. `figures` is a list. If the request describes two people
- shaking hands, handing something over, dancing, one watching another - give
two entries, each with its own preset and its own commands. They are laid out
70 cm apart along the figure's right, so turn them to face each other with
`turn` when they should. Do not put two people in one entry, and do not
collapse a two-person request into one figure.

HANDS AND FEET ARE NOT IMPLIED. Nothing about an arm says which way its palm
faces, and nothing about a leg says where the toes point. A pose that leaves
them out gets the rest pose's hands, which read as limp. Almost every real
pose wants at least one `hand` command: a hand on a desk is bend 35 turn 0, a
hand gripping something is bend 60, a hand hanging relaxed is bend 15. Feet
matter when the figure is not standing flat - toes down for a lunge or a
kneel, lifted for a heel strike.

Start from a stance when one is close, then correct it with a few commands.
Prefer eight to sixteen commands. Do not invent op, target or direction names.
"""


def stance_listing(width=74):
    """The stance names, grouped and wrapped.

    Eighty-odd names on one line is a wall a small model reads badly; the same
    names under the heading they belong to are a menu. The compositional
    stances come first because they are the ones meant to be refined.
    """
    import textwrap
    groups = [("basic", [n for n in sorted(STANCES)
                         if n not in everyday.POSES])]
    groups += [(g, everyday.names_in(g)) for g in everyday.GROUPS]
    lines = []
    for heading, names in groups:
        body = textwrap.wrap(", ".join(names), width,
                             initial_indent="    ", subsequent_indent="    ")
        lines.append("  " + heading + ":")
        lines.extend(body)
    return "\n".join(lines)


def outfit_listing(width=74):
    """The outfit names, wrapped. Forty-odd on one line is a wall."""
    import textwrap
    return "\n".join(textwrap.wrap(", ".join(OUTFIT_NAMES), width,
                                   initial_indent="    ",
                                   subsequent_indent="    ")).lstrip()


PLANNING_PROMPT = """You are working out how a human body is arranged for a
pose, before anyone writes it down. Think it through properly; you will be
asked for the commands afterwards, so do not write JSON here.

Everything is named in the FIGURE's own frame, never the viewer's: "left" and
"right" are the figure's own, "forward" is the way it faces, "up" is above its
head.

Work through, in this order:

1. HOW MANY PEOPLE. One unless the request describes more. "Two people
   shaking hands", "a man handing a woman a box", "a couple dancing" are two
   figures; say what each one is doing separately, and say how they are
   arranged relative to each other.
2. WHAT THE WHOLE BODY IS DOING. Standing, sitting, kneeling, lying,
   crouching, walking, mid-stride. Is it leaning or twisting, and which way?
3. THE LEGS. Where is the weight? Are the knees straight or bent, and how far?
4. THE ARMS. For each one separately: where does the upper arm point, is the
   elbow bent and how far, and where does that put the hand.
5. THE HANDS AND THE FEET. These are separate from the arms and legs and are
   NOT implied by them. Which way does each palm face - down on a table, in
   towards the body, forward? Are the fingers curled round something? Are the
   toes flat on the ground, or pointed, or lifted? A pose that does not say is
   a pose with two limp hands in it.
6. THE HEAD. Where is it looking?
7. ANYTHING IT NEEDS. A figure that sits needs something under it. A figure
   at a desk needs a desk. A figure holding something needs the something.
8. THE CAMERA. Which view shows what makes this pose what it is? A crouch
   read head-on looks like standing.

Then say, in a few lines, the plan you have arrived at: a list of what each
part of each figure does. Be specific about angles in degrees where you can.
"""


def planning_prompt():
    return PLANNING_PROMPT


def with_examples(prompt, limit=3):
    """The request, with the nearest catalogue poses shown as worked answers.

    Few-shot from a corpus that is already there and already correct: the
    suite asserts that every one of the 84 applies with no command skipped,
    so an example cannot teach a name that does not exist or a shape that does
    not work. It is the cheapest accuracy there is - the model stops guessing
    at the grammar and spends its thinking on the pose.
    """
    examples = nearest_examples(prompt, limit)
    if not examples:
        return prompt
    return ("%s\n\nSome poses from the catalogue that are near this one, as "
            "worked examples of the command style. Do not copy one unless it "
            "IS the pose asked for; use them for the shape of an answer:\n\n%s"
            % (prompt, "\n\n".join(examples)))


def nearest_examples(prompt, limit=3):
    """Catalogue poses whose names overlap the request, as worked examples.

    The 84 poses in `everyday.py` are a corpus of correct answers in exactly
    the vocabulary the model has to write: real command lists that apply with
    no warnings, because the suite asserts that. Showing the nearest two or
    three is worth more than any amount of describing the grammar, and it
    costs nothing to keep current - a pose added to the catalogue becomes an
    example the same day.

    Matched on words, not embeddings: there is no embedding model here, the
    names are deliberately plain English, and a miss costs an example rather
    than a wrong answer.
    """
    words = set(re.findall(r"[a-z]+", (prompt or "").lower()))
    words -= {"a", "an", "the", "of", "in", "on", "at", "to", "is", "with",
              "person", "man", "woman", "figure", "someone", "who", "and",
              "his", "her", "their", "its", "he", "she", "they", "it"}
    scored = []
    for name, steps in STANCES.items():
        if not isinstance(steps, list) or not steps:
            continue
        theirs = set(name.lower().split("_"))
        shared = len(words & theirs)
        if shared:
            scored.append((shared, -len(steps), name, steps))
    scored.sort(reverse=True)
    out = []
    for _n, _l, name, steps in scored[:limit]:
        out.append("%s:\n%s" % (name, json.dumps(steps)))
    return out


def system_prompt():
    return SYSTEM_PROMPT % {
        "stances": stance_listing(),
        "points": ", ".join(POINT_TARGETS),
        "bends": ", ".join(sorted(BEND_JOINTS)),
        "directions": ", ".join(sorted(DIRECTIONS)),
        "cameras": ", ".join(sorted(CAMERA_VIEWS)),
        "presets": ", ".join(sorted(BODY_PRESETS)),
        "shapes": ", ".join(SHAPE_NAMES),
        "wearables": ", ".join(WEARABLE_NAMES),
        "outfits": outfit_listing(),
        "anchors": ", ".join(sorted(ANCHORS)),
    }


# ---------------------------------------------------------------------------
# Applying commands
# ---------------------------------------------------------------------------


def resolve_direction(skeleton, name):
    """A named direction as a world unit vector in the figure's current frame."""
    coefficients = DIRECTIONS.get(name)
    if coefficients is None:
        return None
    side, up, facing = body_frame(skeleton)
    a, b, c = coefficients
    return vnorm(vadd(vadd(vmul(side, a), vmul(up, b)), vmul(facing, c)))


def _flex(skeleton, pivot, child, axis, toward, degrees):
    """Swing the bone at `pivot` about `axis`, which runs through its own head.

    The sign is read off the geometry rather than tabulated: rotating by a
    positive angle moves the child end at `axis x direction`, so whichever sign
    sends it towards `toward` is the one that flexes the joint the natural way.
    Tabulating it instead needs a different entry per side and per limb and is
    wrong the moment a stance has already turned the figure round.

    `pivot` is a bone index, and a bone's head IS the joint it turns about, so
    there is no origin to pass and no subtree to collect: rotating the bone
    carries everything below it because that is what a local rotation means.
    """
    q = skeleton.pose.positions()
    direction = vnorm(vsub(tuple(q[child]), tuple(q[pivot])))
    sign = 1.0 if vdot(vcross(axis, direction), toward) >= 0.0 else -1.0
    skeleton.pose.rotate(pivot, axis, math.radians(degrees) * sign)


def _bend_axis(skeleton, parent, pivot, child):
    """The axis a joint hinges about.

    Taken from the limb's carried cross-section frame, the same one the depth
    pass sweeps its profiles on, so the hinge stays square to the limb however
    the limb is posed - including when it points along the body's own forward,
    where picking an axis off `facing` has nothing to work with.
    """
    side, up, facing = body_frame(skeleton)
    down = vmul(up, -1.0)
    q = skeleton.pose.positions()
    joints = [tuple(q[pivot]), tuple(q[child])]
    if parent is not None:
        joints.insert(0, tuple(q[parent]))
    frames = carry_chain((down, facing), *joints)
    axis, forward = frames[-2] if parent is not None else frames[-1]
    hinge = vnorm(vcross(axis, forward))
    return hinge if vlen(hinge) > 1e-6 else side


def ground_level(skeleton):
    """The y the figure stands on: the lowest point of its own body.

    The rigged mesh knows where the sole is, so nothing has to allow 8 cm
    below the ankle for one any more - and that allowance was a constant on
    a figure whose foot is a different size in every preset.
    """
    return skeleton.pose.lowest()


def anchor_point(skeleton, at, distance):
    """(where the object goes, which height it lines up by).

    `distance` runs away from the figure along the ground, which is the only
    reading of "80 cm in front of" that stays true once the figure turns.
    """
    where, align = ANCHORS[at]
    side, up, facing = body_frame(skeleton)
    floor = ground_level(skeleton)
    hips = vmul(vadd(skeleton.at("r_hip"),
                     skeleton.at("l_hip")), 0.5)
    flat = (hips[0], floor, hips[2])          # under the figure, on the floor
    # The push anchors return the spot under the figure; `_clear_of` slides
    # the object out from there once its own size is known, because the gap
    # that was asked for is to its near face, not to its centre.
    if where in ("feet_forward", "feet_back", "feet_left", "feet_right"):
        return flat, align, floor
    if where == "hips":
        return (hips[0], floor, hips[2]), align, floor
    if where == "hands":
        hands = vmul(vadd(skeleton.at("r_wrist"),
                          skeleton.at("l_wrist")), 0.5)
        forward = vmul(facing, 12.0)
        return vadd(hands, forward), align, floor
    if where == "head":
        head = skeleton.at("nose")
        return (head[0], head[1] + 16.0 + distance, head[2]), align, floor
    return flat, align, floor


# Which way each pushed-away anchor sends the object, in the figure's frame.
PUSH = {"in_front": (0.0, 0.0, 1.0), "behind": (0.0, 0.0, -1.0),
        "left_of": (1.0, 0.0, 0.0), "right_of": (-1.0, 0.0, 0.0)}


def _clear_of(skeleton, at, distance, position, box, yaw):
    """Slide a pushed-away object out until it clears the figure's trunk.

    `distance` has to mean the gap between the figure and the near face of the
    object, not the offset of its centre: a desk is 70 cm deep, so centring one
    20 cm in front of a figure puts its top surface through the figure's
    thighs. The trunk is what must not be inside the furniture - the legs go
    *under* a table and the arms reach over it - so the clearance is measured
    on the shoulders and hips plus the body's own depth, and a silhouette that
    includes a seated figure's knees would push every desk out of reach.
    """
    coefficients = PUSH.get(at)
    if coefficients is None:
        return position
    side, _up, facing = body_frame(skeleton)
    push = vnorm(vadd(vmul(side, coefficients[0]), vmul(facing, coefficients[2])))
    trunk = max(vdot(vsub(skeleton.at(name), position), push)
                for name in ("l_shoulder", "r_shoulder", "l_hip", "r_hip"))
    body = skeleton.body
    trunk += max(body[part][1] for part in ("chest", "waist", "pelvis"))
    # the object's own half-extent along the same direction, turned as asked
    half = vmul((box[0], 0.0, box[2]), 0.5)
    turn = math.radians(yaw)
    axes = ((math.cos(turn), 0.0, -math.sin(turn)),
            (math.sin(turn), 0.0, math.cos(turn)))
    reach = sum(abs(vdot(axis, push)) * h
                for axis, h in zip(axes, (half[0], half[2])))
    return vadd(position, vmul(push, trunk + distance + reach))


def place_object(skeleton, shape, at="ground", distance=70.0, size=1.0,
                 yaw=0.0):
    """One object, positioned against the figure. Returns the prop.

    A seat is the case worth spelling out: "under_hips" sets the object's
    height so its top surface meets the hip and its base still reaches the
    floor, rather than dropping a default-height chair under a figure whose
    hips are nowhere near 45 cm up. That is what makes "sitting on a chair"
    come out as a figure sitting on a chair rather than one hovering over it.
    """
    position, align, floor = anchor_point(skeleton, at, distance)
    w, h, d = props_module.default_size(shape)
    w, h, d = w * size, h * size, d * size
    position = _clear_of(skeleton, at, distance, position, (w, h, d), yaw)
    if align == "seat":
        seat = "l_hip" if at == "under_hips" else "l_ankle"
        top = skeleton.at(seat)[1] - (4.0 if at == "under_hips" else 0.0)
        h = max(6.0, top - floor)
        position = (position[0], floor, position[2])
    elif align == "centre":
        position = (position[0], position[1] - h / 2.0, position[2])
    return props_module.make(shape, (w, h, d), position, yaw)


def apply_command(skeleton, command, props=None, depth=0, defer=None):
    """Apply one command. Returns None, or a string saying why it was skipped.

    Every path here is a rotation, so no bone can change length - which is what
    lets a model's output be applied at all without a validation pass over the
    proportions afterwards.
    """
    if not isinstance(command, dict):
        return "not an object: %r" % (command,)
    op = command.get("op")
    target = command.get("target")
    name = command.get("name")
    direction_name = command.get("direction")
    try:
        degrees = float(command.get("degrees", 0.0))
    except (TypeError, ValueError):
        return "degrees is not a number: %r" % (command.get("degrees"),)
    if not -360.0 <= degrees <= 360.0:
        return "degrees out of range: %g" % degrees

    if op == "stance":
        if name not in STANCES:
            return "unknown stance %r" % (name,)
        # A stance may start from another - that is what makes the catalogue
        # worth having - so this recurses, and a stance that reaches itself
        # would otherwise blow the stack rather than be skipped like any other
        # bad command. Four is deeper than any real chain.
        if depth >= 4:
            return "stance %r nests too deep" % (name,)
        # Report what went wrong inside it. Discarding these return values
        # made a stance the one place a bad command *was* silent - including
        # the depth guard above, which fired correctly and said so to nobody.
        trouble = [problem for problem in
                   (apply_command(skeleton, step, props, depth + 1, defer)
                    for step in STANCES[name]) if problem]
        if trouble:
            return "%d of %d steps of stance %r skipped: %s" % (
                len(trouble), len(STANCES[name]), name, trouble[0])
        return None

    if op == "point":
        bones = LIMBS.get(target) or ((target,)
                                      if target in rigpose.SEGMENTS else None)
        if bones is None:
            return "unknown point target %r" % (target,)
        direction = resolve_direction(skeleton, direction_name)
        if direction is None:
            return "unknown direction %r" % (direction_name,)
        pose = skeleton.pose
        for bone in bones:
            swing, far = rigpose.SEGMENTS[bone]
            j, c = pose.bone(swing), pose.bone(far)
            q = pose.positions()
            reach = vlen(vsub(q[c], q[j]))
            pose.aim(j, c, vadd(tuple(q[j]), vmul(direction, reach)))
        return None

    if op == "bend":
        spec = rigpose.BEND.get(target)
        if spec is None:
            return "unknown bend joint %r" % (target,)
        parent, pivot, child, toward_name = spec
        side, up, facing = body_frame(skeleton)
        toward = vmul(facing, -1.0) if toward_name == "-facing" else facing
        pose = skeleton.pose
        parent_i = pose.bone(parent) if parent else None
        pivot_i, child_i = pose.bone(pivot), pose.bone(child)
        axis = _bend_axis(skeleton, parent_i, pivot_i, child_i)
        _flex(skeleton, pivot_i, child_i, axis, toward, degrees)
        return None

    if op in ("turn", "lean"):
        side, up, facing = body_frame(skeleton)
        if direction_name in ("left", "right"):
            axis = up if direction_name == "left" else vmul(up, -1.0)
        elif direction_name in ("forward", "back"):
            axis = side if direction_name == "forward" else vmul(side, -1.0)
            # positive degrees must tip the head the named way whichever side
            # the frame came out on
            probe = vcross(axis, up)
            if vdot(probe, facing) < 0.0:
                axis = vmul(axis, -1.0)
        else:
            return "%s needs left, right, forward or back, not %r" % (
                op, direction_name)
        # A turn rotates the root and a lean rotates the waist bone. What each
        # carries is settled by the armature - everything below the bone - so
        # there is no list of joints to get wrong, and no axis that can move
        # the bone it rotates about. The two bugs the randomizer's length
        # check caught on the keypoint version were both of that kind.
        pose = skeleton.pose
        bone = pose.roots[0] if op == "turn" else pose.bone(rigpose.WAIST)
        pose.rotate(bone, axis, math.radians(degrees))
        return None

    if op == "look":
        direction = resolve_direction(skeleton, direction_name)
        if direction is None:
            return "unknown direction %r" % (direction_name,)
        # Aim the GAZE, not the neck-to-nose line. The nose sits high on the
        # head, so that line stands 74 degrees above horizontal at rest, and
        # aiming *it* at "forward" swings the head 74 degrees - chin on the
        # chest - while the gaze it was meant to set has barely moved.
        # "forward_down" came out at 119 degrees and "down" at 164, which is
        # what made half the catalogue look hunched. The gaze is the ear
        # midpoint to the nose, which is level at rest, and the whole head
        # turns about the neck, so no bone changes length.
        # The face has no bones of its own worth aiming, so the gaze is read
        # off the keypoints the posed rig reports - which is the sound
        # direction: they describe the skull rather than claiming anything
        # about it.
        kp = skeleton.keypoints()
        ear_mid = vmul(vadd(tuple(kp["l_ear"]), tuple(kp["r_ear"])), 0.5)
        gaze = vsub(tuple(kp["nose"]), ear_mid)
        if vlen(gaze) < 1e-6:
            return "the head has no gaze to aim"
        gaze = vnorm(gaze)
        axis = vcross(gaze, direction)
        angle = math.atan2(vlen(axis), vdot(gaze, direction))
        if angle < 1e-9:
            return None
        if vlen(axis) < 1e-9:      # exactly behind: no minimal rotation
            _side, up, _facing = body_frame(skeleton)
            axis = up
        skeleton.pose.rotate(skeleton.pose.bone("neck01"), vnorm(axis), angle)
        return None

    if op == "grip":
        side = str(command.get("side") or target or "both").lower()
        if side not in SIDES:
            return "grip needs a side: left, right or both, not %r" % (side,)
        try:
            amount = float(command.get("amount",
                                       command.get("degrees", 1.0)))
        except (TypeError, ValueError):
            return "grip amount should be a number between 0 and 1"
        for letter in (("r", "l") if side == "both"
                       else ("l" if side == "left" else "r",)):
            skeleton.grip(letter, amount)
        return None

    if op in ("hand", "foot"):
        # The one thing eighteen keypoints cannot say. The wrist and the ankle
        # end their chains, so nothing in a pose reports which way a palm
        # faces or whether a toe points in, and a model has to be able to say
        # it in words or it cannot be said at all.
        side = str(command.get("side") or target or "both").lower()
        if side not in SIDES:
            return "%s needs a side: left, right or both, not %r" % (op, side)
        names = EXTREMITY_ANGLES[op]
        angles = []
        for label, low, high in names:
            try:
                angles.append(max(low, min(high, float(
                    command.get(label, command.get("degrees", 0.0)) or 0.0))))
            except (TypeError, ValueError):
                return "%s %s should be a number" % (op, label)
        # On the rig these are bones like any other, so the two angles are two
        # rotations rather than a pair carried beside the pose and applied by
        # a special case at the end of the solve. A wrist that also deviates
        # sideways, or a foot that rolls, is a third rotation away now rather
        # than a change to the format.
        for letter in (("r", "l") if side == "both"
                       else ("l" if side == "left" else "r",)):
            skeleton.set_extremity(op, letter, angles[0], angles[1])
        return None

    if op == "hide":
        names = LIMBS.get(target) or ((target,)
                                      if target in rigpose.SEGMENTS else None)
        if names is None:
            return "unknown hide target %r" % (target,)
        pose = skeleton.pose
        for bone in names:
            # everything below the segment goes with it, which is what "hide
            # the left arm" means and what the keypoint version had to spell
            # out joint by joint
            for j in pose.subtree(pose.bone(rigpose.SEGMENTS[bone][0])):
                skeleton.visible[j] = False
        return None

    if op == "outfit":
        name = command.get("outfit") or command.get("name") or target
        dressed = wearables.dress(getattr(skeleton, "outfit", None), name)
        if dressed is None:
            return "no outfit called %r" % (name,)
        skeleton.outfit = dressed
        return None

    if op == "wear":
        name = command.get("wears") or command.get("name") or target
        slot = WEARABLES.get(name)
        if slot is None:
            # a model reaching for a whole look through the nearer op is
            # asking for something that exists; give it rather than a warning
            dressed = wearables.dress(getattr(skeleton, "outfit", None), name)
            if dressed is not None:
                skeleton.outfit = dressed
                return None
            return "nothing called %r to wear" % (name,)
        outfit = dict(getattr(skeleton, "outfit", None) or {})
        outfit[slot] = name
        skeleton.outfit = outfit
        return None

    if op == "place":
        if props is None:
            return "place needs a scene to put the object in"
        # Hold it until the figure is finished. An anchor is resolved once, in
        # the figure's own frame, and the figure it was resolved against has
        # to be the posed one: "sit down AND put a chair under the hips" reads
        # naturally in either order, but placing first anchors the chair to a
        # standing hip, and since `ground_level` is the figure's own feet the
        # chair comes out 86 cm tall with the desk in front of it ending up
        # below the seat.
        if defer is not None:
            defer.append(command)
            return None
        shape = command.get("shape") or target
        if shape not in props_module.SHAPES:
            return "unknown shape %r" % (shape,)
        at = command.get("at", "ground")
        if at not in ANCHORS:
            return "unknown anchor %r" % (at,)
        try:
            distance = float(command.get("distance", 70.0))
            size = float(command.get("size", 1.0))
        except (TypeError, ValueError):
            return "distance and size must be numbers"
        if not 0.0 <= distance <= 600.0 or not 0.05 <= size <= 12.0:
            return "distance or size out of range"
        if len(props) >= 24:
            return "too many objects in the scene already"
        props.append(place_object(skeleton, shape, at, distance, size, degrees))
        return None

    return "unknown op %r" % (op,)


def apply_commands(skeleton, commands, props=None):
    """Apply a list of commands. Returns the warnings, one per bad command.

    A bad command is skipped, not fatal. A local model gets one wrong every so
    often, and losing a whole pose over a misspelled joint would make the CLI
    useless exactly when the model is small enough to be worth running locally.

    `props` is the scene's object list; objects are appended to it. Passing
    None refuses `place` rather than dropping it silently, so a caller that
    has no scene to put an object in hears about it.
    """
    warnings = []
    if not isinstance(commands, list):
        return ["commands is not a list"]
    held = [] if props is not None else None
    for i, command in enumerate(commands):
        problem = apply_command(skeleton, command, props, 0, held)
        if problem:
            warnings.append("command %d skipped: %s" % (i + 1, problem))
    # Put the figure back on the ground before anything is anchored to it.
    #
    # Posing is rotation and nothing moves the pelvis, so folding the legs for
    # a sit lifts the whole body's lowest point off the floor and leaves the
    # figure hovering - by 83 cm on a cross-legged sit, which is most of a
    # person. The keypoint version never noticed because its "floor" was the
    # lowest ankle less 8 cm for a sole, so it moved the floor to the figure
    # instead of the figure to the floor; a chair anchored under the hips then
    # reached up from wherever the feet had ended up.
    #
    # Here, and not inside each command, because it is the finished pose that
    # stands on something: re-grounding after every step would fight a stance
    # halfway through building itself.
    if hasattr(skeleton, "pose"):
        skeleton.pose.stand()
    for command in held or ():
        problem = apply_command(skeleton, command, props)
        if problem:
            warnings.append("object not placed: %s" % problem)
    return warnings


# ---------------------------------------------------------------------------
# Building the scene
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Checking the result
# ---------------------------------------------------------------------------
#
# The point of this is that it is NOT the path that built the pose.
#
# Asking a model to re-read its own commands and spot the mistake mostly gets
# the same commands back, because the reasoning that produced them is the
# reasoning being asked to find the fault - it already believes this is a
# person kneeling. What it cannot do, at any amount of thinking, is work out
# that the left hand ended up 4 cm inside the ribcage, or that the figure's
# weight is 30 cm behind its heels. Those are facts about thirteen thousand
# skinned vertices, arrived at by arithmetic, and handing them back is new
# information rather than a second opinion.
#
# So this measures, in centimetres, and says what it found in plain words. A
# revision pass gets the prompt, the plan it wrote, and these.

FLOOR_SLACK = 1.5          # cm a foot may sink before it is worth saying
TOPPLE_SLACK = 6.0         # cm the centre of mass may sit outside the feet


def _torso_axis(figure):
    """(hip midpoint, neck, radius): the trunk as one capsule."""
    q = figure.pose.positions()
    hips = 0.5 * (q[figure.pose.bone("l_hip")] + q[figure.pose.bone("r_hip")])
    neck = q[figure.pose.index["spine01"]]
    half = 0.5 * float(np.linalg.norm(q[figure.pose.bone("l_shoulder")]
                                      - q[figure.pose.bone("r_shoulder")]))
    return hips, neck, max(8.0, half * 0.85)


def _inside_capsule(point, a, b, radius):
    """How far inside a capsule a point sits, in cm; 0 when outside."""
    axis = b - a
    length = float(axis @ axis)
    t = 0.0 if length < 1e-9 else float(np.clip((point - a) @ axis / length,
                                                0.0, 1.0))
    return max(0.0, radius - float(np.linalg.norm(point - (a + axis * t))))


def critique(figures, props=()):
    """What is measurably wrong, or measurably worth knowing, about a scene.

    Plain sentences, because they go back to a language model. Every number in
    them comes from the posed geometry.
    """
    notes = []
    for index, figure in enumerate(figures):
        who = ("the figure" if len(figures) == 1
               else "figure %d" % (index + 1))
        pose = figure.pose
        q = pose.positions()
        surface = figure.surface()
        at = lambda name: q[pose.bone(name)]

        # -- through the floor, or floating above it ---------------------
        low = float(surface[:, 1].min())
        if low < -FLOOR_SLACK:
            notes.append("%s is %.0f cm through the floor." % (who, -low))
        elif low > FLOOR_SLACK:
            notes.append("%s is floating %.0f cm above the floor." % (who, low))

        # -- would it stand up? ------------------------------------------
        #
        # Centre of mass against the ground it is actually touching. A pose
        # whose weight is outside its own feet is a person falling over, which
        # is the single most common thing a plausible-looking command list
        # gets wrong - and the one thing no amount of reasoning about limbs
        # will notice.
        # What is holding it up: the floor, and the top of anything it is
        # sitting or standing on. A chair counts - a seated figure's weight is
        # behind its feet by design, and without the seat every `sitting` pose
        # reads as a person toppling backwards.
        touching = [surface[:, 1] < low + 5.0]
        for prop in props or ():
            plo, phi = props_module.bounds(prop)
            touching.append(
                (surface[:, 1] < phi[1] + 5.0) & (surface[:, 1] > phi[1] - 5.0)
                & (surface[:, 0] > plo[0]) & (surface[:, 0] < phi[0])
                & (surface[:, 2] > plo[2]) & (surface[:, 2] < phi[2]))
        held = touching[0]
        for extra in touching[1:]:
            held = held | extra
        contact = surface[held]
        if len(contact) > 8:
            com = surface.mean(axis=0)
            lo = contact.min(axis=0)
            hi = contact.max(axis=0)
            over_x = max(lo[0] - com[0], com[0] - hi[0])
            over_z = max(lo[2] - com[2], com[2] - hi[2])
            out = max(over_x, over_z)
            if out > TOPPLE_SLACK:
                way = ("forward" if com[2] > hi[2] else
                       "backward" if com[2] < lo[2] else
                       "sideways")
                notes.append(
                    "%s would fall over %s: its weight is %.0f cm outside "
                    "the ground it is standing on. Something has to take the "
                    "weight - a hand down, a wider stance, or an object."
                    % (who, way, out))

        # -- a limb inside the body --------------------------------------
        a, b, radius = _torso_axis(figure)
        for part, bone in (("left hand", "wrist.L"), ("right hand", "wrist.R"),
                           ("left elbow", "lowerarm01.L"),
                           ("right elbow", "lowerarm01.R"),
                           ("left knee", "lowerleg01.L"),
                           ("right knee", "lowerleg01.R")):
            depth = _inside_capsule(at(bone), a, b, radius)
            if depth > 3.0:
                notes.append("%s's %s is %.0f cm inside its own torso."
                             % (who, part, depth))
        head = at("head")
        for part, bone in (("left hand", "wrist.L"), ("right hand", "wrist.R")):
            gap = float(np.linalg.norm(at(bone) - head))
            if gap < 9.0:
                notes.append("%s's %s is inside its own head." % (who, part))

        # -- hands and feet left at rest ---------------------------------
        if not figure.extremities:
            notes.append(
                "%s has no `hand` or `foot` command, so both palms and both "
                "feet are at the rest pose. Say which way they face." % who)

        # -- where things actually ended up ------------------------------
        #
        # Not a fault - a reading. A model that asked for a hand on a desk can
        # only tell whether it got one by being told where the hand is.
        side, up, facing = figure.body_frame()
        hips = 0.5 * (at("l_hip") + at("r_hip"))
        for part, bone in (("left hand", "wrist.L"), ("right hand", "wrist.R")):
            rel = at(bone) - hips
            notes.append(
                "%s's %s is %.0f cm %s, %.0f cm %s and %.0f cm off the "
                "ground." % (who, part,
                             abs(rel @ np.asarray(facing)),
                             "in front of the hips" if rel @ np.asarray(facing) >= 0
                             else "behind the hips",
                             abs(rel @ np.asarray(up)),
                             "above the hips" if rel @ np.asarray(up) >= 0
                             else "below the hips",
                             at(bone)[1]))
        notes.append("%s's head is %.0f cm off the ground; it stands %.0f cm "
                     "tall in this pose." % (who, at("head")[1],
                                             float(surface[:, 1].max() - low)))

    for prop in props or ():
        low, high = props_module.bounds(prop)
        notes.append("there is a %s in the scene, %.0f cm wide and %.0f cm "
                     "tall, its top at %.0f cm."
                     % (prop["shape"], high[0] - low[0], high[1] - low[1],
                        high[1]))
    return notes


def build_scene(plan, view_w=900, view_h=700, aspect=512.0 / 768.0):
    """A plan -> (figures, props, camera, warnings), ready to render."""
    warnings = []
    figures = []
    props = []
    owned = []              # props per figure, so they travel with their owner
    entries = plan.get("figures") if isinstance(plan, dict) else None
    if not isinstance(entries, list) or not entries:
        entries = [{"commands": []}]
        warnings.append("no figures in the plan; posing one default figure")
    for entry in entries[:7]:             # the editor tints seven at most
        if not isinstance(entry, dict):
            warnings.append("figure entry is not an object; skipped")
            continue
        preset = entry.get("preset")
        if preset not in BODY_PRESETS:
            if preset is not None:
                warnings.append("unknown preset %r; using %s"
                                % (preset, DEFAULT_PRESET))
            preset = DEFAULT_PRESET
        skeleton = rigpose.figure_for(preset)
        mine = []
        warnings.extend(apply_commands(skeleton, entry.get("commands", []),
                                       mine))
        figures.append(skeleton)
        owned.append(mine)
        props.extend(mine)
    if not figures:
        figures = [rigpose.figure_for(DEFAULT_PRESET)]
    for i, skeleton in enumerate(figures[1:], start=1):
        skeleton.translate((70.0 * i, 0.0, 0.0))    # stand them side by side
        # an object was anchored against its figure where that figure stood,
        # so it travels with it rather than staying where the chair was
        for prop in owned[i]:
            prop["position"][0] += 70.0 * i

    camera = Camera(view_w, view_h)
    view = plan.get("camera") if isinstance(plan, dict) else None
    if view is not None and view not in CAMERA_VIEWS:
        warnings.append("unknown camera view %r; choosing one that reads"
                        % (view,))
        view = None
    export = frame_rect(view_w, view_h, aspect)
    if view is None:
        view = legible_view(figures, props=props, rect=export)
    yaw, pitch = CAMERA_VIEWS[view]
    camera.yaw, camera.pitch = math.radians(yaw), math.radians(pitch)
    frame_scene(figures, camera, export, props=props)
    return figures, props, camera, warnings


def buried(figures, camera, props, limit=0.3, rect=None, crowd=0.34):
    """Is the figure lost behind the objects from here?

    A desk placed in front of a seated figure is in front of it from the
    figure's side of the room, and a camera that agrees renders a desk with a
    head over it. Counts the keypoints an object covers from nearer than they
    are; more than `limit` of them and the view is no good, however well it
    shows the pose.

    Keypoints alone are not enough, because eighteen of them are a thin
    sample of a picture: a wall placed right in front covers most of the frame
    while leaving a dozen keypoints technically unobscured, and since a depth
    map shows the nearest surface, the conditioning image is then a slab. So
    `rect` also gets an area test - the share of the export rectangle taken by
    objects standing in front of the figure. Sampled on a grid rather than
    unioning the polygons, because the answer only has to be right to a
    percent or two and the polygons overlap freely.

    A third of the frame, not a share of the figure's own area. Scoring it
    against the figure rejects a desk seen from the side - which is a fair
    picture of someone at a desk, legs behind it as in any photograph from
    that angle - and the next view that reads is the back of their head.
    """
    if not props:
        return False
    solids = solid_quads(props, camera)
    hidden = total = 0
    depths = []
    for figure in figures:
        for i, point in enumerate(figure.points):
            if not figure.visible[i]:
                continue
            total += 1
            sx, sy, depth = camera.project(point)
            depths.append(depth)
            if any(near < depth and inside_polygon(poly, sx, sy)
                   for poly, near, _index, _picked in solids):
                hidden += 1
    if total > 0 and hidden > limit * total:
        return True
    if rect is None or not depths:
        return False
    depths.sort()
    middle = depths[len(depths) // 2]
    nearer = [poly for poly, near, _i, _p in solids if near < middle]
    if not nearer:
        return False
    x0, y0, x1, y1 = rect
    steps_x, steps_y = 40, 56
    covered = 0
    for row in range(steps_y):
        py = y0 + (y1 - y0) * (row + 0.5) / steps_y
        for col in range(steps_x):
            px = x0 + (x1 - x0) * (col + 0.5) / steps_x
            if any(inside_polygon(poly, px, py) for poly in nearer):
                covered += 1
    return covered > crowd * steps_x * steps_y


def view_scores(figures, order=None):
    """{view: how much of the pose it shows}, or None if nothing moved.

    Split out of `legible_view` so a test can assert the contract it actually
    offers - the plainest view that clears the bar, else the best there is -
    rather than measuring something adjacent and hoping the two agree.
    """
    order = order or VIEW_ORDER
    rest = {}
    for name, (a, b) in BONES.items():
        for figure in figures:
            key = (id(figure), name)
            fresh = rigpose.figure_for(figure.body.get('preset', DEFAULT_PRESET))
            rest[key] = vnorm(vsub(fresh.at(b), fresh.at(a)))

    moved = []                       # (change direction, bone direction) pairs
    for figure in figures:
        for name, (a, b) in BONES.items():
            posed = vnorm(vsub(figure.at(b), figure.at(a)))
            change = vsub(posed, rest[(id(figure), name)])
            if vlen(change) > 0.25:  # about 14 degrees; below that it is noise
                moved.append((vnorm(change), posed))
    if not moved:
        return None

    scores = {}
    for name in order:
        yaw, pitch = CAMERA_VIEWS[name]
        camera = Camera()
        camera.yaw, camera.pitch = math.radians(yaw), math.radians(pitch)
        right, up, _fwd = camera.basis()
        seen = lambda d: math.hypot(vdot(d, right), vdot(d, up))
        scores[name] = min(min(seen(change), seen(bone))
                           for change, bone in moved)
    return scores


def legible_view(figures, order=None, readable=0.8, props=(), rect=None):
    """The named view that shows most of what makes this pose that pose.

    Not "is every bone visible": a crouch seen head-on still shows 71% of the
    thigh and reads as a figure standing up straight, because what a crouch
    *is* - the thigh swinging forward from hanging - is the part pointing at
    the camera. So score each view on how much of every bone's departure from
    the rest pose survives projection, and ignore the bones that did not move,
    which have no opinion about the view. A standing figure constrains nothing
    and keeps the front; a seated one turns until the thighs read.

    A bone that moved must also be readable *itself*, not only its change. The
    two come apart: an arm brought up to carry a box swings from hanging to
    pointing forward, and the change from one to the other is mostly vertical,
    so a front view shows 82% of the departure while showing 25% of the arm.
    The score is therefore the smaller of the two, per moved bone - which put
    `carrying_box` on a profile that shows 99% of both instead of a front view
    that hides the arms doing the carrying.

    A threshold rather than a maximum, walking VIEW_ORDER plainest first: only
    a flat profile foreshortens nothing, and a three-quarter that clears the
    bar is the better reference. If nothing clears it, the best is still
    better than guessing - some poses have no good view at all, a cross-legged
    sit being the plain case: its shins point at the lens from everywhere.

    A view where the objects bury the figure is last in every case, not merely
    demoted: a buried figure is not a conditioning image at all, because the
    depth map is then a picture of the desk. A seated figure at a desk reads
    99% from the side and 51% from three-quarters, and the side view puts a
    140 cm desk between the lens and the person - so the three-quarter wins
    even though it is the worse view of the pose. Burial is judged on the
    framed camera, because framing is what decides whether the desk covers
    the figure or sits below it.
    """
    order = order or VIEW_ORDER
    scores = view_scores(figures, order)
    if scores is None:                     # nothing moved: nothing to read
        return order[0]
    rect = rect if rect is not None else frame_rect(900, 700, 512.0 / 768.0)
    clear = []
    for name in order:
        yaw, pitch = CAMERA_VIEWS[name]
        camera = Camera(900, 700)
        camera.yaw, camera.pitch = math.radians(yaw), math.radians(pitch)
        frame_scene(figures, camera, rect, props=props)
        if not buried(figures, camera, props, rect=rect):
            clear.append(name)

    for name in order:
        if name in clear and scores[name] >= readable:
            return name
    pool = clear or order
    return max(pool, key=lambda name: (scores[name], -order.index(name)))





def as_drawn(figures, meshes=None):
    """The figures as the pose map draws them: keypoints off their own rigs.

    There is no second opinion left to reconcile. A figure IS its armature, so
    this asks each one to describe itself in the eighteen the OpenPose format
    wants - `keypoints_of`, read off the posed rig - and hands back something
    the pose rasteriser can draw. `meshes` is accepted and ignored: a figure
    carries its own body now, so there is nothing to pair it with.
    """
    return [figure.as_skeleton() for figure in figures]


def render_scene(figures, camera, out_w, out_h, view_w=900, view_h=700,
                 thickness=1.0, with_depth=True, props=(), meshes=None,
                 anatomy=False, ground=True):
    """(pose image, depth image or None, export rect) for a built scene.

    Objects reach the depth map only. The pose map is the OpenPose skeleton and
    nothing else - a chair drawn into it would be read as a limb.

    The depth map comes from rigged geometry. `meshes` is one loaded mesh per
    figure; leave it out and they are resolved from the body set. There is no
    quiet fallback to the swept anatomy: that is a stack of tapering
    cross-sections, right about where every limb is and approximate about what
    a person looks like, and an export of it is the wrong picture with nothing
    to say so. `anatomy=True` asks for it deliberately - the viewport preview
    does, and so do the tests that check the sweep itself.

    On the rigged path the pose map is drawn from the keypoints the posed rig
    lays out, not from the ones that drove it. The rig is posed rather than
    fitted, so it keeps its own proportions - which are a measured body's and
    not the table's - and the two differ by a few centimetres at the shoulder.
    Drawing the authored keypoints beside a mesh that is not shaped like them
    would put out a pair that disagrees with itself; read off the rig, the
    skeleton is a description of the very thing the depth map shows.
    """
    rect = frame_rect(view_w, view_h, out_w / out_h)
    depth = None
    drawn = figures
    if with_depth and anatomy:
        depth = anatomy_depth_image(figures, camera, rect, out_w, out_h,
                                    thickness, props, ground=ground)
    elif with_depth:
        import bodies_lib
        if meshes is None:
            meshes = bodies_lib.for_figures(figures)
        jobs = [(figure, mesh, getattr(figure, "assets", ()) or ())
                for figure, mesh in zip(figures, meshes) if mesh is not None]
        if not jobs:
            raise bodies_lib.MissingBodies(
                "Nothing to render the depth map from.\n\n" + bodies_lib.HOW)
        depth = rigged_depth_image(jobs, camera, rect, out_w, out_h, props,
                                   ground=ground)
        drawn = as_drawn(figures, meshes)
    pose = pose_image(drawn, camera, rect, out_w, out_h)
    return pose, depth, rect


# ---------------------------------------------------------------------------
# Reading the prompt without a model
# ---------------------------------------------------------------------------

KEYWORDS = [
    # (regex, commands) - first match wins for the stance, all limb rules apply
    (r"\bt.?pose\b", [{"op": "stance", "name": "t_pose"}]),
    (r"\b(sit|sitting|seated|sits)\b", [{"op": "stance", "name": "sitting"}]),
    (r"\b(crawl\w*|on all fours)\b",
     [{"op": "stance", "name": "kneeling"},
      {"op": "lean", "direction": "forward", "degrees": 80},
      {"op": "point", "target": "l_arm", "direction": "down"},
      {"op": "point", "target": "r_arm", "direction": "down"}]),
    (r"\b(kneel\w*)\b", [{"op": "stance", "name": "kneeling"}]),
    (r"\b(crouch|crouching|squat|squatting)\b",
     [{"op": "stance", "name": "crouching"}]),
    (r"\b(run|runner|running|runs|sprint\w*|jog\w*|dash\w*)\b",
     [{"op": "stance", "name": "running"}]),
    (r"\b(walk\w*|strolling|stroll|marching|march)\b",
     [{"op": "stance", "name": "walking"}]),
    (r"\b(box|boxer|boxing|punch\w*|fight\w*|guard|combat|martial)\b",
     [{"op": "stance", "name": "fighting_stance"}]),
    (r"\b(lying|lie|lies|laying|prone|supine)\b",
     [{"op": "stance", "name": "lying_down"}]),
    (r"\b(cheer|cheering|celebrat\w*|arms up|hands up|surrender|reach\w* up)\b",
     [{"op": "stance", "name": "arms_up"}]),
    (r"\b(jump|jumping|leap\w*)\b",
     [{"op": "stance", "name": "arms_up"},
      {"op": "bend", "target": "l_knee", "degrees": 55},
      {"op": "bend", "target": "r_knee", "degrees": 55}]),
]

KEYWORD_CAMERAS = [
    (r"\b(from behind|from the back|rear view|back view)\b", "back"),
    (r"\b(three.?quarters?|3/4)\b", "three_quarter_left"),
    (r"\b(profile|side view|from the side)\b", "left"),
    (r"\b(from above|top.?down|bird)\b", "high"),
    (r"\b(from below|low angle|worm)\b", "low"),
    (r"\b(front view|from the front|facing camera)\b", "front"),
]

KEYWORD_PRESETS = [
    (r"\b(child|kid|boy|girl|young)\b", "Child, about 7"),
    (r"\b(woman|female|she|her|lady)\b", "Female, average"),
    (r"\b(athletic|muscular|fit|bodybuilder)\b", "Male, athletic"),
    (r"\b(heavy|large|fat|stout)\b", "Male, heavy"),
    (r"\b(slim|thin|slender|lean)\b", "Male, slim"),
]


KEYWORD_OBJECTS = [
    (r"\b(chair|armchair)\b", "chair", "under_hips"),
    (r"\b(stool)\b", "stool", "under_hips"),
    (r"\b(bench)\b", "bench", "under_hips"),
    (r"\b(sofa|couch)\b", "bench", "under_hips"),
    (r"\b(desk)\b", "desk", "in_front"),
    (r"\b(table)\b", "table", "in_front"),
    (r"\b(bed)\b", "bed", "under_hips"),
    (r"\b(crate|box)\b", "crate", "in_front"),
    (r"\b(barrel)\b", "barrel", "in_front"),
    (r"\b(wall)\b", "wall", "behind"),
    (r"\b(pillar|column)\b", "pillar", "behind"),
    (r"\b(stairs|steps|staircase)\b", "steps", "in_front"),
    (r"\b(doorway|archway|door frame)\b", "archway", "ground"),
    (r"\b(ball|football|basketball)\b", "ball", "at_hands"),
    (r"\b(floor|ground|standing on)\b", "floor", "ground"),
]


KEYWORD_WEARABLES = [
    (r"\b(long hair|flowing hair)\b", "long"),
    (r"\b(ponytail|pony.?tail)\b", "ponytail"),
    (r"\b(bun|top.?knot|chignon)\b", "bun"),
    (r"\b(bob|bobbed)\b", "bob"),
    (r"\b(afro)\b", "afro"),
    (r"\b(bald|shaved head|buzz.?cut)\b", "shaved"),
    (r"\b(short hair)\b", "short"),
    (r"\b(helmet|knight|visor)\b", "helmet"),
    (r"\b(cap|baseball cap)\b", "cap"),
    (r"\b(hat|fedora|sun.?hat)\b", "hat"),
    (r"\b(beanie|woolly hat|wool hat)\b", "beanie"),
    (r"\b(t.?shirt|tee)\b", "t_shirt"),
    (r"\b(tank top|vest|singlet)\b", "tank_top"),
    (r"\b(hoodie|hooded)\b", "hoodie"),
    (r"\b(jacket|blazer)\b", "jacket"),
    (r"\b(coat|overcoat|trench)\b", "coat"),
    (r"\b(armou?r|breastplate|knight)\b", "armour"),
    (r"\b(long.?sleeve|jumper|sweater|shirt)\b", "long_sleeve"),
    (r"\b(shorts)\b", "shorts"),
    (r"\b(trousers|pants|jeans|leggings)\b", "trousers"),
    (r"\b(long skirt|maxi skirt)\b", "long_skirt"),
    (r"\b(skirt)\b", "skirt"),
    (r"\b(dress|gown)\b", "dress"),
    (r"\b(robe|cloak|wizard|monk)\b", "robe"),
    (r"\b(tall boots|riding boots|thigh boots)\b", "tall_boots"),
    (r"\b(boots)\b", "boots"),
    (r"\b(shoes|trainers|sneakers)\b", "shoes"),
]


# A look the prompt names outright. Read before the single garments, so
# "a chef in a beanie" is a chef who swapped the cap.
KEYWORD_OUTFITS = [
    (r"\b(chef|cook)\b", "chef"),
    (r"\b(barista)\b", "barista"),
    (r"\b(waiter|waitress|server)\b", "waiter"),
    (r"\b(nurse)\b", "nurse"),
    (r"\b(doctor|surgeon|lab coat)\b", "doctor"),
    (r"\b(builder|construction|site worker)\b", "builder"),
    (r"\b(mechanic)\b", "mechanic"),
    (r"\b(farmer)\b", "farmer"),
    (r"\b(gardener|gardening)\b", "gardener"),
    (r"\b(soldier|army|military)\b", "soldier"),
    (r"\b(biker|motorcyclist)\b", "biker"),
    (r"\b(knight|armou?red)\b", "knight"),
    (r"\b(wizard|sorcerer|mage)\b", "wizard"),
    (r"\b(monk)\b", "monk"),
    (r"\b(priest|vicar)\b", "priest"),
    (r"\b(superhero|super.?hero)\b", "superhero"),
    (r"\b(queen|king|royal)\b", "royal"),
    (r"\b(office|at work|desk job)\b", "office"),
    (r"\b(business|meeting)\b", "business"),
    (r"\b(suit|smart)\b", "suit"),
    (r"\b(commuter|commuting)\b", "commuter"),
    (r"\b(student|schoolboy|college)\b", "student"),
    (r"\b(schoolgirl|school uniform)\b", "school"),
    (r"\b(evening dress|ball|gala)\b", "evening"),
    (r"\b(party)\b", "party"),
    (r"\b(sun.?dress)\b", "sundress"),
    (r"\b(beach)\b", "beach"),
    (r"\b(swim\w*)\b", "swimming"),
    (r"\b(gym|workout|weights)\b", "gym"),
    (r"\b(jogging|running|runner)\b", "running"),
    (r"\b(yoga|pilates)\b", "yoga"),
    (r"\b(dancer|dancing|ballet)\b", "dancer"),
    (r"\b(cycling|cyclist)\b", "cycling"),
    (r"\b(hiking|hiker)\b", "hiking"),
    (r"\b(tourist|sightseeing)\b", "tourist"),
    (r"\b(explorer|expedition|safari)\b", "explorer"),
    (r"\b(winter|snow|cold)\b", "winter"),
    (r"\b(rain|raining|wet)\b", "rain"),
    (r"\b(summer|hot day)\b", "summer"),
    (r"\b(pyjamas|pajamas|bedtime)\b", "pyjamas"),
    (r"\b(underwear|in their underwear)\b", "underwear"),
    (r"\b(casual)\b", "casual"),
]


def keyword_outfits(text):
    """A named look the prompt asked for. The first match wins: a sentence
    that reads as two jobs at once is one figure, and it can only have one."""
    for pattern, name in KEYWORD_OUTFITS:
        if re.search(pattern, text):
            return [{"op": "outfit", "outfit": name}]
    return []


def keyword_wearables(text):
    """Clothes and hair the prompt named. First match per slot wins, so
    "a long skirt" is a long skirt rather than a skirt and then a long one."""
    out, taken = [], set()
    for pattern, name in KEYWORD_WEARABLES:
        slot = WEARABLES.get(name)
        if slot in taken or not re.search(pattern, text):
            continue
        taken.add(slot)
        out.append({"op": "wear", "wears": name})
    return out


def keyword_objects(text, commands):
    """Objects the prompt named, plus the one a seated figure cannot do without.

    A figure told to sit with nothing under it reads as a figure hovering, and
    the depth map is the part that gives that away, so a sitting stance with no
    seat named gets a chair. Nothing else is invented: an object nobody asked
    for in the frame is worse than a bare one.
    """
    out = []
    sittable = {"chair", "stool", "bench", "bed", "crate", "barrel",
                "platform", "steps", "box"}
    sitting = any(c.get("name") == "sitting" for c in commands)
    for pattern, shape, at in KEYWORD_OBJECTS:
        if not re.search(pattern, text):
            continue
        # "sitting on a crate" puts the figure on the crate, not next to one:
        # whatever was named goes under the hips if it can be sat on
        if sitting and shape in sittable and re.search(r"\b(on|onto|astride)\b",
                                                      text):
            at = "under_hips"
        out.append({"op": "place", "shape": shape, "at": at})
    if sitting and not any(c["at"] == "under_hips" for c in out):
        out.append({"op": "place", "shape": "chair", "at": "under_hips"})
    return out


def keyword_plan(prompt):
    """What can be read out of a prompt with no model at all.

    Deliberately shallow. It exists so the CLI still does something sensible
    with no model running, so the self-test can exercise the whole path in CI,
    and so a run can say plainly which of the two read the prompt.
    """
    text = " " + prompt.lower() + " "
    commands = []
    for pattern, steps in KEYWORDS:
        if re.search(pattern, text):
            commands.extend(steps)
            break
    if re.search(r"\b(arms? (out|wide)|spread)\b", text) and not commands:
        commands.append({"op": "stance", "name": "t_pose"})
    commands.extend(keyword_outfits(text))
    commands.extend(keyword_wearables(text))
    commands.extend(keyword_objects(text, commands))
    if re.search(r"\b(lean\w* forward|bent over|bowing|bow)\b", text):
        commands.append({"op": "lean", "direction": "forward", "degrees": 30})
    if re.search(r"\b(lean\w* back|arch\w*)\b", text):
        commands.append({"op": "lean", "direction": "back", "degrees": 20})
    if re.search(r"\blook\w* (up|upward)\b", text):
        commands.append({"op": "look", "direction": "forward_up"})
    if re.search(r"\blook\w* (down|downward)\b", text):
        commands.append({"op": "look", "direction": "forward_down"})
    if re.search(r"\blook\w* (left|to the left)\b", text):
        commands.append({"op": "look", "direction": "forward_left"})
    if re.search(r"\blook\w* (right|to the right)\b", text):
        commands.append({"op": "look", "direction": "forward_right"})

    preset = DEFAULT_PRESET
    for pattern, name in KEYWORD_PRESETS:
        if re.search(pattern, text):
            preset = name
            break
    plan = {"figures": [{"preset": preset, "commands": commands}],
            "notes": "read by keyword; no local model answered"}
    for pattern, name in KEYWORD_CAMERAS:
        if re.search(pattern, text):
            plan["camera"] = name     # only when the prompt asked for a view;
            break                     # otherwise build_scene picks one that reads
    return plan


# ---------------------------------------------------------------------------
# The local model
# ---------------------------------------------------------------------------

DEFAULT_ENDPOINTS = [
    ("ollama", "http://localhost:11434"),      # Ollama
    ("openai", "http://localhost:1234"),       # LM Studio
    ("openai", "http://localhost:8080"),       # llama.cpp server
    ("openai", "http://localhost:8000"),       # vLLM
]


class Sampling:
    """What the model is asked to do with its probabilities, in one place.

    The defaults are Qwen3's own published recommendation for THINKING mode -
    temperature 0.6, top_p 0.95, top_k 20, min_p 0 - rather than something
    picked here. What they replaced was a hard-coded temperature of 0.3 and
    nothing else, which is a poor setting for any reasoning model and has no
    answer at all for a runtime that wants top_k.

    `reasoning` is "high", "medium", "low" or "off". Ollama takes it as the
    top-level `think` field, which accepts a bool or one of those words;
    everything OpenAI-compatible takes `chat_template_kwargs.enable_thinking`,
    and some servers also read `reasoning_effort`. All three are sent, because
    a server that does not know a field ignores it, and a server that does is
    the one we wanted to reach.
    """

    FIELDS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
              "repeat_penalty", "max_tokens", "seed", "reasoning")

    def __init__(self, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                 presence_penalty=0.0, repeat_penalty=1.0, max_tokens=None,
                 seed=None, reasoning="high"):
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.min_p = float(min_p)
        self.presence_penalty = float(presence_penalty)
        self.repeat_penalty = float(repeat_penalty)
        self.max_tokens = max_tokens
        self.seed = seed
        self.reasoning = str(reasoning).lower()

    def replace(self, **changes):
        out = Sampling(**{f: getattr(self, f) for f in self.FIELDS})
        for k, v in changes.items():
            setattr(out, k, v)
        return out

    def as_dict(self):
        return {f: getattr(self, f) for f in self.FIELDS}

    @property
    def thinking(self):
        return self.reasoning not in ("off", "none", "false", "0", "")

    def ollama_options(self):
        out = {"temperature": self.temperature, "top_p": self.top_p,
               "top_k": self.top_k, "min_p": self.min_p,
               "repeat_penalty": self.repeat_penalty}
        if self.presence_penalty:
            out["presence_penalty"] = self.presence_penalty
        if self.max_tokens:
            out["num_predict"] = int(self.max_tokens)
        if self.seed is not None:
            out["seed"] = int(self.seed)
        return out

    def openai_body(self):
        out = {"temperature": self.temperature, "top_p": self.top_p}
        if self.presence_penalty:
            out["presence_penalty"] = self.presence_penalty
        if self.max_tokens:
            out["max_tokens"] = int(self.max_tokens)
        if self.seed is not None:
            out["seed"] = int(self.seed)
        # top_k and min_p are not in the OpenAI spec; llama.cpp, vLLM and
        # LM Studio all read them anyway, and a server that does not simply
        # ignores them.
        out["top_k"] = self.top_k
        out["min_p"] = self.min_p
        return out


# Qwen publishes one set of numbers for thinking and a different set for
# answering without it, and the gap is not small - temperature 0.6 against
# 0.7, top_p 0.95 against 0.8. The extraction pass below runs with thinking
# off, so it gets the second set rather than the first.
NON_THINKING = {"temperature": 0.7, "top_p": 0.8}


class LocalLLM:
    """A local chat model over HTTP.

    `backend` is "ollama" for Ollama's own /api/chat, or "openai" for the
    OpenAI-compatible /v1/chat/completions that LM Studio, llama.cpp's server
    and vLLM all serve. A local server usually wants no key; one that does
    takes it from --api-key or the API key box on the Pose tab.
    """

    def __init__(self, host, backend="openai", model=None, timeout=600.0,
                 api_key=None, sampling=None):
        self.host = host.rstrip("/")
        self.backend = backend
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.sampling = sampling or Sampling()
        self.last_thinking = ""

    # -- wire ------------------------------------------------------------
    def _post(self, path, payload):
        request = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        if self.api_key:
            request.add_header("Authorization", "Bearer " + self.api_key)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _get(self, path, timeout=None):
        request = urllib.request.Request(self.host + path, method="GET")
        if self.api_key:
            request.add_header("Authorization", "Bearer " + self.api_key)
        with urllib.request.urlopen(request,
                                    timeout=timeout or self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def models(self):
        if self.backend == "ollama":
            return [m.get("name") or m.get("model")
                    for m in self._get("/api/tags", timeout=5.0).get("models", [])]
        return [m.get("id") for m in self._get("/v1/models",
                                               timeout=5.0).get("data", [])]

    def resolve_model(self):
        if self.model:
            return self.model
        available = [m for m in self.models() if m]
        if not available:
            raise RuntimeError("%s has no model loaded" % self.host)
        self.model = available[0]
        return self.model

    # -- one exchange ----------------------------------------------------
    def _bodies(self, messages, sampling, schema):
        """Every spelling of one request, best first.

        Runtimes disagree about all three of the things that matter here - how
        a schema is passed, how thinking is switched on, and where the sampler
        settings live - so each is sent in the spelling the server is most
        likely to know and the next one is tried on a refusal. A server
        ignores a field it does not recognise, which is what makes sending
        several at once safe.
        """
        model = self.resolve_model()
        if self.backend == "ollama":
            base = {"model": model, "messages": messages, "stream": False,
                    "options": sampling.ollama_options()}
            if sampling.thinking:
                base["think"] = (True if sampling.reasoning == "on"
                                 else sampling.reasoning)
            else:
                base["think"] = False
            out = []
            if schema:
                out.append(dict(base, format=schema))
                out.append(dict(base, format="json"))
            else:
                out.append(dict(base))
            # a build too old for `think` refuses the whole request
            out += [dict(b) for b in out]
            for b in out[len(out) // 2:]:
                b.pop("think", None)
            return "/api/chat", out

        base = dict({"model": model, "messages": messages, "stream": False},
                    **sampling.openai_body())
        if sampling.thinking:
            base["chat_template_kwargs"] = {"enable_thinking": True}
            base["reasoning_effort"] = (
                sampling.reasoning if sampling.reasoning in
                ("high", "medium", "low") else "high")
        else:
            base["chat_template_kwargs"] = {"enable_thinking": False}
        out = []
        if schema:
            out.append(dict(base, response_format={
                "type": "json_schema",
                "json_schema": {"name": "pose_plan", "strict": True,
                                "schema": schema}}))
            out.append(dict(base, response_format={"type": "json_object",
                                                   "schema": schema}))
            out.append(dict(base, response_format={"type": "json_object"}))
        else:
            out.append(dict(base))
        plain = []
        for b in out:                      # same again without the thinking
            c = dict(b)                    # fields, for a server that refuses
            c.pop("chat_template_kwargs", None)
            c.pop("reasoning_effort", None)
            plain.append(c)
        return "/v1/chat/completions", out + plain

    def _say(self, messages, sampling, schema=None):
        path, attempts = self._bodies(messages, sampling, schema)
        last = None
        for payload in attempts:
            try:
                reply = self._post(path, payload)
            except (urllib.error.HTTPError, urllib.error.URLError,
                    ValueError) as exc:
                last = exc
                continue
            text, thinking = self._content(reply)
            if thinking:
                self.last_thinking = thinking
            if text and text.strip():
                return text
            last = RuntimeError("empty reply")
        raise RuntimeError("%s: %s" % (self.host, last))

    # -- generation ------------------------------------------------------
    def think_aloud(self, prompt):
        """Work the pose out in words, with no schema in the way.

        This is the half that a constrained call cannot do. A JSON grammar
        forces the first token to be `{`, so the model commits to an answer
        before it has considered anything; and on llama.cpp asking for both a
        grammar and thinking at once silently drops the grammar (ggml-org
        issue #20345), so "both" is not on offer either. Reason here, extract
        below, and each half gets the settings it wants.
        """
        return self._say([{"role": "system", "content": planning_prompt()},
                          {"role": "user", "content": with_examples(prompt)}],
                         self.sampling)

    def extract(self, prompt, notes, schema):
        """Turn a worked-out plan into commands, under the schema.

        Thinking off and Qwen's non-thinking sampler settings: there is
        nothing left to work out, the answer is in `notes`, and the only job
        is to write it in the vocabulary without inventing a name.
        """
        sampling = self.sampling.replace(reasoning="off", **NON_THINKING)
        user = prompt if not notes else (
            "%s\n\nA plan for this pose, worked out already. Turn it into "
            "commands, changing nothing about what it decided:\n\n%s"
            % (prompt, notes))
        text = self._say([{"role": "system", "content": system_prompt()},
                          {"role": "user", "content": with_examples(user)}],
                         sampling, schema)
        parsed = parse_json_object(text)
        if parsed is None:
            raise RuntimeError("model did not return JSON: %.200r" % (text,))
        return parsed

    def revise(self, prompt, plan, findings, schema):
        """A second pass, given measurements of what the first one built.

        The findings are the point. Re-reading its own commands gets the same
        commands back, because the reasoning that wrote them is the reasoning
        being asked to find the fault. Being told that the left hand came out
        4 cm inside the ribcage is something it could not have worked out and
        cannot argue with.
        """
        told = ("Here is the pose you asked for:\n\n%s\n\nHere is what was "
                "built from it, measured:\n\n%s\n\nFix what is wrong and "
                "leave what is right alone. Answer with the whole plan again."
                % (json.dumps(plan, indent=1), "\n".join("- " + f
                                                          for f in findings)))
        return self.extract(prompt, told, schema)

    def complete(self, prompt, schema, notes=None):
        """A plan for this prompt: reason, then write it down.

        Two calls rather than one. Kept as `complete` because that is what
        every caller and the test stub already ask for.
        """
        self.last_thinking = ""
        thought = ""
        if self.sampling.thinking:
            try:
                thought = self.think_aloud(prompt)
            except RuntimeError:
                thought = ""              # a model that cannot, still answers
        if notes:
            thought = (thought + "\n\n" + notes) if thought else notes
        return self.extract(prompt, thought, schema)

    @staticmethod
    def _content(reply):
        """(answer, thinking). Runtimes put the reasoning in three places:
        Ollama in `message.thinking`, OpenAI-compatible servers in
        `message.reasoning_content` or `message.reasoning`."""
        message = reply.get("message")
        if message is None:
            choices = reply.get("choices") or [{}]
            message = choices[0].get("message") or {}
        return (message.get("content", ""),
                message.get("thinking")
                or message.get("reasoning_content")
                or message.get("reasoning") or "")


def parse_json_object(text):
    """The first JSON object in a reply, fence or chatter included.

    Constrained decoding makes this unnecessary; unconstrained fallbacks make
    it essential, because a small model will wrap its answer in ```json more
    often than not.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except ValueError:
            pass
    start = text.find("{")
    while start >= 0:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


def discover(backend="auto", host=None, model=None, api_key=None,
             timeout=600.0, sampling=None):
    """The first local model server that answers, or None.

    `auto` probes the default ports of the four runtimes worth probing. A run
    with no server up is not an error - it falls back to reading the prompt by
    keyword - so this returns None rather than raising.
    """
    if host:
        kinds = [backend] if backend != "auto" else ["ollama", "openai"]
        for kind in kinds:
            client = LocalLLM(host, kind, model, timeout, api_key,
                              sampling)
            try:
                client.resolve_model()
                return client
            except Exception:
                continue
        return None
    for kind, endpoint in DEFAULT_ENDPOINTS:
        if backend not in ("auto", kind):
            continue
        client = LocalLLM(endpoint, kind, model, timeout, api_key,
                          sampling)
        try:
            client.resolve_model()
            return client
        except Exception:
            continue
    return None


def plan_for(prompt, llm=None):
    """(plan, source, warnings). `source` says which route read the prompt."""
    if llm is None:
        return keyword_plan(prompt), "keywords", []
    try:
        plan = llm.complete(prompt, response_schema())
    except Exception as exc:
        plan = keyword_plan(prompt)
        return plan, "keywords", ["%s failed, read the prompt by keyword: %s"
                                  % (llm.host, exc)]
    if not isinstance(plan, dict):
        return keyword_plan(prompt), "keywords", [
            "model returned %s, not an object" % type(plan).__name__]
    return plan, "%s (%s)" % (llm.model, llm.host), []


def pose_from_prompt(prompt, llm=None, view_w=900, view_h=700,
                     aspect=512.0 / 768.0, passes=2, progress=None):
    """Everything between a sentence and a posed scene.

    `passes` is how many times the model gets to answer. The first builds a
    scene; each one after that is shown what was actually built, measured in
    centimetres by `critique`, and asked to fix what is wrong. Two is the
    useful default: the first pass gets the shape of the pose right and the
    second catches the figure standing inside its own arm.

    Returns (figures, props, camera, report), report carrying the plan, which
    route read the prompt, every command that was skipped, and what each pass
    was told.
    """
    say = progress or (lambda _text: None)
    say("thinking about the pose")
    plan, source, warnings = plan_for(prompt, llm)
    figures, props, camera, more = build_scene(plan, view_w, view_h, aspect)
    rounds = []
    for extra in range(max(0, int(passes) - 1)):
        if llm is None:
            break
        findings = [f for f in critique(figures, props)
                    if not f.startswith("there is a")]
        faults = [f for f in findings if " is " in f and
                  ("floor" in f or "fall over" in f or "inside" in f
                   or "no `hand`" in f)]
        rounds.append(findings)
        if not faults:
            break
        say("checking pass %d: %s" % (extra + 2, faults[0][:60]))
        try:
            revised = llm.revise(prompt, plan, findings, response_schema())
        except (RuntimeError, urllib.error.URLError):
            break
        built = build_scene(revised, view_w, view_h, aspect)
        if not built[0]:
            break
        plan, source = revised, source + "+checked"
        figures, props, camera, more = built
    say("done")
    return figures, props, camera, {"plan": plan, "source": source,
                                    "warnings": warnings + more,
                                    "findings": rounds,
                                    "thinking": getattr(llm, "last_thinking",
                                                        "")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def describe_vocabulary():
    return "\n".join([
        "ops        : " + ", ".join(OPS),
        "stances    : " + ", ".join(sorted(STANCES)),
        "point      : " + ", ".join(POINT_TARGETS),
        "bend       : " + ", ".join(sorted(BEND_JOINTS)),
        "directions : " + ", ".join(sorted(DIRECTIONS)),
        "cameras    : " + ", ".join(sorted(CAMERA_VIEWS)),
        "wearables  : " + ", ".join(WEARABLE_NAMES),
        "outfits    : " + ", ".join(OUTFIT_NAMES),
        "shapes     : " + ", ".join(SHAPE_NAMES),
        "anchors    : " + ", ".join(sorted(ANCHORS)),
        "presets    : " + ", ".join(sorted(BODY_PRESETS)),
    ])


def run(args):
    # --size is the same setting as --width/--height and easier to get right,
    # since nobody remembers what 9:16 is in pixels. Resolved once, here, so
    # everything downstream sees one pair of numbers.
    if getattr(args, "size", None):
        args.width, args.height = parse_size(args.size)
    llm = None
    if not args.no_llm:
        llm = discover(args.backend, args.host, args.model, args.api_key,
                       args.timeout)
        if llm is None and args.require_llm:
            print("no local model answered on %s"
                  % (args.host or ", ".join(e for _k, e in DEFAULT_ENDPOINTS)),
                  file=sys.stderr)
            return 2
    figures, props, camera, report = pose_from_prompt(
        args.prompt, llm, aspect=args.width / args.height)
    print("prompt read by: %s%s" % (report["source"],
                                    "" if not props else
                                    "; %d object(s) placed" % len(props)))
    for warning in report["warnings"]:
        print("  warning: %s" % warning, file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    out_w, out_h = args.width, args.height
    pose, depth, rect = render_scene(figures, camera, out_w, out_h,
                                     thickness=args.thickness,
                                     with_depth=not args.no_depth, props=props,
                                     ground=not args.no_ground)
    written = []
    pose_path = os.path.join(args.out, "pose.png")
    pose.save(pose_path)
    written.append(pose_path)
    if depth is not None:
        depth_path = os.path.join(args.out, "depth.png")
        depth.save(depth_path)
        written.append(depth_path)

    scene = scene_to_dict(figures, camera,
                          [pts for pts, _vis in
                           project_people(figures, camera, rect, out_w, out_h)],
                          out_w, out_h, props)
    scene["prompt"] = args.prompt
    scene["plan"] = report["plan"]
    scene_path = os.path.join(args.out, "scene.json")
    with open(scene_path, "w", encoding="utf-8") as fh:
        json.dump(scene, fh, indent=2)
    written.append(scene_path)
    print("wrote " + ", ".join(written))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pose the 3D OpenPose figure from a text prompt with a "
                    "local LLM, and export the pose and depth conditioning "
                    "images.")
    parser.add_argument("--prompt", help="what the figure should be doing")
    parser.add_argument("--out", default="out", help="folder to write into")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--size", metavar="WxH|RATIO",
                        help="export shape, e.g. 768x512 or 16:9; overrides "
                             "--width/--height")
    parser.add_argument("--thickness", type=float, default=1.0,
                        help="body thickness for the depth map")
    parser.add_argument("--no-ground", action="store_true",
                        help="leave the floor out of the depth map; the "
                             "figure then reads as floating")
    parser.add_argument("--no-depth", action="store_true",
                        help="pose image only")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "ollama", "openai"],
                        help="ollama = /api/chat, openai = /v1/chat/completions "
                             "(LM Studio, llama.cpp, vLLM)")
    parser.add_argument("--host", default=os.environ.get("POSE_AGENT_HOST"),
                        help="e.g. http://localhost:11434; probes the usual "
                             "ports when omitted")
    parser.add_argument("--model", default=os.environ.get("POSE_AGENT_MODEL"),
                        help="model name; the first one loaded when omitted")
    parser.add_argument("--api-key", default=os.environ.get("POSE_AGENT_KEY"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-llm", action="store_true",
                        help="read the prompt by keyword, contact nothing")
    parser.add_argument("--require-llm", action="store_true",
                        help="fail instead of falling back to keywords")
    parser.add_argument("--list", action="store_true",
                        help="print the command vocabulary and exit")
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.list:
        print(describe_vocabulary())
        return 0
    if args.selftest:
        return _selftest()
    if not args.prompt:
        build_parser().print_help()
        return 1
    return run(args)


# ---------------------------------------------------------------------------
# Self-test: no model, no network, no model files
# ---------------------------------------------------------------------------

def _selftest():
    ok = True

    def check(label, condition, extra=""):
        nonlocal ok
        ok = ok and bool(condition)
        print(("PASS " if condition else "FAIL ") + label
              + ("  " + extra if extra else ""))

    def lengths(figure):
        """Every bone of the armature, not the seventeen keypoint limbs.

        A hundred and three segments rather than seventeen, and the invariant
        is structural now rather than hopeful: every op applies a rotation to
        a bone, and a rotation cannot change a length. It stays because what
        it is really testing is that nothing in the vocabulary has quietly
        started writing a coordinate again.
        """
        q = figure.pose.positions()
        parents = figure.pose.parents
        return [float(vlen(vsub(tuple(q[j]), tuple(q[int(parents[j])]))))
                for j in range(len(figure.pose.names)) if parents[j] >= 0]

    print("pose_agent self-test (editor %s)" % VERSION)

    # the vocabulary the model is shown must be the vocabulary that works
    schema = response_schema()
    item = schema["properties"]["figures"]["items"]["properties"]["commands"]
    enums = item["items"]["properties"]
    check("schema offers exactly the ops that are implemented",
          set(enums["op"]["enum"]) == set(OPS))
    text = system_prompt()
    check("every stance in the prompt is a stance that exists",
          all(name in text for name in STANCES))
    check("every direction named in a stance resolves",
          all(step.get("direction", "up") in DIRECTIONS
              for steps in STANCES.values() for step in steps))
    check("every target named in a stance resolves",
          all(step.get("target", "l_arm") in POINT_TARGETS or
              step.get("target") in BEND_JOINTS
              for steps in STANCES.values() for step in steps
              if step["op"] in ("point", "bend")))

    # the invariant that made commands the right shape in the first place
    worst = 0.0
    for name in sorted(STANCES):
        skeleton = rigpose.figure_for(DEFAULT_PRESET)
        before = lengths(skeleton)
        # with a scene to put things in: a stance may seat the figure on a
        # chair, and "place needs a scene" is the right refusal when there is
        # nowhere to put one, not a broken stance
        warnings = apply_commands(skeleton, [{"op": "stance", "name": name}],
                                  [])
        after = lengths(skeleton)
        worst = max(worst, max(abs(a - b) for a, b in zip(before, after)))
        if warnings:
            check("stance %s applies cleanly" % name, False, str(warnings))
    check("no stance changes a bone length", worst < 1e-9, "%.2e cm" % worst)

    every = [{"op": "point", "target": t, "direction": d}
             for t in POINT_TARGETS for d in DIRECTIONS]
    every += [{"op": "bend", "target": t, "degrees": d}
              for t in BEND_JOINTS for d in (-90, -30, 30, 90)]
    every += [{"op": o, "direction": d, "degrees": g}
              for o in ("turn", "lean")
              for d in ("left", "right", "forward", "back") for g in (-45, 45)]
    every += [{"op": "look", "direction": d} for d in DIRECTIONS]
    skeleton = rigpose.figure_for(DEFAULT_PRESET)
    before = lengths(skeleton)
    warnings = apply_commands(skeleton, every)
    drift = max(abs(a - b) for a, b in zip(before, lengths(skeleton)))
    check("every command in the vocabulary applies", not warnings, str(warnings[:3]))
    check("and %d of them in a row change no bone length" % len(every),
          drift < 1e-6, "%.2e cm" % drift)

    # A bend has to CLOSE the joint it names, by the angle it was given, and
    # do it the same way whatever the figure has already been turned to.
    #
    # What this asked before was that an elbow moves the wrist "forwards" and
    # a knee moves the ankle "back". That is a property of the rest pose it
    # was written against - a keypoint figure whose arms stick out sideways,
    # where folding the elbow does carry the hand forward. A real body rests
    # in an A-pose with the arms down and the palms inward, and folding that
    # elbow brings the hand up and across the chest, which is what a curl
    # looks like and reads as -0.3 cm forward. Measuring the included angle
    # instead tests what "bend" means rather than where the arms happened to
    # start, and still catches a sign that opens the joint instead of closing
    # it, which is what the original was guarding.
    def included(figure, a, b, c):
        pose = figure.pose
        q = pose.positions()
        i, j, k = pose.bone(a), pose.bone(b), pose.bone(c)
        u = vnorm(vsub(tuple(q[i]), tuple(q[j])))
        v = vnorm(vsub(tuple(q[k]), tuple(q[j])))
        return math.degrees(math.acos(max(-1.0, min(1.0, vdot(u, v)))))

    for turned in (0.0, 90.0, 180.0):
        skeleton = rigpose.figure_for(DEFAULT_PRESET)
        apply_commands(skeleton, [{"op": "turn", "direction": "left",
                                   "degrees": turned}])
        elbow = ("upperarm01.L", "lowerarm01.L", "wrist.L")
        was = included(skeleton, *elbow)
        apply_commands(skeleton, [{"op": "bend", "target": "l_elbow",
                                   "degrees": 90}])
        closed = was - included(skeleton, *elbow)
        check("an elbow closes 90 deg with the figure turned %.0f deg" % turned,
              abs(closed - 90.0) < 1.0, "closed %.1f deg" % closed)
        knee = ("upperleg01.L", "lowerleg01.L", "foot.L")
        was = included(skeleton, *knee)
        apply_commands(skeleton, [{"op": "bend", "target": "l_knee",
                                   "degrees": 90}])
        closed = was - included(skeleton, *knee)
        check("and a knee closes by the same 90",
              abs(closed - 90.0) < 1.0, "closed %.1f deg" % closed)
        # and it folds the way a leg folds: the heel goes behind the figure
        _side, _up, facing = body_frame(skeleton)
        heel = vsub(skeleton.at("l_ankle"), skeleton.at("l_knee"))
        check("and the shin ends up behind the knee",
              vdot(vnorm(heel), facing) < -0.2,
              "cos %.2f" % vdot(vnorm(heel), facing))

    # pointing is in the figure's frame, not the world's
    skeleton = rigpose.figure_for(DEFAULT_PRESET)
    apply_commands(skeleton, [{"op": "turn", "direction": "left", "degrees": 90},
                              {"op": "point", "target": "l_arm",
                               "direction": "forward"}])
    _side, _up, facing = body_frame(skeleton)
    arm = vnorm(vsub(skeleton.at("l_elbow"),
                     skeleton.at("l_shoulder")))
    check("a limb pointed forward follows the figure, not the world",
          vdot(arm, facing) > 0.999, "cos %.4f" % vdot(arm, facing))

    # bad input is survivable
    skeleton = rigpose.figure_for(DEFAULT_PRESET)
    rubbish = [{"op": "wave"}, {"op": "point", "target": "tail",
                                "direction": "up"},
               {"op": "bend", "target": "l_elbow", "degrees": "lots"},
               {"op": "bend", "target": "l_elbow", "degrees": 1e9},
               "not an object", {"op": "stance", "name": "moonwalk"},
               {"op": "point", "target": "l_arm", "direction": "widdershins"},
               {"op": "bend", "target": "l_elbow", "degrees": 60}]
    warnings = apply_commands(skeleton, rubbish)
    check("seven bad commands are reported and skipped", len(warnings) == 7,
          "%d warnings" % len(warnings))
    check("and the eighth, which was good, still applied",
          vlen(vsub(skeleton.at("l_wrist"),
                    rigpose.figure_for(DEFAULT_PRESET).at("l_wrist"))) > 1.0)

    # replies a small model actually sends
    good = {"figures": [{"commands": [{"op": "stance", "name": "running"}]}]}
    for label, text in (
            ("bare JSON", json.dumps(good)),
            ("fenced JSON", "```json\n%s\n```" % json.dumps(good)),
            ("JSON after chatter", "Sure! Here you go:\n%s" % json.dumps(good)),
            ("JSON with a brace in a string",
             '{"figures":[{"commands":[]}],"notes":"use { carefully"}')):
        check("a reply as %s parses" % label,
              isinstance(parse_json_object(text), dict))
    check("a reply with no JSON at all parses to nothing",
          parse_json_object("I am afraid I cannot do that") is None)

    # objects
    every = [{"op": "place", "shape": shape, "at": at}
             for shape in SHAPE_NAMES for at in ANCHORS]
    skeleton = rigpose.figure_for(DEFAULT_PRESET)
    before = lengths(skeleton)
    trouble = [w for command in every
               for w in apply_commands(skeleton, [command], [])]
    check("every shape can be placed at every anchor", not trouble,
          "%d shapes x %d anchors; %s"
          % (len(SHAPE_NAMES), len(ANCHORS), trouble[:2]))
    check("and placing objects never touches the skeleton",
          max(abs(a - b) for a, b in zip(before, lengths(skeleton))) < 1e-9)
    scene = []
    warnings = apply_commands(skeleton, every[:30], scene)
    check("the scene fills up to a limit rather than without bound",
          len(scene) == 24 and len(warnings) == 6
          and "too many" in warnings[-1], "%d placed" % len(scene))

    # clothes
    skeleton = rigpose.figure_for(DEFAULT_PRESET)
    before = lengths(skeleton)
    trouble = [w for name in WEARABLE_NAMES
               for w in apply_commands(skeleton,
                                       [{"op": "wear", "wears": name}], [])]
    check("every wearable can be worn", not trouble, str(trouble[:2]))
    check("and wearing them changes no bone length",
          max(abs(a - b) for a, b in zip(before, lengths(skeleton))) < 1e-9)
    check("the last of each slot is what stays on",
          skeleton.outfit == {slot: sorted(n for n in WEARABLE_NAMES
                                           if WEARABLES[n] == slot)[-1]
                              for slot in wearables.SLOTS},
          str(skeleton.outfit))

    skeleton = rigpose.figure_for(DEFAULT_PRESET)
    apply_commands(skeleton, [{"op": "wear", "wears": "jacket"},
                              {"op": "wear", "wears": "trousers"},
                              {"op": "wear", "wears": "nothing at all"}], [])
    check("slots stack rather than replacing one another",
          skeleton.outfit == {"top": "jacket", "bottom": "trousers"},
          str(skeleton.outfit))
    check("and a garment nobody has is reported, not worn",
          apply_command(rigpose.figure_for(DEFAULT_PRESET),
                        {"op": "wear", "wears": "a sou'wester"}) is not None)

    # named outfits
    skeleton = rigpose.figure_for(DEFAULT_PRESET)
    before = lengths(skeleton)
    trouble = [w for name in OUTFIT_NAMES
               for w in apply_commands(skeleton,
                                       [{"op": "outfit", "outfit": name}], [])]
    check("every outfit can be put on", not trouble, str(trouble[:2]))
    check("and putting them on changes no bone length",
          max(abs(a - b) for a, b in zip(before, lengths(skeleton))) < 1e-9)
    check("an outfit nobody has is reported, not worn",
          apply_command(rigpose.figure_for(DEFAULT_PRESET),
                        {"op": "outfit", "outfit": "black tie"}) is not None)

    # An outfit sets the slots it names and leaves the rest alone, which is
    # what lets a haircut chosen either side of it survive.
    skeleton = rigpose.figure_for(DEFAULT_PRESET)
    apply_commands(skeleton, [{"op": "wear", "wears": "ponytail"},
                              {"op": "outfit", "outfit": "winter"}], [])
    check("an outfit leaves the slots it does not name alone",
          skeleton.outfit.get("hair") == "ponytail"
          and skeleton.outfit.get("top") == "coat", str(skeleton.outfit))
    apply_commands(skeleton, [{"op": "wear", "wears": "cap"}], [])
    check("and a garment after it still lands",
          skeleton.outfit.get("headgear") == "cap", str(skeleton.outfit))
    apply_commands(skeleton, [{"op": "outfit", "outfit": "bare"}], [])
    check("\"bare\" is the one that names every slot",
          not wearables.worn(skeleton.outfit), str(skeleton.outfit))
    check("a whole look asked for through `wear` is still understood",
          apply_command(skeleton, {"op": "wear", "wears": "chef"}) is None
          and skeleton.outfit.get("headgear") == "cap", str(skeleton.outfit))
    check("a prompt that names a job is dressed for it",
          keyword_plan("a chef chopping onions")["figures"][0]["commands"]
          [-1:] != []
          and any(c.get("outfit") == "chef" for c in
                  keyword_plan("a chef chopping onions")["figures"][0]
                  ["commands"]))

    # clothes have to reach the depth map and stay out of the pose map
    dressed, _scene, camera, _r = build_scene({"figures": [{"commands": [
        {"op": "wear", "wears": "coat"}, {"op": "wear", "wears": "long"}]}]})
    bare, _s2, _c2, _r2 = build_scene({"figures": [{"commands": []}]})
    rect = frame_rect(900, 700, 512.0 / 768.0)
    lit = render_scene(dressed, camera, 128, 192)
    plain = render_scene(bare, camera, 128, 192)
    check("a coat shows up in the depth map",
          sum(1 for v in lit[1].tobytes() if v)
          > sum(1 for v in plain[1].tobytes() if v) + 200,
          "%d px vs %d" % (sum(1 for v in lit[1].tobytes() if v),
                           sum(1 for v in plain[1].tobytes() if v)))
    check("and never in the pose map",
          [p for p, _v in project_people(dressed, camera, rect, 128, 192)]
          == [p for p, _v in project_people(bare, camera, rect, 128, 192)])

    # A seat has to reach from the floor to the hips, whatever the figure did,
    # and it has to do it for a child as well as an adult - which is the whole
    # point of anchoring a height rather than writing one.
    #
    # `sitting_on_floor` used to be the third case here and is not any more.
    # It passed on a bug: nothing put the posed figure back on the ground, so
    # folding the legs left it hovering 83 cm up, and the "chair" that reached
    # from the apparent floor to its hips was that gap. Grounded properly the
    # hips ARE on the floor, there is no gap, and a chair under someone
    # already sitting on the floor is a contradiction rather than a
    # measurement. What it collapses to is checked just below instead.
    for stance, preset in (("sitting", DEFAULT_PRESET),
                           ("sitting", "Child, about 7"),
                           ("sitting", "Female, average")):
        skeleton = rigpose.figure_for(preset)
        seat = []
        apply_commands(skeleton, [{"op": "stance", "name": stance},
                                  {"op": "place", "shape": "chair",
                                   "at": "under_hips"}], seat)
        low, high = props_module.bounds(seat[0])
        hips = 0.5 * (skeleton.at("l_hip")[1]
                      + skeleton.at("r_hip")[1])
        floor = ground_level(skeleton)
        check("a chair under %s in %s reaches floor to hip"
              % (stance, preset.split(",")[0].lower()),
              abs(low[1] - floor) < 1e-6 and hips - 14.0 < high[1] <= hips,
              "floor %.0f seat %.0f hip %.0f" % (floor, high[1], hips))

    # anchors are read in the figure's frame, like every other command
    for turn in (0.0, 90.0, 180.0):
        skeleton = rigpose.figure_for(DEFAULT_PRESET)
        scene = []
        apply_commands(skeleton, [{"op": "turn", "direction": "left",
                                   "degrees": turn},
                                  {"op": "place", "shape": "crate",
                                   "at": "in_front", "distance": 90.0}], scene)
        _side, _up, facing = body_frame(skeleton)
        hips = vmul(vadd(skeleton.at("r_hip"),
                         skeleton.at("l_hip")), 0.5)
        to_crate = vsub(scene[0]["position"], (hips[0], scene[0]["position"][1],
                                               hips[2]))
        check("an object in front of a figure turned %.0f deg is in front of it"
              % turn, vdot(vnorm(to_crate), facing) > 0.999,
              "cos %.4f" % vdot(vnorm(to_crate), facing))

    # and a figure with nothing under its hips gets the minimum, not a slab
    floored = rigpose.figure_for("Female, average")
    seat = []
    apply_commands(floored, [{"op": "stance", "name": "sitting_on_floor"},
                             {"op": "place", "shape": "chair",
                              "at": "under_hips"}], seat)
    low, high = props_module.bounds(seat[0])
    check("a chair under a figure already on the floor collapses to nothing",
          high[1] - low[1] <= 6.0 + 1e-6,
          "%.1f cm tall, the minimum" % (high[1] - low[1]))

    skeleton = rigpose.figure_for(DEFAULT_PRESET)
    scene = []
    bad = [{"op": "place", "shape": "spaceship"},
           {"op": "place", "shape": "chair", "at": "in_orbit"},
           {"op": "place", "shape": "chair", "distance": "far"},
           {"op": "place", "shape": "chair", "size": 900.0},
           {"op": "place", "shape": "chair", "at": "ground"}]
    warnings = apply_commands(skeleton, bad, scene)
    check("four bad placements are reported and the good one still lands",
          len(warnings) == 4 and len(scene) == 1, str(warnings)[:80])
    check("and `place` with nowhere to put it says so rather than vanishing",
          apply_command(rigpose.figure_for(DEFAULT_PRESET),
                        {"op": "place", "shape": "chair"}) is not None)

    # objects belong to the depth map only
    figures, scene, camera, _r = build_scene(
        {"figures": [{"commands": [{"op": "place", "shape": "wall",
                                    "at": "behind"}]}]})
    rect = frame_rect(900, 700, 512.0 / 768.0)
    # Both sides through `render_scene`, because the pose map it draws is the
    # rig's own keypoints and `pose_image` on the authored ones is a different
    # picture for a reason that has nothing to do with the wall.
    bare = render_scene(figures, camera, 128, 192, props=[])[0]
    with_wall = render_scene(figures, camera, 128, 192, props=scene)[0]
    check("an object never reaches the pose map",
          bare.tobytes() == with_wall.tobytes())
    # and the skeleton drawn is the one the body has, not the one it was
    # authored with: a pair that disagrees with itself is worse than either
    # half of it
    drawn = as_drawn(figures, __import__("bodies_lib").for_figures(figures))
    check("the pose map describes the mesh beside it",
          drawn[0].points != figures[0].points,
          "identical, so the rig's own keypoints are not reaching the PNG")
    lit = render_scene(figures, camera, 128, 192, props=scene, ground=False)[1]
    empty = render_scene(figures, camera, 128, 192, props=[], ground=False)[1]
    covered = (sum(1 for v in lit.tobytes() if v),
               sum(1 for v in empty.tobytes() if v))
    check("but does reach the depth map", covered[0] > covered[1] * 1.5,
          "%d px vs %d" % covered)

    # -- hands and feet ---------------------------------------------------
    #
    # The one thing eighteen keypoints cannot say, so the only way it can be
    # said is by someone saying it. What has to hold: the angles reach the
    # rig, they turn nothing above the wrist or the ankle, and a positive
    # angle means the same thing on the left as on the right - a rig's own
    # axes come out mirrored, and a control whose +30 pointed one toe in and
    # the other out is a control nobody can use.
    base = rigpose.figure_for("Male, average")
    side0, up0, _facing0 = base.body_frame()

    def at(figure, bone):
        return np.asarray(figure.points[figure.pose.bone(bone)])

    turned = rigpose.figure_for("Male, average")
    turned.set_extremity("hand", "l", 50.0, 0.0)
    turned.set_extremity("foot", "l", 0.0, 30.0)
    above = max(vlen(vsub(tuple(at(turned, n)), tuple(at(base, n))))
                for n in base.pose.names
                if not any(k in n.lower() for k in
                           ("finger", "metacarp", "toe", "wrist", "foot")))
    check("turning a hand or a foot moves nothing above it",
          above < 1e-9, "worst %.2e cm" % above)
    fingers = vlen(vsub(tuple(at(turned, "finger3-3.L")),
                        tuple(at(base, "finger3-3.L"))))
    check("but does carry the fingers with it", fingers > 5.0,
          "%.1f cm" % fingers)

    # Same angle, both sides, mirrored. The signs were measured on the body
    # set, not reasoned about, and this measures them again: a canonical axis
    # still leaves each rotation's handedness open, and a control whose +30
    # points one toe in and the other out is a control nobody can use.
    out, rise = {}, {}
    for letter in ("l", "r"):
        spun = rigpose.figure_for("Male, average")
        spun.set_extremity("foot", letter, 0.0, 30.0)
        toe = "toe3-1.%s" % letter.upper()
        away = vsub(tuple(at(spun, toe)), tuple(at(base, toe)))
        out[letter] = vdot(away, side0) * (1.0 if letter == "l" else -1.0)
        lifted = rigpose.figure_for("Male, average")
        lifted.set_extremity("foot", letter, 20.0, 0.0)
        rise[letter] = vdot(vsub(tuple(at(lifted, toe)), tuple(at(base, toe))),
                            up0)
    check("a positive turn points BOTH toes outward",
          out["l"] > 1.0 and out["r"] > 1.0
          and abs(out["l"] - out["r"]) < 0.1,
          "left %+.1f cm, right %+.1f cm outward" % (out["l"], out["r"]))
    check("and a positive lift raises BOTH sets of toes",
          rise["l"] > 1.0 and rise["r"] > 1.0
          and abs(rise["l"] - rise["r"]) < 0.1,
          "left %+.1f cm, right %+.1f cm up" % (rise["l"], rise["r"]))

    # a positive hand turn is pronation: the thumb goes down, on both hands
    thumb = {}
    for letter in ("l", "r"):
        pro = rigpose.figure_for("Male, average")
        pro.set_extremity("hand", letter, 0.0, 40.0)
        tip = "finger1-3.%s" % letter.upper()
        thumb[letter] = vdot(vsub(tuple(at(pro, tip)), tuple(at(base, tip))),
                             up0)
    check("and a positive hand turn is pronation on both hands",
          thumb["l"] < -1.0 and thumb["r"] < -1.0
          and abs(thumb["l"] - thumb["r"]) < 0.1,
          "thumbs %+.1f / %+.1f cm" % (thumb["l"], thumb["r"]))

    # -- the ground -------------------------------------------------------
    #
    # The floor is what says where in the room the figure is standing. It has
    # to obey the same rule every object does - depth map only - and two more
    # of its own, because it is not a placed object: it must not take part in
    # the framing, and it must not count as something burying the figure. It
    # covers most of the lower frame by design, so a floor that counted as an
    # obstruction would reject every camera with any pitch at all, which is
    # every camera it is any use to.
    standing, _p, cam, _w = build_scene({"figures": [
        {"preset": "Male, average", "commands": []}],
        "camera": "high_three_quarter"})
    floored = render_scene(standing, cam, 160, 240, ground=True)
    floating = render_scene(standing, cam, 160, 240, ground=False)
    check("the ground never reaches the pose map",
          floored[0].tobytes() == floating[0].tobytes())
    on_floor = sum(1 for v in floored[1].tobytes() if v)
    in_air = sum(1 for v in floating[1].tobytes() if v)
    check("but fills the depth map under the figure",
          on_floor > in_air * 2.0, "%d px against %d" % (on_floor, in_air))
    # A gradient, not a flat slab: the whole point is that it says how far
    # away things are. Measured down the middle column, below the feet.
    import numpy as _np
    grey = _np.asarray(floored[1], float)
    column = grey[:, grey.shape[1] // 2]
    lit_rows = _np.nonzero(column)[0]
    band = column[lit_rows.min():lit_rows.max() + 1]
    check("and the floor recedes rather than sitting flat",
          band.max() - band.min() > 60.0,
          "%.0f grey levels down the middle" % (band.max() - band.min()))

    # The figure is graded by its OWN depth range, so putting a floor under it
    # changes nothing about the subject. This is the whole reason the ground
    # has a curve of its own: one curve over both spent the range on the room
    # and left the body a handful of flat shades.
    bare = _np.asarray(floating[1], float)
    lit = bare > 0
    same = grey == bare
    agree = float(same[lit].mean())
    check("and the figure is graded exactly as it is with no floor at all",
          agree > 0.98, "%.1f%% of body pixels identical" % (100.0 * agree))
    # The pixels that do differ are the ones the floor COVERS, not ones it
    # regrades: the slab's near face sits at the figure's own nearest depth,
    # so it wins the z-test over anything further back at the same pixel -
    # the ankles, where the floor in front of the feet crosses them. They are
    # all near the foot of the frame, and nowhere else.
    rows = _np.nonzero((lit & ~same).any(axis=1))[0]
    check("and the only pixels it changes are ones it covers, down at the feet",
          rows.size == 0 or rows.min() > 0.6 * grey.shape[0],
          "highest changed row %d of %d"
          % (rows.min() if rows.size else -1, grey.shape[0]))

    # The floor fades OUT rather than stopping at a fixed grey. `far` is the
    # darkest the subject is allowed to be, so a floor that reaches below it
    # is one that runs into the background instead of ending in a horizontal
    # band across the picture - which reads as a platform, not a floor.
    only_floor = (grey > 0) & ~lit
    check("and fades into the background instead of ending on a line",
          grey[only_floor].min() < 45.0,
          "dimmest floor pixel %.0f, against far=45"
          % grey[only_floor].min())
    check("while staying under the figure it is beside",
          grey[only_floor].max() < grey[lit].max(),
          "floor peaks at %.0f, figure at %.0f"
          % (grey[only_floor].max(), grey[lit].max()))

    before = (cam.zoom, tuple(cam.target))
    frame_scene(standing, cam, frame_rect(900, 700, 160 / 240.0))
    check("the ground is not a prop, so it never moves the framing",
          (cam.zoom, tuple(cam.target)) == before,
          "%s against %s" % ((cam.zoom, tuple(cam.target)), before))
    check("and never counts as burying the figure",
          not buried(standing, cam, [], rect=frame_rect(900, 700, 160 / 240.0)))

    # a backdrop must not shrink the subject out of the frame
    tall = build_scene({"figures": [{"commands": []}]})[2].zoom
    walled = build_scene({"figures": [{"commands": [
        {"op": "place", "shape": "floor", "at": "ground"},
        {"op": "place", "shape": "wall", "at": "behind"}]}]})[2].zoom
    check("a wall and a floor do not shrink the figure away",
          walled > tall * 0.45, "zoom %.2f -> %.2f" % (tall, walled))

    # and the camera must not end up behind the furniture
    figures, scene, camera, _r = build_scene({"figures": [{"commands": [
        {"op": "stance", "name": "sitting"},
        {"op": "place", "shape": "chair", "at": "under_hips"},
        {"op": "place", "shape": "desk", "at": "in_front", "distance": 55.0}]}]})
    check("a view is not chosen that buries the figure behind an object",
          not buried(figures, camera, scene),
          "yaw %.0f" % math.degrees(camera.yaw))

    # a plan end to end, with no model anywhere
    figures, scene, camera, report = pose_from_prompt(
        "a woman with long hair in a dress, sitting on a chair, seen from "
        "three quarters")
    check("keyword route reads the prompt", report["source"] == "keywords")
    check("and dresses her as asked",
          figures[0].outfit.get("bottom") == "dress"
          and figures[0].outfit.get("hair") == "long",
          str(figures[0].outfit))
    check("and picks the figure out of it",
          figures[0].body.get("preset") == "Female, average",
          figures[0].body.get("preset"))
    check("and sits her down",
          vdot(vnorm(vsub(figures[0].at("l_knee"),
                          figures[0].at("l_hip"))),
               (0.0, 0.0, 1.0)) > 0.9)
    check("and turns the camera", abs(camera.yaw) > 0.1,
          "yaw %.0f deg" % math.degrees(camera.yaw))
    check("no warnings on a clean run", not report["warnings"],
          str(report["warnings"]))
    check("and the chair she was told to sit on is in the scene",
          [o["shape"] for o in scene] == ["chair"], str(scene))

    # a plan a model might hand back, wrong parts included
    figures, _scene, camera, report = build_scene({
        "figures": [{"preset": "Nonexistent", "commands": [
            {"op": "stance", "name": "t_pose"}]}],
        "camera": "from the moon"})
    check("an unknown preset and camera are reported, not fatal",
          len(report) == 2 and len(figures) == 1, str(report))

    # The chooser offers one contract: the plainest view that clears the bar,
    # and if nothing clears it the best there is. Assert that, rather than
    # something adjacent - "near the best available" was measuring bone and
    # departure visibility, which is not what it optimises, and it drifted
    # from a pass to a fail the moment more views existed to be better than
    # the one chosen.
    export = frame_rect(900, 700, 512.0 / 768.0)
    broke, settled, worst_bone = [], [], (None, 1.0)
    for name in sorted(STANCES):
        figures, scene, camera, _r = build_scene(
            {"figures": [{"commands": [{"op": "stance", "name": name}]}]})
        chosen = legible_view(figures, props=scene, rect=export)
        scores = view_scores(figures)
        if scores is None:
            continue
        unburied = []
        for view, (yaw, pitch) in CAMERA_VIEWS.items():
            other = Camera(900, 700)
            other.yaw, other.pitch = math.radians(yaw), math.radians(pitch)
            frame_scene(figures, other, export, props=scene)
            if not buried(figures, other, scene, rect=export):
                unburied.append(view)
        best = max((scores[v] for v in unburied), default=0.0)
        if scores[chosen] < 0.8 and scores[chosen] < best - 1e-6:
            broke.append("%s: %s at %.2f, %.2f available"
                         % (name, chosen, scores[chosen], best))
        if scores[chosen] < 0.8:
            settled.append(name)

        # Only the bones that MOVED, because those are the ones the chooser
        # scores and the ones the pose is about. This used to check all eight,
        # and on a figure whose rest arms stick out sideways that made no
        # difference. A real body rests with its arms down, so a resting arm
        # can point straight at the lens - `kneeling_on_one_knee` from
        # `profile_high` shows the right upper arm at 19% of its length while
        # every bone the kneel actually moved reads at 0.91 or better. The
        # chooser is documented to give bones that did not move no say, so a
        # check over all of them is testing the opposite of the contract.
        right, up, _fwd = camera.basis()
        fresh = rigpose.figure_for(figures[0].body.get('preset', DEFAULT_PRESET))
        for a, b in BONES.values():
            bone = vsub(figures[0].at(b), figures[0].at(a))
            rest = vnorm(vsub(fresh.at(b), fresh.at(a)))
            if vlen(vsub(vnorm(bone), rest)) <= 0.25:
                continue                      # it is still where it started
            seen = math.hypot(vdot(bone, right), vdot(bone, up)) / vlen(bone)
            if seen < worst_bone[1]:
                worst_bone = (name, seen)

    check("every stance gets the plainest view that reads, or the best there "
          "is", not broke, "" if not broke else "%d wrong, e.g. %s"
          % (len(broke), broke[0]))
    # The plain-view preference has to be live: `legible_view` offers the
    # PLAINEST view that clears its bar, and only falls back to the best
    # available below it. If the bar sat above everything, that preference
    # would be dead code and every figure would come out on whichever
    # dramatic angle happened to score highest.
    #
    # The share was under a quarter when rest meant a keypoint figure with its
    # arms out sideways. Against a real A-pose the scores sit lower - a pose
    # moves fewer bones, so the score is a minimum over a smaller set - while
    # the chooser itself is unchanged: it still returns the best available
    # view in every case that settles, which the check above asserts
    # separately. So this counts what it is actually guarding rather than
    # restating a constant calibrated against a body that no longer exists.
    check("the plainest-view preference is doing something",
          len(settled) < 0.7 * len(STANCES) and len(settled) > 0,
          "%d of %d settled for the best available; %d took a plain view"
          % (len(settled), len(STANCES), len(STANCES) - len(settled)))
    check("no stance is shown from a view that hides a limb entirely",
          worst_bone[1] > 0.35,
          "worst: %s at %.0f%% of its length" % (worst_bone[0],
                                                 100.0 * worst_bone[1]))

    # A purpose-built case rather than one borrowed from the catalogue. It
    # used to be a seated figure at a desk, which stopped burying itself the
    # moment `place` started clearing the object of the trunk - so the check
    # was passing on a bug rather than on the behaviour it names.
    walled, wall = rigpose.figure_for(DEFAULT_PRESET), []
    apply_commands(walled, [{"op": "stance", "name": "t_pose"},
                            {"op": "place", "shape": "wall", "at": "in_front",
                             "distance": 2.0}], wall)
    front = Camera(900, 700)
    frame_scene([walled], front, export, props=wall)
    check("a wall right in front of the figure buries it from the front",
          buried([walled], front, wall))
    chosen = legible_view([walled], props=wall, rect=export)
    picked = Camera(900, 700)
    picked.yaw = math.radians(CAMERA_VIEWS[chosen][0])
    picked.pitch = math.radians(CAMERA_VIEWS[chosen][1])
    frame_scene([walled], picked, export, props=wall)
    check("so the view chosen is not the front, however well it reads",
          not buried([walled], picked, wall), "chose %s" % chosen)
    check("and a standing figure still lands on a plain view, not a dramatic "
          "one", legible_view([rigpose.figure_for(DEFAULT_PRESET)],
                              rect=export) == "front")
    check("a standing figure is still shown from the front",
          build_scene({"figures": [{"commands": []}]})[2].yaw == 0.0)
    check("and an explicit camera is never overruled",
          abs(math.degrees(build_scene(
              {"figures": [{"commands": [{"op": "stance", "name": "sitting"}]}],
               "camera": "front"})[2].yaw)) < 1e-9)

    # framing has to hold the pose, not the rest figure
    from PIL import Image as _Image                      # noqa: F401 - optional
    for prompt in ("a person cheering with both arms up",
                   "someone lying down", "a runner mid stride"):
        figures, _scene, camera, _r = pose_from_prompt(prompt)
        rect = frame_rect(900, 700, 512 / 768)
        people = project_people(figures, camera, rect, 512, 768)
        xs = [p[0] for p in people[0][0]]
        ys = [p[1] for p in people[0][0]]
        check("%r is framed inside the export" % prompt,
              min(xs) > 0 and max(xs) < 512 and min(ys) > 0 and max(ys) < 768,
              "x %.0f..%.0f  y %.0f..%.0f" % (min(xs), max(xs), min(ys), max(ys)))

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
