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
import math
import os
import re
import sys
import urllib.error
import urllib.request

from openpose3d_editor import (
    BODY_PRESETS, DEFAULT_PRESET, KEYPOINT_NAMES, VERSION, Camera, Skeleton,
    anatomy_depth_image, carry_chain, frame_rect, pose_image, preset_params,
    project_people, scene_to_dict, vadd, vcross, vdot, vlen, vmul, vnorm, vsub,
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
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "left": (90.0, 0.0),                  # looking at the figure's left side
    "right": (-90.0, 0.0),
    "three_quarter_left": (40.0, 0.0),
    "three_quarter_right": (-40.0, 0.0),
    "high": (25.0, 35.0),
    "overhead": (0.0, 70.0),
    "low": (25.0, -20.0),
}

# Tried in this order when nothing named a view, so a pose that reads equally
# well from several gets the plainest of them.
VIEW_ORDER = ["front", "three_quarter_left", "three_quarter_right", "left",
              "right", "high", "overhead", "back", "low"]

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

OPS = ("stance", "point", "bend", "turn", "lean", "look", "hide")

POINT_TARGETS = sorted(set(BONES) | set(LIMBS))
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
  {"op":"hide","target":JOINT_OR_LIMB}         mark it off-frame or occluded

stance NAME: %(stances)s
point target: %(points)s
bend target:  %(bends)s
  positive degrees flexes the joint the natural way: an elbow closes, a knee
  folds back, a hip or shoulder swings the limb forward. Negative reverses it.
  A straight limb is 0. A hard bend is about 120.
direction:    %(directions)s
camera:       %(cameras)s
preset:       %(presets)s

Shape:
{"figures":[{"preset":"Male, average","commands":[ ... ]}],"camera":"front"}

Start from a stance when one is close, then correct it with a few commands.
Prefer six to twelve commands. Do not invent op, target or direction names.
"""


def system_prompt():
    return SYSTEM_PROMPT % {
        "stances": ", ".join(sorted(STANCES)),
        "points": ", ".join(POINT_TARGETS),
        "bends": ", ".join(sorted(BEND_JOINTS)),
        "directions": ", ".join(sorted(DIRECTIONS)),
        "cameras": ", ".join(sorted(CAMERA_VIEWS)),
        "presets": ", ".join(sorted(BODY_PRESETS)),
    }


# ---------------------------------------------------------------------------
# Applying commands
# ---------------------------------------------------------------------------

def body_frame(skeleton):
    """(side, up, facing) of the figure as it stands now, side to its left."""
    frame = Skeleton.torso_frame(skeleton.points)
    if frame is None:                     # a degenerate torso; fall back to world
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    return frame


def resolve_direction(skeleton, name):
    """A named direction as a world unit vector in the figure's current frame."""
    coefficients = DIRECTIONS.get(name)
    if coefficients is None:
        return None
    side, up, facing = body_frame(skeleton)
    a, b, c = coefficients
    return vnorm(vadd(vadd(vmul(side, a), vmul(up, b)), vmul(facing, c)))


def _flex(skeleton, pivot, child, axis, toward, degrees):
    """Swing `child`'s chain about `axis` through `pivot`.

    The sign is read off the geometry rather than tabulated: rotating by a
    positive angle moves the child end at `axis x direction`, so whichever sign
    sends it towards `toward` is the one that flexes the joint the natural way.
    Tabulating it instead needs a different entry per side and per limb and is
    wrong the moment a stance has already turned the figure round.
    """
    origin = skeleton.points[pivot]
    direction = vnorm(vsub(skeleton.points[child], origin))
    sign = 1.0 if vdot(vcross(axis, direction), toward) >= 0.0 else -1.0
    skeleton.rotate_about_axis(skeleton.subtree(child), origin, axis,
                               math.radians(degrees) * sign)


def _bend_axis(skeleton, parent, pivot, child):
    """The axis a joint hinges about.

    Taken from the limb's carried cross-section frame, the same one the depth
    pass sweeps its profiles on, so the hinge stays square to the limb however
    the limb is posed - including when it points along the body's own forward,
    where picking an axis off `facing` has nothing to work with.
    """
    side, up, facing = body_frame(skeleton)
    down = vmul(up, -1.0)
    joints = [skeleton.points[pivot], skeleton.points[child]]
    if parent is not None:
        joints.insert(0, skeleton.points[parent])
    frames = carry_chain((down, facing), *joints)
    axis, forward = frames[-2] if parent is not None else frames[-1]
    hinge = vnorm(vcross(axis, forward))
    return hinge if vlen(hinge) > 1e-6 else side


def apply_command(skeleton, command):
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
        for step in STANCES[name]:
            apply_command(skeleton, step)
        return None

    if op == "point":
        bones = LIMBS.get(target) or ((target,) if target in BONES else None)
        if bones is None:
            return "unknown point target %r" % (target,)
        direction = resolve_direction(skeleton, direction_name)
        if direction is None:
            return "unknown direction %r" % (direction_name,)
        for bone in bones:
            start, end = BONES[bone]
            a, b = INDEX[start], INDEX[end]
            length = vlen(vsub(skeleton.points[b], skeleton.points[a]))
            skeleton.move_joint(b, vadd(skeleton.points[a],
                                        vmul(direction, length)))
        return None

    if op == "bend":
        spec = BEND_JOINTS.get(target)
        if spec is None:
            return "unknown bend joint %r" % (target,)
        parent, pivot, child, toward_name = spec
        side, up, facing = body_frame(skeleton)
        toward = vmul(facing, -1.0) if toward_name == "-facing" else facing
        parent_i = INDEX[parent] if parent else None
        pivot_i, child_i = INDEX[pivot], INDEX[child]
        axis = _bend_axis(skeleton, parent_i, pivot_i, child_i)
        _flex(skeleton, pivot_i, child_i, axis, toward, degrees)
        return None

    if op in ("turn", "lean"):
        side, up, facing = body_frame(skeleton)
        hip_mid = vmul(vadd(skeleton.points[INDEX["r_hip"]],
                            skeleton.points[INDEX["l_hip"]]), 0.5)
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
        joints = (range(len(skeleton.points)) if op == "turn"
                  else [INDEX[n] for n in ABOVE_WAIST])
        skeleton.rotate_about_axis(list(joints), hip_mid, axis,
                                   math.radians(degrees))
        return None

    if op == "look":
        direction = resolve_direction(skeleton, direction_name)
        if direction is None:
            return "unknown direction %r" % (direction_name,)
        neck, nose = INDEX["neck"], INDEX["nose"]
        length = vlen(vsub(skeleton.points[nose], skeleton.points[neck]))
        skeleton.move_joint(nose, vadd(skeleton.points[neck],
                                       vmul(direction, length)))
        return None

    if op == "hide":
        names = LIMBS.get(target)
        if names is not None:
            joints = set()
            for bone in names:
                joints.update(BONES[bone])
        elif target in INDEX:
            joints = {target}
        else:
            return "unknown hide target %r" % (target,)
        for joint in joints:
            skeleton.visible[INDEX[joint]] = False
        return None

    return "unknown op %r" % (op,)


def apply_commands(skeleton, commands):
    """Apply a list of commands. Returns the warnings, one per bad command.

    A bad command is skipped, not fatal. A local model gets one wrong every so
    often, and losing a whole pose over a misspelled joint would make the CLI
    useless exactly when the model is small enough to be worth running locally.
    """
    warnings = []
    if not isinstance(commands, list):
        return ["commands is not a list"]
    for i, command in enumerate(commands):
        problem = apply_command(skeleton, command)
        if problem:
            warnings.append("command %d skipped: %s" % (i + 1, problem))
    return warnings


# ---------------------------------------------------------------------------
# Building the scene
# ---------------------------------------------------------------------------

def build_scene(plan, view_w=900, view_h=700, aspect=512.0 / 768.0):
    """A plan -> (figures, camera, warnings), ready to render."""
    warnings = []
    figures = []
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
        skeleton = Skeleton(preset_params(preset))
        warnings.extend(apply_commands(skeleton, entry.get("commands", [])))
        figures.append(skeleton)
    if not figures:
        figures = [Skeleton(preset_params(DEFAULT_PRESET))]
    for i, skeleton in enumerate(figures[1:], start=1):
        skeleton.translate((70.0 * i, 0.0, 0.0))    # stand them side by side

    camera = Camera(view_w, view_h)
    view = plan.get("camera") if isinstance(plan, dict) else None
    if view is not None and view not in CAMERA_VIEWS:
        warnings.append("unknown camera view %r; choosing one that reads"
                        % (view,))
        view = None
    if view is None:
        view = legible_view(figures)
    yaw, pitch = CAMERA_VIEWS[view]
    camera.yaw, camera.pitch = math.radians(yaw), math.radians(pitch)
    frame_scene(figures, camera, frame_rect(view_w, view_h, aspect))
    return figures, camera, warnings


def legible_view(figures, order=None, readable=0.8):
    """The named view that shows most of what makes this pose that pose.

    Not "is every bone visible": a crouch seen head-on still shows 71% of the
    thigh and reads as a figure standing up straight, because what a crouch
    *is* - the thigh swinging forward from hanging - is the part pointing at
    the camera. So score each view on how much of every bone's departure from
    the rest pose survives projection, and ignore the bones that did not move,
    which have no opinion about the view. A standing figure constrains nothing
    and keeps the front; a seated one turns until the thighs read.

    A threshold rather than a maximum, walking VIEW_ORDER plainest first: only
    a flat profile foreshortens nothing, and a three-quarter that clears the
    bar is the better reference. If nothing clears it, the best is still
    better than guessing.
    """
    rest = {}
    for name, (a, b) in BONES.items():
        for figure in figures:
            key = (id(figure), name)
            fresh = Skeleton(figure.body)
            rest[key] = vnorm(vsub(fresh.points[INDEX[b]], fresh.points[INDEX[a]]))

    moved = []                       # (unit direction of the change, per bone)
    for figure in figures:
        for name, (a, b) in BONES.items():
            posed = vnorm(vsub(figure.points[INDEX[b]], figure.points[INDEX[a]]))
            change = vsub(posed, rest[(id(figure), name)])
            if vlen(change) > 0.25:  # about 14 degrees; below that it is noise
                moved.append(vnorm(change))
    if not moved:
        return (order or VIEW_ORDER)[0]

    scores = {}
    for name in (order or VIEW_ORDER):
        yaw, pitch = CAMERA_VIEWS[name]
        camera = Camera()
        camera.yaw, camera.pitch = math.radians(yaw), math.radians(pitch)
        right, up, _fwd = camera.basis()
        scores[name] = min(math.hypot(vdot(d, right), vdot(d, up))
                           for d in moved)
    for name in (order or VIEW_ORDER):
        if scores[name] >= readable:
            return name
    return max(scores, key=lambda name: (scores[name],
                                         -VIEW_ORDER.index(name)))


def silhouette_points(figure):
    """Keypoints plus the ends of the parts that have no keypoint of their own.

    The crown, the hands and the feet all reach well past the last keypoint on
    their chain - a hand is another 17 cm past the wrist - so framing on the
    keypoints alone crops them off, which is exactly what it did to a figure
    lying down.
    """
    points = list(figure.points)
    at = lambda name: figure.points[INDEX[name]]
    head_up = vnorm(vsub(at("nose"), at("neck")))
    points.append(vadd(at("nose"), vmul(head_up, 14.0)))
    for side in ("r", "l"):
        forearm = vnorm(vsub(at(side + "_wrist"), at(side + "_elbow")))
        points.append(vadd(at(side + "_wrist"), vmul(forearm, 19.0)))
        shin = vnorm(vsub(at(side + "_ankle"), at(side + "_knee")))
        sole = vadd(at(side + "_ankle"), vmul(shin, 8.0))
        _side, _up, facing = body_frame(figure)
        points.append(vadd(sole, vmul(facing, 21.0)))
        points.append(vadd(sole, vmul(facing, -9.0)))
    return points


def frame_scene(figures, camera, rect, margin=1.06):
    """Point the camera at the scene and zoom so the whole of it fits `rect`.

    A posed figure is not the same size on screen as the rest pose - arms up
    adds a head's height, lying down turns it on its side - so a fixed zoom
    crops exactly the poses a prompt is most likely to ask for. The fit is to
    the export rectangle rather than the whole view, because that is the part
    that becomes the PNG.
    """
    points = [p for figure in figures for p in silhouette_points(figure)]
    if not points:
        return
    right, up, _fwd = camera.basis()
    xs = [vdot(p, right) for p in points]
    ys = [vdot(p, up) for p in points]
    mid_x, mid_y = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    # put the bounding box centre on the camera target, in the camera's own
    # plane: the mean of the keypoints would pull the frame towards the head,
    # which carries five of the eighteen
    zero = vmul(right, 0.0)
    camera.target = vadd(zero, vadd(vmul(right, mid_x), vmul(up, mid_y)))
    half_w = (max(xs) - min(xs)) / 2.0
    half_h = (max(ys) - min(ys)) / 2.0
    rect_w, rect_h = rect[2] - rect[0], rect[3] - rect[1]
    camera.zoom = max(0.3, min(40.0,
                               min(rect_w / (2.0 * max(1.0, half_w) * margin),
                                   rect_h / (2.0 * max(1.0, half_h) * margin))))


def render_scene(figures, camera, out_w, out_h, view_w=900, view_h=700,
                 thickness=1.0, with_depth=True):
    """(pose image, depth image or None, export rect) for a built scene."""
    rect = frame_rect(view_w, view_h, out_w / out_h)
    pose = pose_image(figures, camera, rect, out_w, out_h)
    depth = None
    if with_depth:
        depth = anatomy_depth_image(figures, camera, rect, out_w, out_h,
                                    thickness)
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


class LocalLLM:
    """A local chat model over HTTP, constrained to the pose schema.

    `backend` is "ollama" for Ollama's own /api/chat, or "openai" for the
    OpenAI-compatible /v1/chat/completions that LM Studio, llama.cpp's server
    and vLLM all serve. Nothing here needs an API key; a local server that
    wants one takes it from --api-key.
    """

    def __init__(self, host, backend="openai", model=None, timeout=120.0,
                 api_key=None):
        self.host = host.rstrip("/")
        self.backend = backend
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

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

    # -- generation ------------------------------------------------------
    def complete(self, prompt, schema):
        """The model's reply as a parsed object, schema-constrained if it can be.

        Schema enforcement is asked for first and dropped on error. Runtimes
        disagree about how a schema is passed - llama.cpp has shipped releases
        that reject the OpenAI spelling outright - and a model answering
        unconstrained JSON is still worth having, so a refusal falls back
        rather than failing the run.
        """
        model = self.resolve_model()
        messages = [{"role": "system", "content": system_prompt()},
                    {"role": "user", "content": prompt}]
        attempts = []
        if self.backend == "ollama":
            attempts.append({"model": model, "messages": messages,
                             "stream": False, "format": schema,
                             "options": {"temperature": 0.3}})
            attempts.append({"model": model, "messages": messages,
                             "stream": False, "format": "json",
                             "options": {"temperature": 0.3}})
            path = "/api/chat"
        else:
            attempts.append({"model": model, "messages": messages,
                             "temperature": 0.3, "stream": False,
                             "response_format": {
                                 "type": "json_schema",
                                 "json_schema": {"name": "pose_plan",
                                                 "strict": True,
                                                 "schema": schema}}})
            attempts.append({"model": model, "messages": messages,
                             "temperature": 0.3, "stream": False,
                             "response_format": {"type": "json_object",
                                                 "schema": schema}})
            attempts.append({"model": model, "messages": messages,
                             "temperature": 0.3, "stream": False,
                             "response_format": {"type": "json_object"}})
            path = "/v1/chat/completions"

        last = None
        for payload in attempts:
            try:
                reply = self._post(path, payload)
            except (urllib.error.HTTPError, urllib.error.URLError,
                    ValueError) as exc:
                last = exc
                continue
            text = self._content(reply)
            parsed = parse_json_object(text)
            if parsed is not None:
                return parsed
            last = RuntimeError("model did not return JSON: %.200r" % (text,))
        raise RuntimeError("%s: %s" % (self.host, last))

    @staticmethod
    def _content(reply):
        if "message" in reply:                        # ollama
            return reply["message"].get("content", "")
        choices = reply.get("choices") or [{}]        # openai compatible
        return (choices[0].get("message") or {}).get("content", "")


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


def discover(backend="auto", host=None, model=None, api_key=None, timeout=120.0):
    """The first local model server that answers, or None.

    `auto` probes the default ports of the four runtimes worth probing. A run
    with no server up is not an error - it falls back to reading the prompt by
    keyword - so this returns None rather than raising.
    """
    if host:
        kinds = [backend] if backend != "auto" else ["ollama", "openai"]
        for kind in kinds:
            client = LocalLLM(host, kind, model, timeout, api_key)
            try:
                client.resolve_model()
                return client
            except Exception:
                continue
        return None
    for kind, endpoint in DEFAULT_ENDPOINTS:
        if backend not in ("auto", kind):
            continue
        client = LocalLLM(endpoint, kind, model, timeout, api_key)
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
                     aspect=512.0 / 768.0):
    """Everything between a sentence and a posed scene.

    Returns (figures, camera, report), report carrying the plan, which route
    read the prompt and every command that was skipped.
    """
    plan, source, warnings = plan_for(prompt, llm)
    figures, camera, more = build_scene(plan, view_w, view_h, aspect)
    return figures, camera, {"plan": plan, "source": source,
                             "warnings": warnings + more}


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
        "presets    : " + ", ".join(sorted(BODY_PRESETS)),
    ])


def run(args):
    llm = None
    if not args.no_llm:
        llm = discover(args.backend, args.host, args.model, args.api_key,
                       args.timeout)
        if llm is None and args.require_llm:
            print("no local model answered on %s"
                  % (args.host or ", ".join(e for _k, e in DEFAULT_ENDPOINTS)),
                  file=sys.stderr)
            return 2
    figures, camera, report = pose_from_prompt(args.prompt, llm,
                                              aspect=args.width / args.height)
    print("prompt read by: %s" % report["source"])
    for warning in report["warnings"]:
        print("  warning: %s" % warning, file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    out_w, out_h = args.width, args.height
    pose, depth, rect = render_scene(figures, camera, out_w, out_h,
                                     thickness=args.thickness,
                                     with_depth=not args.no_depth)
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
                          out_w, out_h)
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
    parser.add_argument("--thickness", type=float, default=1.0,
                        help="body thickness for the depth map")
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

    def lengths(skeleton):
        from openpose3d_editor import LIMB_SEQ
        return [vlen(vsub(skeleton.points[c], skeleton.points[p]))
                for p, c in LIMB_SEQ]

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
        skeleton = Skeleton(preset_params(DEFAULT_PRESET))
        before = lengths(skeleton)
        warnings = apply_commands(skeleton, [{"op": "stance", "name": name}])
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
    skeleton = Skeleton(preset_params(DEFAULT_PRESET))
    before = lengths(skeleton)
    warnings = apply_commands(skeleton, every)
    drift = max(abs(a - b) for a, b in zip(before, lengths(skeleton)))
    check("every command in the vocabulary applies", not warnings, str(warnings[:3]))
    check("and %d of them in a row change no bone length" % len(every),
          drift < 1e-6, "%.2e cm" % drift)

    # a bend has to bend the named way, whatever the figure has already done
    for turned in (0.0, 90.0, 180.0):
        skeleton = Skeleton(preset_params(DEFAULT_PRESET))
        apply_commands(skeleton, [{"op": "turn", "direction": "left",
                                   "degrees": turned}])
        _side, _up, facing = body_frame(skeleton)
        wrist_before = skeleton.points[INDEX["l_wrist"]]
        apply_commands(skeleton, [{"op": "bend", "target": "l_elbow",
                                   "degrees": 90}])
        moved = vsub(skeleton.points[INDEX["l_wrist"]], wrist_before)
        check("an elbow bends forwards with the figure turned %.0f deg" % turned,
              vdot(moved, facing) > 5.0, "%.1f cm forward" % vdot(moved, facing))
        knee_before = skeleton.points[INDEX["l_ankle"]]
        apply_commands(skeleton, [{"op": "bend", "target": "l_knee",
                                   "degrees": 90}])
        moved = vsub(skeleton.points[INDEX["l_ankle"]], knee_before)
        check("and a knee folds backwards", vdot(moved, facing) < -5.0,
              "%.1f cm back" % vdot(moved, facing))

    # pointing is in the figure's frame, not the world's
    skeleton = Skeleton(preset_params(DEFAULT_PRESET))
    apply_commands(skeleton, [{"op": "turn", "direction": "left", "degrees": 90},
                              {"op": "point", "target": "l_arm",
                               "direction": "forward"}])
    _side, _up, facing = body_frame(skeleton)
    arm = vnorm(vsub(skeleton.points[INDEX["l_elbow"]],
                     skeleton.points[INDEX["l_shoulder"]]))
    check("a limb pointed forward follows the figure, not the world",
          vdot(arm, facing) > 0.999, "cos %.4f" % vdot(arm, facing))

    # bad input is survivable
    skeleton = Skeleton(preset_params(DEFAULT_PRESET))
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
          vlen(vsub(skeleton.points[INDEX["l_wrist"]],
                    Skeleton(preset_params(DEFAULT_PRESET))
                    .points[INDEX["l_wrist"]])) > 1.0)

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

    # a plan end to end, with no model anywhere
    figures, camera, report = pose_from_prompt(
        "a woman sitting on a chair, seen from three quarters")
    check("keyword route reads the prompt", report["source"] == "keywords")
    check("and picks the figure out of it",
          figures[0].body.get("preset") == "Female, average",
          figures[0].body.get("preset"))
    check("and sits her down",
          vdot(vnorm(vsub(figures[0].points[INDEX["l_knee"]],
                          figures[0].points[INDEX["l_hip"]])),
               (0.0, 0.0, 1.0)) > 0.9)
    check("and turns the camera", abs(camera.yaw) > 0.1,
          "yaw %.0f deg" % math.degrees(camera.yaw))
    check("no warnings on a clean run", not report["warnings"],
          str(report["warnings"]))

    # a plan a model might hand back, wrong parts included
    figures, camera, report = build_scene({
        "figures": [{"preset": "Nonexistent", "commands": [
            {"op": "stance", "name": "t_pose"}]}],
        "camera": "from the moon"})
    check("an unknown preset and camera are reported, not fatal",
          len(report) == 2 and len(figures) == 1, str(report))

    # the view has to show the pose, not a foreshortened guess at it
    worst_view, worst_bone, worst_change = None, 1.0, (None, 1.0)
    for name in sorted(STANCES):
        figures, camera, _r = build_scene(
            {"figures": [{"commands": [{"op": "stance", "name": name}]}]})
        right, up, _fwd = camera.basis()
        fresh = Skeleton(figures[0].body)
        for a, b in BONES.values():
            bone = vsub(figures[0].points[INDEX[b]], figures[0].points[INDEX[a]])
            seen = math.hypot(vdot(bone, right), vdot(bone, up)) / vlen(bone)
            if seen < worst_bone:
                worst_view, worst_bone = name, seen
            change = vsub(vnorm(bone), vnorm(vsub(fresh.points[INDEX[b]],
                                                  fresh.points[INDEX[a]])))
            if vlen(change) > 0.25:
                shown = math.hypot(vdot(vnorm(change), right),
                                   vdot(vnorm(change), up))
                if shown < worst_change[1]:
                    worst_change = (name, shown)
    check("no stance is shown from a view that hides a limb", worst_bone > 0.34,
          "worst: %s at %.0f%% of its length" % (worst_view, 100.0 * worst_bone))
    check("and every posed limb's departure from rest is visible in it",
          worst_change[1] > 0.75,
          "worst: %s shows %.0f%% of its change"
          % (worst_change[0], 100.0 * worst_change[1]))
    check("a standing figure is still shown from the front",
          build_scene({"figures": [{"commands": []}]})[1].yaw == 0.0)
    check("and an explicit camera is never overruled",
          abs(math.degrees(build_scene(
              {"figures": [{"commands": [{"op": "stance", "name": "sitting"}]}],
               "camera": "front"})[1].yaw)) < 1e-9)

    # framing has to hold the pose, not the rest figure
    from PIL import Image as _Image                      # noqa: F401 - optional
    for prompt in ("a person cheering with both arms up",
                   "someone lying down", "a runner mid stride"):
        figures, camera, _r = pose_from_prompt(prompt)
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
