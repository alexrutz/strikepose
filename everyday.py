#!/usr/bin/env python3
"""A catalogue of specific everyday poses, and a generator that sweeps them
across every body type.

    python3 everyday.py --list                     # every pose, by group
    python3 everyday.py --selftest                 # no model, no display
    python3 everyday.py --render out/everyday      # the whole set, every body
    python3 everyday.py --render out/set --group "At a desk" --preset female

Why a catalogue rather than prompts
-----------------------------------
`pose_agent` exists so a local model can pose the figure from a sentence. That
is the right interface for one pose at a time and the wrong one for a hundred:
a 7B model asked for "tying a shoelace" a hundred times gives a hundred
slightly different answers, several of them wrong, and none of them repeatable
next week. A ControlNet conditioning set wants the opposite - the same pose,
by name, every time, on every body.

So these are written in exactly the command vocabulary the model emits, and go
through the same `apply_commands`. Nothing here writes a coordinate. Every
bone-length invariant the editor has therefore holds for free, and a pose in
this table can be handed to the model as a starting point and refined by
further commands, because it *is* further commands.

What a pose is allowed to assume
--------------------------------
The figure's own frame, never the viewer's: "forward" is the way it faces and
"left" is its left, which is what lets a pose survive a `turn`. Anything a
pose leans on - a chair under the hips, a desk in front, a wall behind - is
placed by the same `place` command, anchored against the figure at the size it
happens to be, so a chair fits the child and the tall man without a second
entry in the table.
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# The command constructors.
#
# These build the same plain dicts a local model returns over the wire. Writing
# the table through them rather than as JSON literals keeps it readable at 70
# entries and keeps it honest: there is no way to express a coordinate here.
# ---------------------------------------------------------------------------

def point(target, direction):
    return {"op": "point", "target": target, "direction": direction}


def bend(target, degrees):
    return {"op": "bend", "target": target, "degrees": degrees}


def turn(direction, degrees):
    return {"op": "turn", "direction": direction, "degrees": degrees}


def lean(direction, degrees):
    return {"op": "lean", "direction": direction, "degrees": degrees}


def look(direction):
    return {"op": "look", "direction": direction}


def stance(name):
    return {"op": "stance", "name": name}


def place(shape, at="ground", distance=70.0, size=1.0):
    return {"op": "place", "shape": shape, "at": at,
            "distance": distance, "size": size}


# Shorthands for the leg arrangements several poses share. They are command
# lists, spliced in, not a second kind of thing to resolve.
SEATED = [point("l_thigh", "forward"), point("r_thigh", "forward"),
          point("l_shin", "down"), point("r_shin", "down")]
# A deep squat, heels under the hips. The thigh rises forward from the hip
# rather than dropping: nothing moves the pelvis - posing is rotation - so how
# deep a squat reads is the gap between the hip and the feet, and dropping the
# thigh 45 degrees opens that gap to 80 cm, which is a figure dipping its knees.
SQUAT = [point("l_thigh", "forward_up"), point("r_thigh", "forward_up"),
         point("l_shin", "down"), point("r_shin", "down")]
# Sit-depth: thighs level, the depth a gym squat and a chair share.
HALF_SQUAT = [point("l_thigh", "forward"), point("r_thigh", "forward"),
              point("l_shin", "down"), point("r_shin", "down")]
CROSS_LEGGED = [point("l_thigh", "forward_left"), point("l_shin", "forward_right"),
                point("r_thigh", "forward_right"), point("r_shin", "forward_left")]
# Hands meeting in front of the chest. The upper arm goes forward and down so
# the elbow leads, then each forearm is aimed across the midline; aiming the
# whole arm instead puts the wrists where the elbows should be.
HANDS_TOGETHER = [point("l_upper_arm", "forward_down"),
                  point("r_upper_arm", "forward_down"),
                  point("l_forearm", "right_up"), point("r_forearm", "left_up")]
# Hand to the face. Same trick backwards: elbow forward, forearm back and up,
# which is the only way a 26 cm forearm reaches the chin from a hanging elbow.
L_HAND_TO_FACE = [point("l_upper_arm", "forward_down"),
                  point("l_forearm", "back_up")]
R_HAND_TO_FACE = [point("r_upper_arm", "forward_down"),
                  point("r_forearm", "back_up")]


CATALOGUE = [
 ("Standing and waiting", [
  ("arms_crossed", "arms folded across the front",
   # The forearms go straight across, not up: aimed up they meet at the chest
   # and read as two raised fists. Crossed arms are a horizontal pair at the
   # bottom of the ribs, which is where an elbow hanging at the waist puts them.
   [point("l_upper_arm", "left_down"), point("r_upper_arm", "right_down"),
    point("l_forearm", "right"), point("r_forearm", "left")]),
  ("arms_crossed_looking_away", "folded arms, head turned aside",
   [stance("arms_crossed"), look("forward_left")]),
  ("hands_on_hips", "elbows out, hands at the waist",
   [point("l_upper_arm", "left_down"), point("l_forearm", "right_down"),
    point("r_upper_arm", "right_down"), point("r_forearm", "left_down")]),
  ("hands_in_pockets", "arms hanging, forearms turned slightly in",
   [bend("l_shoulder", 8), bend("r_shoulder", 8),
    bend("l_elbow", 22), bend("r_elbow", 22)]),
  ("hands_behind_back", "at ease, forearms crossed behind",
   [bend("l_shoulder", -10), bend("r_shoulder", -10),
    point("l_forearm", "back"), point("r_forearm", "back")]),
  ("hands_clasped_front", "hands joined low in front",
   [bend("l_shoulder", 8), bend("r_shoulder", 8),
    point("l_forearm", "forward_right"), point("r_forearm", "forward_left")]),
  ("weight_on_one_leg", "hip cocked, one knee soft",
   [bend("l_knee", 14), bend("l_hip", -8), lean("back", 6),
    bend("r_shoulder", 8)]),
  ("shrugging", "elbows in, palms turned up and out",
   [point("l_upper_arm", "left_down"), point("l_forearm", "left_up"),
    point("r_upper_arm", "right_down"), point("r_forearm", "right_up"),
    look("forward_up")]),
  ("bowing", "a bow from the waist, arms at the sides",
   [lean("forward", 45), look("forward_down")]),
 ]),

 ("Phones and screens", [
  ("checking_phone", "both hands up at chest height, head down",
   [bend("l_shoulder", 20), bend("r_shoulder", 20),
    bend("l_elbow", 100), bend("r_elbow", 100), look("forward_down")]),
  ("texting_walking", "thumbs up, still walking",
   [stance("walking"), bend("l_shoulder", 25), bend("r_shoulder", 25),
    bend("l_elbow", 105), bend("r_elbow", 105), look("forward_down")]),
  ("phone_to_ear", "one hand at the side of the head",
   [point("l_upper_arm", "left_down"), point("l_forearm", "right_up"),
    bend("r_shoulder", 6)]),
  ("taking_selfie", "one arm out in front, head to the lens",
   [point("r_arm", "forward_up"), bend("r_elbow", 25), look("forward_up")]),
  ("holding_tablet", "one hand under, one hand tapping",
   [bend("l_shoulder", 25), bend("l_elbow", 90),
    point("l_forearm", "forward"), bend("r_shoulder", 30),
    bend("r_elbow", 85), look("forward_down")]),
  ("checking_watch", "one wrist turned up, the other hand steadying it",
   [point("l_upper_arm", "forward_down"), point("l_forearm", "right_up"),
    point("r_upper_arm", "forward_down"), point("r_forearm", "left_up"),
    look("forward_down")]),
 ]),

 ("At a desk", [
  ("sitting_at_desk", "seated square to a desk",
   [place("chair", "under_hips"), place("desk", "in_front", 20.0)]
   + SEATED + [bend("l_shoulder", 20), bend("r_shoulder", 20),
               bend("l_elbow", 80), bend("r_elbow", 80)]),
  ("typing_at_desk", "forearms level, head to the screen",
   [place("chair", "under_hips"), place("desk", "in_front", 20.0)]
   + SEATED + [point("l_upper_arm", "forward_down"),
               point("r_upper_arm", "forward_down"),
               point("l_forearm", "forward"), point("r_forearm", "forward"),
               lean("forward", 10), look("forward_down")]),
  ("laptop_on_lap", "seated, nothing in front, forearms low",
   [place("chair", "under_hips")] + SEATED + [bend("l_shoulder", 15), bend("r_shoulder", 15),
             bend("l_elbow", 70), bend("r_elbow", 70),
             lean("forward", 14), look("forward_down")]),
  ("writing_at_desk", "one forearm on the desk, head down",
   [place("chair", "under_hips"), place("desk", "in_front", 18.0)]
   + SEATED + [point("l_upper_arm", "forward_down"),
               point("l_forearm", "forward_right"), bend("r_shoulder", 30),
               bend("r_elbow", 95), lean("forward", 20),
               look("forward_down")]),
  ("leaning_back_in_chair", "pushed back, hands behind the head",
   [place("chair", "under_hips")] + SEATED
   + [lean("back", 22), point("l_upper_arm", "left_up"),
      point("l_forearm", "back_down"), point("r_upper_arm", "right_up"),
      point("r_forearm", "back_down")]),
  ("elbows_on_knees", "leaned forward off the chair, chin on the hands",
   [place("chair", "under_hips")] + SEATED
   + [lean("forward", 34), point("l_upper_arm", "forward_down"),
      point("l_forearm", "back_up"), point("r_upper_arm", "forward_down"),
      point("r_forearm", "back_up")]),
 ]),

 ("Sitting elsewhere", [
  ("sitting_on_stool", "perched, feet under the knees",
   [place("stool", "under_hips")] + SEATED),
  ("slouching_on_sofa", "sunk back, legs stretched out",
   [place("bench", "under_hips")]
   + [point("l_thigh", "forward"), point("r_thigh", "forward"),
      point("l_shin", "forward_down"), point("r_shin", "forward_down"),
      lean("back", 26)]),
  ("sitting_cross_legged", "on the floor, shins crossed",
   CROSS_LEGGED),
  ("meditating", "cross-legged, hands resting on the knees",
   CROSS_LEGGED + [point("l_upper_arm", "left_down"),
                   point("l_forearm", "forward_down"),
                   point("r_upper_arm", "right_down"),
                   point("r_forearm", "forward_down")]),
  ("sitting_on_floor_hands_back", "legs out, leaning on the hands behind",
   [stance("sitting_on_floor"), lean("back", 24),
    point("l_arm", "back_down"), point("r_arm", "back_down")]),
  ("sitting_knees_up", "knees drawn up, arms around them",
   [point("l_thigh", "forward_up"), point("r_thigh", "forward_up"),
    point("l_shin", "down"), point("r_shin", "down"),
    lean("forward", 16), point("l_arm", "forward_down"),
    point("r_arm", "forward_down")]),
  ("sitting_on_steps", "seated on a step, elbows on the thighs",
   [place("steps", "under_hips")] + SEATED
   + [lean("forward", 20), point("l_upper_arm", "forward_down"),
      point("l_forearm", "forward_down"),
      point("r_upper_arm", "forward_down"),
      point("r_forearm", "forward_down")]),
 ]),

 ("Carrying and lifting", [
  ("carrying_box", "a box held at the waist, leaning back against it",
   [bend("l_shoulder", 22), bend("r_shoulder", 22),
    bend("l_elbow", 85), bend("r_elbow", 85), lean("back", 8),
    place("crate", "at_hands", 0.0, 0.55)]),
  ("holding_tray", "forearms level, held away from the body",
   [point("l_upper_arm", "forward_down"), point("r_upper_arm", "forward_down"),
    point("l_forearm", "forward"), point("r_forearm", "forward")]),
  ("carrying_bag_one_hand", "one arm loaded, the other out for balance",
   [bend("l_shoulder", 6), bend("r_shoulder", 14), bend("r_elbow", 10),
    lean("back", 5)]),
  ("lifting_from_the_floor", "a squat lift, back straight, arms down",
   HALF_SQUAT + [lean("forward", 26), point("l_arm", "forward_down"),
                 point("r_arm", "forward_down"), look("forward_down")]),
  ("stooping_to_pick_up", "legs straight, folded at the hips",
   [lean("forward", 62), bend("l_knee", 14), bend("r_knee", 10),
    point("l_arm", "forward_down"), look("forward_down")]),
  ("pushing_a_cart", "arms out, weight into it",
   [bend("l_shoulder", 48), bend("r_shoulder", 48),
    bend("l_elbow", 22), bend("r_elbow", 22), lean("forward", 16),
    place("crate", "at_hands", 0.0, 0.9)]),
  ("pulling_a_suitcase", "walking, one arm trailing behind",
   [stance("walking"), point("r_upper_arm", "back_down"),
    bend("r_elbow", 14)]),
  ("handing_something_over", "one arm out, elbow soft",
   [bend("r_shoulder", 55), bend("r_elbow", 35), lean("forward", 6)]),
 ]),

 ("Reaching and pointing", [
  ("reaching_high_shelf", "one arm straight overhead",
   [point("l_arm", "up"), bend("l_elbow", 12), look("forward_up"),
    lean("back", 5)]),
  ("reaching_both_arms_up", "both arms overhead",
   [point("l_arm", "up"), point("r_arm", "up"), look("forward_up")]),
  ("reaching_across_a_table", "folded at the hips, one arm out",
   [place("table", "in_front", 45.0), lean("forward", 30),
    point("l_arm", "forward"), look("forward_down")]),
  ("opening_a_door", "one arm out at waist height, turning with it",
   [bend("r_shoulder", 50), bend("r_elbow", 30),
    turn("right", 18), place("panel", "in_front", 55.0)]),
  ("pressing_a_button", "one arm up, forefinger out",
   [point("r_upper_arm", "forward_down"), point("r_forearm", "forward_up"),
    look("forward")]),
  ("hailing_a_taxi", "one arm up and out, stepping off the kerb",
   [point("l_arm", "left_up"), lean("forward", 8), bend("l_hip", 12)]),
  ("pointing_ahead", "one arm straight out, head following it",
   [point("l_arm", "forward_left"), look("forward_left")]),
  ("waving", "one arm up, forearm vertical",
   [point("r_upper_arm", "right_up"), point("r_forearm", "up")]),
  ("waving_both_arms", "both arms up and out",
   [point("l_upper_arm", "left_up"), point("l_forearm", "up"),
    point("r_upper_arm", "right_up"), point("r_forearm", "up")]),
 ]),

 ("Talking to people", [
  ("shaking_hands", "one arm out, elbow square, a small bow",
   [bend("r_shoulder", 38), bend("r_elbow", 58), lean("forward", 7)]),
  ("clapping", "hands meeting in front of the chest",
   HANDS_TOGETHER),
  ("thinking_chin", "one hand at the chin, the other arm folded under it",
   L_HAND_TO_FACE + [bend("r_shoulder", 10), point("r_forearm", "left_up"),
                     look("forward_down")]),
  ("scratching_head", "one hand over the crown",
   [point("l_upper_arm", "left_up"), point("l_forearm", "right_up"),
    look("forward_up")]),
  ("covering_face", "both hands up over the face",
   L_HAND_TO_FACE + R_HAND_TO_FACE + [look("forward_down")]),
  ("arms_out_explaining", "both forearms out, palms up",
   [bend("l_shoulder", 25), bend("r_shoulder", 25),
    point("l_forearm", "forward_left"), point("r_forearm", "forward_right")]),
  ("crouching_to_a_child", "down on the heels, arms forward",
   SQUAT + [lean("forward", 12), point("l_arm", "forward_down"),
            point("r_arm", "forward_down"), look("forward")]),
 ]),

 ("Around the house", [
  ("leaning_on_a_wall", "shoulders back against it, one knee crossed",
   [place("panel", "behind", 22.0), lean("back", 9),
    bend("l_knee", 18), bend("l_hip", -10), bend("l_shoulder", -12),
    bend("r_shoulder", -12)]),
  ("leaning_on_a_counter", "forearms down on the surface",
   [place("table", "in_front", 42.0), lean("forward", 22),
    point("l_upper_arm", "forward_down"), point("l_forearm", "forward"),
    point("r_upper_arm", "forward_down"), point("r_forearm", "forward")]),
  ("washing_hands", "folded slightly, both hands low in front",
   [place("table", "in_front", 40.0), lean("forward", 18),
    bend("l_shoulder", 26), bend("r_shoulder", 26),
    bend("l_elbow", 72), bend("r_elbow", 72), look("forward_down")]),
  ("stirring_a_pot", "one arm out and bent, watching it",
   [place("table", "in_front", 45.0), bend("r_shoulder", 34),
    bend("r_elbow", 68), lean("forward", 10), look("forward_down")]),
  ("sweeping", "both hands on a handle angled down",
   [bend("l_shoulder", 34), bend("l_elbow", 26), bend("r_shoulder", 12),
    bend("r_elbow", 48), lean("forward", 18), look("forward_down"),
    place("pole", "at_hands", 0.0, 1.0)]),
  ("watering_a_plant", "one arm out and down, tipped forward",
   [bend("r_shoulder", 42), bend("r_elbow", 40), lean("forward", 10),
    look("forward_down")]),
  ("drinking_from_a_mug", "one hand at the mouth, head tipped back",
   R_HAND_TO_FACE + [lean("back", 6), look("forward_up")]),
  ("reading_a_book", "both hands up in front, head down",
   [bend("l_shoulder", 22), bend("r_shoulder", 22),
    bend("l_elbow", 96), bend("r_elbow", 96), look("forward_down")]),
  ("tying_shoelaces", "down on the heels, both hands at one foot",
   SQUAT + [lean("forward", 50), point("l_arm", "forward_down"),
            point("r_arm", "forward_down"), look("forward_down")]),
  ("putting_on_a_coat", "one arm back and out behind the shoulder",
   [point("r_upper_arm", "back_down"), point("r_forearm", "back_up"),
    bend("l_shoulder", 20), bend("l_elbow", 40), turn("right", 12)]),
 ]),

 ("Getting about", [
  ("standing_still", "the rest pose, arms at the sides", []),
  ("walking", "a mid-stride walk", [stance("walking")]),
  ("running", "a full running stride", [stance("running")]),
  ("climbing_stairs", "the lead leg up on the next step",
   [place("steps", "in_front", 30.0), bend("l_hip", 52), bend("l_knee", 74),
    bend("r_shoulder", 22), bend("l_shoulder", -18), lean("forward", 14)]),
  ("stepping_up", "one foot high, the other still down",
   [place("platform", "in_front", 35.0, 0.6), bend("l_hip", 44),
    bend("l_knee", 60), lean("forward", 10)]),
  ("lunging", "front knee square, back leg long",
   [bend("l_hip", 46), bend("l_knee", 76), bend("r_hip", -28),
    bend("r_knee", 22), lean("forward", 8), bend("l_shoulder", -20),
    bend("r_shoulder", 20)]),
  ("jumping", "off the ground, arms up, knees tucked",
   [point("l_arm", "up"), point("r_arm", "up"), bend("l_hip", 24),
    bend("r_hip", 20), bend("l_knee", 52), bend("r_knee", 46)]),
  ("kicking", "one leg swung through, torso back",
   [bend("l_hip", 68), bend("l_knee", 16), lean("back", 14),
    point("l_upper_arm", "left_down"), bend("r_shoulder", 25)]),
  ("throwing", "arm cocked back overhead, stepping in",
   [point("r_upper_arm", "back_up"), point("r_forearm", "up"),
    bend("l_shoulder", 40), bend("l_hip", 24), lean("back", 10)]),
  ("catching", "both arms out in front, elbows soft",
   [point("l_arm", "forward_up"), point("r_arm", "forward_up"),
    bend("l_elbow", 28), bend("r_elbow", 28), look("forward_up")]),
 ]),

 ("Exercise and rest", [
  ("stretching_overhead", "both arms up, a long arch back",
   [point("l_arm", "up"), point("r_arm", "up"), lean("back", 16),
    look("forward_up")]),
  ("yawning_stretch", "arms up and back, head tipped up",
   [point("l_arm", "back_up"), point("r_arm", "back_up"),
    lean("back", 18), look("back_up")]),
  ("touching_toes", "legs straight, folded right over",
   [lean("forward", 78), point("l_arm", "forward_down"),
    point("r_arm", "forward_down")]),
  ("squatting", "a gym squat, thighs level, arms forward",
   HALF_SQUAT + [lean("forward", 16), point("l_arm", "forward"),
                 point("r_arm", "forward")]),
  ("deep_squat", "down on the heels, hips low",
   SQUAT + [lean("forward", 12), point("l_arm", "forward_down"),
            point("r_arm", "forward_down")]),
  ("kneeling_upright", "both knees down, back straight",
   [stance("kneeling")]),
  ("kneeling_on_one_knee", "one knee down, the other foot planted",
   [point("l_thigh", "down"), point("l_shin", "back"),
    point("r_thigh", "forward"), point("r_shin", "down"),
    point("r_forearm", "forward_down")]),
  ("press_up", "face down, arms straight under the shoulders",
   [turn("forward", 88), point("l_arm", "forward"),
    point("r_arm", "forward")]),
  ("lying_on_back", "flat out, face up",
   [turn("back", 90), point("l_upper_arm", "left_down"),
    point("r_upper_arm", "right_down")]),
  ("lying_on_side", "rolled onto one side, knees drawn up",
   [turn("back", 90), turn("left", 90), bend("l_hip", 48),
    bend("r_hip", 42), bend("l_knee", 62), bend("r_knee", 56)]),
  # No bed under either of these on purpose. An anchor is resolved against the
  # figure's own ground, and a figure's ground is its lowest foot: lay it down
  # and the ankles come up to hip height, so anything anchored under it lands
  # at the height of a hip rather than a floor. A lying figure reads without
  # one; inventing a mattress at a height the pose cannot supply would not.
  ("sleeping_curled", "on one side, curled in, hands up under the chin",
   [turn("back", 90), turn("left", 90), bend("l_hip", 66),
    bend("r_hip", 60), bend("l_knee", 78), bend("r_knee", 72),
    bend("l_elbow", 84), bend("r_elbow", 78), lean("forward", 14)]),
  ("propped_up_in_bed", "on the back, legs out flat, chest raised",
   [point("l_thigh", "forward"), point("r_thigh", "forward"),
    point("l_shin", "forward"), point("r_shin", "forward"),
    lean("back", 34)]),
 ]),
]

# name -> commands, and the two lookups a caller wants
POSES = {name: commands
         for _group, entries in CATALOGUE for name, _about, commands in entries}
ABOUT = {name: about
         for _group, entries in CATALOGUE for name, about, _cmd in entries}
GROUP = {name: group
         for group, entries in CATALOGUE for name, _about, _cmd in entries}
GROUPS = [group for group, _entries in CATALOGUE]
NAMES = sorted(POSES)


def names_in(group):
    """Pose names in one group, in catalogue order."""
    for name, entries in CATALOGUE:
        if name.lower() == group.lower():
            return [n for n, _a, _c in entries]
    return []


def describe():
    """One line per pose, for `--list` and for the model's system prompt."""
    out = []
    for group, entries in CATALOGUE:
        out.append("")
        out.append(group)
        for name, about, commands in entries:
            out.append("  %-26s %-52s %2d commands"
                       % (name, about, len(commands)))
    return "\n".join(out[1:])


# ---------------------------------------------------------------------------
# Rendering a set
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rigged bodies
#
# The swept-anatomy depth source is a stack of tapering cross-sections; it is
# right about where every limb is and approximate about what a body looks like.
# A folder of .glb bodies exported from MPFB2 replaces it with real skinned
# geometry - hands with fingers, a face, a chest that belongs to the body it
# is on - driven by the same eighteen keypoints. The file for a preset is its
# name, lowercased, spaces and commas turned into underscores, so
# "Female, curvy" is `female_curvy.glb`, with an `mpfb_` prefix accepted too.
# ---------------------------------------------------------------------------

def body_slug(preset):
    return preset.lower().replace(",", "").replace(" ", "_")


def find_bodies(folder, presets):
    """{preset: loaded mesh} for every preset the folder has a file for."""
    import mesh_backend
    found = {}
    missing = []
    for preset in presets:
        slug = body_slug(preset)
        for candidate in (slug, "mpfb_" + slug):
            for suffix in (".glb", ".gltf"):
                path = os.path.join(folder, candidate + suffix)
                if os.path.exists(path):
                    mesh = mesh_backend.load_rigged_mesh(path)
                    mesh["roles"] = mesh_backend.resolve_bones(
                        mesh["joint_names"])
                    problems = mesh_backend.validate_roles(mesh, mesh["roles"])
                    if problems:
                        raise RuntimeError(
                            "%s does not map onto the editor's roles: %s\n"
                            "Run: python3 mesh_backend.py --inspect %s"
                            % (os.path.basename(path), problems[0], path))
                    found[preset] = mesh
                    break
            if preset in found:
                break
        else:
            missing.append(preset)
    return found, missing


def build(name, preset, view=None, out_w=512, out_h=768):
    """One pose on one body -> (figures, props, camera, warnings).

    Goes through `pose_agent.build_scene`, which is the same path a model's
    answer takes, so a pose that renders here is a pose the CLI can produce.
    """
    import pose_agent
    plan = {"figures": [{"preset": preset, "commands": POSES[name]}]}
    if view:
        plan["camera"] = view
    return pose_agent.build_scene(plan, aspect=out_w / float(out_h))


def render(name, preset, out_w=512, out_h=768, view=None, mesh=None):
    """(pose image, depth image, warnings) for one pose on one body.

    With `mesh` the depth comes from that rigged body; without it, from the
    swept anatomy. The pose PNG is the same either way.
    """
    import pose_agent
    figures, objects, camera, warnings = build(name, preset, view, out_w, out_h)
    pose, depth, _rect = pose_agent.render_scene(
        figures, camera, out_w, out_h, props=objects,
        meshes=[mesh] if mesh is not None else None)
    return pose, depth, warnings


def contact_sheet(rows, cell_w, cell_h, labels=(), heading=""):
    """A grid of images with a caption strip down the left and along the top."""
    from PIL import Image, ImageDraw
    # The heading gets a band of its own. Sharing a row with the column labels
    # put "Around the house - 10 poses x 9 bodies" straight through "Male,
    # average", which is the sort of thing a contact sheet exists to avoid.
    pad, left = 4, 150
    band = 22 if heading else 0
    top = band + (18 if labels else 0)
    cols = max(len(row) for _label, row in rows)
    width = left + cols * (cell_w + pad) + pad
    height = top + len(rows) * (cell_h + pad) + pad
    sheet = Image.new("RGB", (width, height), (14, 14, 16))
    draw = ImageDraw.Draw(sheet)
    if heading:
        draw.text((8, 7), heading, fill=(235, 235, 235))
    for c, label in enumerate(labels):
        draw.text((left + c * (cell_w + pad) + 4, band + 4), label,
                  fill=(150, 150, 155))
    for r, (label, row) in enumerate(rows):
        y = top + r * (cell_h + pad) + pad
        draw.text((8, y + cell_h // 2 - 6), label, fill=(215, 215, 215))
        for c, cell in enumerate(row):
            sheet.paste(cell.convert("RGB"),
                        (left + c * (cell_w + pad) + pad, y))
    return sheet


def render_set(out_dir, poses=None, presets=None, out_w=512, out_h=768,
               sheets=True, cell=(190, 285), quiet=False, bodies=None):
    """Write the matched pose/depth pair for every pose on every body.

    The pair is the deliverable: ControlNet wants the OpenPose PNG and the
    depth map of the same figure in the same frame, and the framing is solved
    per pose because a figure lying down is twice as wide as a standing one.
    """
    from PIL import Image
    from openpose3d_editor import BODY_PRESETS

    poses = list(poses or NAMES)
    presets = list(presets or BODY_PRESETS)
    meshes, without = ({}, presets)
    if bodies:
        meshes, without = find_bodies(bodies, presets)
        if not meshes:
            raise SystemExit("no body file in %s matched any preset; expected "
                             "names like %s.glb"
                             % (bodies, body_slug(presets[0])))
        if without and not quiet:
            print("no rigged body for %s - swept anatomy instead"
                  % ", ".join(without))
    os.makedirs(out_dir, exist_ok=True)
    index, trouble = [], []
    cells = {}
    for name in poses:
        for preset in presets:
            pose, depth, warnings = render(name, preset, out_w, out_h,
                                           mesh=meshes.get(preset))
            slug = "%s__%s" % (name, preset.lower().replace(", ", "_")
                               .replace(" ", "_"))
            pose.save(os.path.join(out_dir, slug + "_pose.png"))
            depth.save(os.path.join(out_dir, slug + "_depth.png"))
            index.append({"pose": name, "about": ABOUT[name],
                          "group": GROUP[name], "preset": preset,
                          "files": [slug + "_pose.png", slug + "_depth.png"]})
            if warnings:
                trouble.append((name, preset, warnings))
            if sheets:
                cells[(name, preset)] = depth.resize(cell, Image.LANCZOS)
            if not quiet:
                print("  %-26s %-18s %s" % (name, preset,
                                            "ok" if not warnings else warnings))
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        import json
        json.dump(index, fh, indent=1)

    if sheets:
        for group in GROUPS:
            rows = [(name, [cells[(name, p)] for p in presets])
                    for name in names_in(group) if name in poses]
            if not rows:
                continue
            sheet = contact_sheet(rows, cell[0], cell[1], presets,
                                  "%s - %d poses x %d bodies"
                                  % (group, len(rows), len(presets)))
            path = os.path.join(out_dir, "sheet_%s.png"
                                % group.lower().replace(" ", "_"))
            sheet.save(path)
            if not quiet:
                print("wrote", path)
    return index, trouble


# ---------------------------------------------------------------------------

def _selftest():
    import pose_agent
    from openpose3d_editor import BODY_PRESETS, Skeleton, preset_params, vlen, vsub
    ok = True

    def check(label, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(("PASS " if cond else "FAIL ") + label
              + ("  " + extra if extra else ""))

    check("the catalogue is a decent size", len(POSES) >= 60,
          "%d poses in %d groups" % (len(POSES), len(GROUPS)))
    check("every name is a plain identifier",
          all(n.replace("_", "").isalnum() and n == n.lower() for n in NAMES))
    check("every pose says what it is",
          all(ABOUT[n] and len(ABOUT[n]) < 60 for n in NAMES))
    check("no name is used twice",
          len(NAMES) == sum(len(e) for _g, e in CATALOGUE))
    check("pose_agent resolves every one of them",
          all(n in pose_agent.STANCES for n in NAMES),
          "missing: %s" % [n for n in NAMES if n not in pose_agent.STANCES][:3])

    # Every command must be one the editor accepts. A pose that silently drops
    # half its commands still renders - as the rest pose - which is exactly the
    # failure this has to catch, so the bar is zero warnings, not "it ran".
    bad = []
    for name in NAMES:
        skeleton = Skeleton()
        objects = []
        warnings = pose_agent.apply_commands(skeleton, POSES[name], objects)
        if warnings:
            bad.append((name, warnings[0]))
    check("every pose applies with no command skipped", not bad,
          "" if not bad else "%d bad: %s" % (len(bad), bad[:2]))

    # the invariant that makes any of this safe to generate in bulk
    worst = (0.0, None)
    for name in NAMES:
        rest = Skeleton()
        lengths = dict(rest.lengths)
        pose = Skeleton()
        pose_agent.apply_commands(pose, POSES[name], [])
        for key, was in lengths.items():
            gap = abs(pose.lengths[key] - was)
            if gap > worst[0]:
                worst = (gap, "%s %s" % (name, key))
    check("no pose changed a bone length", worst[0] < 1e-6,
          "worst %.2e cm (%s)" % worst)

    # a pose that does not move the figure is a typo, not a pose
    still = []
    for name in NAMES:
        if name == "standing_still":
            continue
        rest, pose = Skeleton(), Skeleton()
        pose_agent.apply_commands(pose, POSES[name], [])
        moved = max(vlen(vsub(a, b))
                    for a, b in zip(rest.points, pose.points))
        if moved < 4.0:
            still.append((name, round(moved, 2)))
    check("every pose actually departs from the rest pose", not still,
          "" if not still else "barely moved: %s" % still[:3])

    # and it has to work on every body, not just the 175 cm male the table
    # was written against - a child's arm is 30 cm shorter and its head twice
    # the fraction of its height
    bad = []
    for preset in BODY_PRESETS:
        for name in NAMES:
            skeleton = Skeleton(preset_params(preset))
            if pose_agent.apply_commands(skeleton, POSES[name], []):
                bad.append((preset, name))
    check("every pose applies to every body type", not bad,
          "%d bodies x %d poses" % (len(BODY_PRESETS), len(NAMES))
          if not bad else str(bad[:3]))

    # the props a pose leans on have to land under it, not through it
    floated = []
    for name in NAMES:
        skeleton, objects = Skeleton(), []
        pose_agent.apply_commands(skeleton, POSES[name], objects)
        for prop in objects:
            if not all(abs(v) < 600.0 for v in prop["position"]):
                floated.append((name, prop["shape"], prop["position"]))
    check("objects are placed within reach of the figure", not floated,
          "" if not floated else str(floated[:2]))

    # end to end on a sample, through the same path the CLI uses
    sample = ["typing_at_desk", "tying_shoelaces", "lying_on_side", "waving",
              "climbing_stairs", "carrying_box"]
    empty = []
    for name in sample:
        pose, depth, warnings = render(name, "Female, average", 128, 192)
        import numpy as np
        covered = float((np.asarray(depth) > 0).mean())
        if not 0.02 < covered < 0.9 or warnings:
            empty.append((name, round(covered, 3), warnings))
    check("a sample renders to a depth map with a figure in it", not empty,
          "" if not empty else str(empty))

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Everyday poses, across every body type.")
    parser.add_argument("--list", action="store_true",
                        help="print the catalogue and stop")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--render", metavar="DIR",
                        help="write the matched pose/depth pairs here")
    parser.add_argument("--group", action="append", default=[],
                        help="only this group (repeatable)")
    parser.add_argument("--pose", action="append", default=[],
                        help="only this pose (repeatable)")
    parser.add_argument("--preset", action="append", default=[],
                        help="only bodies whose name contains this")
    parser.add_argument("--size", default="512x768",
                        help="export size, default 512x768")
    parser.add_argument("--bodies", metavar="DIR",
                        help="folder of rigged .glb bodies, one per preset, "
                             "named after it (female_curvy.glb). The depth "
                             "map then comes from real geometry")
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.list or not args.render:
        print(describe())
        print("\n%d poses in %d groups" % (len(POSES), len(GROUPS)))
        return 0

    from openpose3d_editor import BODY_PRESETS
    poses = list(args.pose)
    for group in args.group:
        poses.extend(names_in(group))
    unknown = [p for p in poses if p not in POSES]
    if unknown:
        print("no such pose or group: %s" % ", ".join(unknown), file=sys.stderr)
        return 2
    presets = [p for p in BODY_PRESETS
               if not args.preset
               or any(w.lower() in p.lower() for w in args.preset)]
    if not presets:
        print("no body preset matched", file=sys.stderr)
        return 2
    out_w, out_h = (int(v) for v in args.size.lower().split("x"))
    index, trouble = render_set(args.render, poses or None, presets,
                                out_w, out_h, sheets=not args.no_sheets,
                                bodies=args.bodies)
    print("\n%d images in %s" % (2 * len(index), args.render))
    for name, preset, warnings in trouble:
        print("  %s on %s: %s" % (name, preset, warnings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
