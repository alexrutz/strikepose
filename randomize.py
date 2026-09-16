#!/usr/bin/env python3
"""Random poses, for finding the edge cases a catalogue never reaches.

    python3 randomize.py --list
    python3 randomize.py --render out/random --count 24 --parts arms,legs
    python3 randomize.py --render out/random --seed 7 --amount 1.0

The pose catalogue in `everyday.py` is 84 poses somebody chose, which means it
is 84 poses somebody *thought of*. What breaks a depth map is the pose nobody
thought of: an arm swung round to the far side of the body, where aiming a
bone has no minimal rotation and the roll is unstable; a shin pointing at the
lens; a figure folded until its own wrist is inside its own chest. This walks
into those on purpose.

Two things make it useful rather than merely noisy.

**Every joint is moved by ROTATION about its parent**, which is what a mouse
drag does, so no bone can change length whatever the dice say and the whole
`check_lengths` guard holds for free. A randomiser that wrote coordinates
would generate figures the editor itself cannot represent, and every failure
it found would be its own.

**A seed names a pose.** `--seed 7` is the same figure on this build and the
next one, so "the left elbow is inside the ribcage at seed 412" is a bug
report somebody can open rather than a screenshot. The seed is printed on
every render and stored in the scene file.

`amount` is what separates a survey from a stress test. At 0.3 the figures
read as people standing oddly; at 1.0 they are contortionists and that is the
point - the rasteriser, the roll carry and the framing all have to survive
them, because a user can drag a limb anywhere too.
"""

from __future__ import annotations

import math
import random

import rigpose
from skeleton import CHILDREN, KEYPOINT_NAMES, Skeleton
from vecmath import vadd, vcross, vlen, vmul, vnorm, vsub

INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def _j(*names):
    """Rig bone names. A part names bones now rather than keypoint indices,
    because the rig is the only thing the randomizer moves."""
    return tuple(names)


# What each selectable part moves, and how far it is allowed to at amount 1.0.
#
# The limits are not anatomical. A shoulder that only ever moved through the
# 130 degrees a shoulder really has would never produce the far-side-of-the-
# body case that makes roll unstable, and that case is exactly what this is
# for. They are wide enough to reach the awkward geometry and no wider: a
# joint swung a full 360 degrees is the same set of poses sampled twice.
RANDOM_PARTS = {
    "arms": dict(
        about="Shoulders, elbows and wrists",
        # the joint you would grab in a drag: an elbow swings the upper arm,
        # a wrist the forearm, a knuckle the hand
        joints=_j("lowerarm01.R", "lowerarm01.L", "wrist.R", "wrist.L",
                  "metacarpal3.R", "metacarpal3.L"),
        spread=150.0),
    "legs": dict(
        about="Hips, knees and ankles",
        joints=_j("lowerleg01.R", "lowerleg01.L", "foot.R", "foot.L",
                  "toe3-1.R", "toe3-1.L"),
        spread=95.0),
    "head": dict(
        about="Where the figure is looking",
        joints=_j("head"),
        spread=55.0),
    "torso": dict(
        about="Lean and twist above the hips",
        joints=(),                       # handled whole, see _bend_torso
        spread=45.0),
    "turn": dict(
        about="Which way the whole figure faces",
        joints=(),
        spread=180.0),
    "preset": dict(about="Which of the nine bodies", joints=(), spread=0.0),
    "outfit": dict(about="Hair and clothes", joints=(), spread=0.0),
    "camera": dict(about="The view it is seen from", joints=(), spread=0.0),
}

# The order a panel and the CLI list them in: the pose first, then the things
# that are about the picture rather than the pose.
PART_ORDER = ("arms", "legs", "torso", "head", "turn",
              "preset", "outfit", "camera")

DEFAULT_PARTS = ("arms", "legs", "torso", "head")


def _cone_direction(rng, axis, half_angle):
    """A unit vector within `half_angle` of `axis`, uniform on that cap.

    Uniform on the spherical cap, not uniform in the angle: sampling the angle
    flat piles the results up around the axis, because a cap's area grows with
    the sine of it. That matters here - the whole job is reaching the rim.
    """
    axis = vnorm(axis)
    lo = math.cos(half_angle)
    c = lo + (1.0 - lo) * rng.random()
    s = math.sqrt(max(0.0, 1.0 - c * c))
    phi = rng.uniform(0.0, 2.0 * math.pi)
    side = vcross(axis, (0.0, 0.0, 1.0))
    if vlen(side) < 1e-6:
        side = vcross(axis, (1.0, 0.0, 0.0))
    side = vnorm(side)
    other = vcross(axis, side)
    return vadd(vmul(axis, c),
                vadd(vmul(side, s * math.cos(phi)),
                     vmul(other, s * math.sin(phi))))


def _swing(skeleton, joint, rng, half_angle):
    """Aim one bone somewhere within a cone of where it already points.

    `move_joint` with the default stretch=False, which is the same call a drag
    makes: the bone is rotated onto the new direction and keeps its length, and
    everything downstream of it comes along.
    """
    idx = skeleton.pose.bone(joint) if isinstance(joint, str) else joint
    parent = skeleton.parent_of(idx)
    if parent < 0:
        return
    here = vsub(skeleton.points[idx], skeleton.points[parent])
    length = vlen(here)
    if length < 1e-6:
        return
    aim = _cone_direction(rng, here, half_angle)
    skeleton.move_joint(idx, vadd(skeleton.points[parent],
                                  vmul(aim, length)))


def _bend_torso(skeleton, rng, spread):
    """Lean and twist the torso, spread along the spine a bone at a time.

    The rig has a five-bone spine, so the torso is posed the way everything
    else is: rotate a bone and what hangs off it follows. Spreading the lean
    along the chain rather than spending it at one joint is what a back
    actually does, and it stops the figure hinging at the waist like a door.

    The keypoint version could do none of this - there is no spine in eighteen
    points - so it rotated a hand-picked list of everything above the hips
    about an axis it had to choose. Both bugs its own length check caught on
    the first run were about where that axis ran: a twist about the vertical
    through the hip midpoint stretched a neck-to-hip bone by 17 cm once the
    legs had already moved the hips, and re-reading the neck before the lean
    had moved it put 5 cm back. Neither is expressible here. A bone rotation
    cannot move the bone it turns about, and cannot change a length at all -
    so the class of error is gone rather than fixed.
    """
    pose = skeleton.pose
    chain = [n for n in ("spine05", "spine04", "spine03", "spine02", "spine01")
             if n in pose.index]
    if not chain:
        return
    side, up, _facing = skeleton.body_frame()
    lean = math.radians(rng.uniform(-spread, spread)) / len(chain)
    twist = math.radians(rng.uniform(-spread, spread)) * 0.6 / len(chain)
    # a cone rather than a fixed plane, so the lean is not always straight
    # forward or straight back
    axis = _cone_direction(rng, side, math.radians(90.0))
    for name in chain:
        j = pose.bone(name)
        pose.rotate(j, axis, lean)
        pose.rotate(j, up, twist)


def _turn(skeleton, rng, spread):
    """Yaw the whole figure - one rotation of the root bone, which carries
    everything because everything hangs off it."""
    skeleton.turn(math.radians(rng.uniform(-spread, spread)))


def normalise_parts(parts):
    """Accept a string, a sequence or None; return the names that exist."""
    if parts is None:
        parts = DEFAULT_PARTS
    if isinstance(parts, str):
        parts = [p.strip() for p in parts.replace(",", " ").split()]
    if "all" in parts:
        return tuple(PART_ORDER)
    return tuple(p for p in PART_ORDER if p in set(parts))


def randomize_figure(skeleton, parts=None, amount=0.6, seed=None, rng=None):
    """Move the chosen parts of one figure. Returns the parts it moved.

    Rotations only, so every bone comes out the length it went in at - the
    randomiser cannot generate a figure the editor could not have been dragged
    into, which is what makes a failure it finds a real one.
    """
    rng = rng or random.Random(seed)
    parts = normalise_parts(parts)
    amount = max(0.0, min(1.5, float(amount)))
    for name in parts:
        spec = RANDOM_PARTS[name]
        spread = spec["spread"] * amount
        if name == "torso":
            _bend_torso(skeleton, rng, spread)
        elif name == "turn":
            _turn(skeleton, rng, spread)
        elif name == "head":
            # The head rides the nose: the eyes and ears are its subtree, so
            # swinging the nose carries the face and nothing stretches.
            _swing(skeleton, INDEX["nose"], rng, math.radians(spread))
        else:
            for joint in spec["joints"]:
                _swing(skeleton, joint, rng, math.radians(spread))
    return parts


def random_look(rng):
    """An outfit name, weighted so a figure is dressed more often than not."""
    import wearables
    names = [n for n in wearables.OUTFIT_NAMES if n != "bare"]
    return rng.choice(names) if names else "bare"


def random_scene(parts=None, amount=0.6, seed=None, count=1, preset=None):
    """A whole random scene: figures, their looks, and the view to see it from.

    Returns (figures, outfits, view, seed) - the view by name rather than a
    camera, so the caller frames it the same way an export does rather than
    inventing a second projection.
    """
    from anthro import BODY_PRESETS, DEFAULT_PRESET, preset_params

    if seed is None:
        seed = random.randrange(1, 10 ** 9)
    rng = random.Random(seed)
    parts = normalise_parts(parts)

    figures, outfits = [], []
    for _ in range(max(1, count)):
        if "preset" in parts and preset is None:
            name = rng.choice(list(BODY_PRESETS))
        else:
            name = preset or DEFAULT_PRESET
        figure = rigpose.figure_for(name)
        figure.preset = name
        randomize_figure(figure, parts, amount, rng=rng)
        figures.append(figure)
        outfits.append(random_look(rng) if "outfit" in parts else "bare")

    view = None
    if "camera" in parts:
        import pose_agent
        view = rng.choice(list(pose_agent.VIEW_ORDER))
    return figures, outfits, view, seed



def frame_for(figures, view=None, out_w=512, out_h=768, view_w=900, view_h=700):
    """A camera framing these figures, built exactly as `build_scene` builds
    one: same view table, same export rectangle, same framing pass. Written
    out rather than borrowed because a randomizer that framed its own way
    would be previewing a picture the export does not produce - which is the
    mistake the phone made and the suite now pins.
    """
    import math
    import pose_agent
    from camera import Camera

    camera = Camera(view_w, view_h)
    export = pose_agent.frame_rect(view_w, view_h, float(out_w) / out_h)
    if view is None or view not in pose_agent.CAMERA_VIEWS:
        view = pose_agent.legible_view(figures, rect=export)
    yaw, pitch = pose_agent.CAMERA_VIEWS[view]
    camera.yaw, camera.pitch = math.radians(yaw), math.radians(pitch)
    pose_agent.frame_scene(figures, camera, export)
    return camera


# ---------------------------------------------------------------------------

def describe():
    rows = ["  %-8s %s" % (n, RANDOM_PARTS[n]["about"]) for n in PART_ORDER]
    return "\n".join(rows)


def _selftest():
    ok = True

    def check(label, condition, extra=""):
        nonlocal ok
        ok = ok and bool(condition)
        print(("PASS " if condition else "FAIL ") + label
              + ("  " + extra if extra else ""))

    from anthro import DEFAULT_PRESET, preset_params
    from skeleton import LIMB_SEQ

    def lengths(figure):
        """Every bone of the armature - 103 of them, not the seventeen
        keypoint limbs. The randomizer goes through `move_joint` and
        `pose.rotate` and never writes a coordinate, so this cannot fail
        without something new having started to."""
        q = figure.pose.positions()
        parents = figure.pose.parents
        return [vlen(vsub(tuple(q[j]), tuple(q[int(parents[j])])))
                for j in range(len(figure.pose.names)) if parents[j] >= 0]

    # The one that matters: rotations only, so nothing resizes. Run hard,
    # across the whole range, because a randomiser that stretched a limb once
    # in fifty would be worse than none at all.
    worst = 0.0
    for seed in range(60):
        sk = rigpose.figure_for(DEFAULT_PRESET)
        before = lengths(sk)
        randomize_figure(sk, "all", amount=1.5, seed=seed)
        after = lengths(sk)
        worst = max(worst, max(abs(a - b) for a, b in zip(before, after)))
    check("no bone changes length, over 60 seeds at full spread",
          worst < 1e-9, "worst %.2e cm" % worst)

    # A seed names a pose, or a bug report is a screenshot.
    a = rigpose.figure_for(DEFAULT_PRESET)
    b = rigpose.figure_for(DEFAULT_PRESET)
    randomize_figure(a, "all", amount=0.8, seed=1234)
    randomize_figure(b, "all", amount=0.8, seed=1234)
    check("the same seed gives the same pose",
          max(vlen(vsub(p, q)) for p, q in zip(a.points, b.points)) < 1e-12)
    c = rigpose.figure_for(DEFAULT_PRESET)
    randomize_figure(c, "all", amount=0.8, seed=1235)
    check("and a different one does not",
          max(vlen(vsub(p, q)) for p, q in zip(a.points, c.points)) > 1.0)

    # Selectable means selectable: an unchosen part must not move. Legs only,
    # so every joint above the hips has to come out exactly where it started.
    rest = rigpose.figure_for(DEFAULT_PRESET)
    legs = rigpose.figure_for(DEFAULT_PRESET)
    randomize_figure(legs, "legs", amount=1.0, seed=99)
    # every bone of the head, spine and both arms: far more than the twelve
    # keypoints this could name before, and it catches a spine bone moving
    upper = [j for j, n in enumerate(rest.pose.names)
             if rigpose.region_of(n) in ("head", "torso", "left arm",
                                         "right arm", "left hand",
                                         "right hand")]
    check("choosing legs leaves everything above the hips alone",
          max(vlen(vsub(rest.points[i], legs.points[i])) for i in upper) < 1e-9)
    moved = max(vlen(vsub(rest.points[rest.pose.bone(n)],
                          legs.points[legs.pose.bone(n)]))
                for n in ("lowerleg01.R", "lowerleg01.L", "foot.R", "foot.L"))
    check("and actually moves the legs", moved > 1.0, "%.1f cm" % moved)

    # Amount has to mean something, or the control is decoration.
    def departure(amount, seed):
        sk = rigpose.figure_for(DEFAULT_PRESET)
        randomize_figure(sk, "arms", amount=amount, seed=seed)
        return sum(vlen(vsub(p, q)) for p, q in zip(sk.points, rest.points))
    small = sum(departure(0.2, s) for s in range(12))
    large = sum(departure(1.0, s) for s in range(12))
    check("a larger amount departs further from rest", large > small * 2.0,
          "%.0f cm against %.0f" % (large, small))

    # Zero is a real setting: it is how you ask for "this pose, new outfit".
    still = rigpose.figure_for(DEFAULT_PRESET)
    randomize_figure(still, "all", amount=0.0, seed=5)
    check("amount 0 leaves the pose where it was",
          max(vlen(vsub(p, q)) for p, q in zip(still.points, rest.points))
          < 1e-9)

    check("every part in the order has a description",
          all(n in RANDOM_PARTS for n in PART_ORDER)
          and len(PART_ORDER) == len(RANDOM_PARTS))
    check("nonsense in the part list is dropped, not fatal",
          normalise_parts("arms,nope, legs") == ("arms", "legs"),
          str(normalise_parts("arms,nope, legs")))

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true",
                        help="what can be randomized")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--render", metavar="DIR",
                        help="write pose/depth pairs here")
    parser.add_argument("--count", type=int, default=8,
                        help="how many figures to render (default 8)")
    parser.add_argument("--parts", default=",".join(DEFAULT_PARTS),
                        help="comma separated, or 'all'")
    parser.add_argument("--amount", type=float, default=0.6,
                        help="0 is the rest pose, 1 is a contortionist")
    parser.add_argument("--seed", type=int,
                        help="reproduce one run exactly")
    parser.add_argument("--preset", help="hold the body fixed")
    parser.add_argument("--size", default="512x768",
                        help="768x512, or a ratio like 16:9")
    parser.add_argument("--no-ground", action="store_true",
                        help="leave the floor out of the depth maps")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.list or not args.render:
        print("Randomizable parts:\n" + describe())
        print("\nDefault: " + ",".join(DEFAULT_PARTS))
        return 0

    import json
    import os
    import pose_agent
    from exporting import parse_size
    width, height = parse_size(args.size)
    os.makedirs(args.render, exist_ok=True)
    index = []
    base = args.seed if args.seed is not None else random.randrange(1, 10 ** 9)
    for n in range(args.count):
        seed = base + n
        figures, outfits, view, seed = random_scene(
            args.parts, args.amount, seed=seed, count=1, preset=args.preset)
        for figure, look in zip(figures, outfits):
            import wearables
            figure.outfit = wearables.dress({}, look) or {}
        camera = frame_for(figures, view, width, height)
        pose, depth, _rect = pose_agent.render_scene(figures, camera,
                                                     width, height,
                                                     ground=not args.no_ground)
        stem = "%s/seed_%d" % (args.render, seed)
        pose.save(stem + "_pose.png")
        depth.save(stem + "_depth.png")
        index.append({"seed": seed, "parts": args.parts,
                      "amount": args.amount,
                      "preset": getattr(figures[0], "preset", None),
                      "outfit": outfits[0]})
        print("seed %-10d %s" % (seed, getattr(figures[0], "preset", "")))
    with open(os.path.join(args.render, "index.json"), "w") as handle:
        json.dump(index, handle, indent=1)
    print("\n%d pairs in %s" % (len(index), args.render))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
