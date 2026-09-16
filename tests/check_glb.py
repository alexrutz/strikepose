#!/usr/bin/env python3
"""Put the body set through the whole depth pipeline, headless.

    python3 tests/check_glb.py bodies/*.glb
    python3 tests/check_glb.py bodies/female_average.glb --assets hair.glb

For each body: load it, map its bones onto the editor's roles, pose it through
several poses, skin it, render the depth map, and check what only a real
export can be checked against. Writes a contact sheet of every pose so the
result can be looked at as well as asserted.

`mesh_backend.py --selftest` proves the same maths against a rig built in
memory. That rig is Anny-shaped on purpose - the same bone names, the same
A-pose, authored at the stature its keypoints describe - but it cannot catch
what a real export carries: twist bones with their own rest rolls, morph
targets at non-zero weights, nine skinning influences cut down to four. This
is the script for that, and it needs no display.

Not in `run_all.sh`, because it needs `bodies/` on disk. Build it with
`python3 tools/make_bodies.py bodies/`.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import mesh_backend
from openpose3d_editor import (KEYPOINT_NAMES, Camera, Skeleton,
                               build_rest_points, frame_rect,
                               frame_scene,
                               rigged_depth_image, vlen, vnorm, vsub)

INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def preset_of(path):
    """Which body preset a file in `bodies/` is, by its name.

    The rig is posed by keypoints, and those keypoints have to describe THIS
    body: the child is 122 cm and the average man 175, and a rig is no longer
    stretched onto whatever keypoints it is handed. Posing the child with an
    adult's skeleton is a 60 cm disagreement that says nothing about the rig.
    """
    import bodies_lib
    from anthro import BODY_PRESETS, DEFAULT_PRESET
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    for preset in BODY_PRESETS:
        if bodies_lib.slug(preset).lower() == stem:
            return preset
    return DEFAULT_PRESET


def posed(name, preset):
    """A skeleton in one of the poses worth putting a rig through."""
    import pose_agent
    from anthro import preset_params
    import rigpose
    skeleton = rigpose.figure_for(preset)
    commands = {
        "rest": [],
        "t_pose": [{"op": "stance", "name": "t_pose"}],
        "arms_forward": [{"op": "stance", "name": "arms_forward"}],
        "sitting": [{"op": "stance", "name": "sitting"}],
        "reaching": [{"op": "point", "target": "l_arm",
                      "direction": "forward_up"},
                     {"op": "point", "target": "r_arm", "direction": "right"},
                     {"op": "bend", "target": "r_elbow", "degrees": 95}],
        "running": [{"op": "stance", "name": "running"}],
    }[name]
    pose_agent.apply_commands(skeleton, commands)
    return skeleton


POSES = ["rest", "t_pose", "arms_forward", "sitting", "reaching", "running"]


def check_file(path, asset_paths=(), out_dir="out/glb",
               width=320, height=440, verbose=False):
    ok = True

    def check(label, condition, extra=""):
        nonlocal ok
        ok = ok and bool(condition)
        print(("  PASS " if condition else "  FAIL ") + label
              + ("  " + extra if extra else ""))

    print("\n%s" % os.path.basename(path))
    mesh = mesh_backend.load_rigged_mesh(path)
    print("  %d vertices, %d faces, %d bones"
          % (len(mesh["vertices"]), len(mesh["faces"]),
             len(mesh["joint_names"])))
    size = mesh["vertices"].max(axis=0) - mesh["vertices"].min(axis=0)
    print("  bounding box %.3f x %.3f x %.3f as exported" % tuple(size))
    dropped = mesh.get("dropped_vertices")
    if dropped:
        print("  %d vertices dropped as loose helper geometry" % dropped)

    roles = mesh_backend.resolve_bones(mesh["joint_names"])
    mesh["roles"] = roles
    problems = mesh_backend.validate_roles(mesh, roles)
    check("this is the Anny rig, every bone named", not problems,
          "" if not problems else "%d problem(s): %s" % (len(problems),
                                                         problems[0]))
    if problems:
        print("    this program poses one armature, the 104-bone MakeHuman")
        print("    skeleton every body in bodies/ is built on. Rebuild with")
        print("    python3 tools/make_bodies.py bodies/")
        return False
    if verbose:
        for role in sorted(roles):
            print("    %-12s -> %s" % (role, mesh["joint_names"][roles[role]]))

    assets = []
    for asset_path in asset_paths:
        asset = mesh_backend.load_rigged_mesh(asset_path)
        assets.append(asset)
        print("  asset %s: %d vertices"
              % (os.path.basename(asset_path), len(asset["vertices"])))

    os.makedirs(out_dir, exist_ok=True)
    camera = Camera(900, 700)
    camera.yaw = math.radians(35.0)
    rect = frame_rect(900, 700, width / float(height))
    sheet = []
    worst_land, worst_scale = 0.0, None
    worst_pinch = (0.0, None)
    girths = {}                       # rest girth per bone, to compare against

    preset = preset_of(path)
    for name in POSES:
        skeleton = posed(name, preset)
        solution = skeleton.solution()
        bones = solution["bones"]
        points = skeleton.keypoints()

        # The figure IS the armature, so there is no solve to check and
        # nothing for a joint to land near: the elbow keypoint is where the
        # forearm bone starts, by construction. What is worth checking is the
        # other direction - that the eighteen this rig reports still describe
        # the same body they always did - so the gap measured is against the
        # ANSUR table's idea of where they go, which is the five or six
        # centimetres at the shoulder that no amount of work on either side
        # closes.
        # The OpenPose PNG is a description of this mesh, never a second
        # opinion about it. Ten of the eighteen ARE joints of the rig, so they
        # have to come back as exactly those joints and not as anything
        # derived; and the neck has to be the midpoint of the shoulders, or
        # the format is not the format.
        #
        # This used to compare each joint against the keypoint the solver was
        # aiming it at, which measured how far a solve had missed. There is no
        # solve any more - the figure is the armature - so what is left to
        # check is that reading it back does not quietly invent anything.
        for role in ("l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip",
                     "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"):
            if role in roles and role in points:
                worst_land = max(worst_land, float(np.linalg.norm(
                    np.asarray(points[role], float)
                    - bones[mesh["joint_names"][roles[role]]][1])))
        worst_land = max(worst_land, float(np.linalg.norm(
            np.asarray(points["neck"], float)
            - 0.5 * (np.asarray(points["l_shoulder"], float)
                     + np.asarray(points["r_shoulder"], float)))))

        verts = mesh_backend.skin_with(mesh, solution)
        check("%s: skinning produced finite geometry" % name,
              bool(np.isfinite(verts).all()))
        # Like for like: the mesh's height against the *keypoints'* height,
        # not against nose-to-ankle. A running figure raises an arm above its
        # head, so nose-to-ankle stops describing how tall the pose is and the
        # check failed every rig it was given in a pose that was not upright.
        # The crown and the soles reach past the keypoints, hence the margin.
        ys = [p[1] for p in points.values()]
        span = float(verts.max(axis=0)[1] - verts.min(axis=0)[1])
        target = (max(ys) - min(ys)) + 0.16 * abs(
            points["l_ankle"][1] - points["nose"][1])
        if worst_scale is None or abs(span - target) > abs(worst_scale[1]
                                                           - worst_scale[2]):
            worst_scale = (name, span, target)

        # a wrung limb keeps its joints in the right places and collapses the
        # mesh between them, so the girth is measured on the vertices rather
        # than on any frame the solver reports
        for role, child in (("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
                            ("l_hip", "l_knee"), ("l_knee", "l_ankle")):
            if role not in roles or child not in roles:
                continue
            got = limb_girth(mesh, verts, roles, bones, role, child)
            was = girths.setdefault((role, child), got)
            if was > 1e-6 and abs(got - was) / was > worst_pinch[0]:
                worst_pinch = (abs(got - was) / was, "%s %s" % (name, role))

        # Frame it, rather than trusting a fixed camera. There is one framing
        # in this program and both the window and the export use it; a camera
        # left at its defaults happened to suit where a keypoint figure sat
        # and puts a rigged one, which stands with its feet at y=0 and its
        # crown at its own stature, mostly above the top of the frame.
        frame_scene([skeleton], camera, rect)
        image = rigged_depth_image([(skeleton, mesh, assets)], camera, rect,
                                   width, height)
        pixels = np.asarray(image)
        check("%s: the depth map has a figure in it" % name,
              0.02 < (pixels > 0).mean() < 0.85,
              "%.1f%% of the frame" % (100.0 * (pixels > 0).mean()))
        sheet.append((name, image))

    # 25 cm was the bar when a rig could be posed by keypoints describing a
    # different body - the child stretched onto an adult's skeleton. Now the
    # keypoints are built from this body's own preset and the worst across the
    # nine is 10.1 cm, so the bar is where it can actually catch something.
    check("the keypoints it reports are read off the rig, not re-derived",
          worst_land < 1e-9,
          "worst %.2e cm between a joint keypoint and the bone it is" %
          worst_land)
    check("the rig is scaled to the figure", worst_scale is not None
          and abs(worst_scale[1] - worst_scale[2]) < 0.10 * worst_scale[2],
          "worst: %s spans %.0f cm, the pose %.0f cm" % worst_scale)
    if worst_pinch[1] is None:
        # a rig whose named bones drive no vertices on their own - a mock, or
        # one where every vertex is shared with a twist bone. Say so rather
        # than pass, or the check quietly measures nothing forever
        print("  ---- limb girth not measurable: no bone owns 8 vertices "
              "outright")
    else:
        # Linear blend skinning narrows a limb at a hard bend - the classic
        # candy wrapper - and a running stance folds a knee 75 degrees, so
        # some of this is the method rather than the rig. Real exports sit at
        # 9-14%; a quarter is the line between "bends" and "collapses". This
        # bar was once 40%, to accommodate a rig that was losing 36% - and
        # that turned out not to be the skinning at all but a mesh loaded
        # without its morph targets, so the body no longer matched the rig
        # driving it. A tolerance wide enough to pass a broken export checks
        # nothing.
        check("no limb is pinched by the skinning", worst_pinch[0] < 0.12,
              "worst: %s lost %.0f%% of its girth"
              % (worst_pinch[1], 100.0 * worst_pinch[0]))

    # Continuity, which no roll formula can satisfy by construction. A bone
    # swung right round must not jump - except through the one direction
    # opposite its own rest direction, where a minimal rotation has no axis to
    # pick and no solution of this kind can be continuous. That case is
    # reported rather than failed, because it is a property of the maths, not
    # of this rig.
    jump, turned = biggest_frame_step(mesh, roles)
    near_reversal = turned > 140.0
    check("a limb swung right round does not jump",
          jump < 12.0 or near_reversal,
          "worst %.0f deg step, with the arm %.0f deg from its rest direction%s"
          % (jump, turned,
             " - the reversal, where no minimal rotation is defined"
             if near_reversal and jump >= 12.0 else ""))

    name = os.path.splitext(os.path.basename(path))[0]
    contact = os.path.join(out_dir, "%s_poses.png" % name)
    save_sheet(sheet, contact, width, height)
    print("  wrote %s" % contact)
    return ok


def limb_girth(mesh, verts, roles, bones, role, child):
    """Mean distance from the bone axis of the vertices this bone drives.

    Measured on the posed vertices, which is the only place the failure the
    whole roll invariant is about actually shows: a wrung forearm keeps its
    wrist exactly where it was asked to be and collapses the mesh in between.
    Vertices are taken by skin weight, so a rig with twist bones contributes
    whatever it really drives rather than whatever the names suggest.
    """
    names = mesh["joint_names"]
    j, c = roles[role], roles[child]
    start, end = bones[names[j]][1], bones[names[c]][1]
    axis = mesh_backend.unit(end - start)
    if np.linalg.norm(axis) < 1e-9:
        return 0.0
    weights = np.zeros(len(verts))
    for slot in range(mesh["skin_joints"].shape[1]):
        weights += np.where(mesh["skin_joints"][:, slot] == j,
                            mesh["skin_weights"][:, slot], 0.0)
    picked = weights > 0.5
    if picked.sum() < 8:
        return 0.0
    offset = verts[picked] - start
    along = offset @ axis
    across = offset - np.outer(along, axis)
    return float(np.linalg.norm(across, axis=1).mean())


def biggest_frame_step(mesh, roles, step=4):
    """Swing an arm right round; return the worst jump in the forearm's frame,
    and how far the arm had been turned from its rest direction when it jumped.

    The second number is what tells a bug from a fact. Aiming a bone is a
    minimal rotation, and there is no minimal rotation onto the direction
    exactly opposite where the bone started: the axis is undefined there and
    ill-conditioned near it. So a rig whose arms rest out to the side has an
    unstable roll when an arm is swung round to the far side of the body, and
    no solver of this shape can avoid it. A jump anywhere *else* is a bug.
    """
    names = mesh["joint_names"]
    rest = mesh["rest_position"]
    orient = mesh["rest_global"][:, :3, :3]
    if "l_elbow" not in roles or "l_wrist" not in roles:
        return 0.0, 180.0
    rest_fore = mesh_backend.unit(rest[roles["l_wrist"]]
                                  - rest[roles["l_elbow"]])
    rest_upper = mesh_backend.unit(rest[roles["l_elbow"]]
                                   - rest[roles["l_shoulder"]])
    k = int(np.argmin([abs(float(np.dot(orient[roles["l_elbow"]][:, i],
                                        rest_fore))) for i in range(3)]))

    worst, turned, prev = 0.0, 180.0, None
    for degrees in range(-170, 171, step):
        angle = math.radians(degrees)
        aim = vnorm((math.sin(angle), -math.cos(angle), 0.0))
        skeleton = Skeleton()
        for parent, child in (("l_shoulder", "l_elbow"),
                              ("l_elbow", "l_wrist")):
            origin = skeleton.points[INDEX[parent]]
            length = vlen(vsub(skeleton.points[INDEX[child]], origin))
            skeleton.move_joint(INDEX[child],
                                tuple(a + b * length
                                      for a, b in zip(origin, aim)))
        points = {n: skeleton.points[i] for i, n in enumerate(KEYPOINT_NAMES)}
        bones = mesh_backend.pose_rig(mesh, points, roles)["bones"]
        rotation, origin = bones[names[roles["l_elbow"]]]
        axis = mesh_backend.unit(bones[names[roles["l_wrist"]]][1] - origin)
        vector = rotation @ orient[roles["l_elbow"]][:, k]
        frame = mesh_backend.unit(vector - axis * float(np.dot(vector, axis)))
        if prev is not None:
            jump = math.degrees(math.acos(float(np.clip(np.dot(prev, frame),
                                                        -1.0, 1.0))))
            if jump > worst:
                worst = jump
                # how far the upper arm has been turned from where the rig
                # authored it, which is the rotation that goes ill-conditioned
                turned = math.degrees(math.acos(float(np.clip(
                    np.dot(rest_upper, np.asarray(aim, float)), -1.0, 1.0))))
        prev = frame
    return worst, turned


def save_sheet(sheet, path, width, height):
    from PIL import Image, ImageDraw
    out = Image.new("L", (width * len(sheet), height), 0)
    for i, (name, image) in enumerate(sheet):
        out.paste(image, (i * width, 0))
        ImageDraw.Draw(out).text((i * width + 6, 6), name, fill=255)
    out.save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="+", help=".glb bodies to check")
    parser.add_argument("--assets", nargs="*", default=[],
                        help="hair, clothing and the like, on the same rig")
    parser.add_argument("--out", default="out/glb")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=440)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="list the bone mapping")
    args = parser.parse_args(argv)

    results = [check_file(path, args.assets, args.out,
                          args.width, args.height, args.verbose)
               for path in args.files]
    print("\n" + ("ALL PASS" if all(results) else "FAILURES PRESENT"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
