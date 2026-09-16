#!/usr/bin/env python3
"""Reading and writing a scene, including the ones older builds wrote.

A scene holds world centimetres, never an anchor: `pose_agent`'s anchors are
resolved once, at placement, in the figure's own frame, and the anchor stops
being true the moment anything is dragged.
"""

from __future__ import annotations

import props as props_module

from anthro import (DEFAULT_PRESET, REST_POSE, merge_body,
                    preset_params)
import rigpose
from skeleton import (KEYPOINT_NAMES, LIMB_SEQ, Skeleton,
                      clean_extremities)
from vecmath import vlen, vsub


def _bone_name(skeleton, index):
    """What to call bone `index` in a saved file: the rig's own name, or the
    keypoint it used to be when there is no rig behind the figure."""
    rig = getattr(skeleton, "pose", None)
    return rig.names[index] if rig is not None else KEYPOINT_NAMES[index]


def scene_to_dict(figures, camera, points_list, out_w, out_h, props=()):
    """Scene with any number of people. people[] is OpenPose's own format;
    figures[] carries what the editor needs to reload the scene, and objects[]
    the props standing in it."""
    if not isinstance(figures, (list, tuple)):
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
        # The pose as the armature holds it, which is the record; the
        # keypoints beside it are a description of that, kept so anything
        # reading a COCO-18 file still finds one.
        #
        # A bare keypoint `Skeleton` can still be written - the web prototype
        # and the pure-maths tests hold one - and comes out in the older,
        # keypoint-only form, which `scene_from_dict` imports.
        rig = getattr(skeleton, "pose", None)
        stored = rig.to_dict() if rig is not None else None
        keypoints = (skeleton.keypoints() if rig is not None else
                     dict(zip(KEYPOINT_NAMES, skeleton.points)))
        saved.append({
            "bones": stored["bones"] if stored else None,
            "offset": stored["offset"] if stored else None,
            "pose_3d": {n: [round(float(v), 6) for v in keypoints[n]]
                        for n in KEYPOINT_NAMES},
            "visible": list(skeleton.visible),
            "assets": list(getattr(skeleton, "assets", [])),
            "outfit": dict(getattr(skeleton, "outfit", {}) or {}),
            # Hands and feet: not in the keypoints, so not recoverable from
            # them either - a scene that dropped these would come back with
            # every palm and toe at the rig's own default.
            "extremities": {name: list(pair) for name, pair
                            in (getattr(skeleton, "extremities", {})
                                or {}).items()},
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
        # the top level repeats the first figure, for readers that predate
        # multi-figure scenes
        "bones": saved[0]["bones"],
        "offset": saved[0]["offset"],
        "pose_3d": saved[0]["pose_3d"],
        "visible": list(skeleton.visible),
        "bone_lengths": {_bone_name(skeleton, j): round(float(v), 4)
                         for j, v in skeleton.lengths.items()},
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
        skeleton = rigpose.figure_for(DEFAULT_PRESET)
        # Forward the whole entry rather than a hand-picked four keys: a key
        # left out of that list is not an error, it is a silent fall-through
        # to whatever the loader does without it - which is how `bones` went
        # missing and every saved scene came back through the keypoint
        # importer, 10 cm from where it was saved.
        scene_from_dict(dict(entry, camera=None), skeleton, camera)
        skeleton.assets = list(entry.get("assets", []))
        skeleton.outfit = dict(entry.get("outfit") or {})
        skeleton.extremities = clean_extremities(entry.get("extremities"))
        figures.append(skeleton)
    return figures or [rigpose.figure_for(DEFAULT_PRESET)]


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
    """Restore one figure.

    A scene written by this version holds `bones`: the armature's own joint
    angles, which is what the editor poses and what the depth map is skinned
    from, restored exactly. An older one holds only eighteen keypoints, and
    those are IMPORTED - run once through the solver that used to run on every
    redraw, and kept as angles from then on. That is the only direction
    keypoints still travel, and refusing the old files instead would be
    throwing away work somebody saved.
    """
    body = data.get("body")
    if body:
        skeleton.apply_body(merge_body(body))
    rig = getattr(skeleton, "pose", None)
    bones = data.get("bones")
    if rig is not None and bones:
        rig.from_dict({"bones": bones,
                       "offset": data.get("offset") or [0.0] * 3})
    elif data.get("pose_3d"):
        pose = data["pose_3d"]
        points = [tuple(pose.get(n, REST_POSE[n])) for n in KEYPOINT_NAMES]
        if rig is not None:
            skeleton.from_keypoints(dict(zip(KEYPOINT_NAMES, points)))
        else:                       # a bare keypoint skeleton, as it was
            skeleton.points = points
            skeleton.lengths = {c: vlen(vsub(points[c], points[p]))
                                for p, c in LIMB_SEQ}
    vis = data.get("visible")
    if vis and len(vis) == len(skeleton.visible):
        skeleton.visible = [bool(v) for v in vis]
    if body and data.get("body_scale") is not None:
        skeleton.body_scale = float(data.get("body_scale", 1.0))
    cam = data.get("camera") or {}
    camera.yaw = float(cam.get("yaw", camera.yaw))
    camera.pitch = float(cam.get("pitch", camera.pitch))
    camera.zoom = float(cam.get("zoom", camera.zoom))
    camera.target = tuple(cam.get("target", camera.target))
