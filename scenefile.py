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
from skeleton import KEYPOINT_NAMES, LIMB_SEQ, Skeleton
from vecmath import vlen, vsub


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
