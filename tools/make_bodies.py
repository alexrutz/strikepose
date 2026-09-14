#!/usr/bin/env python3
"""Build the body set as skinned .glb, one file per editor preset.

    pip install anny
    python3 tools/make_bodies.py bodies/
    python3 everyday.py --render out/set --bodies bodies/

Why Anny
--------
Anny (naver/anny) is the MakeHuman base mesh - the same geometry MPFB2 builds
from - wrapped in a parametric shape space calibrated on WHO anthropometry,
with a 104-bone rig whose names are MakeHuman's. Three things it gives that
building the same bodies in Blender does not:

  * it is a pip package, so a body set is reproducible on a machine with no
    Blender and no add-on to clone;
  * `age` is anthropometric rather than a slider - 0.0 is a 66 cm newborn,
    0.3 a 130 cm nine-year-old - and `height` can be bisected onto an exact
    stature, so a preset that says 162 cm gets a body 162 cm tall;
  * the licence is stated: Apache 2.0 code over CC0 MakeHuman assets.

What it does *not* give is a better-looking body. It is the same mesh, so the
depth maps are the same depth maps; what improves is where the numbers come
from and whether anyone can rebuild them. The one measurable difference is the
skinning, which carries nine influences per vertex against MPFB2's four and
loses 8-9% of a limb's girth at a hard bend where MPFB2's loses 9-14%.

Bone names come out as `upperarm01.L`, `lowerleg01.R`, `spine05`, so
`mesh_backend.resolve_bones` matches every role with no roles file.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

# Anny's `gender` runs 0 = male, 1 = female. MPFB2 uses the same MakeHuman
# assets and the same slider name the other way round, so this is measured
# rather than assumed: at 0.0 the figure has 54 cm shoulders on a 190 cm
# frame, at 1.0 44.5 cm on 176. Getting it backwards is silent - you get a
# complete, plausible body set with every sex inverted.
MALE, FEMALE = 0.0, 1.0

# (preset name, stature cm, traits). `height` is solved for, not set.
BODIES = [
    ("Male, average",   175.0, dict(gender=MALE,   age=0.78, muscle=0.50, weight=0.50)),
    ("Male, athletic",  178.0, dict(gender=MALE,   age=0.72, muscle=0.88, weight=0.42)),
    ("Male, heavy",     175.0, dict(gender=MALE,   age=0.82, muscle=0.38, weight=0.90)),
    ("Male, slim",      176.0, dict(gender=MALE,   age=0.70, muscle=0.44, weight=0.24)),
    ("Female, average", 162.0, dict(gender=FEMALE, age=0.78, muscle=0.42, weight=0.50)),
    ("Female, athletic",166.0, dict(gender=FEMALE, age=0.72, muscle=0.85, weight=0.40)),
    ("Female, curvy",   162.0, dict(gender=FEMALE, age=0.80, muscle=0.34, weight=0.72)),
    ("Female, slim",    165.0, dict(gender=FEMALE, age=0.72, muscle=0.40, weight=0.26)),
    ("Child, about 7",  122.0, dict(gender=0.5,    age=0.26, muscle=0.45, weight=0.45)),
]


def slug(preset):
    """The file name `everyday.py --bodies` looks for."""
    return preset.lower().replace(",", "").replace(" ", "_")


def to_y_up(points):
    """Anny is Z-up; glTF is Y-up."""
    points = np.asarray(points, float)
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)


def write_glb(path, verts, faces, heads, parents, names, bone_idx, bone_weights):
    """A skinned GLB: mesh, armature, and four influences per vertex."""
    verts = to_y_up(verts).astype(np.float32)
    heads = to_y_up(heads)
    faces = np.asarray(faces, np.uint32)

    # glTF's first joint set carries four influences and Anny gives nine, so
    # keep the four heaviest and renormalise. The fifth weight on a vertex
    # that has one is worth well under a millimetre of its position.
    order = np.argsort(-bone_weights, axis=1)[:, :4]
    rows = np.arange(len(bone_weights))[:, None]
    joints = bone_idx[rows, order].astype(np.uint16)
    weights = bone_weights[rows, order].astype(np.float64)
    total = weights.sum(axis=1, keepdims=True)
    weights = (weights / np.where(total < 1e-12, 1.0, total)).astype(np.float32)

    # Bones are written as pure translations, so a bone's rest global is a
    # translation by its head and the inverse bind is the negative of it.
    ibm = np.zeros((len(heads), 4, 4), np.float32)
    for b, head in enumerate(heads):
        M = np.eye(4, dtype=np.float32)
        M[:3, 3] = -head
        ibm[b] = M.T                                   # column major on disk

    blobs = [verts.tobytes(), faces.tobytes(), joints.tobytes(),
             weights.tobytes(), ibm.tobytes()]
    offsets, payload = [], b""
    for blob in blobs:
        payload += b"\0" * ((-len(payload)) % 4)
        offsets.append(len(payload))
        payload += blob

    gltf = {
        "asset": {"version": "2.0", "generator": "strikepose tools/make_bodies.py"},
        "scene": 0, "scenes": [{"nodes": [0, 1]}],
        "buffers": [{"byteLength": len(payload)}],
        "bufferViews": [{"buffer": 0, "byteOffset": offsets[i],
                         "byteLength": len(blobs[i])} for i in range(5)],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(verts), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": faces.size, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5123, "count": len(joints), "type": "VEC4"},
            {"bufferView": 3, "componentType": 5126, "count": len(weights), "type": "VEC4"},
            {"bufferView": 4, "componentType": 5126, "count": len(heads), "type": "MAT4"}],
        "meshes": [{"primitives": [{"attributes": {
            "POSITION": 0, "JOINTS_0": 2, "WEIGHTS_0": 3}, "indices": 1}]}],
        "skins": [{"joints": list(range(2, 2 + len(heads))),
                   "inverseBindMatrices": 4}],
        "nodes": [{"name": "body", "mesh": 0, "skin": 0}],
    }
    children = {}
    for b, parent in enumerate(parents):
        if parent is not None and parent >= 0:
            children.setdefault(int(parent), []).append(b + 2)
    roots = [b + 2 for b, p in enumerate(parents) if p is None or p < 0]
    gltf["nodes"].append({"name": "armature", "children": roots})
    for b, name in enumerate(names):
        parent = parents[b]
        base = heads[parent] if parent is not None and parent >= 0 else np.zeros(3)
        node = {"name": name,
                "translation": [float(v) for v in (heads[b] - base)]}
        if b in children:
            node["children"] = children[b]
        gltf["nodes"].append(node)

    raw = json.dumps(gltf).encode("utf-8")
    raw += b" " * ((-len(raw)) % 4)
    payload += b"\0" * ((-len(payload)) % 4)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2,
                             12 + 8 + len(raw) + 8 + len(payload)))
        fh.write(struct.pack("<II", len(raw), 0x4E4F534A))
        fh.write(raw)
        fh.write(struct.pack("<II", len(payload), 0x004E4942))
        fh.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the body set as skinned .glb, one per editor preset.")
    parser.add_argument("out", help="folder to write the .glb files into")
    parser.add_argument("--only", action="append", default=[],
                        help="just this preset (repeatable)")
    args = parser.parse_args(argv)

    try:
        import torch
        import roma
        import anny
    except ImportError as problem:
        print("needs Anny and its dependencies: pip install anny\n  (%s)"
              % problem, file=sys.stderr)
        return 2

    model = anny.Anny(skinning_method="lbs")
    dtype = model.template_vertices.dtype
    count = len(model.bone_labels)
    rest_pose = roma.Rigid(
        torch.eye(3, dtype=dtype).expand(count, 3, 3),
        torch.zeros(count, 3, dtype=dtype))[None].to_homogeneous()

    def build(traits, height):
        out = model(pose_parameters=rest_pose,
                    phenotype_kwargs=dict(traits, height=height, proportions=0.5))
        return (out["rest_vertices"][0].detach().numpy(),
                out["rest_bone_heads"][0].detach().numpy())

    def stature_of(verts):
        return 100.0 * float(verts[:, 2].max() - verts[:, 2].min())

    def solve_height(traits, want):
        """`height` is a 0..1 trait, not centimetres. Bisect onto the mark."""
        low, high = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (low + high)
            if stature_of(build(traits, mid)[0]) < want:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    os.makedirs(args.out, exist_ok=True)
    faces = model.faces.detach().numpy()
    bone_idx = model.vertex_bone_indices.detach().numpy()
    bone_weights = model.vertex_bone_weights.detach().numpy()
    parents = list(model.bone_parents)
    names = list(model.bone_labels)

    for preset, stature, traits in BODIES:
        if args.only and preset not in args.only:
            continue
        height = solve_height(traits, stature)
        verts, heads = build(traits, height)
        path = os.path.join(args.out, slug(preset) + ".glb")
        write_glb(path, verts, faces, heads, parents, names, bone_idx,
                  bone_weights)
        print("%-18s %6.1f cm (asked %.0f)  %5d vertices  %d bones  -> %s"
              % (preset, stature_of(verts), stature, len(verts), len(heads),
                 os.path.basename(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
