#!/usr/bin/env python3
"""The real garment meshes, and which slot each one fills.

    python3 garments_lib.py --list

`wearables.py` builds a garment out of the body's own swept profile. That is
the right thing for the viewport, where it has to redraw at sixty frames a
second, and it is an approximation - a sleeve is the arm, slightly larger.
What goes into a conditioning image should not be an approximation, so a
rigged export dresses the figure in real meshes instead: the CC0 MakeHuman
garment library, fitted to every body in the set by `tools/make_wearables.py`.

An asset carries no rig of its own. It is skinned to the body's armature, so
`mesh_backend.skin_with` drives it from the body's own pose solution, which is
what keeps it from drifting off the figure it is on. It also carries, per body,
the body vertices it stands in for: MakeHuman's `delete_verts`, which is how a
shirt keeps a chest from coming through it.
"""

from __future__ import annotations

import os

FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "garments")

# slot -> preset name -> the mesh that fills it. The names on the left are
# `wearables`' vocabulary, unchanged: what a model emits does not depend on
# whether the export path has a mesh for it.
#
# A garment authored for one sex is used for both where it reads the same in a
# silhouette - a depth map has no colour and no seams - and the split is kept
# only where the cut differs enough to see.
CATALOGUE = {
    "hair": {
        "short": "cortu_short_messy_hair",
        "medium": "littleright_bobcut_hair",
        "bob": "toigo_blunt_bob",
        "long": "o4saken_long01",
        "braid": "elvs_double_mh_braid",
        "bun": "rehmanpolanski_hair_bun_brown",
        "curly": "culturalibre_hair_01",
        "ponytail": "elvs_double_mh_braid",
        "pigtails": "elvs_double_mh_braid",
    },
    "headgear": {
        "hat": "fedora01",
        "cap": "fedora01",
        "sun_hat": "fedora01",
    },
    "top": {
        "t_shirt": "male_casualsuit04",
        "long_sleeve": "male_casualsuit01",
        "tank_top": "female_sportsuit01",
        "jacket": "male_elegantsuit01",
        "hoodie": "male_casualsuit01",
        "coat": "male_worksuit01",
    },
    "bottom": {
        "trousers": "male_casualsuit01",
        "shorts": "female_sportsuit01",
        "leggings": "female_casualsuit01",
        "dress": "female_elegantsuit01",
        "skirt": "female_casualsuit02",
        "long_skirt": "female_elegantsuit01",
    },
    "shoes": {
        "shoes": "shoes01",
        "boots": "shoes03",
        "sandals": "shoes01",
        "tall_boots": "shoes03",
    },
}

_cache = {}


def available():
    """The garment files actually on disk."""
    if not os.path.isdir(FOLDER):
        return set()
    return {f[:-4] for f in os.listdir(FOLDER) if f.endswith(".npz")}


def named(slot, name):
    """The mesh filling this slot, or None - nothing here is mandatory."""
    stem = CATALOGUE.get(slot, {}).get(name)
    if stem and stem in available():
        return stem
    return None


def load(stem, preset, body):
    """A garment as a mesh `skin_with` can drive, plus what it hides.

    `body` is the loaded body it was fitted to; the garment borrows its rig
    wholesale, because it is skinned to that armature and solving it
    separately is the one thing that makes an asset drift.
    """
    import numpy as np
    import bodies_lib
    key = (stem, preset)
    if key in _cache:
        return _cache[key]
    path = os.path.join(FOLDER, stem + ".npz")
    if not os.path.exists(path):
        return None
    blob = np.load(path)
    slug = bodies_lib.slug(preset)
    if slug not in blob.files:
        return None
    mesh = dict(body)
    mesh["vertices"] = blob[slug].astype(float)
    mesh["faces"] = blob["faces"].astype(np.int64)
    mesh["skin_joints"] = blob["skin_joints"].astype(np.int64)
    mesh["skin_weights"] = blob["skin_weights"].astype(float)
    hide = blob["hide_" + slug] if ("hide_" + slug) in blob.files else None
    _cache[key] = (mesh, hide)
    return _cache[key]


def dress(figure, preset, body):
    """(garment meshes, the body with what they cover taken out).

    The body is handed back with those faces gone rather than the garment
    pushed off it: a shirt lies on a chest, and a depth map of a chest 5 mm
    in front of a shirt is a chest.
    """
    import numpy as np
    import wearables
    outfit = wearables.clean(getattr(figure, "outfit", None))
    worn, hidden = [], []
    for slot, name in outfit.items():
        if name == "none":
            continue
        stem = named(slot, name)
        if not stem:
            continue
        found = load(stem, preset, body)
        if found is None:
            continue
        mesh, hide = found
        worn.append(mesh)
        if hide is not None and len(hide):
            hidden.append(hide)
    if not worn:
        return [], body
    if hidden:
        drop = np.zeros(len(body["vertices"]), bool)
        drop[np.concatenate(hidden)] = True
        body = dict(body)
        faces = body["faces"]
        body["faces"] = faces[~drop[faces].any(axis=1)]
    return worn, body


def describe():
    have = available()
    lines = []
    for slot in ("hair", "headgear", "top", "bottom", "shoes"):
        for name, stem in sorted(CATALOGUE.get(slot, {}).items()):
            lines.append("  %-9s %-12s %-32s %s"
                         % (slot, name, stem, "" if stem in have else "MISSING"))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if "--list" in sys.argv:
        print(describe())
        print("\n%d garment files in %s" % (len(available()), FOLDER))
    else:
        print(__doc__)
