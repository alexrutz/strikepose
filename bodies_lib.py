#!/usr/bin/env python3
"""Where the rigged bodies are, and the refusal to export without them.

A depth map is a picture of a body, and the built-in swept anatomy is a stack
of tapering cross-sections: it knows where every limb is and only approximates
what a person looks like. It exists to draw the viewport at sixty frames a
second and to give `wearables` a profile to clip a garment out of. It is not
what anyone wants out of an export, so it is no longer allowed to become one.

Every path that writes a depth map resolves its bodies through here and fails
with `MissingBodies` - which names the command that fixes it - rather than
quietly producing the basic version. Silently falling back is the whole
problem: the file still appears, it is just the wrong picture, and nothing
says so.

Looked for, in order:

    $STRIKEPOSE_BODIES
    ./bodies/                 next to the repo
    ~/.strikepose/bodies/

One .glb per body preset, named after it with spaces and commas turned into
underscores - `female_curvy.glb`, `child_about_7.glb`. `tools/make_bodies.py`
writes exactly that from Anny, and only from Anny: the depth path names the
104 bones it poses, so a rig that is not Anny's matches nothing at all.
"""

from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))

SEARCH = [
    lambda: os.environ.get("STRIKEPOSE_BODIES"),
    lambda: os.path.join(HERE, "bodies"),
    lambda: os.path.expanduser("~/.strikepose/bodies"),
]

HOW = ("Build them once:\n"
       "    pip install anny\n"
       "    python3 tools/make_bodies.py bodies/\n"
       "or point $STRIKEPOSE_BODIES at a folder of rigged .glb bodies named\n"
       "after the presets (male_average.glb, female_curvy.glb, ...).")


class MissingBodies(RuntimeError):
    """No rigged body for a figure the export needs one for."""


_cache = {}


def folders():
    """Every place a body set might be, in order, that exists.

    `$STRIKEPOSE_BODIES` is authoritative rather than first: someone who sets
    it means *there*, and quietly topping the set up from `./bodies` would
    make it impossible to test what happens when a body is absent.
    """
    named = os.environ.get("STRIKEPOSE_BODIES")
    if named:
        return [named] if os.path.isdir(named) else []
    seen, out = set(), []
    for find in SEARCH[1:]:
        path = find()
        if path and path not in seen and os.path.isdir(path):
            seen.add(path)
            out.append(path)
    return out


def slug(preset):
    return preset.lower().replace(",", "").replace(" ", "_")


def load(folder=None, presets=None, required=True):
    """{preset: loaded mesh}. Raises `MissingBodies` when a preset has none.

    Cached on the folder, because a body is thirteen thousand vertices and an
    export that reloaded one per frame would spend its life in the parser.
    """
    import everyday
    from openpose3d_editor import BODY_PRESETS

    presets = list(presets or BODY_PRESETS)
    places = [folder] if folder else folders()
    if not places:
        if not required:
            return {}
        raise MissingBodies(
            "No rigged bodies found. Looked in: %s\n\n%s"
            % (", ".join(filter(None, (f() for f in SEARCH))), HOW))

    found = {}
    for place in places:
        key = os.path.abspath(place)
        if key not in _cache:
            # every preset, not the ones this call happens to want: the cache
            # is keyed on the folder, so a first call for one preset would
            # otherwise make that folder look like it holds only that one
            _cache[key] = everyday.find_bodies(place, BODY_PRESETS)[0]
        for preset, mesh in _cache[key].items():
            found.setdefault(preset, mesh)

    missing = [p for p in presets if p not in found]
    if missing and required:
        raise MissingBodies(
            "No rigged body for %s.\nFound %d of %d in: %s\n\n%s"
            % (", ".join(missing), len(found), len(presets),
               ", ".join(places), HOW))
    return found


def for_figures(figures, folder=None, required=True):
    """One mesh per figure, in order, for `rigged_depth_image`."""
    presets = [f.body.get("preset") for f in figures]
    bank = load(folder, [p for p in presets if p], required)
    meshes = [bank.get(p) for p in presets]
    absent = [p for p, m in zip(presets, meshes) if m is None]
    if absent and required:
        raise MissingBodies("No rigged body for %s.\n\n%s"
                            % (", ".join(sorted(set(map(str, absent)))), HOW))
    return meshes


def available():
    """What is on disk, without raising - for a UI that wants to say so."""
    try:
        return load(required=False)
    except MissingBodies:
        return {}
