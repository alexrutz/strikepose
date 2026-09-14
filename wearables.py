#!/usr/bin/env python3
"""Hair and clothing, draped over whatever body is underneath.

    python3 wearables.py --list        # every preset
    python3 wearables.py --selftest    # no display, no model files

A garment here is not a mesh. It is the body's own swept profile, taken over
the stretch of the body it covers and pushed outward by a few millimetres of
cloth. That is the whole idea: a sleeve built that way fits every preset and
every pose for nothing, because it *is* the arm, slightly larger. Nothing to
rig, nothing to skin, nothing to drift, and it goes through the same analytic
rasteriser as the body and the objects.

What it costs is that cloth cannot hang. A skirt flares because it is told to
flare, not because it falls; a sleeve follows the arm rather than gathering at
the elbow. For a depth map that conditions a generator this is the right
trade: the silhouette is what does the work, and the silhouette of a person in
a t-shirt is a person slightly thicker through the chest with a hem across the
hips.

The frame a preset is written in
--------------------------------
`t` runs along the segment's own profile: 0 at the shoulder for the torso, 0
at the chin for the head, 0 at the elbow for a forearm. Values outside 0..1 are
allowed where the body's profile carries them - the torso runs to 1.19, the
crotch - which is how a jacket reaches below the hip.

Presets and outfits
-------------------
A preset is one garment in one slot; an outfit is a named set of them - "chef",
"winter", "knight". The same split the pose side makes between a command and a
stance, and for the same reason: the garment names are picked on *every* `wear`
command so that list stays short, while an outfit is picked at most once and
saves five guesses, so the catalogue can be long. An outfit names only the
slots it sets, so `outfit winter` leaves the haircut alone.
"""

from __future__ import annotations

import math

from openpose3d_editor import (_blob, _tube, sample_profile, vadd, vcross,
                               vdot, vlen, vmul, vnorm, vsub)

# ---------------------------------------------------------------------------
# the vocabulary
#
# Each entry is a list of pieces. A "drape" piece rides a body segment:
#   (segment, t0, t1, pad, flare)
# pad is the cloth thickness in centimetres and flare is extra width added by
# the hem, so a skirt widens as it falls. A "shell" piece is a free solid, used
# for the things cloth does that a body does not: a bun, a hood, a brim.
# ---------------------------------------------------------------------------

HAIR = {
    "none": [],
    "short": [("drape", "head", 0.46, 1.00, 0.9, 0.0)],
    "medium": [("drape", "head", 0.34, 1.00, 1.6, 0.0)],
    "curly": [("drape", "head", 0.38, 1.00, 3.4, 0.0)],
    "bob": [("drape", "head", 0.30, 1.00, 2.4, 0.0),
            ("fall", 0.32, 11.0, 2.8)],
    "long": [("drape", "head", 0.36, 1.00, 2.0, 0.0),
             ("fall", 0.50, 46.0, 3.4)],
    "ponytail": [("drape", "head", 0.46, 1.00, 1.1, 0.0),
                 ("tail", 0.78, 26.0, 3.6)],
    "braid": [("drape", "head", 0.46, 1.00, 1.1, 0.0),
              ("tail", 0.74, 44.0, 2.6)],
    "pigtails": [("drape", "head", 0.46, 1.00, 1.1, 0.0),
                 ("tail", 0.60, 21.0, 3.0, 9.0)],
    "bun": [("drape", "head", 0.46, 1.00, 1.1, 0.0),
            ("knot", 0.86, -0.55, 4.4)],
    "afro": [("drape", "head", 0.40, 1.00, 5.2, 0.0)],
    "shaved": [("drape", "head", 0.50, 1.00, 0.25, 0.0)],
}

TOPS = {
    "none": [],
    "tank_top": [("drape", "torso", 0.10, 0.80, 0.45, 0.0)],
    "t_shirt": [("drape", "torso", -0.02, 0.82, 0.55, 0.0),
                ("sleeve", "upper_arm", 0.00, 0.50, 0.75, 0.3)],
    "long_sleeve": [("drape", "torso", -0.02, 0.84, 0.55, 0.0),
                    ("sleeve", "upper_arm", 0.00, 1.00, 0.75, 0.0),
                    ("sleeve", "forearm", 0.00, 0.94, 0.75, 0.0)],
    "jacket": [("drape", "torso", -0.04, 1.02, 1.5, 0.6),
               ("sleeve", "upper_arm", 0.00, 1.00, 1.5, 0.0),
               ("sleeve", "forearm", 0.00, 0.96, 1.5, 0.2)],
    "hoodie": [("drape", "torso", -0.04, 1.00, 1.9, 0.5),
               ("sleeve", "upper_arm", 0.00, 1.00, 1.9, 0.0),
               ("sleeve", "forearm", 0.00, 0.96, 1.9, 0.2),
               ("hood", 0.20, -9.0, 8.0)],
    "coat": [("drape", "torso", -0.05, 1.34, 2.2, 1.6),
             ("sleeve", "upper_arm", 0.00, 1.00, 2.2, 0.0),
             ("sleeve", "forearm", 0.00, 1.00, 2.2, 0.3)],
    "armour": [("drape", "torso", -0.03, 0.86, 2.6, 0.0),
               ("pauldron", 0.0, 0.0, 0.0)],
}

BOTTOMS = {
    "none": [],
    "briefs": [("drape", "torso", 0.86, 1.16, 0.5, 0.0)],
    "shorts": [("drape", "torso", 0.84, 1.18, 1.0, 0.4),
               ("sleeve", "thigh", 0.00, 0.42, 1.1, 0.5)],
    "leggings": [("drape", "torso", 0.84, 1.18, 0.5, 0.0),
                 ("sleeve", "thigh", 0.00, 1.00, 0.5, 0.0),
                 ("sleeve", "calf", 0.00, 0.96, 0.5, 0.0)],
    "trousers": [("drape", "torso", 0.82, 1.18, 1.0, 0.2),
                 ("sleeve", "thigh", 0.00, 1.00, 1.1, 0.0),
                 ("sleeve", "calf", 0.00, 0.96, 1.1, 0.5)],
    "skirt": [("drape", "torso", 0.80, 1.24, 1.2, 3.2)],
    "long_skirt": [("skirt", 0.80, 62.0, 7.0)],
    "dress": [("drape", "torso", 0.06, 0.86, 0.7, 0.0),
              ("skirt", 0.84, 46.0, 6.0)],
    "robe": [("drape", "torso", 0.02, 0.90, 1.6, 0.0),
             ("sleeve", "upper_arm", 0.00, 1.00, 2.4, 0.8),
             ("sleeve", "forearm", 0.00, 0.80, 3.2, 1.2),
             ("skirt", 0.88, 70.0, 9.0)],
}

FEET = {
    "none": [],
    "sandals": [("sleeve", "foot", -0.02, 1.02, 0.35, 0.0)],
    "shoes": [("sleeve", "foot", -0.04, 1.04, 0.9, 0.0)],
    "boots": [("sleeve", "foot", -0.05, 1.05, 1.2, 0.0),
              ("sleeve", "calf", 0.52, 1.02, 1.4, 0.2)],
    "tall_boots": [("sleeve", "foot", -0.05, 1.05, 1.2, 0.0),
                   ("sleeve", "calf", 0.06, 1.02, 1.5, 0.3)],
}

HEADGEAR = {
    "none": [],
    # A hat is a crown and a brim. With a crown only a centimetre proud the
    # head reads as bare and the brim as a plank stuck through it, which is
    # what an edge-on disc is; the crown is what says "hat" from the side.
    "cap": [("drape", "head", 0.70, 1.06, 1.6, 0.0),
            ("brim", 0.72, 8.5, 1.2)],
    "hat": [("drape", "head", 0.64, 1.10, 2.4, 0.0),
            ("brim", 0.66, 12.0, 1.5)],
    "sun_hat": [("drape", "head", 0.64, 1.09, 2.0, 0.0),
                ("brim", 0.66, 15.0, 1.7)],
    "beanie": [("drape", "head", 0.62, 1.04, 1.4, 0.0)],
    "helmet": [("drape", "head", 0.44, 1.04, 1.8, 0.0)],
}

# Worn over a top: things whose silhouette is their own, not the body's. An
# apron hangs flat down the front, a cape down the back, a pack stands off it.
OVER = {
    "none": [],
    "apron": [("panel", 0.18, 1.10, 1.1, 2.0, 0.86)],
    "scarf": [("drape", "head", 0.00, 0.16, 2.8, 0.0),
              ("panel", 0.02, 0.30, 1.8, 2.4, 0.30)],
    "cape": [("cloak", 0.02, 74.0, 1.9, 7.0)],
    "backpack": [("pack", 0.30, 13.0, 9.5, 19.0)],
}

SLOTS = {"hair": HAIR, "headgear": HEADGEAR, "top": TOPS,
         "over": OVER, "bottom": BOTTOMS, "shoes": FEET}

# The order a panel and a listing show the slots in: head down, and "over"
# after "top" because that is the order they go on.
SLOT_ORDER = ("hair", "headgear", "top", "over", "bottom", "shoes")

DEFAULT_OUTFIT = {slot: "none" for slot in SLOTS}


# ---------------------------------------------------------------------------
# Named outfits
#
# Long on purpose. An outfit is picked at most once per figure, so it costs a
# model nothing to have forty to choose from and saves it five separate
# guesses when one fits - the same trade `everyday.POSES` makes against the
# command vocabulary.
#
# An entry names only the slots it sets. Everything it leaves out stays as it
# was, so "outfit winter" puts a coat and boots on a figure without shaving
# its head, and "wear ponytail" after it still lands. "bare" is the one that
# names every slot, because taking it all off is the one thing that cannot be
# said by omission.
# ---------------------------------------------------------------------------

OUTFITS = {
    "bare": dict(DEFAULT_OUTFIT),
    "underwear": {"top": "tank_top", "bottom": "briefs"},
    "pyjamas": {"top": "t_shirt", "bottom": "shorts"},

    "casual": {"top": "t_shirt", "bottom": "trousers", "shoes": "shoes"},
    "jeans_and_tee": {"top": "t_shirt", "bottom": "trousers",
                      "shoes": "boots"},
    "hoodie_and_jeans": {"top": "hoodie", "bottom": "trousers",
                         "shoes": "shoes"},
    "office": {"top": "long_sleeve", "bottom": "trousers", "shoes": "shoes"},
    "business": {"top": "jacket", "bottom": "trousers", "shoes": "shoes"},
    "suit": {"top": "jacket", "bottom": "trousers", "shoes": "shoes",
             "hair": "short"},
    "commuter": {"top": "jacket", "bottom": "trousers", "shoes": "shoes",
                 "over": "backpack"},
    "student": {"top": "hoodie", "bottom": "trousers", "shoes": "shoes",
                "headgear": "cap", "over": "backpack"},
    "evening": {"bottom": "dress", "shoes": "shoes", "hair": "bun"},
    "party": {"bottom": "dress", "shoes": "shoes", "hair": "long"},
    "sundress": {"bottom": "dress", "shoes": "sandals",
                 "headgear": "sun_hat", "hair": "long"},
    "school": {"top": "long_sleeve", "bottom": "skirt", "shoes": "shoes",
               "hair": "pigtails"},

    "summer": {"top": "tank_top", "bottom": "shorts", "shoes": "sandals"},
    "beach": {"top": "none", "bottom": "briefs", "shoes": "none",
              "headgear": "sun_hat"},
    "swimming": {"top": "none", "bottom": "briefs", "shoes": "none",
                 "hair": "bun"},
    "winter": {"top": "coat", "bottom": "trousers", "shoes": "boots",
               "headgear": "beanie", "over": "scarf"},
    "rain": {"top": "coat", "bottom": "trousers", "shoes": "boots",
             "headgear": "hat"},

    "gym": {"top": "tank_top", "bottom": "shorts", "shoes": "shoes"},
    "running": {"top": "t_shirt", "bottom": "shorts", "shoes": "shoes",
                "hair": "ponytail"},
    "yoga": {"top": "tank_top", "bottom": "leggings", "shoes": "none",
             "hair": "bun"},
    "dancer": {"top": "tank_top", "bottom": "leggings", "shoes": "none",
               "hair": "bun"},
    "cycling": {"top": "t_shirt", "bottom": "shorts", "shoes": "shoes",
                "headgear": "helmet"},
    "hiking": {"top": "long_sleeve", "bottom": "shorts", "shoes": "boots",
               "headgear": "cap", "over": "backpack"},
    "tourist": {"top": "t_shirt", "bottom": "shorts", "shoes": "shoes",
                "headgear": "cap", "over": "backpack"},
    "explorer": {"top": "long_sleeve", "bottom": "trousers", "shoes": "boots",
                 "headgear": "hat", "over": "backpack"},

    "chef": {"top": "long_sleeve", "bottom": "trousers", "shoes": "shoes",
             "headgear": "cap", "over": "apron"},
    "barista": {"top": "t_shirt", "bottom": "trousers", "shoes": "shoes",
                "over": "apron"},
    "waiter": {"top": "long_sleeve", "bottom": "trousers", "shoes": "shoes",
               "over": "apron"},
    "artist": {"top": "t_shirt", "bottom": "trousers", "shoes": "shoes",
               "over": "apron"},
    "doctor": {"top": "coat", "bottom": "trousers", "shoes": "shoes",
               "hair": "short"},
    "nurse": {"top": "long_sleeve", "bottom": "trousers", "shoes": "shoes",
              "hair": "bun"},
    "builder": {"top": "long_sleeve", "bottom": "trousers", "shoes": "boots",
                "headgear": "helmet"},
    "mechanic": {"top": "long_sleeve", "bottom": "trousers", "shoes": "boots",
                 "headgear": "cap", "over": "apron"},
    "farmer": {"top": "long_sleeve", "bottom": "trousers", "shoes": "boots",
               "headgear": "sun_hat"},
    "gardener": {"top": "t_shirt", "bottom": "trousers", "shoes": "boots",
                 "headgear": "sun_hat", "over": "apron"},
    "soldier": {"top": "long_sleeve", "bottom": "trousers", "shoes": "boots",
                "headgear": "helmet", "over": "backpack"},
    "biker": {"top": "jacket", "bottom": "trousers", "shoes": "boots",
              "headgear": "helmet"},

    "knight": {"top": "armour", "bottom": "trousers", "shoes": "tall_boots",
               "headgear": "helmet"},
    "wizard": {"bottom": "robe", "shoes": "boots", "headgear": "hat",
               "hair": "long"},
    "monk": {"bottom": "robe", "shoes": "sandals", "hair": "shaved"},
    "priest": {"bottom": "robe", "shoes": "shoes", "hair": "short"},
    "superhero": {"top": "tank_top", "bottom": "leggings", "shoes": "boots",
                  "over": "cape"},
    "royal": {"bottom": "dress", "shoes": "shoes", "over": "cape",
              "hair": "long"},
}

OUTFIT_NAMES = sorted(OUTFITS)


def options(slot):
    return sorted(SLOTS[slot]) if slot in SLOTS else []


def dress(outfit, name):
    """`outfit` with the named look applied over it, or None if there is none.

    Over it, not instead of it: an outfit names only the slots it sets, so a
    figure that has already been given a ponytail keeps it under a coat.
    """
    look = OUTFITS.get(name)
    if look is None:
        return None
    return clean(dict(clean(outfit), **look))


def clean(outfit):
    """An outfit dict with only slots and presets that exist."""
    out = dict(DEFAULT_OUTFIT)
    for slot, name in (outfit or {}).items():
        if slot in SLOTS and name in SLOTS[slot]:
            out[slot] = name
    return out


def worn(outfit):
    """Is anything actually being worn? Saves building a frame for nothing."""
    return any(v != "none" for v in clean(outfit).values())


# ---------------------------------------------------------------------------
# draping
# ---------------------------------------------------------------------------

def drape(profile, t0, t1, pad, flare=0.0, steps=11):
    """The body's profile between t0 and t1, pushed out by `pad` centimetres.

    Sampled rather than sliced: the stations a garment needs are not the
    stations the body happens to be authored with, and a hem has to land where
    the hem is, not at the nearest control point. The forward offset is carried
    through unchanged, so a garment sits on the mass underneath rather than on
    the bone.
    """
    out = []
    for i in range(steps):
        s = i / (steps - 1.0)
        t = t0 + (t1 - t0) * s
        w, d, off = sample_profile(profile, t)
        grow = pad + flare * s
        out.append((t, max(0.2, w + grow), max(0.2, d + grow), off))
    return out


def _piece(segment, t0, t1, pad, flare, coarsen):
    return _tube(segment["a"], segment["b"], segment["ref"],
                 drape(segment["profile"], t0, t1, pad, flare),
                 segment["scale_w"], segment["scale_d"], coarsen=coarsen,
                 round_start=False, round_end=False)


def _cloth_tube(start, direction, length, half_w, half_d, frame_axes,
                grow_w=0.0, grow_d=0.0, spacing=0.9):
    """A free-hanging piece of cloth: hair down a back, a skirt below a hem.

    Straight, because nothing here simulates cloth. A skirt that falls straight
    and widens reads as a skirt in a silhouette, which is all a depth map is
    asked for.

    `grow_w` and `grow_d` are centimetres added by the time it reaches the hem,
    not multipliers - a hem is a measurement, and a multiplier on a wide hip
    and a narrow one gives two different garments. Stations are spaced by
    length rather than counted, for the same reason the body's sweep is: eight
    of them down a 60 cm skirt are 7 cm apart and read as stairs.
    """
    axis = vnorm(direction)
    if vlen(axis) < 1e-9 or length <= 0.0:
        return []
    side, fwd = frame_axes
    steps = max(6, int(length / spacing) + 1)
    # half again the gap between stations, not half of it: a tapering stack
    # seen end-on shows every flat cylinder rim it has, and a ponytail came out
    # looking like a comb. The body's own sweep overlaps for the same reason.
    half = 0.78 * length / steps
    out = []
    for i in range(steps):
        s = (i + 0.5) / steps
        centre = vadd(start, vmul(axis, length * s))
        out.append(("slab", centre, (side, fwd, axis),
                    (max(0.2, half_w + grow_w * s),
                     max(0.2, half_d + grow_d * s), half)))
    return out


def layers(skeleton, segments, frame, coarsen=1.0):
    """Everything worn, as (slot, standoff, part) in the body's own units.

    The standoff is how far the piece stands proud of the body underneath, in
    centimetres, or None where there is no body underneath it. A rigged export
    needs the distinction: `wearables` cuts a garment out of the body's *swept*
    profile, which is thinner than a rigged mesh through a chest or a shoulder,
    so a drape has to be pushed out until it clears whatever the rig actually
    put there - by its own thickness, because a t-shirt clears by half a
    centimetre and an afro by five, and one number for both flattens the afro
    onto the skull. A fall of hair or a hat brim is not over the body at all;
    pushing *those* out drags them round to the front of the head.

    Each garment is one part per standoff, so the rasteriser hard-unions its
    pieces - a sleeve has to meet its body with an edge - while the garment as
    a whole still blends with the figure like any other part.
    """
    outfit = clean(getattr(skeleton, "outfit", None))
    if not worn(outfit):
        return []

    side, up = frame["side"], frame["up"]
    facing, down = frame["facing"], frame["down"]
    head_axis, face = frame["head_axis"], frame["face"]
    head_side, ear_mid = frame["head_side"], frame["ear_mid"]
    hip_mid, sh_mid = frame["hip_mid"], frame["sh_mid"]
    B = frame["body"]

    def head_at(t):
        """A point on the head's own axis, t along its profile."""
        seg = segments.get("head")
        if seg is None:
            return None
        return vadd(seg["a"], vmul(vsub(seg["b"], seg["a"]), t))

    def head_radius(t, pad=0.0):
        seg = segments.get("head")
        if seg is None:
            return 0.0, 0.0
        w, d, _off = sample_profile(seg["profile"], t)
        return w * seg["scale_w"] + pad, d * seg["scale_d"] + pad

    def torso_at(t):
        """(centre on the surface's own axis, half width, half depth) at t."""
        seg = segments.get("torso")
        if seg is None:
            return None
        at = vadd(seg["a"], vmul(vsub(seg["b"], seg["a"]), t))
        w, d, off = sample_profile(seg["profile"], t)
        return (vadd(at, vmul(facing, off)),
                w * seg["scale_w"], d * seg["scale_d"])

    out = []
    for slot in SLOT_ORDER:
        name = outfit.get(slot, "none")
        if name == "none":
            continue
        # Pieces that lie on the body are grouped by how far they stand off
        # it; the free ones go under None and are never pushed anywhere.
        buckets = {}

        def emit(standoff, pieces):
            if pieces:
                buckets.setdefault(standoff, []).extend(pieces)

        for piece in SLOTS[slot][name]:
            kind = piece[0]

            if kind == "drape":
                _k, seg_name, t0, t1, pad, flare = piece
                seg = segments.get(seg_name)
                if seg is not None:
                    emit(pad, _piece(seg, t0, t1, pad, flare, coarsen))

            elif kind in ("sleeve", "pauldron"):
                if kind == "pauldron":
                    dw, dd, dh = B["deltoid"]
                    for sd in ("r", "l"):
                        joint = skeleton.points[
                            [i for i, n in enumerate(_NAMES)
                             if n == sd + "_shoulder"][0]]
                        emit(2.2, _blob(vadd(joint, vmul(up, -1.4)),
                                        (side, facing, up),
                                        (dw + 2.2, dd + 2.2, dh + 1.4)))
                    continue
                _k, limb, t0, t1, pad, flare = piece
                for sd in ("r", "l"):
                    seg = segments.get("%s_%s" % (sd, limb))
                    if seg is not None:
                        emit(pad, _piece(seg, t0, t1, pad, flare, coarsen))

            elif kind == "fall":            # hair down the back
                _k, t, length, thick = piece
                anchor = head_at(t)
                if anchor is None:
                    continue
                w, d = head_radius(t, 0.0)
                # Hair lies on a back, and a back is further back than the
                # skull it hangs from. Starting it on the skull buries it in
                # the shoulders; starting it behind the shoulders leaves a
                # plank hovering with a gap above it. So it starts touching the
                # head and *leans* back over its length by the difference, which
                # is what hair on a back actually does.
                head_back = d + thick * 0.45
                lean = 0.0
                torso = segments.get("torso")
                if torso is not None:
                    at = vadd(torso["a"],
                              vmul(vsub(torso["b"], torso["a"]), 0.18))
                    _tw, td, toff = sample_profile(torso["profile"], 0.18)
                    at = vadd(at, vmul(facing, toff))
                    behind = vdot(vsub(anchor,
                                       vadd(at, vmul(facing,
                                                     -td * torso["scale_d"]))),
                                  face)
                    lean = max(0.0, behind + thick * 0.5 - head_back)
                start = vadd(anchor, vmul(face, -head_back))
                run = vadd(vmul(down, length), vmul(face, -lean))
                shell = _cloth_tube(
                    start, run, vlen(run), w * 0.95, thick, (head_side, face),
                    grow_w=w * 0.10, grow_d=thick * 0.25)
                # a rounded end, so the hair stops rather than being cut off
                tip = vadd(start, run)
                shell += _blob(tip, (head_side, face, vnorm(run)),
                               (w * 1.05, thick * 1.25, thick * 1.1))
                emit(None, shell)

            elif kind == "tail":            # a ponytail, clear of the skull
                _k, t, length, thick = piece[:4]
                spread = piece[4] if len(piece) > 4 else 0.0
                anchor = head_at(t)
                if anchor is None:
                    continue
                _w, d = head_radius(t, 0.0)
                # a single tail down the centre, or a pair out to the sides
                offsets = ((0.0,) if spread <= 0.0 else (-spread, spread))
                for dx in offsets:
                    root = vadd(vadd(anchor, vmul(head_side, dx)),
                                vmul(face, -d - thick * 0.5))
                    # down and back, so it hangs behind the shoulder rather
                    # than standing off the crown. A pair also splays out as
                    # it falls, or the two hang in the same place as one and
                    # the silhouette has nothing in it that says "two".
                    run = vadd(down, vmul(face, -0.22))
                    if dx:
                        run = vadd(run, vmul(head_side,
                                             0.30 * (1.0 if dx > 0 else -1.0)))
                    emit(None, _cloth_tube(
                        root, vnorm(run), length, thick, thick,
                        (head_side, face),
                        grow_w=-thick * 0.35, grow_d=-thick * 0.35))

            elif kind == "knot":            # a bun
                _k, t, behind, radius = piece
                anchor = head_at(t)
                if anchor is None:
                    continue
                _w, d = head_radius(t, 0.0)
                centre = vadd(anchor, vmul(face, behind * d - radius * 0.3))
                emit(None, _blob(centre, (head_side, face, head_axis),
                                 (radius, radius, radius)))

            elif kind == "hood":            # sits behind the neck, not on it
                _k, t, behind, radius = piece
                anchor = head_at(t)
                if anchor is None:
                    continue
                _w, d = head_radius(t, 0.0)
                centre = vadd(anchor, vmul(face, behind - d * 0.1))
                emit(None, _blob(centre, (head_side, face, head_axis),
                                 (radius * 0.95, radius * 0.8, radius)))

            elif kind == "brim":
                _k, t, reach, thick = piece
                anchor = head_at(t)
                if anchor is None:
                    continue
                w, d = head_radius(t, 0.6)
                # Round, near enough. A head is wider than it is deep, so
                # giving the reach to the depth alone and 45% of it to the
                # width made a brim that is a plank end-on: a walking figure
                # in a sun hat came out wearing a diving board.
                emit(None, [("slab", anchor, (head_side, face, head_axis),
                             (w + reach * 0.85, d + reach, thick))])

            elif kind == "skirt":           # hangs from a hem on the torso
                _k, t, length, flare = piece
                seg = segments.get("torso")
                if seg is None:
                    continue
                anchor = vadd(seg["a"], vmul(vsub(seg["b"], seg["a"]), t))
                w, d, off = sample_profile(seg["profile"], t)
                anchor = vadd(anchor, vmul(facing, off))
                # 1.2 of clearance at the hem, which is also what it has to
                # clear the rigged hip by where the two still overlap
                emit(1.2, _cloth_tube(
                    anchor, down, length, w + 1.2, d + 1.2, (side, facing),
                    grow_w=flare, grow_d=flare * 0.8))

            elif kind == "panel":           # an apron: flat down the front
                _k, t0, t1, thick, stand, narrow = piece
                torso = segments.get("torso")
                if torso is None:
                    continue
                run = vsub(torso["b"], torso["a"])
                probe = [torso_at(t0 + (t1 - t0) * i / 10.0)
                         for i in range(11)]
                probe = [v for v in probe if v is not None]
                if not probe:
                    continue
                # One flat slab, not a stack that follows the body station by
                # station. A panel of cloth has one width and hangs in one
                # plane; giving each station the profile's own width and depth
                # put a step between every pair of them, and an apron came out
                # of the depth map in horizontal bands. It clears the rigged
                # body by the standoff wherever the body is deeper than the
                # plane, which is what makes it drape over a belly.
                half_w = max(w for _c, w, _d in probe) * narrow
                front = max(vdot(c, facing) + d for c, _w, d in probe)
                mid = vadd(torso["a"], vmul(run, 0.5 * (t0 + t1)))
                mid = vadd(mid, vmul(facing,
                                     front - thick - vdot(mid, facing)))
                emit(stand, [("slab", mid, (side, facing, vnorm(run)),
                              (max(0.4, half_w), thick,
                               0.5 * (t1 - t0) * vlen(run)))])

            elif kind == "cloak":           # a cape down the back
                _k, t, length, thick, flare = piece
                at = torso_at(t)
                if at is None:
                    continue
                centre, w, d = at
                start = vadd(centre, vmul(face, -(d + thick)))
                shell = _cloth_tube(start, down, length, w, thick,
                                    (side, face), grow_w=flare, grow_d=0.0)
                # a rounded hem, for the same reason hair gets one: a stack of
                # cylinders cut off square reads as a plank
                shell += _blob(vadd(start, vmul(down, length)),
                               (side, face, down),
                               (w + flare, thick * 1.2, thick * 1.6))
                emit(None, shell)

            elif kind == "pack":            # a rucksack behind the shoulders
                _k, t, half_w, half_d, half_h = piece
                at = torso_at(t)
                if at is None:
                    continue
                centre, w, d = at
                # A box, not a blob. Swept as an ellipsoid a rucksack is a
                # beach ball on someone's back; the corners are most of what
                # says "pack" in a silhouette.
                emit(None, [("box", vadd(centre, vmul(face, -(d + half_d))),
                             (side, face, up),
                             (min(half_w, w * 1.15), half_d, half_h))])

        for standoff in sorted(buckets, key=lambda v: (v is None, v)):
            out.append((slot, standoff, buckets[standoff]))
    return out


def parts(skeleton, segments, frame, coarsen=1.0):
    """Everything worn, one part per garment. See `layers` for the standoffs.

    This is what the viewport sweep and the anatomy depth want: they draw a
    garment into the same blended figure the body is drawn into, so nothing
    has to clear anything and a garment's pieces stay one solid.
    """
    grouped, seen = [], {}
    for slot, _standoff, part in layers(skeleton, segments, frame, coarsen):
        if slot in seen:
            seen[slot].extend(part)
        else:
            seen[slot] = list(part)
            grouped.append(seen[slot])
    return grouped


_NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder",
          "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip",
          "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]


def describe():
    lines = ["  %-9s %s" % (slot, ", ".join(options(slot)))
             for slot in SLOT_ORDER]
    lines.append("  %-9s %s" % ("outfits", ", ".join(OUTFIT_NAMES)))
    return "\n".join(lines)


def _selftest():
    from openpose3d_editor import (BODY_PRESETS, Skeleton, body_parts,
                                   body_segments, preset_params)
    ok = True

    def check(label, condition, extra=""):
        nonlocal ok
        ok = ok and bool(condition)
        print(("PASS " if condition else "FAIL ") + label
              + ("  " + extra if extra else ""))

    print("wearables self-test")

    def extent(skeleton):
        lo = [1e18] * 3
        hi = [-1e18] * 3
        for part in body_parts(skeleton):
            for _kind, centre, axes, radii in part:
                for i in range(3):
                    reach = sum(abs(r * a[i]) for a, r in zip(axes, radii))
                    lo[i] = min(lo[i], centre[i] - reach)
                    hi[i] = max(hi[i], centre[i] + reach)
        return lo, hi

    def volume(skeleton):
        """Rough solid count: how many primitives and how much they span."""
        return sum(len(part) for part in body_parts(skeleton))

    # every preset has to build, on every body, without blowing up
    bare = Skeleton(preset_params("Male, average"))
    bare_lo, bare_hi = extent(bare)
    trouble = []
    for slot in SLOTS:
        for name in options(slot):
            for preset in ("Male, average", "Female, average", "Child, about 7"):
                skeleton = Skeleton(preset_params(preset))
                skeleton.outfit = {slot: name}
                try:
                    built = body_parts(skeleton)
                except Exception as exc:                    # pragma: no cover
                    trouble.append("%s/%s on %s: %s" % (slot, name, preset, exc))
                    continue
                for part in built:
                    for _k, centre, _ax, radii in part:
                        sane = (all(abs(v) < 1e5 for v in centre)
                                and all(0.0 < r < 500.0 for r in radii))
                        if not sane:
                            trouble.append("%s/%s on %s: %r %r"
                                           % (slot, name, preset, centre, radii))
                            break
    check("every preset builds on every body", not trouble,
          "%d problem(s): %s" % (len(trouble), trouble[:1]))

    # a garment has to be bigger than the body, never smaller
    shrunk = []
    for slot in SLOTS:
        for name in options(slot):
            if name == "none":
                continue
            skeleton = Skeleton(preset_params("Male, average"))
            skeleton.outfit = {slot: name}
            lo, hi = extent(skeleton)
            for i in range(3):
                if lo[i] > bare_lo[i] + 1e-6 or hi[i] < bare_hi[i] - 1e-6:
                    shrunk.append("%s/%s" % (slot, name))
                    break
    check("nothing worn makes the figure smaller", not shrunk, str(shrunk[:3]))

    # and it has to add something, or it is not being worn
    missing = []
    for slot in SLOTS:
        for name in options(slot):
            if name == "none":
                continue
            skeleton = Skeleton(preset_params("Male, average"))
            skeleton.outfit = {slot: name}
            if volume(skeleton) <= volume(bare):
                missing.append("%s/%s" % (slot, name))
    check("everything worn adds solids to the figure", not missing,
          str(missing[:3]))

    # the slots have to stack rather than replace one another
    dressed = Skeleton(preset_params("Female, average"))
    dressed.outfit = {"hair": "long", "top": "jacket", "bottom": "trousers",
                      "shoes": "boots", "headgear": "hat"}
    full = volume(dressed)
    one = []
    for slot, name in dressed.outfit.items():
        single = Skeleton(preset_params("Female, average"))
        single.outfit = {slot: name}
        one.append(volume(single) - volume(Skeleton(preset_params(
            "Female, average"))))
    bare_f = volume(Skeleton(preset_params("Female, average")))
    check("five slots at once add what five slots add",
          abs(full - bare_f - sum(one)) < 1, "%d vs %d" % (full - bare_f,
                                                           sum(one)))

    # a hem lands where the preset says, not at the nearest control point
    from openpose3d_editor import P_THIGH
    hem = drape(P_THIGH, 0.0, 0.42, 1.0)
    check("a drape starts and ends exactly where it was asked to",
          abs(hem[0][0] - 0.0) < 1e-9 and abs(hem[-1][0] - 0.42) < 1e-9,
          "%.3f .. %.3f" % (hem[0][0], hem[-1][0]))
    w0, d0, _o = sample_profile(P_THIGH, 0.0)
    check("and it is exactly the padding wider than the body",
          abs(hem[0][1] - (w0 + 1.0)) < 1e-9 and abs(hem[0][2] - (d0 + 1.0)) < 1e-9)
    flared = drape(P_THIGH, 0.0, 0.42, 1.0, flare=4.0)
    check("a flare widens towards the hem and nowhere else",
          abs(flared[0][1] - hem[0][1]) < 1e-9
          and flared[-1][1] > hem[-1][1] + 3.9)

    # Every layer says how far it stands off the body, and a rigged export
    # leans on that: a drape clears the rig by its own padding, a piece with
    # nothing under it - a fall of hair, a hat brim, a rucksack - clears
    # nothing and keeps its own depth. One figure for all of them flattens a
    # five-centimetre afro onto the skull; pushing the free ones out drags a
    # ponytail round to the front of the face.
    dressed = Skeleton(preset_params("Female, average"))
    segments, frame = body_segments(dressed)
    wrong = []
    for slot in SLOTS:
        for name in options(slot):
            if name == "none":
                continue
            dressed.outfit = {slot: name}
            pads = {p[4] for p in SLOTS[slot][name]
                    if p[0] in ("drape", "sleeve")}
            got = {standoff for _s, standoff, _p
                   in layers(dressed, segments, frame, 1.0)}
            if pads - got:
                wrong.append("%s/%s wants %s, got %s"
                             % (slot, name, sorted(pads), sorted(
                                 v for v in got if v is not None)))
    check("a drape stands off the body by its own padding", not wrong,
          str(wrong[:2]))

    free = []
    for slot in SLOTS:
        for name in options(slot):
            kinds = {p[0] for p in SLOTS[slot][name]}
            if kinds - {"fall", "tail", "knot", "hood", "brim", "cloak",
                        "pack"}:
                continue                       # something rides the body too
            dressed.outfit = {slot: name}
            got = [standoff for _s, standoff, _p
                   in layers(dressed, segments, frame, 1.0)]
            if got and any(v is not None for v in got):
                free.append("%s/%s: %s" % (slot, name, got))
    check("and a piece with nothing under it stands off nothing", not free,
          str(free[:2]))

    dressed.outfit = {"hair": "long"}
    stack = layers(dressed, segments, frame, 1.0)
    check("hair on a head is two layers: the cap on it and the fall off it",
          sorted(v is None for _s, v, _p in stack) == [False, True],
          str([v for _s, v, _p in stack]))
    check("and `parts` hands the whole garment over as one solid",
          len(parts(dressed, segments, frame, 1.0)) == 1
          and sum(len(p) for _s, _v, p in stack)
          == len(parts(dressed, segments, frame, 1.0)[0]))

    # outfits
    dressed.outfit = {}
    trouble = []
    for name, look in OUTFITS.items():
        for slot, garment in look.items():
            if slot not in SLOTS or garment not in SLOTS[slot]:
                trouble.append("%s: %s/%s" % (name, slot, garment))
    check("every outfit names slots and garments that exist", not trouble,
          str(trouble[:3]))
    check("\"bare\" is the only one that names every slot",
          [n for n, look in OUTFITS.items()
           if set(look) == set(SLOTS)] == ["bare"],
          str([n for n, look in OUTFITS.items() if set(look) == set(SLOTS)]))
    check("an outfit sets the slots it names and leaves the rest",
          dress({"hair": "ponytail"}, "winter")
          == dict(DEFAULT_OUTFIT, hair="ponytail", top="coat",
                  bottom="trousers", shoes="boots", headgear="beanie",
                  over="scarf"),
          str(dress({"hair": "ponytail"}, "winter")))
    check("and one nobody has is refused rather than guessed at",
          dress({}, "black tie") is None)
    check("every outfit puts something on", all(worn(dress({}, n))
                                                for n in OUTFIT_NAMES
                                                if n != "bare"))
    check("and \"bare\" takes it all off",
          not worn(dress({"top": "coat", "hair": "afro"}, "bare")))

    # nonsense in, default out
    check("an unknown slot or preset falls back rather than failing",
          clean({"hat": "sombrero", "hair": "mullet", "top": "jacket"})
          == dict(DEFAULT_OUTFIT, top="jacket"))
    check("an empty outfit is not worn", not worn({}) and not worn(None))
    check("and one with something in it is", worn({"hair": "bob"}))

    # the segments a garment rides must exist for every body
    for preset in BODY_PRESETS:
        segments, _frame = body_segments(Skeleton(preset_params(preset)))
        need = {"torso", "head", "l_upper_arm", "r_forearm", "l_thigh",
                "r_calf", "l_foot"}
        check("%-16s offers every segment a garment rides" % preset,
              need <= set(segments), str(sorted(need - set(segments))))

    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    if "--list" in sys.argv:
        print(describe())
        raise SystemExit(0)
    print(__doc__)
