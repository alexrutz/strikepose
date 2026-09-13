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
    "bob": [("drape", "head", 0.30, 1.00, 2.4, 0.0),
            ("fall", 0.32, 11.0, 2.8)],
    "long": [("drape", "head", 0.36, 1.00, 2.0, 0.0),
             ("fall", 0.50, 46.0, 3.4)],
    "ponytail": [("drape", "head", 0.46, 1.00, 1.1, 0.0),
                 ("tail", 0.78, 26.0, 3.6)],
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
    "shoes": [("sleeve", "foot", -0.04, 1.04, 0.9, 0.0)],
    "boots": [("sleeve", "foot", -0.05, 1.05, 1.2, 0.0),
              ("sleeve", "calf", 0.52, 1.02, 1.4, 0.2)],
    "tall_boots": [("sleeve", "foot", -0.05, 1.05, 1.2, 0.0),
                   ("sleeve", "calf", 0.06, 1.02, 1.5, 0.3)],
}

HEADGEAR = {
    "none": [],
    "cap": [("drape", "head", 0.70, 1.02, 1.1, 0.0),
            ("brim", 0.72, 8.5, 1.0)],
    "hat": [("drape", "head", 0.66, 1.04, 1.3, 0.0),
            ("brim", 0.68, 13.0, 1.0)],
    "beanie": [("drape", "head", 0.62, 1.04, 1.4, 0.0)],
    "helmet": [("drape", "head", 0.44, 1.04, 1.8, 0.0)],
}

SLOTS = {"hair": HAIR, "top": TOPS, "bottom": BOTTOMS,
         "shoes": FEET, "headgear": HEADGEAR}

DEFAULT_OUTFIT = {slot: "none" for slot in SLOTS}


def options(slot):
    return sorted(SLOTS[slot]) if slot in SLOTS else []


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


def parts(skeleton, segments, frame, coarsen=1.0):
    """Everything the figure is wearing, as parts in the body's own units.

    Each garment is one part, so the rasteriser hard-unions its pieces - a
    sleeve has to meet its body with an edge - while the garment as a whole
    still blends with the figure like any other part.
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

    out = []
    for slot, name in outfit.items():
        if name == "none":
            continue
        piece_list = []
        for piece in SLOTS[slot][name]:
            kind = piece[0]

            if kind == "drape":
                _k, seg_name, t0, t1, pad, flare = piece
                seg = segments.get(seg_name)
                if seg is not None:
                    piece_list += _piece(seg, t0, t1, pad, flare, coarsen)

            elif kind in ("sleeve", "pauldron"):
                if kind == "pauldron":
                    dw, dd, dh = B["deltoid"]
                    for sd in ("r", "l"):
                        joint = skeleton.points[
                            [i for i, n in enumerate(_NAMES)
                             if n == sd + "_shoulder"][0]]
                        piece_list += _blob(vadd(joint, vmul(up, -1.4)),
                                            (side, facing, up),
                                            (dw + 2.2, dd + 2.2, dh + 1.4))
                    continue
                _k, limb, t0, t1, pad, flare = piece
                for sd in ("r", "l"):
                    seg = segments.get("%s_%s" % (sd, limb))
                    if seg is not None:
                        piece_list += _piece(seg, t0, t1, pad, flare, coarsen)

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
                piece_list += _cloth_tube(
                    start, run, vlen(run), w * 0.95, thick, (head_side, face),
                    grow_w=w * 0.10, grow_d=thick * 0.25)
                # a rounded end, so the hair stops rather than being cut off
                tip = vadd(start, run)
                piece_list += _blob(tip, (head_side, face, vnorm(run)),
                                    (w * 1.05, thick * 1.25, thick * 1.1))

            elif kind == "tail":            # a ponytail, clear of the skull
                _k, t, length, thick = piece
                anchor = head_at(t)
                if anchor is None:
                    continue
                _w, d = head_radius(t, 0.0)
                root = vadd(anchor, vmul(face, -d - thick * 0.5))
                # down and back, so it hangs behind the shoulder rather than
                # standing off the crown
                piece_list += _cloth_tube(
                    root, vnorm(vadd(down, vmul(face, -0.22))),
                    length, thick, thick, (head_side, face),
                    grow_w=-thick * 0.35, grow_d=-thick * 0.35)

            elif kind == "knot":            # a bun
                _k, t, behind, radius = piece
                anchor = head_at(t)
                if anchor is None:
                    continue
                _w, d = head_radius(t, 0.0)
                centre = vadd(anchor, vmul(face, behind * d - radius * 0.3))
                piece_list += _blob(centre, (head_side, face, head_axis),
                                    (radius, radius, radius))

            elif kind == "hood":            # sits behind the neck, not on it
                _k, t, behind, radius = piece
                anchor = head_at(t)
                if anchor is None:
                    continue
                _w, d = head_radius(t, 0.0)
                centre = vadd(anchor, vmul(face, behind - d * 0.1))
                piece_list += _blob(centre, (head_side, face, head_axis),
                                    (radius * 0.95, radius * 0.8, radius))

            elif kind == "brim":
                _k, t, reach, thick = piece
                anchor = head_at(t)
                if anchor is None:
                    continue
                w, d = head_radius(t, 0.6)
                piece_list.append(("slab", anchor, (head_side, face, head_axis),
                                   (w + reach * 0.45, d + reach, thick)))

            elif kind == "skirt":           # hangs from a hem on the torso
                _k, t, length, flare = piece
                seg = segments.get("torso")
                if seg is None:
                    continue
                anchor = vadd(seg["a"], vmul(vsub(seg["b"], seg["a"]), t))
                w, d, off = sample_profile(seg["profile"], t)
                anchor = vadd(anchor, vmul(facing, off))
                piece_list += _cloth_tube(
                    anchor, down, length, w + 1.2, d + 1.2, (side, facing),
                    grow_w=flare, grow_d=flare * 0.8)

        if piece_list:
            out.append(piece_list)
    return out


_NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder",
          "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip",
          "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]


def describe():
    return "\n".join("  %-9s %s" % (slot, ", ".join(options(slot)))
                     for slot in ("hair", "top", "bottom", "shoes", "headgear"))


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
