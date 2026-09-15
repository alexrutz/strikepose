#!/usr/bin/env python3
"""The swept body: measured cross-sections carried along the bones.

This draws the viewport at sixty frames a second and it gives `wearables` a
profile to cut a garment out of. It is an approximation and it is NEVER an
export - see `bodies_lib` and `exporting.rigged_depth_image` for the real
thing, which is rigged geometry or nothing.

Roll is carried ALONG a limb by `carry_frame`, never re-derived per bone from
a fixed reference: a projection of the body's facing onto each bone in turn is
undefined where a bone points along it, flips a full 180 degrees either side,
and is 45 degrees out in plain diagonal poses. That is what used to come out
of the depth map as crooked and twisted limbs.
"""

from __future__ import annotations

import math

from anthro import BASE_BODY, merge_body
from skeleton import KEYPOINT_NAMES, clean_extremities
from vecmath import (any_perpendicular, matvec, rotation_between, vadd,
                     vcross, vdot, vlen, vmul, vnorm, vsub)


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


# ---------------------------------------------------------------------------

P_UPPER_ARM = [(0.00, 5.3, 5.5, 0.0), (0.22, 5.2, 5.7, 0.3),
               (0.58, 4.6, 4.9, 0.0), (1.00, 3.7, 4.1, -0.3)]
P_FOREARM = [(0.00, 4.0, 4.4, 0.0), (0.18, 4.3, 4.8, 0.1),
             (0.58, 3.4, 3.7, 0.0), (1.00, 2.6, 3.0, 0.0)]
P_THIGH = [(0.00, 8.4, 8.2, 0.2), (0.18, 8.2, 8.6, -0.4),
           (0.60, 6.6, 6.9, -0.3), (1.00, 5.3, 5.6, 0.0)]
P_CALF = [(0.00, 5.2, 5.6, 0.0), (0.22, 5.1, 6.2, -1.7),
          (0.62, 3.6, 4.1, -0.9), (1.00, 2.7, 3.2, 0.2)]
P_NECK = [(0.00, 6.6, 7.0, -0.6), (0.55, 6.0, 6.2, -0.2), (1.00, 5.5, 5.7, 0.0)]
# head runs chin (0) to crown (1); the chin sits forward of the ear axis
P_HEAD = [(0.00, 2.0, 2.8, 3.6), (0.05, 3.6, 4.8, 3.3), (0.13, 5.2, 6.8, 2.6),
          (0.27, 6.6, 8.4, 1.4), (0.42, 7.4, 9.2, 0.4), (0.57, 7.6, 9.1, 0.0),
          (0.74, 7.0, 7.9, -0.4), (0.86, 6.1, 6.8, -0.8),
          (0.94, 4.8, 5.3, -0.9), (0.98, 3.4, 3.8, -1.0),
          (1.00, 1.8, 2.0, -1.0)]
# hand: thin across the palm, wide front to back, as it hangs beside the thigh
P_HAND = [(0.00, 2.4, 4.2, 0.0), (0.35, 2.3, 4.7, 0.0),
          (0.75, 2.0, 4.1, 0.0), (1.00, 1.5, 2.6, 0.0)]
# foot: swept heel to toe, "depth" is thickness above the sole
P_FOOT = [(0.00, 3.6, 4.2, 0.0), (0.30, 4.2, 3.8, 0.0),
          (0.72, 3.8, 2.7, 0.0), (1.00, 2.6, 1.9, 0.0)]


def torso_profile(body):
    """Shoulder line to hip line, extended past both. Anchored on the preset's
    three measured cross-sections and shaped by the ribcage/waist/pelvis
    relationship that is common to every human torso."""
    cw, cd = body["chest"]
    ww, wd = body["waist"]
    pw, pd = body["pelvis"]
    belly = body.get("belly", 0.0)
    return [
        (-0.085, 0.46 * cw, 0.56 * cd, -0.7),    # base of the neck
        (-0.035, 0.84 * cw, 0.84 * cd, -0.2),    # trapezius sloping outwards
        (0.075, 0.99 * cw, 0.97 * cd, 0.3),      # armpit line
        (0.210, 1.00 * cw, 1.00 * cd, 0.5),      # chest / bust line
        (0.420, 0.88 * cw, 0.92 * cd, 0.3 + 0.35 * belly),
        # the narrowest station is the tenth rib, and it sits higher on a woman:
        # 0.569 of the way from shoulder to hip against 0.593 on a man
        (body.get("waist_t", 0.593), ww, wd, belly),
        (0.780, 0.90 * pw, 0.94 * pd, 0.3 * belly),
        (0.930, pw, pd, -0.5),                   # hips
        (1.090, 0.95 * pw, 1.01 * pd, -1.9),     # seat
        (1.190, 0.78 * pw, 0.88 * pd, -2.6),     # crotch
    ]


def _catmull(p0, p1, p2, p3, s):
    return 0.5 * ((2.0 * p1) + (-p0 + p2) * s
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * s * s
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * s * s * s)


def sample_profile(profile, t):
    """Catmull-Rom through the control stations, so the silhouette curves
    instead of running between straight taper segments."""
    n = len(profile)
    if t <= profile[0][0]:
        return profile[0][1:]
    if t >= profile[-1][0]:
        return profile[-1][1:]
    i = 0
    while i < n - 2 and t > profile[i + 1][0]:
        i += 1
    t0, t1 = profile[i][0], profile[i + 1][0]
    s = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    p0, p1 = profile[max(0, i - 1)], profile[i]
    p2, p3 = profile[i + 1], profile[min(n - 1, i + 2)]
    return tuple(_catmull(p0[j], p1[j], p2[j], p3[j], s) for j in (1, 2, 3))


def carry_frame(ref, axis):
    """Carry a cross-section frame from one bone direction onto another.

    `ref` is (reference axis, reference forward) - the frame the bone inherits
    its roll from, forward perpendicular to the reference axis. The frame is
    swung onto `axis` by the *minimal* rotation between the two directions, so
    the bone gains no twist about itself that the reference did not already
    have. Returns (forward, side), both perpendicular to `axis`.

    Projecting a fixed reference onto the bone instead - which is what this
    used to do - is discontinuous. `facing - axis * (facing . axis)` vanishes
    the moment a bone points along the body's forward direction (a reach, a
    sitting thigh, a kick), so the cross-section snapped 90 degrees there and
    flipped a full 180 either side of it: the calf mass jumped from behind the
    tibia to in front of it and the profile's width and depth swapped over. In
    plain diagonal poses it was 45 degrees out. That is what wrung the limbs in
    the depth map. A swing has no such singularity: it only degenerates when
    the bone points exactly opposite its reference, which down a limb chain
    means folded back on itself.
    """
    ref_axis, ref_fwd = ref
    fwd = matvec(rotation_between(ref_axis, axis), ref_fwd)
    fwd = vsub(fwd, vmul(axis, vdot(fwd, axis)))
    if vlen(fwd) < 1e-6:
        fwd = any_perpendicular(axis)
    fwd = vnorm(fwd)
    return fwd, vnorm(vcross(axis, fwd))


def carry_chain(ref, *joints):
    """One (axis, forward) frame per bone along a chain of joints.

    Each bone inherits the roll of the one above it, so a bent elbow carries
    the forearm round with it instead of letting it pick its own orientation.
    A zero-length bone passes its parent's frame straight through rather than
    restarting the chain from the torso.
    """
    out = []
    for a, b in zip(joints, joints[1:]):
        axis = vnorm(vsub(b, a))
        if vlen(axis) < 1e-9:
            out.append(ref)
            continue
        ref = (axis, carry_frame(ref, axis)[0])
        out.append(ref)
    return out


def _tube(a, b, ref, profile, scale_w=1.0, scale_d=1.0,
          spacing=0.45, round_start=True, round_end=True, coarsen=1.0):
    """Sweep a profile from a to b as a chain of oriented ellipsoids.

    `ref` is the (axis, forward) frame this bone takes its roll from; see
    `carry_frame`. Passing the bone's own axis as the reference axis means
    "use this forward as it stands".
    """
    span = vsub(b, a)
    length = vlen(span)
    if length < 1e-6:
        return []
    axis = vmul(span, 1.0 / length)
    fwd, side = carry_frame(ref, axis)

    t0, t1 = profile[0][0], profile[-1][0]
    steps = max(5, int(abs(t1 - t0) * length / (spacing * coarsen)) + 1)
    half = 0.55 * abs(t1 - t0) * length / (steps - 1.0)
    out = []
    for i in range(steps):
        t = t0 + (t1 - t0) * i / (steps - 1.0)
        w, d, off = sample_profile(profile, t)
        w, d = max(0.2, w * scale_w), max(0.2, d * scale_d)
        centre = vadd(vadd(a, vmul(axis, t * length)), vmul(fwd, off))
        # a stack of touching elliptical slabs sweeps the profile exactly;
        # ellipsoid stations would lose width between stations and scallop
        out.append(("slab", centre, (side, fwd, axis), (w, d, half)))
    for at_end, rounded in ((False, round_start), (True, round_end)):
        if not rounded:
            continue
        t = t1 if at_end else t0
        w, d, off = sample_profile(profile, t)
        w, d = max(0.2, w * scale_w), max(0.2, d * scale_d)
        centre = vadd(vadd(a, vmul(axis, t * length)), vmul(fwd, off))
        out.append(("ball", centre, (side, fwd, axis), (w, d, min(w, d))))
    return out


def _blob(centre, axes, radii):
    return [("ball", centre, axes, radii)]


def body_segments(skeleton, respect_visibility=True):
    """Every swept part of the figure, before any of it is turned into solids.

    Returns (segments, frame). A segment is a dict carrying what `_tube` needs:
    where it runs from and to, the frame it takes its roll from, the profile it
    sweeps and how much to scale that profile by.

    Pulled out of `body_parts` because a garment is the body's own sweep,
    clipped to a stretch of it and padded outward - which is what makes a
    sleeve fit whatever preset and whatever pose it finds, with nothing to fit
    and nothing to drift. `wearables` reads this table; so does the body.
    """
    P = skeleton.points
    ids = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
    B = getattr(skeleton, "body", None)
    if B is None or "deltoid" not in B:      # scene from an older version
        B = merge_body(B)
        skeleton.body = B
    # Hand and foot angles, if the figure carries any. Cleaned here rather
    # than trusted, so a scene file or a local model cannot bend a wrist
    # somewhere a wrist does not go.
    turned = clean_extremities(getattr(skeleton, "extremities", None))

    def pt(n):
        return P[ids[n]]

    def vis(*names):
        return True if not respect_visibility else all(
            skeleton.visible[ids[n]] for n in names)

    r_sh, l_sh = pt("r_shoulder"), pt("l_shoulder")
    r_hip, l_hip = pt("r_hip"), pt("l_hip")
    sh_mid = vmul(vadd(r_sh, l_sh), 0.5)
    hip_mid = vmul(vadd(r_hip, l_hip), 0.5)
    side = vnorm(vsub(l_sh, r_sh))
    up_t = vnorm(vsub(sh_mid, hip_mid))
    if vlen(side) < 1e-6:
        side = (1.0, 0.0, 0.0)
    if vlen(up_t) < 1e-6:
        up_t = (0.0, 1.0, 0.0)
    facing = vnorm(vcross(side, up_t))
    if vlen(facing) < 1e-6:
        facing = (0.0, 0.0, 1.0)

    # Frames. Every swept part takes its roll from the one above it, ending
    # at the torso, so a limb keeps the orientation it has in the rest pose
    # however it is posed. See `carry_frame` for why deriving each bone's
    # frame from `facing` on its own twisted them instead.
    down = vmul(up_t, -1.0)
    trunk_ref = (vnorm(vsub(hip_mid, sh_mid)), facing)

    segments = {}

    def add(name, a, b, ref, profile, scale_w=1.0, scale_d=1.0, **kw):
        segments[name] = dict(a=a, b=b, ref=ref, profile=profile,
                              scale_w=scale_w, scale_d=scale_d, **kw)

    if vis("r_shoulder", "l_shoulder", "r_hip", "l_hip"):
        add("torso", sh_mid, hip_mid, trunk_ref, torso_profile(B),
            round_start=False, round_end=False)

    girth = (B["arm_girth"], B["forearm_girth"], B["thigh_girth"], B["calf_girth"])
    for sd in ("r", "l"):
        sh, el, wr = sd + "_shoulder", sd + "_elbow", sd + "_wrist"
        hp, kn, an = sd + "_hip", sd + "_knee", sd + "_ankle"
        # The chains are built whole, before anything is emitted: a hidden
        # upper arm must not change how the forearm below it is rolled.
        # The rest pose hangs both limbs straight down, so both start from the
        # torso's own forward with `down` as the reference axis.
        upper, fore = carry_chain((down, facing), pt(sh), pt(el), pt(wr))
        thigh, calf = carry_chain((down, facing), pt(hp), pt(kn), pt(an))
        if vis(sh, el):
            add(sd + "_upper_arm", pt(sh), pt(el), upper, P_UPPER_ARM,
                girth[0], girth[0])
        if vis(el, wr):
            add(sd + "_forearm", pt(el), pt(wr), fore, P_FOREARM,
                girth[1], girth[1])
        if vis(wr, el):
            # The hand runs on along the forearm, so it shares its frame: the
            # palm keeps facing the way it does with the arm hanging. Where
            # the figure carries hand angles - which eighteen keypoints cannot
            # give and someone had to set - the sweep is turned by them too,
            # so the viewport shows what the export will render rather than a
            # different hand.
            d = vnorm(vsub(pt(wr), pt(el)))
            bend, roll = turned.get(sd + "_hand", (0.0, 0.0))
            if bend or roll:
                d = _spin(_spin(d, fore[1], math.radians(-bend)),
                          d, math.radians(roll))
            add(sd + "_hand", pt(wr), vadd(pt(wr), vmul(d, 17.0 * B["hand"])),
                fore, P_HAND, B["hand"], B["hand"])
        if vis(hp, kn):
            add(sd + "_thigh", pt(hp), pt(kn), thigh, P_THIGH,
                girth[2], girth[2])
        if vis(kn, an):
            add(sd + "_calf", pt(kn), pt(an), calf, P_CALF, girth[3], girth[3])
        if vis(an):
            drop = vnorm(vsub(pt(an), pt(kn))) if vis(kn) else down
            sole = vadd(pt(an), vmul(drop, 3.2 * B["foot"]))
            ahead = facing
            lift, splay = turned.get(sd + "_foot", (0.0, 0.0))
            if lift or splay:
                # out from the midline is the figure's own side, mirrored, so
                # a positive splay points both toes away from each other
                ahead = _spin(_spin(ahead, vcross(ahead, up_t),
                                    math.radians(-lift)),
                              up_t, math.radians(splay)
                              * (1.0 if sd == "l" else -1.0))
            heel = vadd(sole, vmul(ahead, -6.0 * B["foot"]))
            toe = vadd(sole, vmul(ahead, 19.0 * B["foot"]))
            # the foot turns off the shin, so its thickness stays across the
            # sole however the leg is posed
            add(sd + "_foot", heel, toe, calf, P_FOOT, B["foot"], B["foot"])

    neck = pt("neck")
    if vis("r_ear", "l_ear"):
        ear_mid = _lerp(pt("r_ear"), pt("l_ear"), 0.5)
    else:
        ear_mid = vadd(vsub(pt("nose"), vmul(facing, 5.0)), vmul(up_t, 3.0))
    head_axis = vnorm(vsub(ear_mid, neck))
    if vlen(head_axis) < 1e-6:
        head_axis = up_t
    face = vnorm(vsub(pt("nose"), ear_mid)) if vis("nose") else facing
    if vlen(face) < 1e-6:
        face = facing
    face = vnorm(vsub(face, vmul(head_axis, vdot(face, head_axis))))
    if vlen(face) < 1e-6:
        face = facing

    h = B["head"]
    nk = B["neck_girth"]
    add("neck", vsub(neck, vmul(head_axis, 3.0)),
        vsub(ear_mid, vmul(head_axis, 8.0 * h)), (up_t, facing), P_NECK,
        nk, nk, round_start=False)
    # the skull has a forward of its own - the face - so it is its own
    # reference rather than inheriting the neck's
    add("head", vsub(ear_mid, vmul(head_axis, 10.6 * h)),
        vadd(ear_mid, vmul(head_axis, 10.4 * h)), (head_axis, face), P_HEAD,
        h * B.get("jaw", 1.0), h, round_start=False, round_end=False)

    frame = {"side": side, "up": up_t, "facing": facing, "down": down,
             "sh_mid": sh_mid, "hip_mid": hip_mid, "neck": neck,
             "ear_mid": ear_mid, "head_axis": head_axis, "face": face,
             "head_side": vnorm(vcross(head_axis, face)), "body": B}
    return segments, frame


def sweep(segment, coarsen=1.0):
    """One segment as solids."""
    return _tube(segment["a"], segment["b"], segment["ref"],
                 segment["profile"], segment["scale_w"], segment["scale_d"],
                 coarsen=coarsen,
                 round_start=segment.get("round_start", True),
                 round_end=segment.get("round_end", True))


def _spin(v, axis, angle):
    """Rodrigues: `v` turned about a unit `axis` by `angle` radians."""
    axis = vnorm(axis)
    c, s = math.cos(angle), math.sin(angle)
    return vadd(vadd(vmul(v, c), vmul(vcross(axis, v), s)),
                vmul(axis, vdot(axis, v) * (1.0 - c)))


def body_parts(skeleton, thickness=1.0, respect_visibility=True, coarsen=1.0):
    """Posed body as a list of parts; each part is a list of oriented
    ellipsoids (centre, orthonormal axes, radii) in world space.

    Anything the figure is wearing comes with it, so the depth export, the
    viewport preview and the silhouette all get clothes for free rather than
    each having to remember to ask.
    """
    P = skeleton.points
    ids = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
    segments, frame = body_segments(skeleton, respect_visibility)
    B = frame["body"]
    side, up_t, facing = frame["side"], frame["up"], frame["facing"]
    sh_mid, hip_mid = frame["sh_mid"], frame["hip_mid"]
    k = thickness * getattr(skeleton, "body_scale", 1.0)

    def pt(n):
        return P[ids[n]]

    def vis(*names):
        return True if not respect_visibility else all(
            skeleton.visible[ids[n]] for n in names)

    parts = []

    def emit(part):
        if part:
            # radii are in centimetres; k scales the whole figure
            parts.append([(kind, c, ax, (r[0] * k, r[1] * k, r[2] * k))
                          for kind, c, ax, r in part])

    if "torso" in segments:
        emit(sweep(segments["torso"], coarsen))

    # bust: two masses on the front of the ribcage
    bust = B.get("bust")
    if bust and vis("r_shoulder", "l_shoulder"):
        bw, bd, bh, bt, sep = bust
        centre = _lerp(sh_mid, hip_mid, bt)
        depth = B["chest"][1]
        for sgn in (-1.0, 1.0):
            c = vadd(vadd(centre, vmul(side, sgn * sep)),
                     vmul(facing, depth * 0.74))
            emit(_blob(c, (side, facing, up_t), (bw, bd, bh)))

    # buttocks
    glute = B.get("glute")
    if glute and vis("r_hip", "l_hip"):
        gw, gd, gh, gt, gsep = glute
        centre = _lerp(sh_mid, hip_mid, gt)
        for sgn in (-1.0, 1.0):
            c = vadd(vadd(centre, vmul(side, sgn * gsep)),
                     vmul(facing, -B["pelvis"][1] * 0.45))
            emit(_blob(c, (side, facing, up_t), (gw, gd, gh)))

    # deltoids
    dw, dd, dh = B["deltoid"]
    for shoulder in ("r_shoulder", "l_shoulder"):
        if vis("neck", shoulder):
            emit(_blob(vadd(pt(shoulder), vmul(up_t, -1.6)),
                       (side, facing, up_t), (dw, dd, dh)))

    for name, segment in segments.items():
        if name in ("torso", "head"):
            continue
        emit(sweep(segment, coarsen))

    if "head" in segments:
        head = sweep(segments["head"], coarsen)
        h = B["head"]
        head_side, face = frame["head_side"], frame["face"]
        head_axis = frame["head_axis"]
        if vis("nose"):
            head += _blob(vadd(pt("nose"), vmul(face, -1.4 * h)),
                          (head_side, face, head_axis),
                          (1.9 * h, 3.0 * h, 2.4 * h))
        for ear in ("r_ear", "l_ear"):
            if vis(ear):
                head += _blob(pt(ear), (head_side, face, head_axis),
                              (1.3 * h, 2.6 * h, 3.2 * h))
        emit(head)

    import wearables                  # deferred: it imports from this module
    for part in wearables.parts(skeleton, segments, frame, coarsen):
        emit(part)

    return parts
