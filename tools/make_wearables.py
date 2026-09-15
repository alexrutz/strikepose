#!/usr/bin/env python3
"""Fit the CC0 MakeHuman garment library onto this project's body set.

    python3 tools/make_wearables.py --fetch          # the CC0 packs, ~530 MB
    python3 tools/make_wearables.py --build          # fit them to bodies/
    python3 tools/make_wearables.py --list

Why this exists
---------------
`wearables.py` does not carry meshes: it clips and pads the body's own swept
profile. That is the right thing for the viewport and it is an approximation,
and an approximation is not what a conditioning image wants. The MakeHuman
community publishes a real garment library under CC0 - shirts, trousers,
suits, shoes, hats, twenty-five hairstyles - authored against the same
MakeHuman assets the Anny bodies are built from. These are premade meshes; the
job here is only to get them onto these bodies.

Why it is not just an index lookup
----------------------------------
A `.mhclo` describes a garment as, per vertex, three vertices of the MakeHuman
base mesh, barycentric weights and an offset. Against `base.obj` that is exact
- the garment lands on the head to the millimetre. Against an Anny body it is
nonsense: Anny has 13718 vertices to hm08's 13380 and shares 1967 triangles of
26756, so the orderings have nothing to do with each other. Fitting by index
anyway puts most of a hairstyle roughly on the head and throws the rest across
the room, which looks like a fixable glitch and is not one.

So the garment is fitted to `base.obj`, where the indices mean what they say,
and then carried onto the Anny body geometrically: each garment vertex is
described by which base-body vertices it sits over and how far out along their
normals, and rebuilt from the same description over the Anny surface. What
makes that sound is that the two are the same human shape in the same rest
pose at the same stature - not that they share any numbering.

What comes out
--------------
One `.npz` per garment holding the faces, the skinning, and the fitted
vertices for every body preset. The rig is shared, so a garment needs no
solution of its own: `mesh_backend.skin_with` drives it from the body's, which
is the invariant that keeps an asset from drifting off the figure it is on.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# The base mesh the whole library is authored against. CC0, and not vendored:
# it is a build input, like the packs.
BASE_URL = ("https://raw.githubusercontent.com/makehumancommunity/makehuman"
            "/master/makehuman/data/3dobjs/base.obj")
BODY_VERTS = 13380          # hm08's body; the rest of base.obj is helpers

PACKS = {
    "makehuman_system_assets": "clothes",   # suits, shoes, hats
    "hair01": "hair",
}
PACK_URL = ("https://files2.makehumancommunity.org/asset_packs/%s/%s_cc0.zip")


# ---------------------------------------------------------------------------
# reading what MakeHuman ships
# ---------------------------------------------------------------------------

def read_obj(path):
    import numpy as np
    vs, fs = [], []
    for line in open(path, errors="replace"):
        w = line.split()
        if not w:
            continue
        if w[0] == "v":
            vs.append([float(x) for x in w[1:4]])
        elif w[0] == "f":
            idx = [int(p.split("/")[0]) - 1 for p in w[1:]]
            for i in range(1, len(idx) - 1):
                fs.append([idx[0], idx[i], idx[i + 1]])
    return np.array(vs, float), np.array(fs, "int64")


def read_mhclo(path):
    """(header, per-vertex table, scale references) from a MakeHuman proxy.

    A row is three base-mesh vertices, three barycentric weights and an offset
    in base-mesh units. The three `?_scale` lines name a pair of base vertices
    and the distance between them on the mesh the garment was authored on, so
    the offset can be rescaled to whatever body it is being put on.
    """
    import numpy as np
    head, verts, scales, state = {}, [], {}, None
    hide = []
    for line in open(path, errors="replace"):
        w = line.split()
        if not w:
            continue
        if w[0] in ("x_scale", "y_scale", "z_scale"):
            scales[w[0][0]] = (int(w[1]), int(w[2]), float(w[3]))
            continue
        if w[0] == "obj_file":
            head["obj"] = " ".join(w[1:])
            continue
        if w[0] == "name":
            head["name"] = " ".join(w[1:])
            continue
        if w[0] == "verts":
            state = "verts"
            continue
        if w[0] == "delete_verts":
            # which base-body vertices the garment stands in for. MakeHuman
            # hides them; so do we, or the body pokes through anything tight -
            # a shirt on a chest, a bob against a skull - and in a depth map
            # the skin simply wins the z-test and the garment is not there.
            state = "hide"
            continue
        if state == "hide":
            try:
                run = [int(x) if x != "-" else "-" for x in w]
            except ValueError:
                state = None
                continue
            i = 0
            while i < len(run):
                if i + 2 < len(run) and run[i + 1] == "-":
                    hide.extend(range(run[i], run[i + 2] + 1))
                    i += 3
                else:
                    if run[i] != "-":
                        hide.append(run[i])
                    i += 1
            continue
        if state == "verts":
            # a `delete_verts` block follows, and other keywords may too, so
            # a row is only a row while it starts with a number
            try:
                first = int(w[0])
            except ValueError:
                state = None
                continue
            if len(w) >= 9:
                verts.append([float(x) for x in w[:9]])
            elif len(w) == 1:                 # rigidly on one vertex
                verts.append([float(first)] * 3 + [1., 0., 0., 0., 0., 0.])
            else:
                state = None
    head["hide"] = np.array(sorted(set(hide)), "int64")
    return head, np.array(verts, float), scales


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def vertex_normals(verts, faces):
    import numpy as np
    n = np.zeros_like(verts)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face = np.cross(b - a, c - a)
    for i in range(3):
        np.add.at(n, faces[:, i], face)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(length < 1e-12, 1.0, length)


def nearest(query, cloud, k=6):
    """The k nearest cloud points to each query point, with inverse-square
    weights. A grid bucket rather than a kd-tree: this project depends on
    numpy and Pillow and nothing else, and a body is 13k points in a 2 m box."""
    import numpy as np
    cell = 0.04 * float(cloud[:, 1].max() - cloud[:, 1].min())
    origin = cloud.min(0) - cell
    key = lambda p: np.floor((p - origin) / cell).astype("int64")
    grid = {}
    for i, c in enumerate(map(tuple, key(cloud))):
        grid.setdefault(c, []).append(i)
    idx = np.zeros((len(query), k), "int64")
    dist = np.zeros((len(query), k))
    for n, (p, c) in enumerate(zip(query, key(query))):
        pool, ring = [], 1
        while len(pool) < k * 3 and ring < 12:
            pool = [i for dx in range(-ring, ring + 1)
                    for dy in range(-ring, ring + 1)
                    for dz in range(-ring, ring + 1)
                    for i in grid.get((c[0] + dx, c[1] + dy, c[2] + dz), ())]
            ring += 1
        pool = np.array(pool) if pool else np.arange(len(cloud))
        d = np.linalg.norm(cloud[pool] - p, axis=1)
        take = pool[np.argsort(d)[:k]]
        idx[n] = take
        dist[n] = np.linalg.norm(cloud[take] - p, axis=1)
    w = 1.0 / np.maximum(dist, 1e-6) ** 2
    return idx, w / w.sum(axis=1, keepdims=True)


def on_base(folder, stem, base_all):
    """The garment sitting on the MakeHuman base mesh. Exact - this is the one
    place the .mhclo indices mean what they say."""
    import numpy as np
    head, table, scales = read_mhclo(os.path.join(folder, stem + ".mhclo"))
    if table.ndim != 2 or len(table) == 0 or len(scales) != 3:
        raise ValueError("%s: no usable fitting table" % stem)
    gv, gf = read_obj(os.path.join(folder, head["obj"]))
    axis = {"x": 0, "y": 1, "z": 2}
    k = np.array([abs(float(base_all[i][axis[a]] - base_all[j][axis[a]])) / ref
                  for a, (i, j, ref) in sorted(scales.items())])
    tri = np.clip(table[:, :3].astype(int), 0, len(base_all) - 1)
    w = table[:, 3:6]
    pos = (base_all[tri[:, 0]] * w[:, 0:1] + base_all[tri[:, 1]] * w[:, 1:2]
           + base_all[tri[:, 2]] * w[:, 2:3] + table[:, 6:9] * k)
    return pos, gf, head["hide"]


def body_map(base_body, base_norm, anny, anny_norm):
    """Where the Anny body is, for each vertex of the base body."""
    import numpy as np
    twin, tw = nearest(base_body, anny, 4)
    here = (anny[twin] * tw[..., None]).sum(axis=1)
    facing = (anny_norm[twin] * tw[..., None]).sum(axis=1)
    length = np.linalg.norm(facing, axis=1, keepdims=True)
    return here, facing / np.where(length < 1e-12, 1.0, length)


def carry(garment, base_body, base_norm, here, facing, k=6):
    """The garment, described over the base body and rebuilt over the other."""
    import numpy as np
    idx, w = nearest(garment, base_body, k)
    lift = ((garment[:, None, :] - base_body[idx]) * base_norm[idx]).sum(-1)
    return ((here[idx] + facing[idx] * lift[..., None]) * w[..., None]).sum(1)


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------

def fetch(into):
    import urllib.request
    os.makedirs(into, exist_ok=True)
    base = os.path.join(into, "base.obj")
    if not os.path.exists(base):
        print("base mesh ...", flush=True)
        urllib.request.urlretrieve(BASE_URL, base)
    for pack in PACKS:
        zip_path = os.path.join(into, pack + ".zip")
        if os.path.exists(zip_path):
            continue
        print("%s ..." % pack, flush=True)
        urllib.request.urlretrieve(PACK_URL % (pack, pack), zip_path)
    for pack, folder in PACKS.items():
        out = os.path.join(into, pack)
        if os.path.isdir(out):
            continue
        with zipfile.ZipFile(os.path.join(into, pack + ".zip")) as z:
            for member in z.namelist():
                if member.startswith(folder + "/") and member.endswith(
                        (".mhclo", ".obj")):
                    z.extract(member, out)
    return into


def garments(assets):
    """Every (folder, stem, kind) the packs offer."""
    found = []
    for pack, kind in PACKS.items():
        root = os.path.join(assets, pack, kind)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            folder = os.path.join(root, name)
            if not os.path.isdir(folder):
                continue
            for f in sorted(os.listdir(folder)):
                if f.endswith(".mhclo"):
                    found.append((folder, f[:-6], kind))
                    break
    return found


def build(assets, out_dir, only=None):
    import numpy as np
    import bodies_lib
    import mesh_backend
    from openpose3d_editor import BODY_PRESETS

    base_all, base_faces = read_obj(os.path.join(assets, "base.obj"))
    base_body = base_all[:BODY_VERTS]
    base_norm = vertex_normals(base_all, base_faces)[:BODY_VERTS]

    folders = bodies_lib.folders()
    bank = {}
    for preset in BODY_PRESETS:
        path = os.path.join(folders[0], bodies_lib.slug(preset) + ".glb")
        if not os.path.exists(path):
            continue
        # the whole mesh, helpers and all: the map is geometric, so dropping
        # loose shells would only take points out of it
        bank[preset] = mesh_backend.load_rigged_mesh(path, drop_loose=False)
    if not bank:
        raise SystemExit("No bodies found. " + bodies_lib.HOW)

    maps = {}
    for preset, mesh in bank.items():
        anny = mesh["vertices"]
        lo_b, hi_b = base_body.min(0), base_body.max(0)
        lo_a, hi_a = anny.min(0), anny.max(0)
        scale = (hi_a[1] - lo_a[1]) / (hi_b[1] - lo_b[1])
        shift = 0.5 * (lo_a + hi_a) - 0.5 * (lo_b + hi_b) * scale
        here, facing = body_map(base_body * scale + shift, base_norm, anny,
                                vertex_normals(anny, mesh["faces"]))
        maps[preset] = (scale, shift, here, facing)
        print("  mapped %s" % preset, flush=True)

    os.makedirs(out_dir, exist_ok=True)
    made = []
    for folder, stem, kind in garments(assets):
        if only and not any(o in stem for o in only):
            continue
        blob = {}
        faces = skin_j = skin_w = None
        try:
            for preset, (scale, shift, here, facing) in maps.items():
                pos, gf, hide = on_base(folder, stem, base_all * scale + shift)
                fitted = carry(pos, base_body * scale + shift, base_norm,
                               here, facing)
                blob[bodies_lib.slug(preset)] = fitted.astype("float32")
                if faces is None:
                    faces = gf.astype("int32")
                    mesh = bank[preset]
                    near, nw = nearest(fitted, mesh["vertices"], 4)
                    skin_j = mesh["skin_joints"][near[:, 0]].astype("int32")
                    skin_w = (mesh["skin_weights"][near]
                              * nw[..., None]).sum(axis=1).astype("float32")
                # the hidden base vertices, carried onto this body the same way
                hidden = hide[hide < BODY_VERTS]
                if len(hidden):
                    twin, _tw = nearest(here[hidden], bank[preset]["vertices"], 1)
                    mask = np.zeros(len(bank[preset]["vertices"]), bool)
                    mask[np.unique(twin[:, 0])] = True
                    # Erode by a ring before using it. The deletion list is in the
                    # base mesh's numbering and arrives here through a nearest-
                    # neighbour map, so its boundary lands a vertex or two off -
                    # and a body face dropped just past the garment's hem is a
                    # black slit in the depth map with nothing over it. Eroded,
                    # what is left is the deep interior, which the garment covers
                    # with room to spare.
                    keep = mask.copy()
                    faces_b = bank[preset]["faces"]
                    for tri in faces_b:
                        if not mask[tri].all():
                            keep[tri] = False
                    blob["hide_" + bodies_lib.slug(preset)] = np.where(
                        keep)[0].astype("int32")
        except (ValueError, KeyError, IndexError) as problem:
            # a handful of community assets have no usable fitting table; one
            # bad file is not worth losing the run over
            print("  %-38s skipped: %s" % (stem, problem), flush=True)
            continue
        np.savez_compressed(os.path.join(out_dir, stem + ".npz"),
                            faces=faces, skin_joints=skin_j,
                            skin_weights=skin_w, kind=kind, **blob)
        made.append(stem)
        print("  %-38s %5d verts, %d bodies" % (stem, len(faces), len(blob)),
              flush=True)
    print("\n%d garments in %s" % (len(made), out_dir))
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", default=os.path.join(ROOT, "build", "mhassets"))
    ap.add_argument("--out", default=os.path.join(ROOT, "garments"))
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.assets)
    if args.list:
        for folder, stem, kind in garments(args.assets):
            print("  %-10s %s" % (kind, stem))
    if args.build:
        build(args.assets, args.out, args.only)
    if not (args.fetch or args.build or args.list):
        print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
