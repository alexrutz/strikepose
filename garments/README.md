# Garments

Real clothing and hair meshes, fitted to every body in `bodies/`.

These are the [MakeHuman community asset
packs](https://static.makehumancommunity.org/assets/assetpacks.html), all
**CC0** — public domain, no attribution required, no restrictions on use. They
are authored against MakeHuman's `hm08` base mesh, which is also CC0.

`tools/make_wearables.py` fetches the packs and the base mesh, fits each
garment to the base by its `.mhclo` — the format's own mechanism, exact by
construction — and then carries it onto each Anny body geometrically, because
Anny does not share hm08's vertex ordering (1967 triangles in common out of
26756) and fitting by index puts most of a hairstyle on the head and throws
the rest across the room.

Each `.npz` holds the garment's faces, its skinning against the body's own
armature, the fitted vertices for every body preset, and the body vertices it
stands in for — MakeHuman's `delete_verts`, which is what stops a chest coming
through a shirt.

To rebuild, or to add more of the library:

    python3 tools/make_wearables.py --fetch      # ~530 MB of packs, not vendored
    python3 tools/make_wearables.py --list
    python3 tools/make_wearables.py --build --only <stems>

`garments_lib.CATALOGUE` maps `wearables`' slot names onto these stems. A slot
with nothing behind it is simply not worn on a rigged export; nothing here is
mandatory and the vocabulary a model emits does not depend on it.
