# 3D OpenPose editor

Author OpenPose skeletons in 3D and export a pose map and a matching depth map
for ControlNet. Limbs keep a fixed length: dragging a joint moves it on the
sphere of reach around its parent, so the drop point sets direction, never size.

    python3 openpose3d_editor.py

## Requirements

tkinter is required (`sudo apt install python3-tk`, or reinstall Python on
Windows with the tcl/tk option). Pillow enables PNG export and NumPy enables
depth maps; both are worth installing. See `requirements.txt`.

## Where the proportions come from

Bone lengths and cross-sections are derived from **ANSUR II**, the 2012 US Army
anthropometric survey - 4082 men and 1986 women, 93 measurements each, public
since 2017. The means are computed from the released CSVs rather than quoted,
and `python3 tests/test_proportions.py` checks the figure that comes out
against them: shoulder at acromial height, hip at the trochanter, knee at the
femoral epicondyle, arm span closing on the measured span.

Two measurements the survey cannot settle are marked as such in the code: hip
width, which it takes at the iliac crests rather than the femoral heads, and
the waist, which it takes at the navel while the body profile's waist station
is the tenth rib. There is no child in ANSUR, so that preset is inherited.

## Depth sources

Three, tried in order, each falling back to the next:

1. **Rigged mesh** — a `.glb` exported from Blender with its armature. Build
   bodies with MPFB2, one file per body type in a folder, named after the
   preset (`male_average.glb`). Hair and clothing go in an assets folder and
   are worn per figure.
2. **SMPL-X** — needs `pip install smplx torch` and the model files from
   smpl-x.is.tue.mpg.de.
3. **Built-in anatomy** — always available, no dependencies beyond NumPy.
   Measured cross-sections swept along the bones.

Inspect a rig before loading it:

    python3 mesh_backend.py --inspect body.glb

and put it through the whole depth pipeline, headless, before trusting it:

    python3 tests/check_glb.py body.glb
    python3 tests/check_glb.py body.glb --roles roles.json --assets hair.glb

That loads the file, maps its bones, poses it through six poses, skins it,
renders each depth map and writes a contact sheet, checking that every mapped
joint lands on its keypoint, that the rig is scaled to the figure, that no limb
is pinched by the skinning, and that a limb swung right round does not jump.
If the bone names are not recognised, `--inspect` lists them and a roles file
maps them by hand:

    {"hips": "pelvis", "l_shoulder": "upperarm01.L", ...}

### Where to get a body

**Anny** (`naver/anny`) is the quickest, and `tools/make_bodies.py` drives it:

    pip install anny
    python3 tools/make_bodies.py bodies/
    python3 everyday.py --render out/set --bodies bodies/

It is the MakeHuman base mesh wrapped in a parametric shape space calibrated
on WHO anthropometry, with MakeHuman's own bone names, so nothing here needs a
roles file. Apache 2.0 code over CC0 assets. Three things it gives that
building the same bodies in Blender does not: a pip install instead of Blender
plus an add-on, so the set is reproducible anywhere; an `age` axis that is
anthropometric rather than a slider - 0.0 is a 66 cm newborn, 0.3 a 130 cm
nine-year-old - with `height` bisected onto an exact stature, so a preset that
says 162 cm gets a body 162 cm tall; and nine skinning influences per vertex
against four, which measures out as 8-9% of a limb's girth lost at a hard bend
against 9-14%.

What it does not give is a better-looking body: it is the same mesh, so the
depth maps are the same depth maps. What improves is where the numbers come
from and whether anyone can rebuild them.

**MPFB2** (MakeHuman Plugin for Blender) is the one this is built around: the
add-on is GPLv3, its bundled assets are CC0, and what you make with it is CC0,
so it can be used commercially without conditions. The bone names its default
rig uses - `upperarm01.L`, `lowerleg01.R` - are what the aliases here match
first, so an export needs no roles file. Blender 4.2+.

**Mixamo** characters are royalty-free for commercial and non-commercial use
with no attribution, and its bone names (`LeftArm`, `RightForeArm`) are also
matched. It has had no maintenance since 2015 and its authenticated features
have been unreliable since mid-2025, so treat it as a source that may not be
there tomorrow.

**SMPL-X** is non-commercial research only; commercial use needs a sub-licence
from Meshcapade. That is why it is a separate optional backend here rather than
the default.

Whatever the source, delete MakeHuman's helper geometry before exporting, or
the skirt and tights read as clothing in the depth map - the loader drops small
loose shells, but deleting them in MPFB2 is cleaner.

Shape keys can be left in the export. MPFB2 keeps the whole body - age, weight,
muscle, proportions - in shape keys, which glTF stores as morph targets with
default weights, and those are read and applied here. An exporter set to strip
them writes the unshaped base mesh instead, so every body in a set comes out
the same 167 cm mannequin while its skeleton still carries the real shape.

## Hair and clothes

The **Hair and clothes** panel dresses the active figure from five independent
slots - hair, headgear, top, bottom, feet - so a coat does not take the
trousers off. `python3 wearables.py --list` prints them all.

A garment is not a mesh. It is the body's own swept profile, taken over the
stretch it covers and pushed outward by a few millimetres of cloth, so it fits
every preset and every pose for nothing: it *is* the arm, slightly larger. The
cost is that cloth cannot hang - a skirt flares because it is told to, not
because it falls - which for a depth map is the right trade, since what
conditions the generator is the silhouette.

Outfits save with the scene and ride the undo stack, and the prompt route reads
them: "a woman with long hair in a long skirt and boots" dresses her.

## Objects

A depth map conditions everything in frame, not only the person, and a figure
sitting on nothing reads as a figure floating. The **Objects** panel places
chairs, tables, crates, walls, pillars and the rest - `python3 props.py --list`
prints them all with their real sizes. Drag one in the viewport to move it,
and use the panel to resize, turn or drop it back to the floor. Objects appear
in the depth map only; the pose map stays the OpenPose skeleton and nothing
else.

They are built from the same oriented boxes, cylinders and ellipsoids the body
is, so they go through the same analytic rasteriser and occlude the figure, and
each other, correctly. Scenes save and load with their objects.

## Posing from a text prompt with a local LLM

`pose_agent.py` turns a sentence into a pose and exports both conditioning
images. The editor's own entry point forwards to it, so either works:

    python3 openpose3d_editor.py --prompt "a boxer in a guard, three quarters"
    python3 pose_agent.py --prompt "someone kneeling, looking up" --out out/

writing `pose.png`, `depth.png` and a `scene.json` the editor can load. There is
a **Prompt** box at the top of the editor's panel that does the same to the open
scene; Ctrl+Z puts the old one back.

The model is found automatically on the usual ports - Ollama on 11434, LM Studio
on 1234, llama.cpp's server on 8080, vLLM on 8000 - or name one:

    python3 pose_agent.py --prompt "..." --host http://localhost:11434 \
                          --model qwen2.5:7b-instruct

`--backend ollama` uses Ollama's `/api/chat`, `--backend openai` the
OpenAI-compatible `/v1/chat/completions` everything else serves; `auto` tries
both. Generation is constrained to a JSON schema, which is what makes a 7B model
usable for this. `POSE_AGENT_HOST` and `POSE_AGENT_MODEL` set the defaults. No
extra Python package is needed: it is urllib and the standard library.

The model does not return coordinates - it returns a short list of commands
(`point`, `bend`, `turn`, `lean`, `look`, `stance`, `hide`), applied through the
same rotations a mouse drag uses, so no bone can change length whatever it asks
for and an invented command is reported and skipped. `--list` prints the whole
vocabulary. The model places objects with the same kind of command -
`{"op":"place","shape":"chair","at":"under_hips"}` - anchored against the
figure rather than in absolute coordinates, so "a chair under the hips" comes
out as a chair whose seat meets the hips and whose legs still reach the floor,
whatever the figure's size or pose. `scene.json` carries the plan, so a run can
be hand-edited and
replayed again. With no model running the prompt is still read, by keyword,
and the
run says so - use `--require-llm` if you would rather it failed.

Name a view in the prompt and you get it. Say nothing and the camera turns
until the pose reads: a crouch seen head-on is a figure standing up straight,
because the part that makes it a crouch is the part pointing at the lens.

## A catalogue of everyday poses

`everyday.py` names 84 specific poses - typing at a desk, tying a shoelace,
carrying a box, hailing a taxi, curled up asleep - in ten groups, and renders
any slice of them across all nine body types.

    python3 everyday.py --list
    python3 everyday.py --render out/set                 # 84 poses x 9 bodies
    python3 everyday.py --render out/set --group "At a desk"
    python3 everyday.py --render out/set --preset female --size 768x1024

Each pose is written in exactly the command vocabulary the local model emits,
and goes through the same `apply_commands`, so nothing in the table can write a
coordinate and every bone-length invariant holds for free. That also means the
model can name one: `{"op":"stance","name":"tying_shoelaces"}` is a valid
command, and further commands refine it.

The render writes a matched pair per pose per body - the OpenPose PNG and the
depth map of the same figure in the same frame - plus `index.json` naming every
one, and a contact sheet per group. A figure that sits gets a chair under it, a
figure at a desk gets a desk, and both are sized against that body, so the
child's chair is a child's chair.

Point `--bodies` at a folder of rigged `.glb` bodies and the depth map comes
from real geometry instead of the swept anatomy - hands with fingers, a face,
a chest that belongs to the body it is on:

    python3 everyday.py --render out/set --bodies bodies/

One file per preset, named after it with spaces and commas turned into
underscores - `female_curvy.glb`, `child_about_7.glb`, an `mpfb_` prefix
accepted - and any preset without a file falls back to the swept anatomy. The
pose PNG is identical either way, because it is the same eighteen keypoints:
the depth side of a conditioning pair can be upgraded without the OpenPose
side moving a pixel.

## Tests

    ./tests/run_all.sh

Read `CLAUDE.md` first if you are changing anything. It lists the invariants and
the bugs that produced them.

## Mobile prototype

`web/` is a touch-first version sharing the same maths in JavaScript. Serve the
folder over HTTPS and open it on a phone; it is static, with no build step.
