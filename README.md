# 3D OpenPose editor

Pose a rigged human body in 3D and export a depth map, and a matching OpenPose
pose map, for ControlNet. What you drag is the body's own 104-bone armature:
grab a joint and the bone above it swings, carrying everything below. Bones
cannot change length, because a rotation cannot change one.

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

## The depth map is always the Anny body

Every export renders real skinned geometry: the Anny body set, one `.glb` per
preset, posed on its own armature. It lives in `bodies/` and is found without
being told where - `$STRIKEPOSE_BODIES` first, then `./bodies`, then
`~/.strikepose/bodies`. Build it once:

    pip install anny
    python3 tools/make_bodies.py bodies/

If a body is missing the export **refuses** and says which one and how to
build it. It does not fall back to the built-in swept anatomy: that sweep is
there to draw the viewport and to cut garments out of, and an export of it is
a picture of a mannequin - the trap being that the file still appears and
nothing says so.

This program poses **one armature**, the 104-bone MakeHuman skeleton those
bodies are built on, and it names every bone rather than matching it. A file
that is not that skeleton is refused rather than half-matched into a mangled
limb. There is no SMPL-X backend, no roles file and no way to open some other
rig, and that is deliberate: carrying rigs nobody renders cost accuracy on the
one that ships. Aiming a bone is a minimal rotation and there is none onto
exactly the reverse of where it started, so a rig whose arms rest out to the
side has an unstable roll that Anny's A-pose simply does not have.

## The armature is the pose

The editor used to drag eighteen OpenPose keypoints and solve the rig onto
them. That was a translation between two descriptions of one pose, and it
could only lose whatever the poorer of the two could not say: the whole hand,
both collarbones, the roll of every limb, and 85 of the rig's 104 bones.

There is one description now. A pose is a local rotation per bone; a drag
composes a rotation; the depth map is skinned from the same array with nothing
in between. Every finger joint and every toe is a bone you can drag (press
**K** to show them), a forearm pronates, a collarbone shrugs. Sizing the rig
is a unit conversion, because `tools/make_bodies.py` bisects each body onto
its preset's stature, so nothing scales a bone either.

The eighteen keypoints are still there, as **output**: `keypoints_of` reads
them off the posed rig for the pose PNG, so the two images cannot disagree
about where the figure is. They travel back in exactly twice - opening a scene
file written before this change, and a drag from the phone, which has only
eighteen points to offer. Both are lossy imports and say so.

The built-in swept anatomy draws the viewport and cuts garments out; it is
never an export. Press **P** to see the armature that will be exported, **B**
for the sweep.

Put the body set through the whole depth pipeline, headless:

    python3 tests/check_glb.py bodies/*.glb

### Where the bodies come from

**Anny** (`naver/anny`), driven by `tools/make_bodies.py`:

    pip install anny
    python3 tools/make_bodies.py bodies/
    python3 everyday.py --render out/set --bodies bodies/

It is the MakeHuman base mesh wrapped in a parametric shape space calibrated
on WHO anthropometry, carrying MakeHuman's own bone names. Apache 2.0 code
over CC0 assets, so the bodies and anything made with them can be used
commercially without conditions. Three things it gives that building the same
bodies in Blender with MPFB2 does not: a pip install instead of Blender plus
an add-on, so the set is reproducible anywhere; an `age` axis that is
anthropometric rather than a slider - 0.0 is a 66 cm newborn, 0.3 a 130 cm
nine-year-old - with `height` bisected onto an exact stature, so a preset that
says 162 cm gets a body 162 cm tall, which is what makes sizing the rig a unit
conversion; and nine skinning influences per vertex against four, which
measures out as 8% of a limb's girth lost at a hard bend against 9-14%.

`tools/make_bodies.py` states the one convention that would otherwise be
silent: Anny's `gender` slider runs the opposite way to MPFB2's, and getting
it backwards produces a complete, plausible body set with every sex inverted.
It is pinned as a measurement - at 0.0 the figure has 54 cm shoulders on a
190 cm frame, at 1.0 44.5 cm on 176.

## Hair and clothes

The **Hair and clothes** panel dresses the active figure from five independent
slots - hair, headgear, top, bottom, feet - so a coat does not take the
trousers off. `python3 wearables.py --list` prints them all.

What a depth export actually wears is a real mesh: the CC0 MakeHuman community
garment packs, fitted to every body in `bodies/` and skinned to its armature,
so a coat rides the body's own pose solution instead of drifting off it. They
live in `garments/`; `python3 garments_lib.py --list` says which preset name
maps to which mesh, and a garment hides the body under it rather than hovering
a millimetre off a chest. Rebuild or widen the set with:

    python3 tools/make_wearables.py --fetch     # ~530 MB of packs, not vendored
    python3 tools/make_wearables.py --list
    python3 tools/make_wearables.py --build

The viewport draws something cheaper, because it has to redraw at sixty frames
a second: the body's own swept profile over the stretch the garment covers,
pushed outward by a few millimetres of cloth, so it fits every preset and every
pose for nothing. It is an approximation and it never reaches an export. A slot
the library has no mesh for is simply not worn on the rigged path, which is why
the vocabulary only names garments the library can deliver.

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

The depth map always comes from real geometry - hands with fingers, a face, a
chest that belongs to the body it is on. The body set is found without being
told where; `--bodies` only says *which* folder:

    python3 everyday.py --render out/set --bodies bodies/

One file per preset, named after it with spaces and commas turned into
underscores - `female_curvy.glb`, `child_about_7.glb`, an `mpfb_` prefix
accepted. A preset with no file is a refusal naming the command that builds
it, never a quiet fall back to the sweep: the PNG would still appear and
nothing would say it was a mannequin.

## The ground the figure stands on

Every depth map has a floor in it. Without one the figure is a cut-out against
black, which tells the generator nothing about where in a room the person is -
and a figure with nothing under its feet reads as floating.

The floor is the ground the figures are actually standing on: it sits at the
lowest point any of them reaches, so it fits a crouch, a child and a figure
lying down without being told. Turn it off with the **Ground under the figure**
box on the Export tab, or `--no-ground` on any of the command lines.

What it can show depends on the camera. A horizontal plane seen from a level
camera is exactly edge-on, so on `front`, `side` and the other pitch-zero
views all you get is a ground line at the feet - which is still worth having,
since it says where the floor is. A view with some pitch in it -
`high_three_quarter`, `bird`, `over_shoulder` - shows the floor receding, and
that is what carries the distance.

The floor is graded by a curve of its own: it starts a little under the
figure's brightness at the feet and fades to black within a couple of metres,
so it reads as the ground the person is on rather than a room filling the
frame. The figure is graded over its own depth range, exactly as it is with no
floor at all, so adding the ground never costs the subject any modelling.

## Hands and feet

Nothing in a pose says which way a palm faces or whether a toe points in, so
it has to be said. The **Hands and feet** panel on the Pose tab is the quick
way: pick a hand, a foot, or both of either, and two sliders do it.

Underneath they are ordinary bones. Press **K** and the thirty bones of each
hand, the fifteen of each foot and the face rig appear in the viewport and
drag like anything else, so a single finger is a drag away when the sliders
are not enough.

    bend   the hand curls towards its own palm
    turn   with the arm at rest, the palm rolls back
    lift   the toes come up towards the shin
    turn   the toes point outward, away from the other foot

A positive angle means the same thing on the left as on the right, and the
range of each slider is the joint's own, so a wrist cannot be bent somewhere a
wrist does not go. The prompt route can set them too:

    {"op": "hand", "side": "both", "bend": 40, "turn": -20}
    {"op": "foot", "side": "left", "lift": 15, "turn": 25}

They save with the scene, ride the undo stack, and show in the viewport as
well as the export.

## The export shape

The **Export** tab picks the shape first and the pixels second: 2:3 and 3:4
portrait, 9:16, square, 4:3, 3:2 and 16:9, or type any width and height you
like. The rectangle drawn in the viewport is that ratio and the scene is
re-framed to fit it, so what is inside the rectangle is what comes out of the
PNG - both PNGs, since the pose map and the depth map are always the same
frame. "Turn it on its side" swaps portrait for landscape.

Everything headless takes the same setting, as a size or as a ratio:

    python3 pose_agent.py --prompt "..." --size 16:9
    python3 everyday.py --render out/set --size 3:2
    python3 randomize.py --render out/random --size 1:1

## Random poses, for the edge cases

The pose catalogue is 84 poses somebody chose, which means it is 84 poses
somebody thought of. What breaks a depth map is the pose nobody thought of.
Press **X** in the editor, or:

    python3 randomize.py --list
    python3 randomize.py --render out/random --count 24 --parts arms,legs
    python3 randomize.py --render out/random --seed 7 --amount 1.0

Tick what should move - arms, legs, torso, head, which way the figure faces,
and the body, outfit and camera around it - and how far from rest, where 1.0
is a contortionist. Every joint moves by rotation about its parent, the same
call a mouse drag makes, so no bone changes length whatever the dice say and
anything odd that comes out is a real case rather than an artefact.

A seed names a pose. `--seed 7` is the same figure on this build and the next
one, so an awkward result is something you can report rather than screenshot.
The editor shows the seed it used and puts it in the box; typing one back
reproduces that figure exactly.

## Tests

    ./tests/run_all.sh

Read `CLAUDE.md` first if you are changing anything. It lists the invariants and
the bugs that produced them.

## On a phone

    python3 server.py                 # http://127.0.0.1:8765
    python3 server.py --host 0.0.0.0  # and from the phone on the same wifi
    python3 server.py --bodies bodies/    # depth from the rigged .glb bodies

`server.py` serves `web/` and the engine behind it. The phone keeps only what
has to be local - projecting and dragging at sixty frames a second, which
cannot survive a round trip - and asks Python for everything that decides what
comes *out*: the 84-pose library with its search, the prompt, the body
presets, the objects and clothing, and the export. Tap a pose and the phone
takes the server's keypoints *and its camera*, so what is on screen is the
view the PNG will come out in. "Render pair" returns the real OpenPose and
depth images, full screen, to press-and-hold and save.

It installs to a home screen: there is a manifest and an icon, and it runs
without browser chrome. With nothing listening it still poses a figure - the
library, the prompt and the real export go grey and the local preview stays.

The binding is loopback unless you ask otherwise. `--host 0.0.0.0` is what
puts it on the phone, and it puts it on everything else on that network too:
there is no authentication and a request costs CPU. Use it on a network you
trust.

## Mobile prototype

`web/` is a touch-first version sharing the same maths in JavaScript. Serve the
folder over HTTPS and open it on a phone; it is static, with no build step.
