# 3D OpenPose editor

Author OpenPose skeletons in 3D and export a pose map and a matching depth map
for ControlNet. Limbs keep a fixed length: dragging a joint moves it on the
sphere of reach around its parent, so the drop point sets direction, never size.

    python3 openpose3d_editor.py

## Requirements

tkinter is required (`sudo apt install python3-tk`, or reinstall Python on
Windows with the tcl/tk option). Pillow enables PNG export and NumPy enables
depth maps; both are worth installing. See `requirements.txt`.

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

## Tests

    ./tests/run_all.sh

Read `CLAUDE.md` first if you are changing anything. It lists the invariants and
the bugs that produced them.

## Mobile prototype

`web/` is a touch-first version sharing the same maths in JavaScript. Serve the
folder over HTTPS and open it on a phone; it is static, with no build step.
