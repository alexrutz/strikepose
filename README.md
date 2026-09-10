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

## Tests

    ./tests/run_all.sh

Read `CLAUDE.md` first if you are changing anything. It lists the invariants and
the bugs that produced them.

## Mobile prototype

`web/` is a touch-first version sharing the same maths in JavaScript. Serve the
folder over HTTPS and open it on a phone; it is static, with no build step.
