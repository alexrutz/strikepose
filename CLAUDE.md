# CLAUDE.md

Project context for Claude Code. Read this before changing anything.

## What this is

A desktop editor for authoring OpenPose skeletons in 3D and exporting matched
pose and depth images for ControlNet. `openpose3d_editor.py` is the tkinter
application; `mesh_backend.py` and `smplx_backend.py` are optional depth
sources; `web/` is a mobile prototype sharing the same maths in JavaScript.

## Run the tests before and after every change

```
./tests/run_all.sh                  # everything, needs xvfb on a headless box
python3 tests/test_core.py          # pure maths, no display needed
python3 mesh_backend.py --selftest  # rigged mesh maths, no model files needed
python3 smplx_backend.py --selftest # SMPL-X maths, no model files needed
```

These are not decoration. Every check in them was written because something
actually broke. Several bugs below were invisible for multiple releases because
the test that would have caught them made a wrong assumption.

## Invariants that must never regress

**A bone may not change length unless explicitly asked.** `Skeleton.move_joint`
takes `stretch=False` by default. A target at the wrong distance aims the bone;
it never resizes it. Only the deliberate free-length drag passes `stretch=True`.
Two separate bugs stretched limbs before this existed: symmetric editing fed a
reflected *position* into `move_joint`, and a modifier misread made every drag
look like a free-length drag. There is also a runtime guard, `check_lengths`,
that reverts a drag which resized a bone without being asked.

**Never read modifier keys from `event.state`.** The bit values differ between
Windows, X11 and macOS. Bind `<Alt-ButtonPress-1>` and friends and let Tk map
them. Reading bits directly is what stretched every limb on Windows while all
the Linux tests passed.

**Symmetric editing mirrors a direction, not a position.** A reflected position
only sits at the right distance from the twin's parent while the figure is
perfectly symmetric.

**Exports use the canonical OpenPose palette only.** Viewport tinting for
telling figures apart must never reach `render_openpose`. Tests assert that no
tinted colour appears in an exported PNG.

**Rotating a bone must carry everything below it.** `mesh_backend.pose_globals`
stores *local* rotations composed down the tree. Storing globals and flagging
which were set by hand loses the composition, which is invisible on a rig whose
aimed joints are direct parent and child, and wrong on any rig with twist bones
— which MakeHuman's default rig has throughout. The mock rig in the self-test
has twist bones specifically to catch this.

**Twist is not observable from 18 keypoints.** Bone roll is tied to the body's
rest orientation instead. Rolling a bone about its own axis must not move the
joint it points at.

**Roll is carried along a limb, never re-derived per bone.** Both depth paths
learned this the hard way. `openpose3d_editor.carry_frame` swings a bone's
cross-section frame onto its axis by the *minimal* rotation from the frame
above it; `mesh_backend`'s roll reset does the same against the chain-carried
rest reference. Projecting a fixed reference onto each bone instead - the
body's `facing` for the anatomy, the pelvis-rotated rest frame for the rig -
is discontinuous: it is undefined the moment a bone points along the
reference, flips a full 180 degrees either side of it, and is 45 degrees out
in plain diagonal poses. That is what kept coming out of the depth map as
crooked and twisted limbs. Aiming a bone at a keypoint already composes
minimal rotations down the tree, which *is* parallel transport, so the rig's
roll reset should confirm what aiming worked out, never overrule it.

**A frame test must compare a moved bone with a moved reference.** The check
that was meant to guard the above compared the posed bone against the rest
reference left where it was, which measures the swing, not the twist - so it
read 176 degrees of jump between neighbouring poses as success. Both suites
now sweep a limb right round and assert the frame never steps further than the
pose did.

**Assets ride the body's pose solution.** `solve_pose` returns the result keyed
by bone name; `skin_with` applies it to any mesh on the same armature. Never
solve an asset separately or it will drift from the body.

## Platform and format gotchas

- A tk `Canvas` defaults to 378 px wide. Five in a row overflow their strip and
  the last ones collapse to 1 px. Ask for `width=10` and let `expand` share.
- `pack_forget` drops a widget from the packing order; re-packing appends it,
  and an expanding sibling has already claimed the space. Use `before=`.
- glTF splits vertices at UV seams, so index connectivity reports one body as
  dozens of shells. Weld by position before any connected-component analysis.
- A model's joint array is longer than its parent array. SMPL-X returns 127
  joints and 55 parents; the rest are landmarks posed by skinning. Iterate over
  the parents.
- MakeHuman's base mesh carries helper geometry (skirt, tights, hair helper,
  joint cubes) as loose shells. It reads as clothing in a depth map. Delete
  helpers in MPFB2, or let the loader drop small shells.
- Anthropometric segment measurements do not chain. Adding published
  acromion-radiale, radiale-stylion and hand lengths gives a 194 cm arm span on
  a 175 cm figure. Close the arm on the span invariant instead.

## Conventions

World space is centimetres, +X right, +Y up, +Z towards the default camera; the
figure faces +Z so its right side is at -X. The camera is orthographic, which
is what makes the reach sphere project to a circle and the guide arcs solve
exactly. Keep it that way unless you are prepared to replace the drag maths
with ray-sphere intersection.

Bump `VERSION` in `openpose3d_editor.py` for anything user-visible. It shows in
the title bar, the HUD and on stdout, and it is how a bug report gets pinned to
a build.

## Known weak spots, in rough priority order

1. `openpose3d_editor.py` is one ~3000-line file grown by successive edits. It
   wants splitting into modules — skeleton, anatomy, rendering, UI — with the
   test suite as the guard. Do this first, in small steps, running the tests
   between each.
2. The tests are scripts printing PASS/FAIL, not pytest. Converting them keeps
   the assertions but gains proper reporting and selection.
3. Depth export is CPU rasterisation. Live depth would want the geometry on the
   GPU rather than the analytic rasteriser ported.
4. Wrists and ankles are never rotated; there are no keypoints for them.
5. The web prototype duplicates the maths in JavaScript. It is tested
   independently (`node web/test.mjs`) but will drift from the Python.
