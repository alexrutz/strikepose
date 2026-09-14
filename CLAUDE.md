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

**`look` aims the gaze, not the neck-to-nose line.** The nose sits high on
the head, so that line stands 74 degrees above horizontal at rest. Aiming *it*
at "forward" swings the head 74 degrees - chin on the chest - while the gaze
it was meant to set has barely moved; "forward_down" came out at 119 degrees
and "down" at 164. That is what made half the pose catalogue look hunched. The
gaze is the ear midpoint to the nose, which is level at rest, and the whole
head turns about the neck so no bone changes length.

**The head is turned by how far it moved, never aimed at a keypoint.** There
is no keypoint on the skull. The nearest is the ear midpoint, and a neck bone
does not point at the ears: MakeHuman's runs from C7 forward and up to the
base of the skull, 20 degrees off vertical, where the editor's shoulder-to-ear
line leans 3. Aiming the bone at the ears therefore tipped every skull on
every MPFB2 body 17 degrees back, in the rest pose as much as in any other.
`solve_pose` takes the body's own `rest_points` and turns the neck by the
change in the head's orientation *relative to the torso* - measured in the
body frame on both sides so a figure that has also turned does not count the
torso twice, and composed `now @ rest.T`, which is four degrees away from
`rest.T @ now` on a head turned forty. At rest the change is zero and the rig
keeps the neck it was authored with, which is the point of having a rig.

**A foot under a near-upright shin is flat.** There is no keypoint past the
ankle either, so a foot rides its shin rigidly and points wherever the shin
does - a lunge on pointed toes, a seated figure dangling. Where the shin is
within 40 degrees of vertical the foot is taking weight, so it goes back to
the pitch the rig authored while keeping the heading the leg gave it. Beyond
that the leg is not standing on anything: a kneeling figure's shins point
backwards and a lying one's sideways, and both keep the rig's own
relationship. Gate it on the *keypoints*, not on `q`: the rig has not been
slid onto them yet, so its own bone lengths put the knee elsewhere and its
shin reads 11 degrees steeper than the pose asked for.

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

**The pose catalogue is data, the command vocabulary is not.**
`everyday.py` names 84 specific everyday poses - typing at a desk, tying a
shoelace, carrying a box - and every one of them is a list of the same
commands a model emits, applied through the same `apply_commands`. That is
what makes a conditioning set repeatable: a 7B model asked for "tying a
shoelace" a hundred times gives a hundred slightly different answers and
several wrong ones. What stays short is the vocabulary the enums carry - ops,
directions, joints, limbs - because a model picks from those on *every*
command; a stance name is picked at most once and saves it ten guesses, so the
catalogue can be long. The system prompt lists them grouped rather than as one
line of 84.

**A model never writes coordinates.** `pose_agent` hands a local LLM a command
vocabulary - point a bone, bend a joint, turn the figure - and applies it
through `move_joint` and `rotate_about_axis`, the same rotations a drag uses.
Every bone-length invariant above then holds for free, whatever the model
returns. A 7B model asked for eighteen keypoints returns mismatched limbs, and
writing those in would walk straight through `check_lengths`. Commands are
named in the *figure's* frame, not the world's or the viewer's, which is what
lets them compose after a `turn`; a bend reads its own sign off the geometry
rather than tabulating one per side.

**A bad command is skipped, not fatal.** A local model gets one wrong every so
often. Losing the whole pose to a misspelled joint would make the CLI useless
exactly where a small local model is the point. `apply_commands` returns the
warnings and carries on. A stance recurses into its own steps, so it has a
depth guard - and it must *report* what failed inside it, which is the one
place a bad command used to be silent, the depth guard's own message included.

**A stance may start from another, and a name may only mean one thing.**
`everyday.POSES` is merged into `STANCES` with `setdefault`, never `update`: a
catalogue entry that repeats a basic name - "walking", "running" - is an
*alias* for it, one `stance walking` command, so letting it overwrite points
the name at itself and the first figure to walk takes the process down with a
RecursionError.

**`distance` is the gap to the object's near face.** A desk is 70 cm deep, so
centring one 20 cm in front of a figure puts its top surface through the
figure's thighs. The clearance is measured on the trunk - shoulders and hips
plus the body's own depth - because the legs go *under* a table and the arms
reach over it, and a silhouette that included a seated figure's knees would
push every desk out of reach.

**An object is anchored against the finished figure.** `apply_commands` holds
every `place` until the rest of the list has run. "Sit down AND put a chair
under the hips" reads naturally in either order, and placed first the chair
anchors to a *standing* hip - and since a figure's ground is its own lowest
foot, that chair comes out 86 cm tall with the desk in front of it ending up
below the seat.

**A garment is the body's own sweep, clipped and padded.** `wearables` does
not carry meshes: it reads `body_segments` - the same table `body_parts`
builds solids from - takes the stretch of a segment a garment covers and pushes
the profile outward. That is what makes a sleeve fit every preset and every
pose with nothing to fit and nothing to drift. Padding is centimetres, never a
multiplier: a multiplier on a wide hip and a narrow one is two different
garments. And a pad has to survive `render_depth`'s two-centimetre smooth
minimum, which is why a t-shirt is a centimetre of cloth rather than the two
millimetres it really is.

**Hair falls down a back, and a back is deeper than the skull it hangs from.**
Anchoring a fall on the head's own surface buries it in the shoulders;
anchoring it behind the shoulders leaves a plank hovering with a gap above it.
It starts touching the head and *leans* back over its length by the difference.

**Objects reach the depth map, never the pose map.** A chair drawn into an
OpenPose image is read as a limb. `render_scene` and the editor's exports keep
them apart, and both suites assert the pose PNG is byte-identical with and
without objects in the scene.

**An object is placed against the figure, and stored in world centimetres.**
`pose_agent`'s anchors - `under_hips`, `in_front`, `at_hands` - are resolved
once, at placement, in the figure's own frame; what a scene file holds is the
result. The anchor stops being true the moment anything is dragged, so keeping
it would be keeping a lie. `under_hips` is the one anchor that sets a height
rather than a position: the object's top meets the hip and its base still
reaches the floor, which is what makes a chair fit a child and an adult.

**A head is not a scaled copy of the body under it.** Fitting log head size
on log stature across ANSUR II gives an exponent of 0.07 for head breadth, 0.24
for tragion-to-crown and 0.34 for head length - a quarter - against 0.89 for
the hand. This used stature / 175, an exponent of one, and carried a special
case forcing a child's head back up because the result was absurd. The exponent
does that on its own. The other sex differences worth having are in `ANSUR`
too: a woman's head is 4% larger against her height than straight scaling
gives, and her narrowest waist sits at 0.569 of the way from shoulder to hip
against his 0.593. The limb girth multipliers were already right - checked, not
assumed.

**Proportions come from ANSUR II, and a test says so.** The 2012 US Army
survey: 4082 men, 1986 women, 93 measurements, public since 2017. The `ANSUR`
table holds means computed from the released CSVs, and
`tests/test_proportions.py` repeats the reference values and fails when the
two part company. What this replaced - Drillis & Contini's 1966 constants from
Winter's Biomechanics - put the trochanter at 0.530 of stature against 0.513
measured: three centimetres of leg on a 175 cm figure, taken off the torso,
uncaught for as long as nothing compared the table with anything.

The landmark matters as much as the number. Trochanterion is the hip joint
centre, the lateral femoral epicondyle the knee, the lateral malleolus the
ankle, acromion the shoulder keypoint, tragion the ear. Thigh and shank are
differences of measured heights, so they close on the floor by construction.
Two things the survey cannot settle: hip width, which it measures at the iliac
crests and not at the femoral heads the leg swings from, and the waist, which
it measures at the navel - t = 0.71 along shoulder-to-hip - while the profile's
waist station is the tenth rib at t = 0.59. There is no child in ANSUR either,
so that preset is inherited and unverified.

**OpenPose's neck is the midpoint of the shoulders.** It is inferred that way
from the COCO annotations, which have no neck of their own, so `shoulder_drop`
must be zero. It was 0.011 of stature, which put every exported neck keypoint
two centimetres above where the format puts it and the shoulder line the same
distance below measured acromial height - one constant, wrong against both the
survey and the format.

**Assets ride the body's pose solution.** `solve_pose` returns the result keyed
by bone name; `skin_with` applies it to any mesh on the same armature. Never
solve an asset separately or it will drift from the body.

## Platform and format gotchas

- Local runtimes disagree about how a JSON schema is passed. Ollama takes it
  as `format` on `/api/chat`; everything else takes `response_format` on
  `/v1/chat/completions`, and llama.cpp has shipped releases that reject the
  OpenAI spelling of it. `LocalLLM.complete` tries them in order and drops to
  unconstrained JSON rather than failing, and `parse_json_object` digs the
  object out of a fence, because a small model wraps its answer more often
  than not. `tests/test_pose_agent.py` serves all of that from a stub.
- Aiming a bone is a minimal rotation, and there is none onto the direction
  exactly opposite where the bone started: the axis is undefined there and
  ill-conditioned near it. A rig whose arms rest out to the side therefore has
  an unstable roll when an arm swings round to the far side of the body. No
  solver of this shape avoids it; the roll reset makes it smaller (46 deg to
  14 deg on CesiumMan), and `tests/check_glb.py` reports it rather than failing
  when the jump is within 40 degrees of the reversal. A jump anywhere else is
  a bug. On a rig with arms down at rest - MakeHuman, MPFB2 - the same sweep
  steps 4 degrees and never jumps.
- A stack of tapering cylinders seen end-on shows every flat rim it has. A
  ponytail built with stations half a gap apart came out looking like a comb;
  `_cloth_tube` overlaps by 0.78 of the gap, as the body's own sweep does.
- Objects need a box primitive of their own. Swept as an elliptical cylinder
  a crate has rounded sides, and a depth map of a room built from those reads
  as a room full of cushions. `_box_z` is the same slab clip `_slab_z` does
  against its end planes, three times over.
- `_screen_extent` must be exact per kind, because it sets the pixel window a
  primitive is solved in and anything outside it is not drawn. The ellipsoid
  formula under-measures a cylinder by its end caps and a box by most of a
  corner - invisible on a limb station a quarter of a centimetre thick, a
  shaved edge on a table.
- Framing and the view choice both have to know about objects. A 6 m floor or
  a 4 m wall is a backdrop: framing to hold all of one shrinks the figure the
  image is about to nothing, so objects may widen the frame only so far past
  the people. And a desk in front of a seated figure is in front of it from
  the figure's side of the room, so a view is rejected when objects cover more
  than a third of the keypoints from nearer than they are.
- A view is chosen by how much of the pose's *departure from rest* survives
  projection, not by how much of each bone does. A crouch seen head-on still
  shows 71% of the thigh and reads as a figure standing up straight, because
  what makes it a crouch is the part aimed at the lens. Bones that did not
  move get no say. A camera the prompt or the model names is never overruled.
  But a bone that *did* move has to be readable itself as well, because the
  two come apart: an arm brought up to carry a box swings from hanging to
  pointing forward, and the change between them is mostly vertical, so the
  front shows 82% of the departure and 25% of the arm. `legible_view` scores
  the smaller of the two. The test that guards it compares the chosen view
  with the best of the nine rather than against a fixed bar, because some
  poses have no good view - a cross-legged sit points its shins at the lens
  from everywhere - and a bar low enough to admit that one catches nothing.
- Eighteen keypoints are a thin sample of a picture, so burial is also an
  area test: a wall right in front covers most of the frame while leaving a
  dozen keypoints technically unobscured, and a depth map of that is a slab.
  A third of the *frame*, not a share of the figure's own area - scoring it
  against the figure rejects a desk seen from the side, which is a fair
  picture of someone at a desk with their legs behind it as in any photograph
  from that angle, and the next view that reads is the back of their head.
- A view that buries the figure is last in every case, never merely demoted:
  a buried figure is not a conditioning image at all, because the depth map is
  then a picture of the desk. A seated figure at a desk reads 99% from the
  side and 51% from three-quarters, and the side puts a 140 cm desk between
  the lens and the person, so the three-quarter has to win despite being the
  worse view of the pose. `legible_view` used to take the best-reading view
  whenever no *clear* view cleared its threshold, which is the opposite. The
  same applies to the test: "best available" has to mean best among the views
  that do not bury the figure, or it asserts the behaviour being fixed.
- Burial must be judged on the *framed* camera. `legible_view` was testing it
  on an unframed one, which is a different projection from the one the export
  uses, and framing is exactly what decides whether the desk covers the figure
  or sits below it.
- Framing has to fit the *export rectangle*, not the window. A figure lying
  down is twice as wide as the standing one and was cropped by framing on the
  view, and framing on the keypoints alone cuts the hands off, which reach
  another 17 cm past the wrist. A skeleton also has no thickness and a body
  does, so the fit is padded by the widest cross-section the figure carries: a
  6% margin is ample on a standing figure, where 175 cm of height dwarfs it,
  and nowhere near enough on a deep crouch, where the figure is 90 cm across
  and a 19 cm chest half-width is a fifth of that - a head and two hands over
  the edge. It only showed once a rigged body put real girth in the frame.

- A tk `Canvas` defaults to 378 px wide. Five in a row overflow their strip and
  the last ones collapse to 1 px. Ask for `width=10` and let `expand` share.
- `pack_forget` drops a widget from the packing order; re-packing appends it,
  and an expanding sibling has already claimed the space. Use `before=`.
- glTF splits vertices at UV seams, so index connectivity reports one body as
  dozens of shells. Weld by position before any connected-component analysis.
- A shape key reaches glTF as a *morph target with a default weight*, and a
  body built out of sliders keeps its whole shape there - MakeHuman's macro
  sliders, a Daz morph dial. Reading POSITION alone loads every figure in a
  set as the identical unshaped base mesh, and the giveaway is indirect: the
  rig *is* fitted to the shape, so it no longer matches the mesh it drives.
  Five MPFB2 bodies came in as one 167 cm mannequin with five skeletons
  stretching it, and the only symptom was a limb the skinning appeared to
  pinch by a third. Node weights override the mesh's own, per the spec.
- A morph target is usually stored in a *sparse* accessor: no bufferView of
  its own, a base of zeros, and only the elements that moved. Skipping sparse
  reads such a target as no displacement at all, which looks exactly like a
  mesh that has no targets.
- A model's joint array is longer than its parent array. SMPL-X returns 127
  joints and 55 parents; the rest are landmarks posed by skinning. Iterate over
  the parents.
- Anny and MPFB2 build from the same MakeHuman assets and name the same
  slider `gender`, and they run it the opposite way: Anny's 0.0 is male, MPFB2's
  is female. Getting it backwards is silent - a complete, plausible body set
  comes out with every sex inverted - so `tools/make_bodies.py` states the
  convention as a measurement: at 0.0 the figure has 54 cm shoulders on a
  190 cm frame, at 1.0 44.5 cm on 176.
- Anny is Z-up and glTF is Y-up, and Anny's skinning carries nine influences
  per vertex where glTF's first joint set carries four. Keep the four heaviest
  and renormalise; the fifth weight on a vertex that has one is worth well
  under a millimetre.
- MakeHuman's base mesh carries helper geometry (skirt, tights, hair helper,
  joint cubes) as loose shells. It reads as clothing in a depth map. Delete
  helpers in MPFB2, or let the loader drop small shells.
- Nothing moves the pelvis - posing is rotation - so how deep a squat reads is
  the gap between the hip and the feet, and a figure's floor is its own lowest
  foot rather than a plane in the scene. Dropping the thigh 45 degrees opens
  that gap to 80 cm, which is a figure dipping its knees, not a squat; the
  thigh has to rise *forward* from the hip. The same fact is why a lying
  figure has nothing to anchor a bed to: its ankles come up to hip height, so
  anything placed under it lands at the height of a hip.
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
4. Wrists are never rotated; there is no keypoint for them. Ankles are only
   levelled, not aimed - a foot taking weight goes flat, but nothing turns it
   in or out.
5. The web prototype duplicates the maths in JavaScript. It is tested
   independently (`node web/test.mjs`) and will drift from the Python. Its
   preset table no longer can: it is generated from `preset_params` and
   `tests/test_proportions.py` reads it back and compares. It had drifted by a
   third of the child's torso, 33 cm against 44, before anything checked. The
   rest of the file is still unguarded.
