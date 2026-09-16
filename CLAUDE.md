# CLAUDE.md

Project context for Claude Code. Read this before changing anything.

## What this is

A desktop editor for authoring OpenPose skeletons in 3D and exporting matched
pose and depth images for ControlNet. The layout:

- `openpose3d_editor.py` - the entry point and the public surface. What used
  to be one 4300-line file is now `vecmath`, `anthro`, `camera`, `skeleton`,
  `posemap`, `anatomy`, `raster`, `exporting`, `scenefile`, `randomize`,
  `rigpose` and `ui_app`; this re-exports all of them, so existing callers are
  untouched.
- `rigpose.py` - THE POSE. `RigPose` is a local rotation per bone on Anny's
  104-bone armature, and `Figure` is one posable person: that armature plus
  who the body is. This is what the editor drags, what a command applies to,
  what a scene file stores and what the depth map is skinned from. There is
  no second description of a pose anywhere.
- `mesh_backend.py` - loads an Anny body and poses its own armature. This is
  where a depth export comes from, and it is the only place one comes from.
- `bodies_lib.py` / `bodies/` - the nine rigged bodies, one per preset, built
  by `tools/make_bodies.py`.
- `garments_lib.py` / `garments/` - real CC0 MakeHuman garment meshes fitted
  to each of those bodies by `tools/make_wearables.py`.
- `wearables.py` - the garment *vocabulary*, plus the swept approximation the
  viewport draws at sixty frames a second.
- `pose_agent.py`, `everyday.py`, `props.py` - the command vocabulary a local
  model emits, the 84-pose catalogue and the object library.
- `server.py` + `web/` - a mobile prototype; the maths it duplicates is only
  the projection and the drag.

## Run the tests before and after every change

```
./tests/run_all.sh                  # 23 checks; needs python3-tk and xvfb
python3 tests/test_core.py          # pure maths, no display needed
python3 rigpose.py --selftest       # the pose itself: FK, drags, undo, files
python3 mesh_backend.py --selftest  # rigged mesh maths, no model files needed
python3 randomize.py --selftest     # the randomizer, no display needed
python3 tests/check_glb.py bodies/*.glb   # the body set itself, not in run_all
```

The body set is not optional any more. The editor poses Anny's armature, so
without `bodies/` there is nothing to pose and nothing to draw; `bodies_lib`
raises `MissingBodies`, which names the command that builds them.

These are not decoration. Every check in them was written because something
actually broke. Several bugs below were invisible for multiple releases because
the test that would have caught them made a wrong assumption.

## Invariants that must never regress

**A bone cannot change length, and that is now structural rather than
enforced.** Everything that moves a figure - a drag, a randomizer, a command
from a model - ends in `RigPose.rotate` or `RigPose.aim`, which compose a
rotation onto one bone's local matrix. A rotation cannot resize anything, so
there is no flag to get wrong and no free-length drag to misread. `Skeleton`
still has `move_joint(stretch=...)` because the pure keypoint maths still
exists and is still tested; nothing the editor does goes through it.

What this replaced is worth remembering, because both bugs were in the FLAG
and not in the maths: symmetric editing fed a reflected *position* into
`move_joint`, and a modifier misread made every drag on Windows look like a
free-length drag. `check_lengths` is still there as a backstop and now has
nothing left to catch - if it ever fires, that is a bug report.

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

**Twist is not observable from 18 keypoints - so it is EDITED instead.**
Nothing in a keypoint skeleton says how far a forearm has pronated, and the
old solver had to tie every bone's roll to the body's rest orientation and
hope. On the armature a roll is just another rotation: `RigPose.spin` turns a
bone about its own axis, which by construction does not move the joint it
points at and does turn everything hanging off it. The selftest measures both.

MakeHuman splits each limb segment in two so the second half can carry roll
without wringing the skin at the joint, and `rigpose.TWIST` names those eight
bones - `upperarm02`, `lowerarm02`, `upperleg02`, `lowerleg02`, both sides.
They are NOT joints. A dragged elbow walks up past `upperarm02.L` to the
shoulder so the whole humerus swings as the one rigid piece it is; rotating
the twist bone instead folds the arm 8 cm below the shoulder, which the depth
map shows as a crease no arm has. A roll goes the other way and is applied TO
the twist bone, which is what it is for.

They are named, not detected, and both obvious rules fail. "A bone continuing
its parent's name one number higher" also catches every finger phalanx, every
toe joint and the two lower cervicals, which are real joints. "Near-collinear
with its parent" puts `upperleg02` at 12.9 degrees on the same side as
`finger2-2` at 12.6. There is one rig; the honest thing is to say which eight.

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

**A depth export is rigged geometry, or it does not happen.** The built-in
swept anatomy is a stack of tapering cross-sections: it knows where every limb
is and only approximates what a person looks like. It is there to draw the
viewport at sixty frames a second and to give `wearables` a profile to cut a
garment out of. It must never become an export. Every path that writes a depth
map resolves its bodies through `bodies_lib` and raises `MissingBodies` -
which names the command that builds them - rather than quietly producing the
basic version. The silent fallback is the whole problem: the PNG still
appears, it is just a picture of a mannequin, and nothing says so. Asking for
the sweep deliberately is `anatomy=True`, and only the viewport preview and
the tests of the sweep itself do.

**Cloth lies on the surface, it is not padded until it clears.** `wearables`
clips and pads the body's own *swept* profile, and that profile is thinner
than a rigged mesh wherever the two disagree - a chest, a shoulder - so a coat
dropped straight into the rigged buffer comes out with the body through it.
Where a garment covers the body it takes the body's own depth less the
garment's own thickness, so it follows the rigged shape rather than the
approximation it was cut from; where it reaches past the body - a hem, a fall
of hair - it keeps its own depth, because there is nothing under it. Padding
harder instead would make a coat that fits the sweep and floats off the mesh.

**A garment clears the rigged body by its OWN thickness, and a piece with
nothing under it clears nothing.** One figure for every garment was a flat
centimetre, which is right for a t-shirt and flattens a five-centimetre afro
onto the skull, a helmet into the head and a coat into the chest: every thick
garment there is came out of a rigged export looking like bare skin, and since
the silhouette still grew the PNG looked plausible, which is why nothing
caught it for a release. So `wearables.layers` hands the rigged path one part
per standoff rather than one per garment, and the standoff is the piece's own
padding - `None` for a fall of hair, a hat brim or a bun, which are not over
the body at all and are dragged round to the front of the face by any standoff
whatsoever. The test is not coverage, which the clamp never
touched: it is whether the thick version stands *proud* of what the thin one
reaches.

**The garment names are the vocabulary, the outfits are the catalogue.** The
same split `everyday.POSES` makes against the command ops, for the same
reason: a model picks a garment on every `wear`, so those thirty-odd names
stay lean, while `outfit` is picked at most once per figure and saves four
guesses, so `wearables.OUTFITS` can name thirty-one looks. An outfit sets only
the slots
it names - "outfit winter" leaves the haircut alone, and a `wear` either side
of it still lands - which is why "bare" has to name every slot explicitly:
taking it all off is the one thing that cannot be said by omission.

**A pair of tails has to splay, and a hat brim is round.** Two ponytails at
the same fall hang in the same place and read as one, so a pair splays as it
falls. And giving a brim's reach to the depth alone with 45% of it to the
width put a diving board on a walking figure in a sun hat. The same class of
mistake is why a flat panel of cloth - an apron, a cape - has to be one slab
rather than a stack of stations following the torso's profile, which comes out
of the depth map in horizontal bands, and why a rucksack wants `_box_z` rather
than an ellipsoid, which is a beach ball on someone's back. Those three are
not in the vocabulary now, because the library has no mesh for them; if they
come back, they come back knowing this.

**A rigged export wears real meshes, and only real meshes.** `wearables`
builds a garment out of the body's own swept profile; that draws the viewport
at sixty frames a second and it is an approximation. What goes into a
conditioning image is the CC0 MakeHuman garment library, fitted to every body
by `tools/make_wearables.py` and skinned to its armature, so `skin_with` drives
it from the body's own solution. The two are never mixed on one figure: a
swept scarf next to a fitted coat reads as a slab hanging off the chest,
because the eye goes straight to the join. A slot the library has nothing for
is not worn on that path, and `garments_lib.describe()` says which those are.

**A `.mhclo` is exact against the mesh it names and meaningless against any
other.** It describes a garment as, per vertex, three vertices of MakeHuman's
`hm08`, barycentric weights and an offset. Anny has 13718 vertices to hm08's
13380 and shares 1967 triangles of 26756, so fitting by index puts most of a
hairstyle roughly on the head - which looks like a fixable glitch - and throws
the rest across the room. It is not fixable by patching outliers. The garment
is fitted to `base.obj`, where the indices mean what they say, and carried onto
the Anny body geometrically: each garment vertex is described by which
base-body vertices it sits over and how far out along their normals, and
rebuilt from that description over the Anny surface. What makes it sound is
that the two are the same human shape in the same rest pose at the same
stature, not that they share any numbering.

**A garment hides the body under it.** MakeHuman ships `delete_verts` for
exactly this, and without it a shirt sits a millimetre off a chest and the
chest wins the z-test, so the garment is absent wherever it matters most.
Carried across by the same geometric map the vertices are, then eroded by a
ring: the deletion list is in the base mesh's numbering and its boundary lands
a vertex or two off, and a body face dropped just past a hem is a black slit
with nothing over it.

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

**You drag the armature, because the armature IS the pose.** There is one
description of a figure now and it is `rigpose.Figure`: a local rotation per
bone on Anny's own skeleton. The joints on the canvas are its joints, a drag
rotates the bone above the one you grabbed, everything below follows because
that is what a local rotation composed down a tree means, and the depth map is
skinned from the same array without anything in between.

What this replaced was an eighteen-point OpenPose skeleton that you dragged
and a solver that aimed the rig at it. The intermediate step was drawn as an
overlay so you could at least SEE the armature, which papered over the real
problem: it was a translation between two descriptions of one pose, and the
second could only ever lose what the poorer of the two could not say. That was
the whole hand, both collarbones, the roll of every limb and 85 of the rig's
104 bones. All of it is editable now, and none of it needed new machinery -
they were always bones.

The swept body preview stays on B because it is the only shading that survives
a drag at sixty frames a second, and it is still the approximation, not the
export. The real thing is P.

**One rig, named, and nothing else.** `mesh_backend.ANNY_BONES` maps every
role onto an exact bone name in the 104-bone MakeHuman skeleton that
`tools/make_bodies.py` builds every body in `bodies/` on. All nine share one
bone-name set, so the map is exact and a file that is not that skeleton fails
`validate_roles` rather than being half-matched into a mangled limb.

What this replaced was a table of name fragments - MakeHuman, Mixamo, Daz -
matched longest-first against whatever a file carried, plus an SMPL-X backend,
a hand-loaded `.glb` path, a mesh library and a roles file for rigs whose
names were not recognised. That is the right shape for a program that takes
any rig and the wrong one for a program whose depth export is a specific body
set, and it cost accuracy on the rig that ships: a fragment match cannot fail
loudly, and every solver decision had to hold for rigs nobody here had ever
rendered.

**A rig is sized by a unit conversion, because the body set is authored at
each figure's stature.** `tools/make_bodies.py` bisects the height slider onto
the preset's stature and all nine land on it exactly, so `pose_rig` without a
stature multiplies metres by a hundred and stops.

The fallback it replaced - the ratio of the keypoints' shoulder-to-hip span to
the rig's own - was not merely unnecessary, it was hiding something. The
self-test's mock rig was 1.60 m while the keypoints posing it described 175 cm,
and the span scale silently cancelled that 9%. Authored at the same stature as
its keypoints, the same rig's roll reset agrees with aiming to 3.66 degrees
where it read 6.56 before. A rig the wrong size for the pose it is given has
every joint landing where its own bone lengths did not intend, and everything
measured against a rest reference drifts. `tests/check_glb.py` now builds each
body's keypoints from that body's own preset for the same reason: the worst
joint error across the nine went from 10.96 cm to 10.12, the worst on an adult
from 10.96 to 5.49, and the worst skinning pinch from 14% to 8%.

**The rig is posed, not fitted, and the keypoints are read back off it.** A
depth map is a picture of a body, and the body is the rig's. Fitting inverted
that: every bone was aimed at a keypoint, then slid onto it and scaled until it
landed, so the mesh came out wearing the keypoint skeleton's proportions -
stretched by up to a tenth per segment, thigh and shin pulling opposite ways on
the same leg. Those proportions come from a table and the rig is a measured
body; the two were never going to agree, and reconciling them took six passes
of per-segment scaling, a landmark-offset table, and several hundred lines that
are no longer here. None of it was needed. A pose is a set of joint ANGLES, and
the line from a shoulder keypoint to an elbow keypoint says which way an upper
arm points no matter whose arm it is. So `mesh_backend.pose_rig` takes the
directions and nothing else, every bone is rotated and none is moved or
resized, and each segment comes out at exactly the length it was authored with.
The rig's own hands, feet and spine chain come along for free - 85 of its 104
bones, 30 of them in the hands, had nothing to do under a retarget.

`stature` still sizes the body, because a figure has to be the right height to
stand beside another one - but that is one uniform scale over the whole mesh
and it changes no proportion.

**And the OpenPose PNG is a description of that mesh, never a second opinion
about it.** `keypoint_riders` and `keypoints_of` read the eighteen back off the
posed rig: the ten that are joints the rig already has are simply its joints -
the elbow keypoint is where the forearm bone starts, and nothing describes that
better than the thing itself - while the face rides the skull and the shoulder
carries the acromion offset on the collar, because OpenPose puts it at a bony
corner where nothing rotates. The neck stays the midpoint of the shoulders, or
the format is not the format. Read in this order the pair cannot disagree with
itself; read the other way the skeleton and the body it is drawn beside differ
by five or six centimetres at the shoulder, which is the gap between ANSUR's
table and this particular body, and no amount of work on either side closes it.

**A keypoint is a landmark; a rig joint is a hinge, and two of them are not
the same point.** OpenPose's shoulder is the acromion - the bony corner on
*top* of the shoulder, which is what the COCO format and the ANSUR table both
mean by it - while the bone an arm swings from starts at the glenohumeral
joint, below it and well inboard. Its neck is not a neck at all: COCO has none,
so it is inferred as the *midpoint of the shoulders*, out in the middle of the
upper chest, where a rig's neck bone starts at the top of the thorax. This is
why nothing slides a rig joint onto a keypoint: doing so drags the body with it
- the shoulder line up and outward, the whole head down - and correcting it
with a table of per-joint offsets went out wrong twice in opposite directions.
Applied whole, the shoulder's offset moved the joint 6 cm down the arm while
the elbow keypoint stayed put, so the humerus spanned a gap 6 cm shorter than
it is: 18% of an adult's upper arm and 54% of the child's. Masking the offset
along the bone kept the humerus and threw away the correction, so every figure
stood with its shoulders round its ears. Neither was a fix. Aiming a bone needs
no offset at all, because a direction does not care where either end sits - and
elbow, wrist, knee and ankle keypoints are joint centres anyway.

**So the arm hangs from the joint.** `derive_proportions` closes the arm on the
measured span from `gh_w`, the glenohumeral half-width, and splits it by
`arm_split` taken from the rigs - not from the acromion with ANSUR's
acromion-radiale ratio, which starts at a bony corner 0.022 of stature above
and 0.009 inboard of the joint and overstates the humerus by the difference.
Measured from the joint it closes to a tenth of a percent: gh_w + humerus +
forearm + hand is 0.5172 of stature against a measured half-span of 0.5165 on
the adult male rig, which is what `tests/test_proportions.py` asserts. The
keypoints the editor hands out then put the elbow exactly a humerus from the
shoulder joint rather than from the bony corner above it, and every figure
finishes within 1.5 cm of its stature.

The lesson underneath is duller and worth more. Every check written for the
first attempt was on the shoulder - where it sat, how wide it was, how proud it
stood - and not one was on the arm hanging off it, so a fix that halved a
child's upper arm passed a green suite and shipped. The second attempt was
measured the same way and left the shoulders visibly high, and the number
saying so was in the output at the time. When a change moves a joint, test the
bones on *both* sides of it, and look at the picture at a magnification where
you could see the thing you are claiming to have fixed.

**A rig is sized by the figure's stature, never by a span between two
landmarks.** This divided the keypoints' shoulder-to-hip distance by the rig's
own, and those two spans do not measure the same thing - an acromion-to-
trochanter against a glenohumeral-to-femoral-head - so the whole rig came up
15% oversize on an average adult and 41% on the child before a single bone had
been aimed, and the scaling passes that tried to drag it back down left
shoulders 4 cm high on an average man and a child 8 cm too tall. `pose_rig`
takes `stature` and sizes the rig by it against its own rest height; a body set
built for these presets is authored at the figure's stature, so the rig is left
the size it was drawn. Without one - an outside GLB with no figure behind it -
it falls back to the torso span, which is all such a rig can offer, and that
fallback is the only thing that span is still used for.

The rigged body now fills the same frame as the swept anatomy to within a
tenth. The old detail test passed on the strength of the bug: a body a sixth
too big for its own skeleton covers more of the picture and carries more edge
with it, so normalise that measure per covered pixel or it is measuring size,
not surface.

**There is no child in ANSUR, so the child row is measured against the body
set's own rigs.** It is the male row times the adult-to-child shape change,
which `tools/child_ratios.py` computes from the Anny adult and the Anny child -
both built from the same WHO-calibrated model, so the ratio between them is
sound even where neither absolute is, because the landmark disagreement sits on
both sides of it and cancels. `tests/test_proportions.py` reads it back, the
same arrangement the web preset table has. What it replaced was the male row at
122 cm with `leg_ratio` 0.9 - a 70% scale soldier with its legs cut, which a
child is not. Its shoulders came out 14.4 cm half-width against the 11.8 its own
rig has, 22% too broad, and the leg_ratio put the hip at 56.3 cm where the rig
puts it at 63.0, so the torso ran 8 cm long. Only the ratio is taken from the
rig and never the absolute: the rig's shoulder is the glenohumeral joint and
the table's is the acromion.

**A bone keeps the length it was authored with, and the self-test says so
exactly.** `skin_mesh` still builds a general 4x4 out of Q, because the uniform
stature scale rides in it, but nothing writes a per-bone scale any more: the
selftest asserts every segment matches the rig's own to machine precision
(4e-16) rather than to a tolerance. A tolerance is what let a per-segment
scaling pass hide behind "close enough" for several releases.

**The editor is modules now, and `openpose3d_editor.py` is the front door.**
The 4300-line file is split - `vecmath`, `anthro`, `camera`, `skeleton`,
`posemap`, `anatomy`, `raster`, `exporting`, `scenefile`, `randomize`,
`ui_app` - and every name it used to export is re-exported from it, so all 28
callers and all 26 test files kept working with nothing edited. That is the
point of the facade and it should stay: a split that made every caller change
would have been a rewrite wearing a refactor's clothes. New code should import
the module it wants; old code must keep working through the front door.

**There is ONE framing, and both the window and the export use it.**
`exporting.frame_scene` fits the figure's own silhouette - the crown, the
hands and the soles, none of which is a keypoint, plus the widest
cross-section the body carries - into the export rectangle. The editor used to
have a second one: eighteen keypoints with a flat 18% margin, which cropped
the crown and the feet off a figure standing at rest, and which could frame
about a different centre from the one the PNG used. Two framings mean the
rectangle on screen stops promising what it exists to promise.

**`frame_rect` is measured against the camera, not the canvas widget.** They
are the same size in a live window, because `on_resize` sets one from the
other, but `camera.project` puts its origin at the middle of the camera - so a
rectangle centred on anything else is not centred on the projection. Where the
two drifted apart, keypoints landed outside the exported PNG. Widening the
panel was enough to expose it; it had been latent for as long as the two
existed.

**A random pose is made of the same rotations a drag is.** `randomize` goes
through `move_joint` and `RigPose.rotate` and never writes a coordinate, so an
edge case it finds is a real one rather than an artefact of how it was
generated. A seed names a pose, which is what makes "the elbow is inside the
ribcage at seed 412" a bug report instead of a screenshot.

The two bugs its own length check caught on the first run are worth keeping
even though **neither is expressible any more**, because the lesson generalises
to anything that rotates part of a figure by hand. **Both were about where the
AXIS ran, not about what was being rotated.** Leaning a list of keypoints
about the hip LINE is exact, because both hips lie on that axis and cannot
move. Twisting the same list about the vertical through the hip midpoint is
not, because by then the legs have moved the hips and the neck is no longer
over them - 17 cm on a neck-to-hip bone. Anchoring the twist at the neck fixed
that, and then reading the neck BEFORE the lean had moved it put 5 cm back,
which looked identical to the first bug and needed its own fix.

A bone rotation cannot make either mistake. It turns about its own head, it
carries exactly what hangs off it, and it cannot change a length at all. The
torso leans by rotating the spine chain and the legs stay because they hang
off `root` beside it, so there is no list of what moves and no axis to choose.
That is the general argument for posing a rig rather than a point cloud: a
whole class of error stops being a thing you can get wrong.

**The panel is four tabs, and every section names the tab it lives on.**
Twelve collapsible groups in one column is a list to hunt through even folded,
and folding is not free: the thing you want is three clicks away and you have
to remember which heading it is under. `SECTION_TABS` is the table, and
`tests/test_panel_layout.py` asserts no section is missing from it - one left
out would still be built, into whichever tab came first, and would look like
it belonged there.

That test also had to be fixed to measure what it claimed. It checked that
every control could be scrolled into view using `winfo_y`, which is relative
to the widget's own PARENT - so for a button nested three frames deep it
measured the offset inside that frame and nothing else, and reported the
lowest control on a 31-control tab as 107 pixels down. Position against the
scrolling canvas, plus wherever the canvas is scrolled to, is the only thing
that answers the question.

**A posed figure is put back on the ground.** Posing is rotation and nothing
moves the pelvis, so folding the legs for a sit lifts the whole body's lowest
point off the floor - 83 cm on a cross-legged sit, which is most of a person.
`apply_commands` calls `pose.stand()` once the pose is finished and before
anything is anchored to it, so a chair placed under the hips measures a real
gap. The keypoint version never noticed because its "floor" was the lowest
ankle less 8 cm for a sole: it moved the floor to the figure instead of the
figure to the floor, and a test asserting a chair reached from floor to hip
passed on exactly that hovering gap.

Once, at the end, rather than inside each command - a stance that re-grounded
halfway through building itself would fight its own steps.

**Which way a figure faces is the hips bone's own orientation.** Anny is
authored standing upright facing +Z with its left at +X, so `Figure.body_frame`
is the world axes at rest and whatever the pelvis has been turned to after
that. It stays right for a figure lying down, where a shoulder-to-hip line
says nothing about facing at all.

What this replaced measured `up` from the hips to a point on the spine. That
is exact on a keypoint figure, whose torso is vertical because a table built
it that way, and 8 degrees out on a real body, whose lumbar curve leans back.
Every direction in the command vocabulary resolves against this frame, so
those 8 degrees reached all of them: a crate placed in front of someone at
arm's length landed 12 cm high.

**The figure stands on something, and the floor is not a prop.** A depth map
with nothing under the feet says the person is floating, and a generator
conditioned on it puts them nowhere. `exporting.ground_part` puts a ground
plane in every depth export, on by default.

It is deliberately NOT a placed object. A prop takes part in the framing and
in the buried-figure test; the ground must do neither. Framing to hold a
24-metre slab would shrink the figure to nothing, and a floor covers most of
the lower frame by design, so counting it as something in the way would reject
every camera with any pitch - which is every camera a floor is any use to. It
is passed to the depth paths separately, which is also what keeps it out of
the pose map for free.

**The floor starts at the figure and runs AWAY from the camera.** A real floor
carries on towards the lens, but in a photograph that stretch is below the
bottom of the frame - the frame is fitted to the person, and their feet are
its lower edge. An orthographic camera has no "below the frame" to hide it in:
a slab centred on the scene puts its near edge twelve metres in front of the
figure, that edge becomes the nearest thing in the buffer and takes the whole
bright end of the range, and the figure comes out a black silhouette on every
level view. From a low angle the same slab covered the figure completely.

**A level orthographic camera cannot show a floor, only where it is.** A
horizontal plane seen at pitch zero is exactly edge-on, so all that is left of
it is its front face. At twelve centimetres thick that is a bright bar across
the picture that reads as a step; at three it is a ground line, which is the
most a level view can honestly say. Pitch is what makes a floor carry depth,
and `high_three_quarter`, `bird` and `over_shoulder` are where it earns its
keep.

**The ground is graded by a curve of its own, and the figure never notices
it.** `raster.grade_scene` takes two buffers. The subject - the figures and
anything placed with them - is normalised over its OWN depth range exactly as
`depth_to_grey` does, so a figure comes out pixel-identical whether or not
there is a floor under it; the selftest asserts that, and that the only pixels
that differ are ones the floor COVERS, down at the ankles where the slab's
near face crosses in front of the feet.

One curve over both does not work, and both attempts are worth keeping. A
linear grade spends the range on the room: with a floor three metres back the
body got 61 grey levels of 210 and came out a flat white cut-out. Grading
everything in inverse depth instead - MiDaS's own convention, and ControlNet's
depth models are trained on MiDaS - bought some of that back, 114 levels
against the floor's 116. But it still crossed the whole frame as a slow
gradient and stopped at `far`, so it ended in a horizontal grey line against
the black background and read as a platform the figure was standing on.

A floor should fade OUT, and fading out means reaching the background, which
is below `far` by definition. So the ground gets its own falloff, and it gets
to go all the way to black - which is also the truthful answer, since
disparity really does go to zero at the horizon. `GROUND_FADE` halves its
brightness every 22 cm and `GROUND_BRIGHT` starts it at 0.60 of full white, so
it is a pool of ground around the feet reaching about the bottom fifth of the
frame, under the figure rather than competing with it.

**A hand and a foot are SET, never inferred - and they are ordinary bones.**
Nothing in a pose says which way a palm faces, so it has to be said. It used
to be said in a pair of angles carried BESIDE the pose and applied by a
special case at the end of the solve, because the wrist and the ankle ended
the keypoint chains and there was nowhere else to put them.
`Figure.set_extremity` is two rotations of one bone now, and the wrist, every
knuckle and every toe joint are all draggable in the viewport like anything
else. A wrist that also deviates sideways is a third rotation, not a change to
the format.

**Their axes are derived; their four SIGNS were measured.** A canonical axis
still leaves the handedness of each rotation open, and a control whose +30
points the left toe in and the right toe out is one nobody can use. So the
signs were measured on the body set rather than reasoned about, and the
selftest measures them again: a positive lift raises both sets of toes 8.3 cm,
a positive foot turn points both toes outward 7.4 cm, a positive hand bend
curls both hands toward their own palms, and a positive hand turn is
pronation, taking both thumbs down.

The perpendicular a bend turns about is built from each part's OWN direction,
so it is already mirrored between the sides and must not be given a side sign
as well - that flips it twice on one side and not at all on the other, which
is exactly what made the lift raise one toe and drop the other. The turn axis
IS shared between the sides, so that one does take the sign. Getting this
backwards is silent and symmetrical-looking on a screenshot of one foot.

**A tk Scale fires its `command` from the event loop, not from the
assignment.** So a guard raised and cleared inside the method that sets the
sliders is already down when the callbacks arrive, and they then stamp those
values onto whatever the control points at *now*. That is how switching from
the feet to a hand used to write the feet's angles onto the hand, and how an
undo came back with the wrong ones. The guard has to stay up until the queue
has drained - `after_idle`.

**The export shape is one table, and changing it re-frames.** `exporting.ASPECTS`
names the ratios worth having and `parse_size` accepts either spelling, so the
editor menu, `--size 16:9` on every CLI and the phone's dropdown all read the
same list - the phone's comes over the wire in `/api/vocabulary` rather than
being written out again in JavaScript, for the same reason the preset table is
generated.

Changing the shape must call `frame_scene`, not just write two numbers. The
export rectangle IS the ratio, so a figure fitted to a tall frame is not
fitted to a wide one: turning a 2:3 portrait on its side without re-framing
crops the head and the feet, and the only place that shows is the PNG. And a
shape is recognised by RATIO, never by the exact pixels - 1024x1536 is 2:3 as
much as 512x768 is, and a control that only knew its own defaults would call
every scaled-up export "custom".

**The locked views live in the width the export frame cannot use.** The export
is 2:3 and a window is wide, so fitting that frame into the canvas leaves most
of the width black at any window size - 880 of 1340 pixels at 1600x1000. The
five small views are a column down the left rather than a strip across the
bottom, which fills exactly that space and gives the main view back the height
the strip was taking.

**Keypoints are OUTPUT, with exactly one importer.** `keypoints_of` reads the
eighteen off a posed rig for a pose PNG, for the swept preview and for the
phone, and nothing reads them back into a figure - except `from_keypoints`,
which exists because scene files written before the editor posed the rig hold
eighteen points and nothing else, and because the phone genuinely has only
eighteen points to drag. It runs the old solver once, at load, and keeps the
joint angles. The file is a rig pose from then on.

It is lossy and the number is worth knowing: rig, out to keypoints, back to
rig loses up to 6.6 cm of body - every roll, both collarbones and all thirty
bones of each hand. So the wire format between the phone and `server.py`
carries `bones` as well as `points`: a client echoing a scene back gets the
same pixels, and a client that has actually dragged sends keypoints and is
solved. A test asserts both. Nothing else in the program may import keypoints.

**The pose PNG never sees the armature.** `project_people` asks each figure
for its eighteen and projects those. Handing it `figure.points` instead hands
it 104 joints in the rig's own order, which runs straight off the end of the
OpenPose palette - and that is the friendly failure. The unfriendly one is a
scene with a figure whose bone count happens to be 18.

**One anchor pins a joint; there is no two-anchor hinge.** Pinning means the
figure slides so that joint ends up where it started, which is exactly right
for "hold the hand still and lean the body" and is honestly all that one
anchor can mean. Two anchors used to make a hinge the body swung about, built
by re-rooting the keypoint tree at the pair. A rig has one root and every one
of its 104 bones hangs off it, so the same gesture is an IK solve rather than
a re-parent. It is not implemented, and `rotate_hinge` says so rather than
doing nothing quietly.

**Assets ride the body's pose solution.** `pose_rig` returns the result keyed
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
  an unstable roll when an arm swings round to the far side of the body, and
  no solver of this shape avoids it. Anny rests in an A-pose, so the reversal
  is somewhere an arm rarely goes and the same sweep steps a few degrees -
  which is one of the things committing to one rig buys. `tests/check_glb.py`
  reports a jump rather than failing when it is within 40 degrees of the
  reversal; a jump anywhere else is a bug.
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
- Six of the nine original named views sat at pitch zero, which is why a set
  rendered from them came out looking like a catalogue rather than
  photographs. The six added - `high_three_quarter`, `low_three_quarter`,
  `worm`, `over_shoulder`, `bird`, `profile_high` - combine a yaw with a
  pitch, which is what a camera someone is holding does. They go at the *end*
  of `VIEW_ORDER` on purpose: a figure that constrains nothing should still
  come out on a plain view, and these are there to be asked for.
- Figures are set out along world +X, 70 cm apart, so a near-profile camera
  stacks a group one behind another. A scene with more than one person wants
  a yaw well away from +-90.
- `legible_view` offers one contract - the plainest view that clears the bar,
  and if nothing clears it the best there is - and the test has to assert
  *that*, through `view_scores`, rather than something adjacent. Checking it
  was "near the best available" on a bone-and-departure measure the chooser
  does not optimise passed by luck, and turned into a failure the moment more
  views existed to be better than the one chosen.
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

- A `<canvas>` has intrinsic dimensions, so `position:fixed; inset:0` with an
  auto width does *not* stretch it: it stays 300x150, and since the resize
  handler sets the backing store from `getBoundingClientRect()`, the element
  then doubles on every resize event. The figure is drawn at the wrong scale
  in the corner and nothing says so. Give it `width:100%; height:100%` as
  well.
- The phone frames on the same silhouette the Python does - the crown, the
  hands and the soles reach past the last keypoint on their chain - and then
  needs a wider margin still, because what it draws is capsules *around* the
  skeleton and at a phone's zoom a head capsule alone is a hundred pixels past
  its keypoint.
- Taking the server's keypoints without its camera shows a seated figure
  head-on while the export comes out in profile. `legible_view` chose that
  view for a reason; adopt it too, or the phone is previewing a different
  picture from the one it will produce.
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
  A body set came in as one 167 cm mannequin with nine skeletons stretching
  it, and the only symptom was a limb the skinning appeared to pinch by a
  third. Node weights override the mesh's own, per the spec.
- A morph target is usually stored in a *sparse* accessor: no bufferView of
  its own, a base of zeros, and only the elements that moved. Skipping sparse
  reads such a target as no displacement at all, which looks exactly like a
  mesh that has no targets.
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
  joint cubes) as loose shells, and it reads as clothing in a depth map. The
  loader drops small shells for that reason; `tools/make_bodies.py` should not
  be emitting them in the first place.
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

1. The tests are scripts printing PASS/FAIL, not pytest. Converting them keeps
   the assertions but gains proper reporting and selection.
2. `ui_app.py` is still 2200 lines. It is one class doing input, drawing,
   panel construction and file dialogs; the panel is the part that would come
   out most cleanly.
3. Depth export is CPU rasterisation. Live depth would want the geometry on the
   GPU rather than the analytic rasteriser ported.
4. There is no IK. Every bone is posed from its parent down, so "put the hand
   here and let the arm work it out" is not a thing you can ask for, and the
   two-anchor hinge that used to approximate it is gone with the keypoint
   tree it was built on. One anchor pins a joint by sliding the figure. This
   is the biggest thing the rig path does not do that a poser should.
5. The `hand` and `foot` COMMANDS still take two angles each, though the bones
   underneath take any rotation and every finger is draggable. Nothing is
   inferred from context either - a figure whose palm is flat on a table
   still has to be told so.
6. The garment vocabulary was cut back to what the library can actually put
   on a figure: the `over` slot (apron, cape, backpack, scarf) and eleven
   presets with no mesh behind them are gone rather than silently absent from
   a rigged export. Headgear is one fedora standing in for hat, cap and sun
   hat. The packs have far more - `tools/make_wearables.py --list` shows the
   fifty-odd available - so widening the vocabulary again is a matter of
   building and vendoring meshes first, never of naming things the export
   cannot deliver.
7. The web prototype duplicates the maths in JavaScript. It is tested
   independently (`node web/test.mjs`) and will drift from the Python. Two
   things no longer can: the preset table is generated from `preset_params`
   and `tests/test_proportions.py` reads it back and compares - it had drifted
   by a third of the child's torso, 33 cm against 44, before anything checked
   - and the *export* is no longer duplicated at all. `server.py` renders it
   with the same code the desktop build and the suite use, and
   `tests/test_server.py` asserts that a scene sent back as keypoints comes
   out byte-identical to the plan it came from. What is left in JavaScript is
   the projection and the drag, which have to be local; the capsule depth map
   it used to write is now only the offline preview.
