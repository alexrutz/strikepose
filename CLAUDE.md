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
padding - `None` for a fall of hair, a hat brim, a bun or a rucksack, which
are not over the body at all and are dragged round to the front of the face by
any standoff whatsoever. The test is not coverage, which the clamp never
touched: it is whether the thick version stands *proud* of what the thin one
reaches.

**The garment names are the vocabulary, the outfits are the catalogue.** The
same split `everyday.POSES` makes against the command ops, for the same
reason: a model picks a garment on every `wear`, so those forty-odd names stay
lean, while `outfit` is picked at most once per figure and saves five guesses,
so `wearables.OUTFITS` can name forty-six looks. An outfit sets only the slots
it names - "outfit winter" leaves the haircut alone, and a `wear` either side
of it still lands - which is why "bare" has to name every slot explicitly:
taking it all off is the one thing that cannot be said by omission.

**A panel of cloth is one slab, and a pair of tails has to splay.** An apron
built as a stack of stations following the torso's own profile gives every
station its own width and its own depth, and every step between them comes out
as a band across the apron; it hangs in one plane, so build it as one. Two
ponytails at the same fall hang in the same place and read as one, so a pair
splays as it falls. And a rucksack swept as an ellipsoid is a beach ball on
someone's back - `_box_z` exists, and the corners are most of what says
"pack". A hat brim is round: giving the reach to the depth alone and 45% of it
to the width put a diving board on a walking figure in a sun hat.

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

**A keypoint is a landmark; a rig joint is a hinge, and two of them are not
the same point.** OpenPose's shoulder is the acromion - the bony corner on
*top* of the shoulder, which is what the COCO format and the ANSUR table both
mean by it - while the bone an arm swings from starts at the glenohumeral
joint, below it and well inboard. Its neck is not a neck at all: COCO has none,
so it is inferred as the *midpoint of the shoulders*, out in the middle of the
upper chest, where a rig's neck bone starts at the top of the thorax. Sliding a
rig joint onto either drags the body with it - the shoulder line up and
outward, the whole head down. `mesh_backend.LANDMARK_JOINTS` names them and
`landmark_shift` holds each at the offset the rig itself has at rest, measured
from the hip midpoint and rotated into a common frame on both sides, because
the rig's coordinates and the editor's share neither an origin nor a heading
and a raw difference of positions is dominated by that - which threw the
shoulders further out than leaving them alone did. Elbow, wrist, knee and ankle
keypoints *are* joint centres and must never be in that list.

**A landmark offset says where a joint sits, never how long the bone leaving
it is.** Two releases went out getting this wrong in opposite directions.
Applied whole against keypoints that hang the arm from the acromion, the
shoulder's offset moves the joint 6 cm down the arm while the elbow keypoint
stays put, so the humerus spans a gap 6 cm shorter than it is: 18% of an
adult's upper arm and 54% of the child's. Masking the offset along the bone
keeps the humerus and throws away the correction that lowers the shoulder
line, so every figure stands with its shoulders round its ears. Neither is a
fix, because both are working around keypoints that are wrong.

**So the arm hangs from the joint.** `derive_proportions` closes the arm on the
measured span from `gh_w`, the glenohumeral half-width, and splits it by
`arm_split` taken from the rigs - not from the acromion with ANSUR's
acromion-radiale ratio, which starts at a bony corner 0.022 of stature above
and 0.009 inboard of the joint and overstates the humerus by the difference.
Measured from the joint it closes to a tenth of a percent: gh_w + humerus +
forearm + hand is 0.5172 of stature against a measured half-span of 0.5165 on
the adult male rig. `build_rest_points` then carries `l_gh`/`r_gh` and
`neck_joint` alongside the eighteen keypoints - not keypoints, OpenPose has no
such thing, but the only place that knows where those joints are - and
`mesh_backend.LANDMARK_SOURCE` reads them. The offset is then that figure's own
anthropometry, so moving the rig's shoulder onto it lands the joint exactly a
humerus away from the elbow keypoint and nothing is stretched to reach.

Derived from the rig instead, as `landmark_shift` still must for a rig whose
figure says nothing, the offset came out 6.5 cm where the anthropometry says
3.9: the extra 2.6 is whatever the rig and the preset disagree about between
the hip and the shoulder, and every centimetre of it lands on the humerus.

With both declared, the shoulder line sits within 0.007 of the body's height of
where the rig author put it - against 0.027 before any of this - every figure
finishes within 1.5 cm of its stature, and no arm bone is off the rig's own by
more than 7%.

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
been aimed. The per-segment scaling then spent six passes dragging the limbs
back down, and what it could not reach stayed inflated: shoulders 4 cm high and
4.5 cm broad on an average man, 5.6 and 6.4 on a woman, and a child 8 cm too
tall with its shoulders 21 cm high and half again too wide. `solve_pose` takes
`stature` and sizes the rig by it against its own rest height; a body set built
for these presets is authored at the figure's stature, so the rig is left the
size it was drawn. Without one it falls back to the span, which is all an
outside rig can offer.

Two things say the sizing stayed fixed. The fit is *exact* - every mapped joint
lands on the point it was sent to and every segment matches to 0.00 cm, where
the best before was 1.2 cm - so the sentinel in that test starts at -1.0, or a
perfect fit reports as "none measured". And the rigged body now fills the same
frame as the swept anatomy to within a tenth. The old detail test passed on the
strength of the bug: a body a sixth too big for its own skeleton covers more of
the picture and carries more edge with it, so normalise that measure per
covered pixel or it is measuring size, not surface.

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

**A rig is fitted to the figure segment by segment, never dragged onto it.**
One number - the ratio of the two shoulder-to-hip spans - cannot match a torso
and the limbs hanging off it unless the two bodies have the same proportions.
Against the editor's presets the same rig came out with a forearm 30% long on
an average man and 54% on the child, whose thigh and shin were 58% and 61%
over. Sliding each joint onto its keypoint and carrying its subtree, which is
what closed that, puts the *joint* right and leaves the geometry between at
the rig's own length: a 42 cm shin pulled onto a 26 cm gap overshoots the
ankle, and the child came out with bowed shins and its feet hanging off them.
Scale is what was missing. `skin_mesh` builds a general 4x4 out of Q, so a
uniform scale per bone shortens a segment's geometry along with its length -
a shorter shin is a thinner shin, which is what a child's is - and the mesh is
never torn. Four things it took to work:

- the scale goes into `local`, not `Q`: `propagate` rebuilds every Q from
  local and wipes anything written to Q directly;
- `local` composes down the tree, so scaling the top of a chain scales
  everything hanging below it - the whole leg for a thigh, the whole body for
  the spine. Divide it back out at every branch off the chain, so a hand keeps
  the body's scale rather than its forearm's correction;
- scaling moves joints, which leaves every aimed direction stale: a pelvis
  pulled in carries the thigh with it and the knee ends up inboard of the
  keypoint the thigh was pointed at. Aim again each pass and the two converge;
- and measure the target from the *rig's own parent joint*, not from the
  keypoint standing in for it. Aiming has already put the child on the ray
  from that joint towards the keypoint, so scaling by that ratio lands it
  exactly - whereas aiming a shoulder from the collar and then scaling it from
  the neck are two constraints that place it nowhere in particular.

Once Q can carry a scale, `aim` has to take the rotation out of it before
using the transpose as an inverse, or every re-aim multiplies the limb by the
square of its scale. Worst joint error across the nine bodies went from
30 cm to 1.2 cm, and the worst limb pinch from 14% to 7%.

**A bone keeps its shape, not its size.** The self-test used to assert that
distances inside one bone's vertex set were preserved exactly, and that check
had to go with the above - a scaled segment is a scaled segment. What must not
change is the *shape*: a uniform scale leaves every ratio of distances alone,
while shear, a torn joint, or a limb dragged onto its keypoint do not.

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
5. The garment library covers hair, tops, bottoms and shoes; `over`
   (apron, cape, backpack, scarf) and most headgear have no mesh behind them
   yet, so those slots are silently not worn on a rigged export. The packs
   have hats and more besides - `tools/make_wearables.py --list` shows the
   fifty-odd available - it is a matter of building and vendoring them.
5. The web prototype duplicates the maths in JavaScript. It is tested
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
