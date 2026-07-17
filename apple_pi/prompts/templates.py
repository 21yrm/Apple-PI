from typing import Dict
from apple_pi.types import Subtrack

# Video Generation Model Prompt Templates
VM_PROMPT_TEMPLATES: Dict[Subtrack, str] = {
    Subtrack.PERCEPTION_TEXT: """\
Generate a video where the background, ground, sky, and all physical \
objects gradually fade to pure white, while all human-drawn annotations \
(text labels, coordinate axes, velocity arrows) remain completely \
unchanged. Static camera, no zoom, no rotation. The final frame should \
show only the original annotations on white — crisp, sharp, and \
readable. Do not add any extra annotations or elements not in the \
original.""",

    Subtrack.PERCEPTION_GRAPHIC: """\
Generate a video where the background, ground, sky, and all \
annotations (text labels, coordinate axes, velocity arrows) gradually \
fade to pure white, while the subject physical objects remain still \
and unchanged. Static camera, no zoom, no rotation. The final frame \
should show only the objects with labeled physical properties on \
white — remove supporting structures (ramps, tracks, rails, \
platforms). Do not transform, resize, or replace any object. Each \
object must retain its exact position, size, shape, color, and texture.""",

    Subtrack.FORMULATION_TEXT: """\
This physics infographic shows annotated physical parameters describing \
a specific type of motion. Below are 4 candidate formulas. Exactly ONE \
is correct:
{formula_choices}

Generate a video ending on a white background with THREE lines of \
large, clearly readable, centered text:
  Line 1: the correct option letter (A/B/C/D).
  Line 2: the correct formula exactly as written above.
  Line 3: the formula with annotated values substituted in.

When substituting: if multiple objects exist, pick one consistent set.""",

    Subtrack.FORMULATION_GRAPHIC: """\
Predict the physical state at t = {target_time}s. Generate a video where \
the original annotations (text labels, coordinate axes, arrows) gradually \
fade out and cleanly disappear, while the background, ground, sky, and \
all scene lighting remain completely unchanged throughout. Objects move \
smoothly to their correct positions at t = {target_time}s — each object \
must retain its exact texture, material, size, and shape. \
For each moving object, draw a borderless orange arrow starting from \
the object's center pointing in its velocity direction, with a label \
"v = X.XX" (speed value, no unit). If at rest, show "v = 0" with no \
arrow. Static camera, no zoom, no rotation.""",

    Subtrack.DEDUCTION: """\
The first frame is an annotated physics scene. All annotations follow \
SI units: g is gravitational acceleration, e is restitution (0-1), \
v0 is initial velocity (arrows show direction), mu is friction \
coefficient. Generate {physics_duration} seconds of physically \
accurate motion strictly following these parameters. Every object AND \
the ground surface are perfectly rigid — nothing breaks, cracks, \
deforms, or creates craters. Each object maintains its exact material, \
color, and texture throughout. All annotations must cleanly disappear \
after the first frame. Static camera, fixed background. Smooth, \
continuous motion.""",
}

# Unified Model Prompt Templates
UM_PROMPT_TEMPLATES: Dict[Subtrack, str] = {
    Subtrack.PERCEPTION_TEXT: """\
Given this physics infographic, generate a new image that shows ONLY \
the human-drawn annotations on a pure white background. Keep all text \
labels, coordinate axes, and velocity arrows exactly as they appear — \
same position, content, color, font, and size. Text must be crisp and \
readable. Remove everything else: the background, ground, sky, and all \
physical objects. Do not add any extra elements not in the original.""",

    Subtrack.PERCEPTION_GRAPHIC: """\
Show only the subject physical objects from this image on a pure white \
background. Remove all annotations (text labels, coordinate axes, \
velocity arrows), the ground surface, sky, and supporting structures \
(ramps, tracks, rails, platforms). Keep only the objects that have \
labeled physical properties. Do not transform, resize, or replace any \
object — a cube must stay a cube, a sphere must stay a sphere. Each \
object must retain its exact position, size, shape, color, and texture.""",

    Subtrack.FORMULATION_TEXT: """\
This physics infographic shows a scene with annotated physical \
parameters. Below are 4 candidate formulas. Exactly ONE is the correct \
governing equation for the motion shown:
{formula_choices}

Generate an image with a pure white background containing THREE lines \
of large, clearly readable, centered text:
  Line 1: The correct option letter (A/B/C/D).
  Line 2: The correct formula exactly as written above.
  Line 3: The formula with annotated variables replaced by their numeric \
values from the image, keeping unannotated symbols unchanged.

When substituting: if multiple objects exist, pick exactly one \
consistent set of values.""",

    Subtrack.FORMULATION_GRAPHIC: """\
Predict the physical state at t = {target_time}s. Generate an image\x20
showing the scene at that instant, where the original annotations \
(text labels, coordinate axes, arrows) have been cleanly removed while \
the background, ground, sky, and all scene lighting remain completely \
unchanged from the input. Show each object at its correct position at \
t = {target_time}s — each object must retain its exact texture, \
material, size, and shape. For each moving object, overlay a borderless \
orange arrow starting from the object's center pointing in its \
instantaneous velocity direction, with a label "v = X.XX" (speed value, \
no unit). If an object is at rest, show "v = 0" with no arrow. Keep \
the same viewpoint as the input image — no zoom, no rotation, no crop.""",

    Subtrack.DEDUCTION: """\
This image is an annotated physics scene. All annotations follow SI \
units: g is gravitational acceleration, e is restitution (0-1), v0 is \
initial velocity (arrows show direction), mu is friction coefficient. \
Generate what the scene looks like at t = {time_point} seconds after \
motion begins, following these physical parameters accurately. Remove \
all annotations (text, axes, arrows). Objects are rigid bodies — same \
shape, size, color, texture throughout. Static camera, fixed background \
and ground.""",
}
