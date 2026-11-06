# Pick v1 postmortem: "grasped" meant touching, not gripping

v1 reward: reach + 0.5*down + grasped*(1 + 3*lift), where grasped = both jaws in contact with the box.
After 225k steps (2 seeds): ever-grasped 95–100%, lift flat at about 0.4 cm, return about 2.1 per step.

Diagnosis (rollouts of the 225k policy):
- In half the episodes the gripper target equalled the jaw angle (ctrl ≈ q), so there was zero squeeze.
  The jaws were resting on the box, the reward paid full "grasp", and lifting would only risk losing it.
- In the other half it did squeeze (ctrl = q − 0.15) and the box rose 1–2 cm. Lifting was starting, but the
  reward gave little for it: holding the box 2 cm up paid about 3.0 per step, sitting about 2.4.

v2 fix:
- grasped = both jaws in contact AND squeeze (q − ctrl) > 0.05 rad, so the motor is actually pushing.
- grasped*(0.5 + 6*lift): sitting is worth about 2 per step, holding the box at 8 cm about 8 per step.
- Scripted oracle under v2: 7.0 per step at the end, 90% success; the same pose with no squeeze → not grasped.
Lesson: a reward term's NAME is not its definition. Check that the cheapest state satisfying it is the one you meant.
