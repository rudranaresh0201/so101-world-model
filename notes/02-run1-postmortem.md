# Run 1 post-mortem (reach, 60k SAC steps): 0% success

| steps | success | knocked | median final distance |
|---|---|---|---|
| random | 0% | 0% | 28.0 cm |
| 10k | 0% | 43% | 16.3 cm |
| 20k | 0% | 3% | 17.8 cm |
| 30k | 0% | 17% | 12.8 cm |
| 40k | 0% | 27% | 14.2 cm |
| 50k | 0% | 30% | 13.2 cm |
| 60k | 0% | 3% | 21.5 cm |

**What happened:** it learned to approach the box (28 → about 13 cm), then oscillated between rushing
in (knocking the box) and hanging back (hovering). It never succeeded.

**Root cause:** a task-design bug. Success meant "claw tip within 2 cm of the box CENTRE". The box is
4x4x6 cm, so its centre is 2 cm inside each side face and 3 cm below the top. From outside the box
that's unreachable. The only pose that achieves it is the jaws AROUND the box (the scene's `pickup`
keyframe: 1.8 cm), i.e. the hard part of grasping. Getting that close also risks a knock, which ends
the episode. So hovering was the rational choice.

**Lesson:** a checker can prove the *code* matches the design but not that the *task* is achievable.
Before training, check that the success condition can physically be met.

**Fix for run 2 (one change only):** target a pre-grasp point 2 cm above the box's top face.
