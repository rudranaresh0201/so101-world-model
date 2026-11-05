# Run 2 post-mortem (reach, pre-grasp target, 60k SAC steps): at most 13% success

| steps | success | knocked | median final distance |
|---|---|---|---|
| random | 0% | 0% | 26.1 cm |
| 10k | 3% | 7% | 12.6 cm |
| 20k | 0% | 57% | 13.6 cm |
| 30k | 7% | 0% | 14.0 cm |
| 40k | 13% | 7% | 20.8 cm |
| 50k | 0% | 23% | 12.5 cm |
| 60k | 3% | 30% | 10.6 cm |

**What happened:** the first successes appeared (so the target fix was needed), but learning was
unstable and the video shows the arm curling back toward its base.

**Root cause:** the motors are weak (forcerange ±2.94 N·m vs gravity). Under repeated delta actions the
target wound up to 1.46 rad (84°) ahead of the real joint. The policy only observed the real angles,
not the targets, so identical observations hid very different targets. **The env wasn't Markov.**

**Fix for run 3:** (1) add the 6 motor targets (`data.ctrl`) to the observation (21 → 27 numbers);
(2) anti-wind-up: a target may lead the real joint by at most 0.15 rad.
