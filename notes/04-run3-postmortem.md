# Run 3 post-mortem: 0% success, so is the task even possible?

Runs 1–3 all failed, so we stopped guessing and tested **feasibility** first: a scripted controller
(damped least-squares inverse kinematics via `mj_jacSite`) using the exact same action interface
as the policy.

    scripted IK oracle: success 100%  knocked 0%  median final 1.5 cm

So the robot, the motors and the action interface are fine. The problem is the **incentive**.

**Root cause (present in runs 1–3):** reward = 1 − tanh(5d) is always positive, and success
*terminated* the episode. Ending forfeits all remaining reward, so crossing the 2 cm line cost about
0.9/step × the remaining steps. The rational policy hovers just outside the line. (Same family as the
"knock the box off to stop the negative reward" bug: **reward and ending rules interact.**)

**Fix for run 4:** success no longer ends the episode; it's measured at the final step.
Only a knocked box terminates.

**Process lesson:** when RL fails, first check (1) that an oracle can solve the task through the
same interface, then (2) the incentives. Only then tune the algorithm.
