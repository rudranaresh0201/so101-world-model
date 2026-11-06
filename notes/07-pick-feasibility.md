# Pick feasibility (scripted, no learning)

- Position-only IK reaching above the box: **0% lifted**. The gripper arrives tilted and the fixed jaw lands on the box's top face.
- Site frame: fingers point along local +x, jaws close along local z (fixed jaw on -z), gripper joint axis = local y.
- 6-D damped-least-squares IK (position + "fingers down, jaws close along world x"), grasp 1 cm above box centre: **95% lifted > 5 cm** (20 boxes).
- Video: outputs/pick_oracle.mp4. The grasp is physically possible with this gripper and box (4 cm, 96 g).
- Implication for RL: the policy must learn wrist orientation, not just claw position. The reach target ignored orientation.
