# Ablation after run 4: why can't SAC learn reach? (60k steps, seed 1, 30 unseen eval boxes)

| Run | norm obs | knock ends episode | success @60k | median final dist | trend |
|---|---|---|---|---|---|
| A control | no | yes | 0% | 9.1 cm | flat |
| B | yes | yes | 0% | 12.2 cm | flat |
| C | no | no | 10% | 4.6 cm | improves, then plateaus |
| **D** | **yes** | **no** | **40%** | **2.4 cm** | **still climbing** |

**Main blocker:** terminating on a knock. Every close approach risked ending the episode and losing all
the remaining reward, so the policy never practised the final centimetres. (Third case of
"ending rules interact with the reward".)

**Normalisation:** no effect alone, but needed for stable progress once knocks no longer end episodes.
Raw observations mix joint speeds (~±10 rad/s) with distances (~0.01 m), so without it the network
barely "sees" the small numbers that matter most.

**Evaluation still ends on knocks** in every run, so the success numbers are comparable.

**Next:** D for 200k steps × 2 seeds, to check it converges and isn't luck.
