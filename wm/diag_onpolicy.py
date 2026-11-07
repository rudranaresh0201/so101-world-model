import sys, pickle; sys.path.insert(0, '.')
import numpy as np, torch
from wm.dream_sac import load_wm, ROOT
from envs.so101_pick import SO101PickEnv
from wm.reward import pick_reward
from stable_baselines3 import SAC
torch.set_num_threads(2)
wm = load_wm(ROOT / 'outputs/wm_v1.pt')
D = np.load(ROOT / 'data/pick_v1.npz')
def load(n): return SAC.load(ROOT / f'outputs/{n}.zip', device='cpu'), pickle.load(open(ROOT / f'outputs/{n}_vecnorm.pkl', 'rb'))
groups = {'q': slice(0, 6), 'qvel': slice(6, 12), 'claw': slice(12, 15), 'box': slice(15, 18), 'ctrl': slice(21, 27), 'flags': slice(39, 41)}
for name in sys.argv[1:] or ['dream_v1', 'pickv2_fresh']:
    m, vn = load(name); env = SO101PickEnv()
    err, dis, acts, rr, rd, dist = [], [], [], [], [], []
    for i in range(10):
        o, _ = env.reset(seed=10_000 + i)
        for t in range(150):
            a = m.predict(vn.normalize_obs(o), deterministic=True)[0]
            P = wm.predict(torch.tensor(o, dtype=torch.float32)[None], torch.tensor(a, dtype=torch.float32)[None])[:, 0].numpy()
            o2, r, *_ , info = env.step(a)
            nd = (P.mean(0) - o2) / wm.d_sd.numpy()          # error in units of typical step-change
            err.append(nd ** 2); dis.append(P.std(0) / wm.d_sd.numpy()); acts.append(a)
            rr.append(r); rd.append(pick_reward(P.mean(0))[0]); dist.append(info['distance']); o = o2
    err, dis, acts = np.array(err), np.array(dis), np.array(acts)
    print(f'== {name}: 1-step model error ON THIS POLICY\'S real states (val data was ~0.02-0.12, flags 0.47)')
    print('   err  ', '  '.join(f'{k} {err[:, s].mean():.2f}' for k, s in groups.items()))
    print('   disagreement', '  '.join(f'{k} {dis[:, s].mean():.2f}' for k, s in groups.items()))
    print(f'   |action| mean {np.abs(acts).mean():.2f}, frac saturated(|a|>0.95) {(np.abs(acts) > .95).mean():.2f}   per-joint mean a {np.round(acts.mean(0), 2)}')
    print(f'   real 1-step reward {np.mean(rr):.2f} vs model-predicted {np.mean(rd):.2f};  claw->box distance: start {np.mean(dist[0::150])*100:.1f}cm  end {np.mean(dist[149::150])*100:.1f}cm  min {np.min(dist)*100:.1f}cm')
a = D['act']; print(f'dataset: |action| mean {np.abs(a).mean():.2f}, saturated {(np.abs(a) > .95).mean():.2f}')
