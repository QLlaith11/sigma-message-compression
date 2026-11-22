import os, sys, glob, time, types, random, zipfile, subprocess, warnings, math
warnings.filterwarnings("ignore")

SMOKE          = False
PROBE_K, PROBE_BITS, PROBE_SIDE = 1, 2, True
PROBE_SETTINGS = [(40, 32), (40, 64)]
EPISODES       = 20
MAX_STEPS      = 300
SEED           = 0

REPO_URL = "https://github.com/marmotlab/SIGMA.git"
ROOT     = "/content/SIGMA"
CKPT     = "models/save_model/model_house/84000_house.pth"
OUT_DIR  = "/content/sigma_compressors"
if SMOKE:
    PROBE_SETTINGS, EPISODES, MAX_STEPS = [(40, 8)], 2, 50

def find_root(base):
    for p in glob.glob(os.path.join(base, "**", "model.py"), recursive=True):
        d = os.path.dirname(p)
        if os.path.exists(os.path.join(d, "configs.py")):
            return d
    return None

if not os.path.exists(os.path.join(ROOT, "model.py")):
    r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, ROOT], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(os.path.join(ROOT, "model.py")):
        print("git clone failed -> upload the repo zip")
        from google.colab import files
        up = files.upload(); zname = next(iter(up)); out = "/content/_sigma_zip"
        with zipfile.ZipFile(zname) as z:
            z.extractall(out)
        ROOT = find_root(out); assert ROOT, "model.py/configs.py not found in zip"
print("repo root:", ROOT)

_envpy = os.path.join(ROOT, "environment.py")
_src = open(_envpy).read()
if "if target_agent_id:" in _src:
    open(_envpy, "w").write(_src.replace("if target_agent_id:", "if target_agent_id.size > 0 and bool(target_agent_id.flat[0]):"))
    sys.modules.pop("environment", None)
    print("patched environment.py for NumPy >= 2.2 (behaviour unchanged)")

import numpy as np
for _n, _t in (("int", int), ("bool", bool), ("float", float)):
    try:
        getattr(np, _n)
    except AttributeError:
        setattr(np, _n, _t)
try:
    import skimage
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-image"])
try:
    import wandb
except Exception:
    sys.modules["wandb"] = types.ModuleType("wandb")

import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm

os.chdir(ROOT); sys.path.insert(0, ROOT)
import configs
from model import Network
from environment import Environment
assert configs.test_senario == "house"

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("device:", DEV, "| torch", torch.__version__, "| numpy", np.__version__)
assert PROBE_BITS is not None, "the probe needs an integer-bit compressor (symbols)"

R = configs.obs_radius; G = 2 * R + 1; NC = G * G
H = configs.hidden_dim; KCOMM = configs.max_comm_agents; K1 = KCOMM - 1

teacher = Network()
teacher.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
teacher = teacher.eval().to(DEV)
for p_ in teacher.parameters():
    p_.requires_grad_(False)

def neighbors_from_pos(pos_t):
    n = pos_t.size(0)
    d = pos_t.unsqueeze(0) - pos_t.unsqueeze(1)
    ok = (d.abs().max(-1)[0] <= R) & ~torch.eye(n, dtype=torch.bool, device=DEV)
    dist = d.float().pow(2).sum(-1).sqrt()
    score = torch.where(ok, -dist, torch.full_like(dist, -1e9))
    kk = min(K1, n)
    val, idx = score.topk(kk, dim=1)
    valid = val > -1e8
    if kk < K1:
        idx = torch.cat([idx, torch.zeros(n, K1 - kk, dtype=idx.dtype, device=DEV)], 1)
        valid = torch.cat([valid, torch.zeros(n, K1 - kk, dtype=torch.bool, device=DEV)], 1)
    off = pos_t[idx] - pos_t.unsqueeze(1)
    cell = ((off[..., 0] + R) * G + (off[..., 1] + R)).clamp(0, NC - 1)
    return idx, valid, cell

class Compressor(nn.Module):
    def __init__(self, k, bits, side):
        super().__init__()
        self.k, self.bits, self.side = k, bits, side
        self.enc = nn.ModuleList([nn.Sequential(nn.Linear(H, 256), nn.ReLU(), nn.Linear(256, k)) if k > 0
                                  else nn.Identity() for _ in range(2)])
        self.cell_emb = nn.ModuleList([nn.Embedding(NC, 32) for _ in range(2)])
        din = k + 32 + (H + 1 if side else 0)
        self.dec = nn.ModuleList([nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                                nn.Linear(256, H)) for _ in range(2)])
    def encode(self, l, x):
        if self.k == 0:
            return x.new_zeros(x.size(0), 0)
        z = torch.tanh(self.enc[l](x))
        if self.bits is not None:
            levels = 2 ** self.bits
            q = torch.round((z + 1) / 2 * (levels - 1)) / (levels - 1) * 2 - 1
            z = z + (q - z).detach()
        return z
    def decode(self, l, z_nb, x, m, cell):
        n, k1, _ = z_nb.shape
        parts = [z_nb, self.cell_emb[l](cell)]
        if self.side:
            parts.append(torch.cat([x, m], -1).unsqueeze(1).expand(-1, k1, -1))
        return self.dec[l](torch.cat(parts, -1))

path = f"{OUT_DIR}/learned_k{PROBE_K}_b{PROBE_BITS}_s{int(PROBE_SIDE)}.pt"
if SMOKE and not os.path.exists(path):
    print("SMOKE: no saved compressor found, using an untrained one just to test the code path")
    comp = Compressor(PROBE_K, PROBE_BITS, PROBE_SIDE).to(DEV).eval()
else:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - train it with sigma_learned_compressor_colab.py first "
                                f"(config ({PROBE_K}, {PROBE_BITS}, {PROBE_SIDE})), or set OUT_DIR.")
    comp = Compressor(PROBE_K, PROBE_BITS, PROBE_SIDE).to(DEV)
    comp.load_state_dict(torch.load(path, map_location=DEV)); comp.eval()
print("loaded", path)

LEVELS = 2 ** PROBE_BITS
@torch.no_grad()
def step_with_codes(obs_np, pos_np, hidden):
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=DEV); n = obs.size(0)
    lat = teacher.obs_encoder(obs); m = teacher.mlp(lat)
    h_pre = teacher.recurrent(lat) if hidden is None else teacher.recurrent(lat, hidden)
    pos = torch.as_tensor(pos_np, device=DEV).long()
    j, valid, cell = neighbors_from_pos(pos)
    has = valid.any(-1)
    x, syms = h_pre, []
    if bool(has.any()):
        tv = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=DEV), valid], 1)
        attn_mask = (~tv).unsqueeze(1).expand(n, 1 + K1, 1 + K1)
        for l in range(2):
            z = comp.encode(l, x)
            syms.append(torch.round((z[:, 0] + 1) / 2 * (LEVELS - 1)).long().clamp(0, LEVELS - 1))
            tok = comp.decode(l, z[j], x, m, cell)
            info = teacher.comm.self_attn(torch.cat([x.unsqueeze(1), tok], 1), attn_mask=attn_mask)
            new = teacher.comm.update_cell(info[:, 0], x)
            x = torch.where(has.unsqueeze(-1), new, x)
    comb = torch.cat((x, m), dim=1)
    adv = teacher.adv(comb) if configs.Advantage_all else teacher.adv(x)
    q = teacher.state(x) + adv - adv.mean(1, keepdim=True)
    return q.argmax(1).tolist(), x, syms, j, valid

agent_rows, pair_rows = [], []
for (L, N) in PROBE_SETTINGS:
    for e in tqdm(range(EPISODES), desc=f"{L}x{L} {N} agents", leave=False):
        env = Environment(); obs, pos = env.reset(num_agents=N, map_length=L)
        hid, done = None, False
        while not done and env.steps < MAX_STEPS:
            actions, hid, syms, j, valid = step_with_codes(obs, pos, hid)
            old = env.agents_pos.copy(); goals = env.goals_pos.copy()
            (obs, pos), _, done, _ = env.step(actions)
            if not syms:
                continue
            new = env.agents_pos.copy()
            disp = new - old
            moved = (disp != 0).any(1).astype(int)
            gd = goals - old
            adx, ady = np.abs(gd[:, 0]), np.abs(gd[:, 1])
            goal_dir = np.where((adx + ady) == 0, 0, np.where(adx >= ady, np.where(gd[:, 0] > 0, 1, 2),
                                                               np.where(gd[:, 1] > 0, 3, 4)))
            dist = adx + ady
            dist_bin = np.digitize(dist, [1, 4, 11, 26])
            jn, vn = j.cpu().numpy(), valid.cpu().numpy()
            n_nb = np.minimum(vn.sum(1), 4)
            heard = np.bincount(jn[vn].ravel(), minlength=len(old)) > 0
            s0, s1 = syms[0].cpu().numpy(), syms[1].cpu().numpy()
            act = np.asarray(actions)
            for a in np.nonzero(heard)[0]:
                agent_rows.append((N, s0[a], s1[a], act[a], goal_dir[a], dist_bin[a], n_nb[a], moved[a]))
            rec, snd = np.nonzero(vn)
            snd_idx = jn[rec, snd]
            rel = old[rec] - old[snd_idx]
            toward = (disp[snd_idx] * rel).sum(1) > 0
            cls = np.where(moved[snd_idx] == 0, 0, np.where(toward, 1, 2))
            for sidx, c in zip(snd_idx, cls):
                pair_rows.append((N, s0[sidx], s1[sidx], c))

A = pd.DataFrame(agent_rows, columns=["agents", "sym0", "sym1", "action", "goal_dir", "dist_bin", "n_nb", "moved"])
P = pd.DataFrame(pair_rows, columns=["agents", "sym0", "sym1", "sender_motion"])
print(f"\nlogged {len(A)} heard sender-steps and {len(P)} sender->receiver pairs")

def entropy(a):
    _, c = np.unique(a, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

def mutual_info(a, b):
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((len(ua), len(ub))); np.add.at(joint, (ia, ib), 1); joint /= joint.sum()
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (pa @ pb)[nz])).sum())

NAMES = {"action": "chosen action (5)", "goal_dir": "goal direction (0=at goal, 4 dirs)", "dist_bin": "distance to goal (5 bins)",
         "n_nb": "visible neighbours (0-4+)", "moved": "moved this step (2)"}
for N in sorted(A.agents.unique()):
    a, p = A[A.agents == N], P[P.agents == N]
    print(f"\n================ {N} agents ================")
    for ex, col in ((0, "sym0"), (1, "sym1")):
        print(f"\n-- exchange {ex + 1}: symbol entropy H(S) = {entropy(a[col]):.3f} bits (nominal width {PROBE_BITS} bits)")
        rows = []
        for v, nm in NAMES.items():
            mi = mutual_info(a[col].values, a[v].values)
            rows.append(dict(variable=nm, MI_bits=round(mi, 3), MI_over_H_variable=round(mi / max(entropy(a[v]), 1e-9), 3)))
        mi = mutual_info(p[col].values, p.sender_motion.values)
        rows.append(dict(variable="sender motion vs receiver (stay / toward / other)", MI_bits=round(mi, 3),
                         MI_over_H_variable=round(mi / max(entropy(p.sender_motion), 1e-9), 3)))
        print(pd.DataFrame(rows).to_string(index=False))
    print("\nP(sender motion | symbol of exchange 1)   [0 = stays, 1 = moves toward receiver, 2 = moves elsewhere]")
    print(pd.crosstab(p.sym0, p.sender_motion, normalize="index").round(2).to_string())
    print("\nP(goal direction | symbol of exchange 1)  [0 = at goal, 1-4 = dominant-axis direction]")
    print(pd.crosstab(a.sym0, a.goal_dir, normalize="index").round(2).to_string())
    print("\nP(chosen action | symbol of exchange 1)")
    print(pd.crosstab(a.sym0, a.action, normalize="index").round(2).to_string())
A.to_csv("/content/sigma_probe_agents.csv", index=False); P.to_csv("/content/sigma_probe_pairs.csv", index=False)
print("\nsaved -> /content/sigma_probe_agents.csv, sigma_probe_pairs.csv")
print("Read: a high MI with 'sender motion vs receiver' means the message tells neighbours whether to expect the sender in their way; "
      "a high MI with goal direction or action means it broadcasts intent. H(S) below the nominal bit width means the "
      "channel could be entropy-coded to fewer bits.")
