import gc, os, sys, glob, time, types, pickle, random, zipfile, subprocess, warnings, math, copy
warnings.filterwarnings("ignore")

SMOKE          = False
CONFIGS        = [(1, 2, True), (2, 4, True)]
SHUFFLE_CONTROL = False
SEEDS          = [0, 1]
SETTINGS_CASES = {(40, 8): 100, (40, 16): 100, (40, 32): 100, (40, 64): 60, (40, 128): 30}
DATA_SETTINGS  = [(30, 16), (40, 16), (40, 32), (40, 32), (40, 48), (40, 64), (40, 64), (40, 96)]
TEACHER_EPS    = 40
DATA_MAX_STEPS = 300
STRIDE         = 2
UPD_STAGE1     = 1500
DAGGER_ROUNDS  = 1
DAGGER_EPS     = 24
UPD_DAGGER     = 1000
B_STEPS        = 8
LR             = 1e-3
TAU            = 0.2
W_KL, W_ADV, W_REC = 1.0, 1.0, 0.1
INCLUDE_REFERENCE = True
RESUME         = True
SEED           = 0

REPO_URL = "https://github.com/marmotlab/SIGMA.git"
ROOT     = "/content/SIGMA"
CKPT     = "models/save_model/model_house/84000_house.pth"
TEST_DIR = "house_test_set"
OUT_DIR  = "/content/sigma_compressors"
if SMOKE:
    CONFIGS, TEACHER_EPS, UPD_STAGE1, DAGGER_ROUNDS, DAGGER_EPS, UPD_DAGGER = [(0, None, True), (2, 4, True)], 4, 20, 1, 2, 10
    DATA_SETTINGS, SETTINGS_CASES, B_STEPS, RESUME, SEEDS = [(40, 8)], {(40, 8): 2}, 4, False, [0]

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
os.makedirs(OUT_DIR, exist_ok=True)
import configs
from model import Network
from environment import Environment
assert configs.test_senario == "house"

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("device:", DEV, "| torch", torch.__version__, "| numpy", np.__version__)

R = configs.obs_radius; G = 2 * R + 1; NC = G * G
H = configs.hidden_dim; KCOMM = configs.max_comm_agents; K1 = KCOMM - 1

teacher = Network()
teacher.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
teacher = teacher.eval().to(DEV)
for p_ in teacher.parameters():
    p_.requires_grad_(False)
assert teacher.comm.num_layers == 2

@torch.no_grad()
def team_step_layers(net, obs_np, pos_np, hidden, use_comm=True):
    obs = torch.as_tensor(obs_np.astype(np.float32), device=DEV)
    pos = torch.as_tensor(pos_np.astype(np.float32), device=DEV)
    n = obs.size(0)
    latent = net.obs_encoder(obs)
    m = net.mlp(latent)
    h_pre = net.recurrent(latent) if hidden is None else net.recurrent(latent, hidden)
    pos_mat = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
    dist = (pos_mat[..., 0] ** 2 + pos_mat[..., 1] ** 2).sqrt()
    in_obs = (pos_mat <= R).all(2)
    _, rank = dist.topk(min(KCOMM, n), dim=1, largest=False)
    near = torch.zeros((n, n), dtype=torch.bool, device=DEV)
    near.scatter_(1, rank, True)
    comm_mask = in_obs & near
    xs, x = [h_pre], h_pre
    L = net.comm.num_layers
    upd = comm_mask.sum(-1) > 1
    if use_comm and bool(upd.any()):
        xb, attn_mask = x.unsqueeze(0), ~comm_mask
        for _ in range(L):
            info = net.comm.self_attn(xb, attn_mask=attn_mask)
            new = net.comm.update_cell(info[0], xb[0])
            xb = torch.where(upd[None, :, None], new.unsqueeze(0), xb)
            xs.append(xb[0])
        x = xb[0]
    else:
        xs += [x] * L
    comb = torch.cat((x, m), dim=1)
    adv = net.adv(comb) if configs.Advantage_all else net.adv(x)
    q = net.state(x) + adv - adv.mean(1, keepdim=True)
    return q.argmax(1).tolist(), q, xs, x, m

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

def compressed_team_forward(comp, h_pre, m, j, valid, cell, shuffle=False):
    n = h_pre.size(0)
    has = valid.any(-1)
    x, rec = h_pre, h_pre.new_zeros(())
    if bool(has.any()):
        tv = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=h_pre.device), valid], 1)
        attn_mask = (~tv).unsqueeze(1).expand(n, 1 + K1, 1 + K1)
        for l in range(teacher.comm.num_layers):
            z = comp.encode(l, x)
            jj = j[torch.randperm(n, device=j.device)] if shuffle else j
            tok = comp.decode(l, z[jj], x, m, cell)
            rec = rec + F.mse_loss(tok[valid], x[j][valid].detach())
            info = teacher.comm.self_attn(torch.cat([x.unsqueeze(1), tok], 1), attn_mask=attn_mask)
            new = teacher.comm.update_cell(info[:, 0], x)
            x = torch.where(has.unsqueeze(-1), new, x)
    comb = torch.cat((x, m), dim=1)
    adv = teacher.adv(comb) if configs.Advantage_all else teacher.adv(x)
    q = teacher.state(x) + adv - adv.mean(1, keepdim=True)
    return q, x, rec

def collect(replay, n_eps, acting, comp=None):
    for e in tqdm(range(n_eps), desc=f"collect ({acting})", leave=False):
        L, n = DATA_SETTINGS[e % len(DATA_SETTINGS)]
        env = Environment(); obs, pos = env.reset(num_agents=n, map_length=L)
        t_hid = s_hid = None; done = False
        while not done and env.steps < DATA_MAX_STEPS:
            a_t, q_t, xs, post_t, m_t = team_step_layers(teacher, obs, pos, t_hid)
            t_hid = post_t
            if acting == "teacher":
                actions, h_in, m_in = a_t, xs[0], m_t
            else:
                with torch.no_grad():
                    ob = torch.as_tensor(obs, dtype=torch.float32, device=DEV)
                    lat = teacher.obs_encoder(ob); m_in = teacher.mlp(lat)
                    h_in = teacher.recurrent(lat) if s_hid is None else teacher.recurrent(lat, s_hid)
                    pos_t = torch.as_tensor(pos, device=DEV).long()
                    j, valid, cell = neighbors_from_pos(pos_t)
                    q_s, s_hid, _ = compressed_team_forward(comp, h_in, m_in, j, valid, cell)
                    actions = q_s.argmax(1).tolist()
            if env.steps % STRIDE == 0:
                replay.append((h_in.half().cpu(), m_in.half().cpu(),
                               torch.as_tensor(pos).short().cpu(), q_t.half().cpu()))
            (obs, pos), _, done, _ = env.step(actions)
    return replay

def sample_loss(comp, replay):
    tot = dict(kl=0.0, adv=0.0, rec=0.0)
    loss = 0.0
    for idx in np.random.randint(len(replay), size=B_STEPS):
        h0, m0, pos, qT = [t.to(DEV).float() if i != 2 else t.to(DEV).long() for i, t in enumerate(replay[idx])]
        j, valid, cell = neighbors_from_pos(pos)
        q, _, rec = compressed_team_forward(comp, h0, m0, j, valid, cell)
        kl = F.kl_div(F.log_softmax(q / TAU, -1), F.softmax(qT / TAU, -1), reduction="batchmean")
        l_adv = F.smooth_l1_loss(q - q.mean(1, keepdim=True), qT - qT.mean(1, keepdim=True))
        loss = loss + (W_KL * kl + W_ADV * l_adv + W_REC * rec) / B_STEPS
        tot["kl"] += kl.item() / B_STEPS; tot["adv"] += l_adv.item() / B_STEPS; tot["rec"] += float(rec) / B_STEPS
    return loss, tot

def train(comp, replay, updates, lr, tag):
    opt = torch.optim.Adam(comp.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(updates, 1), eta_min=lr * 0.1)
    t0 = time.time()
    for u in range(1, updates + 1):
        loss, tot = sample_loss(comp, replay)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(comp.parameters(), 5); opt.step(); sched.step()
        if u % 250 == 0 or u == updates:
            print(f"[{tag} {u}/{updates}] " + " ".join(f"{k} {v:.4f}" for k, v in tot.items()) + f" | {time.time()-t0:.0f}s")

def compressed_policy(comp, shuffle=False):
    @torch.no_grad()
    def f(obs_np, pos_np, hid):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=DEV)
        lat = teacher.obs_encoder(obs); m = teacher.mlp(lat)
        h_pre = teacher.recurrent(lat) if hid is None else teacher.recurrent(lat, hid)
        pos = torch.as_tensor(pos_np, device=DEV).long()
        j, valid, cell = neighbors_from_pos(pos)
        q, x, _ = compressed_team_forward(comp, h_pre, m, j, valid, cell, shuffle)
        return q.argmax(1).tolist(), x
    return f

def teacher_policy(use_comm):
    def f(obs, pos, hid):
        a, _, _, post, _ = team_step_layers(teacher, obs, pos, hid, use_comm); return a, post
    return f

def run_case(case, act_fn):
    env = Environment(); env.load(case[0], case[1], case[2])
    obs, pos = env.observe(); hid, done = None, False
    while not done and env.steps < configs.max_episode_length:
        actions, hid = act_fn(obs, pos, hid)
        (obs, pos), _, done, _ = env.step(actions)
    return (np.array_equal(env.agents_pos, env.goals_pos), env.steps,
            float((env.agents_pos == env.goals_pos).all(1).mean()))

def evaluate(policies, plan):
    rows = []
    for (L, N), n_cases in plan.items():
        path = f"{TEST_DIR}/{L}length_{N}agents_0.3density.pth"
        if not os.path.exists(path):
            print("missing test set (skipped):", path); continue
        with open(path, "rb") as fh:
            tests = pickle.load(fh)[:n_cases]
        for name, fn in policies.items():
            res = [run_case(c, fn) for c in tqdm(tests, desc=f"{L}x{L} {N:>3} | {name}", leave=False)]
            rows.append(dict(agents=N, policy=name, success_pct=100 * np.mean([r[0] for r in res]),
                             avg_steps=np.mean([r[1] for r in res]), arrival_pct=100 * np.mean([r[2] for r in res]),
                             n_cases=len(tests)))
            print(f"{L}x{L} {N:>3} | {name:<30} success {rows[-1]['success_pct']:5.1f}% | steps {rows[-1]['avg_steps']:6.1f} "
                  f"| arrival {rows[-1]['arrival_pct']:5.1f}% | n={len(tests)}")
    return pd.DataFrame(rows).round(2)

def wilson(p_pct, n, z=1.96):
    p = p_pct / 100.0
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(centre - half, 0), 100 * min(centre + half, 1)

teacher_replay = collect([], TEACHER_EPS, "teacher")
print(f"teacher data: {len(teacher_replay)} team steps")

comps, meta = {}, {}
for seed in SEEDS:
    for (k, bits, side) in CONFIGS:
        name = f"learned k={k} {'float' if bits is None else str(bits) + 'bit'}{'' if side else ' noside'} seed{seed}"
        base = f"{OUT_DIR}/learned_k{k}_b{bits}_s{int(side)}"
        path = base + ".pt" if seed == 0 else f"{base}_seed{seed}.pt"
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        comp = Compressor(k, bits, side).to(DEV)
        if RESUME and os.path.exists(path):
            comp.load_state_dict(torch.load(path, map_location=DEV)); print("reusing", path)
        else:
            print(f"\n===== training {name} =====")
            replay = list(teacher_replay)
            train(comp, replay, UPD_STAGE1, LR, f"{name} stage1")
            for r in range(DAGGER_ROUNDS):
                collect(replay, DAGGER_EPS, "student", comp)
                print(f"DAgger round {r+1}: {len(replay)} team steps")
                train(comp, replay, UPD_DAGGER, LR * 0.5, f"{name} dagger{r+1}")
            torch.save(comp.state_dict(), path)
            del replay; gc.collect(); torch.cuda.empty_cache()
        comps[name] = comp.eval(); meta[name] = (k, bits, side, seed)

policies = {}
if INCLUDE_REFERENCE:
    policies["REF full_comm"] = teacher_policy(True)
    policies["REF no_comm"] = teacher_policy(False)
for name, comp in comps.items():
    policies[name] = compressed_policy(comp)
    if SHUFFLE_CONTROL:
        policies[name + " SHUFFLED"] = compressed_policy(comp, shuffle=True)

t0 = time.time()
df = evaluate(policies, SETTINGS_CASES)
print(f"\nelapsed {time.time()-t0:.0f}s")

agents = sorted(df.agents.unique())
rows = []
for name in policies:
    r = dict(policy=name)
    if name in meta:
        k, bits, side, seed = meta[name]
        r.update(k=k, bits=bits, seed=seed, bits_per_neighbour=2 * k * (32 if bits is None else bits))
    for N in agents:
        g = df[(df.agents == N) & (df.policy == name)]
        if g.empty:
            continue
        g = g.iloc[0]
        lo, hi = wilson(g.success_pct, g.n_cases)
        r[f"success@{N}"] = f"{g.success_pct:.1f} [{lo:.0f}-{hi:.0f}] n={int(g.n_cases)}"
        r[f"steps@{N}"] = round(g.avg_steps, 1)
    rows.append(r)
summary = pd.DataFrame(rows)
print("\nLearned compressor, closed-loop success (Wilson 95% CI) vs message size:")
print(summary.to_string(index=False))

pooled = []
for (k, bits) in sorted({(m[0], m[1]) for m in meta.values()}, key=lambda t: (t[0], t[1] or 99)):
    names = [n for n, m in meta.items() if m[0] == k and m[1] == bits]
    r = dict(k=k, bits=bits, bits_per_neighbour=2 * k * (32 if bits is None else bits), seeds=len(names))
    for N in agents:
        g = df[(df.agents == N) & (df.policy.isin(names))]
        if not g.empty:
            r[f"mean_success@{N}"] = round(g.success_pct.mean(), 1)
            r[f"seed_spread@{N}"] = round(g.success_pct.max() - g.success_pct.min(), 1)
    pooled.append(r)
print("\nMean over training seeds (spread = max - min across seeds):")
print(pd.DataFrame(pooled).to_string(index=False))
for ref in ("REF full_comm", "REF no_comm"):
    if ref in policies:
        print(ref, {N: f"{df[(df.agents == N) & (df.policy == ref)].success_pct.iloc[0]:.1f}" for N in agents})

df.to_csv("/content/sigma_learned_compressor_results.csv", index=False)
summary.to_csv("/content/sigma_learned_compressor_summary.csv", index=False)
print("\nsaved -> /content/sigma_learned_compressor_results.csv, sigma_learned_compressor_summary.csv")
print("Read: does the compressor stay near full comm at every density, and do seeds agree? "
      "128 agents is beyond the training range (up to 96 agents).")
