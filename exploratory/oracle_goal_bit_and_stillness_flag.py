import gc, os, sys, glob, time, types, pickle, random, zipfile, subprocess, warnings, math
warnings.filterwarnings("ignore")

SMOKE          = False
VARIANTS       = ["goal_bit", "still_bit"]
T_STILL        = None
T_LIST         = [1, 2, 3, 5, 8, 12]
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
SETTINGS_CASES = {(40, 32): 60, (40, 64): 60}
INCLUDE_REFERENCE = True
RESUME         = True
SEED           = 0

REPO_URL = "https://github.com/marmotlab/SIGMA.git"
ROOT     = "/content/SIGMA"
CKPT     = "models/save_model/model_house/84000_house.pth"
TEST_DIR = "house_test_set"
OUT_DIR  = "/content/sigma_semantic"
if SMOKE:
    VARIANTS, TEACHER_EPS, UPD_STAGE1, DAGGER_ROUNDS, DAGGER_EPS, UPD_DAGGER = ["goal_bit", "still_bit"], 4, 20, 1, 2, 10
    DATA_SETTINGS, SETTINGS_CASES, B_STEPS, RESUME, T_LIST = [(40, 8)], {(40, 8): 2}, 4, False, [1, 3]

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

class FlagDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell_emb = nn.ModuleList([nn.Embedding(NC, 32) for _ in range(2)])
        din = 1 + 32 + H + 1
        self.dec = nn.ModuleList([nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                                nn.Linear(256, H)) for _ in range(2)])
    def decode(self, l, z_nb, x, m, cell):
        n, k1, _ = z_nb.shape
        side = torch.cat([x, m], -1).unsqueeze(1).expand(-1, k1, -1)
        return self.dec[l](torch.cat([z_nb, self.cell_emb[l](cell), side], -1))

def team_forward(dec, h_pre, m, j, valid, cell, flag):
    n = h_pre.size(0)
    has = valid.any(-1)
    z = flag.float().unsqueeze(-1) * 2 - 1
    x, rec = h_pre, h_pre.new_zeros(())
    if bool(has.any()):
        tv = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=h_pre.device), valid], 1)
        attn_mask = (~tv).unsqueeze(1).expand(n, 1 + K1, 1 + K1)
        for l in range(teacher.comm.num_layers):
            tok = dec.decode(l, z[j], x, m, cell)
            rec = rec + F.mse_loss(tok[valid], x[j][valid].detach())
            info = teacher.comm.self_attn(torch.cat([x.unsqueeze(1), tok], 1), attn_mask=attn_mask)
            new = teacher.comm.update_cell(info[:, 0], x)
            x = torch.where(has.unsqueeze(-1), new, x)
    comb = torch.cat((x, m), dim=1)
    adv = teacher.adv(comb) if configs.Advantage_all else teacher.adv(x)
    q = teacher.state(x) + adv - adv.mean(1, keepdim=True)
    return q, x, rec

def make_flag(variant, T, at_goal, cnt):
    return at_goal if variant == "goal_bit" else (cnt >= T)

def collect(replay, n_eps, acting, dec=None, variant=None, T=None, stats=None):
    for e in tqdm(range(n_eps), desc=f"collect ({acting})", leave=False):
        L, n = DATA_SETTINGS[e % len(DATA_SETTINGS)]
        env = Environment(); obs, pos = env.reset(num_agents=n, map_length=L)
        t_hid = s_hid = None; done = False
        cnt = np.zeros(n, dtype=np.int64)
        while not done and env.steps < DATA_MAX_STEPS:
            at_goal = (env.agents_pos == env.goals_pos).all(1)
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
                    flag = torch.as_tensor(make_flag(variant, T, at_goal, cnt), device=DEV)
                    q_s, s_hid, _ = team_forward(dec, h_in, m_in, j, valid, cell, flag)
                    actions = q_s.argmax(1).tolist()
            if stats is not None:
                stats.append((at_goal.copy(), cnt.copy()))
            if env.steps % STRIDE == 0:
                replay.append((h_in.half().cpu(), m_in.half().cpu(), torch.as_tensor(pos).short().cpu(),
                               q_t.half().cpu(), torch.as_tensor(at_goal), torch.as_tensor(cnt)))
            old = env.agents_pos.copy()
            (obs, pos), _, done, _ = env.step(actions)
            cnt = np.where((env.agents_pos == old).all(1), cnt + 1, 0)
    return replay

def sample_loss(dec, replay, variant, T):
    tot = dict(kl=0.0, adv=0.0, rec=0.0)
    loss = 0.0
    for idx in np.random.randint(len(replay), size=B_STEPS):
        h0, m0, pos, qT, ag, cnt = replay[idx]
        h0, m0, qT = h0.to(DEV).float(), m0.to(DEV).float(), qT.to(DEV).float()
        pos = pos.to(DEV).long()
        flag = make_flag(variant, T, ag.to(DEV).bool(), cnt.to(DEV))
        j, valid, cell = neighbors_from_pos(pos)
        q, _, rec = team_forward(dec, h0, m0, j, valid, cell, flag)
        kl = F.kl_div(F.log_softmax(q / TAU, -1), F.softmax(qT / TAU, -1), reduction="batchmean")
        l_adv = F.smooth_l1_loss(q - q.mean(1, keepdim=True), qT - qT.mean(1, keepdim=True))
        loss = loss + (W_KL * kl + W_ADV * l_adv + W_REC * rec) / B_STEPS
        tot["kl"] += kl.item() / B_STEPS; tot["adv"] += l_adv.item() / B_STEPS; tot["rec"] += float(rec) / B_STEPS
    return loss, tot

def train(dec, replay, variant, T, updates, lr, tag):
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(updates, 1), eta_min=lr * 0.1)
    t0 = time.time()
    for u in range(1, updates + 1):
        loss, tot = sample_loss(dec, replay, variant, T)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(dec.parameters(), 5); opt.step(); sched.step()
        if u % 250 == 0 or u == updates:
            print(f"[{tag} {u}/{updates}] " + " ".join(f"{k} {v:.4f}" for k, v in tot.items()) + f" | {time.time()-t0:.0f}s")

ENV = None

def flag_policy(dec, variant, T):
    @torch.no_grad()
    def f(obs_np, pos_np, hid):
        if hid is None:
            x_prev, cnt, prev = None, np.zeros(len(pos_np), dtype=np.int64), pos_np.copy()
        else:
            x_prev, cnt, prev = hid
            cnt = np.where((pos_np == prev).all(1), cnt + 1, 0)
        at_goal = (ENV.agents_pos == ENV.goals_pos).all(1)
        flag = torch.as_tensor(make_flag(variant, T, at_goal, cnt), device=DEV)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=DEV)
        lat = teacher.obs_encoder(obs); m = teacher.mlp(lat)
        h_pre = teacher.recurrent(lat) if x_prev is None else teacher.recurrent(lat, x_prev)
        pos = torch.as_tensor(pos_np, device=DEV).long()
        j, valid, cell = neighbors_from_pos(pos)
        q, x, _ = team_forward(dec, h_pre, m, j, valid, cell, flag)
        return q.argmax(1).tolist(), (x, cnt, pos_np.copy())
    return f

def teacher_policy(use_comm):
    def f(obs, pos, hid):
        a, _, _, post, _ = team_step_layers(teacher, obs, pos, hid, use_comm); return a, post
    return f

def run_case(case, act_fn):
    global ENV
    env = Environment(); env.load(case[0], case[1], case[2]); ENV = env
    obs, pos = env.observe(); hid, done = None, False
    while not done and env.steps < configs.max_episode_length:
        actions, hid = act_fn(obs, pos, hid)
        (obs, pos), _, done, _ = env.step(actions)
    return (np.array_equal(env.agents_pos, env.goals_pos), env.steps,
            float((env.agents_pos == env.goals_pos).all(1).mean()))

def wilson(p_pct, n, z=1.96):
    p = p_pct / 100.0
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(centre - half, 0), 100 * min(centre + half, 1)

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
            print(f"{L}x{L} {N:>3} | {name:<26} success {rows[-1]['success_pct']:5.1f}% | steps {rows[-1]['avg_steps']:6.1f} "
                  f"| arrival {rows[-1]['arrival_pct']:5.1f}% | n={len(tests)}")
    return pd.DataFrame(rows).round(2)

stats = []
teacher_replay = collect([], TEACHER_EPS, "teacher", stats=stats)
print(f"teacher data: {len(teacher_replay)} team steps")

AG = np.concatenate([s[0] for s in stats]); CNT = np.concatenate([s[1] for s in stats])
print(f"\nBase rate of agents parked at their goal: {AG.mean()*100:.1f}%  ({len(AG)} agent-steps)")
f1s = {}
rows = []
for T in T_LIST:
    pred = CNT >= T
    tp = float((pred & AG).sum())
    prec = tp / max(pred.sum(), 1); rec = tp / max(AG.sum(), 1)
    f1s[T] = 2 * prec * rec / max(prec + rec, 1e-9)
    rows.append(dict(T=T, precision=round(prec, 3), recall=round(rec, 3), F1=round(f1s[T], 3)))
print("How well does 'stationary for >= T steps' predict 'at goal'?")
print(pd.DataFrame(rows).to_string(index=False))
T_USE = T_STILL if T_STILL is not None else max(f1s, key=f1s.get)
print(f"-> still_bit uses T = {T_USE}")

decs = {}
for variant in VARIANTS:
    name = f"{variant}" + (f" T={T_USE}" if variant == "still_bit" else "")
    path = f"{OUT_DIR}/{variant}{'_T' + str(T_USE) if variant == 'still_bit' else ''}.pt"
    dec = FlagDecoder().to(DEV)
    if RESUME and os.path.exists(path):
        dec.load_state_dict(torch.load(path, map_location=DEV)); print("reusing", path)
    else:
        print(f"\n===== training {name} =====")
        replay = list(teacher_replay)
        train(dec, replay, variant, T_USE, UPD_STAGE1, LR, f"{name} stage1")
        for r in range(DAGGER_ROUNDS):
            collect(replay, DAGGER_EPS, "student", dec, variant, T_USE)
            print(f"DAgger round {r+1}: {len(replay)} team steps")
            train(dec, replay, variant, T_USE, UPD_DAGGER, LR * 0.5, f"{name} dagger{r+1}")
        torch.save(dec.state_dict(), path)
        del replay; gc.collect(); torch.cuda.empty_cache()
    decs[name] = (dec.eval(), variant)

policies = {}
if INCLUDE_REFERENCE:
    policies["REF full_comm"] = teacher_policy(True)
    policies["REF no_comm"] = teacher_policy(False)
for name, (dec, variant) in decs.items():
    policies[name] = flag_policy(dec, variant, T_USE)

t0 = time.time()
df = evaluate(policies, SETTINGS_CASES)
print(f"\nelapsed {time.time()-t0:.0f}s")

agents = sorted(df.agents.unique())
out = []
for name in policies:
    r = dict(policy=name)
    for N in agents:
        g = df[(df.agents == N) & (df.policy == name)]
        if g.empty:
            continue
        g = g.iloc[0]
        lo, hi = wilson(g.success_pct, g.n_cases)
        r[f"success@{N}"] = f"{g.success_pct:.1f} [{lo:.0f}-{hi:.0f}] n={int(g.n_cases)}"
        r[f"steps@{N}"] = round(g.avg_steps, 1)
    out.append(r)
summary = pd.DataFrame(out)
print("\nOne-bit semantic messages vs references (closed loop, Wilson 95% CI):")
print(summary.to_string(index=False))
df.to_csv("/content/sigma_semantic_results.csv", index=False)
summary.to_csv("/content/sigma_semantic_summary.csv", index=False)
print("\nsaved -> /content/sigma_semantic_results.csv, sigma_semantic_summary.csv")
print("Read: goal_bit near full comm => the protocol is essentially 'I am parked'. "
      "still_bit well above k=0 => most of it can be inferred without any communication (optimistic: global tracking).")
