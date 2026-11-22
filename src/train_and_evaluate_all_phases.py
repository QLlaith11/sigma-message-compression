import gc, os, sys, glob, time, types, pickle, random, zipfile, subprocess, warnings, math
warnings.filterwarnings("ignore")

SMOKE          = False
PHASES         = ["A", "B", "C"]
FORCE          = False
OUT_DIR        = "/content/sigma_paper"
USE_DRIVE      = True
CONFIGS        = [(1, 1), (1, 2), (2, 4), (4, 4)]
SEEDS          = [0, 1]
PLAN_A         = {(40, 32): 60, (40, 64): 60}
PLAN_B         = {(40, 128): 40}
PLAN_C         = {(40, 32): 60, (40, 64): 60}
PLAN_D         = {(40, 32): 200, (40, 64): 200}
HEADLINE_CONFIGS = [(1, 2), (2, 4)]
D_ENVS         = ["house", "maze"]
MAZE_ROBUST    = False
PRUNE_KS       = [1, 3]
NAIVE_BITS     = [1, 2]
PCA_CONFIGS    = [(1, 4), (2, 4), (4, 4)]
STATS_EPISODES = 20
STATS_SETTINGS = [(30, 16), (40, 32), (40, 64), (40, 64), (40, 96)]
ROBUST_LEARNED = [(2, 4), (1, 2)]
DROP_PS        = [0.1, 0.3, 0.5]
FLIP_QS        = [0.1, 0.3]
CRC_BITS       = 8
TEACHER_EPS    = 48
DATA_MAX_STEPS = 300
STRIDE         = 2
UPD_STAGE1     = 1500
DAGGER_ROUNDS  = 2
DAGGER_EPS     = 24
UPD_DAGGER     = 1000
B_STEPS        = 8
LR             = 1e-3
TAU            = 0.2
W_KL, W_ADV, W_REC = 1.0, 1.0, 0.1
TRAIN_DROP_MAX    = 0.0
TRAIN_CORRUPT_MAX = 0.0
TAG            = ""
SEED           = 0

ENVS = {
    "house": dict(scenario="house", ckpt="models/save_model/model_house/84000_house.pth", test_dir="house_test_set",
                  data=[(30, 16), (40, 16), (40, 32), (40, 32), (40, 48), (40, 64), (40, 64), (40, 96), (40, 128)]),
    "maze":  dict(scenario="maze", ckpt="models/save_model/model_maze/82000.pth", test_dir="test_set",
                  data=[(40, 16), (40, 32), (40, 32), (40, 48), (40, 64), (40, 64), (40, 96), (40, 128)]),
}
REPO_URL = "https://github.com/marmotlab/SIGMA.git"
ROOT     = "/content/SIGMA"
if SMOKE:
    CONFIGS, SEEDS, TEACHER_EPS, UPD_STAGE1, DAGGER_ROUNDS, DAGGER_EPS, UPD_DAGGER, B_STEPS = [(2, 4)], [0, 1], 4, 20, 1, 2, 10, 4
    PLAN_A, PLAN_B, PLAN_C, PLAN_D, HEADLINE_CONFIGS = {(40, 8): 2}, {(40, 8): 2}, {(40, 8): 2}, {(40, 8): 2}, [(2, 4)]
    PRUNE_KS, NAIVE_BITS, PCA_CONFIGS, ROBUST_LEARNED, DROP_PS, FLIP_QS = [1], [2], [(2, 4)], [(2, 4)], [0.3], [0.3]
    STATS_EPISODES, STATS_SETTINGS, FORCE = 3, [(40, 8)], True
    ENVS["house"]["data"] = [(40, 8)]; ENVS["maze"]["data"] = [(40, 8)]

if USE_DRIVE and not SMOKE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        OUT_DIR = "/content/drive/MyDrive/sigma_paper"
    except Exception as e:
        print("Drive not mounted, files stay in", OUT_DIR, "->", e)

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

os.chdir(ROOT); sys.path.insert(0, ROOT)
os.makedirs(OUT_DIR, exist_ok=True)
import configs
from model import Network
from environment import Environment

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("device:", DEV, "| torch", torch.__version__, "| numpy", np.__version__, "| out:", OUT_DIR)

R = configs.obs_radius; G = 2 * R + 1; NC = G * G
H = configs.hidden_dim; KCOMM = configs.max_comm_agents; K1 = KCOMM - 1
MAXEP = configs.max_episode_length

teacher = None; STATS = None; DATA_SETTINGS = None
ENV_NAME = None; CFG = None; TEST_DIR = None; ENV_DIR = None

def load_teacher(path):
    net = Network()
    net.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    net = net.eval().to(DEV)
    for p_ in net.parameters():
        p_.requires_grad_(False)
    assert net.comm.num_layers == 2
    return net


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
    def __init__(self, k, bits, side=True):
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


def make_env(n, L):
    last = None
    for _ in range(6):
        try:
            env = Environment(fix_density=0.3)
            obs, pos = env.reset(num_agents=n, map_length=L)
            return env, obs, pos
        except Exception as e:
            last = e
    raise last


def compressed_team_forward(comp, h_pre, m, j, valid, cell, drop_p=0.0, flip_q=0.0):
    n = h_pre.size(0)
    if drop_p > 0:
        valid = valid & (torch.rand(valid.shape, device=valid.device) >= drop_p)
    has = valid.any(-1)
    x, rec = h_pre, h_pre.new_zeros(())
    if bool(has.any()):
        tv = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=h_pre.device), valid], 1)
        attn_mask = (~tv).unsqueeze(1).expand(n, 1 + K1, 1 + K1)
        for l in range(teacher.comm.num_layers):
            z = comp.encode(l, x)
            zj = z[j]
            if flip_q > 0 and comp.bits is not None and comp.k > 0:
                levels = 2 ** comp.bits
                bad = torch.rand(zj.shape[:2], device=zj.device) < flip_q
                rnd = torch.randint(0, levels, zj.shape, device=zj.device).float() / (levels - 1) * 2 - 1
                zj = torch.where(bad.unsqueeze(-1), rnd, zj)
            tok = comp.decode(l, zj, x, m, cell)
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
        env, obs, pos = make_env(n, L)
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
                replay.append((h_in.half().cpu(), m_in.half().cpu(), torch.as_tensor(pos).short().cpu(), q_t.half().cpu()))
            (obs, pos), _, done, _ = env.step(actions)
    return replay

def sample_loss(comp, replay):
    tot = dict(kl=0.0, adv=0.0, rec=0.0)
    loss = 0.0
    for idx in np.random.randint(len(replay), size=B_STEPS):
        h0, m0, pos, qT = [t.to(DEV).float() if i != 2 else t.to(DEV).long() for i, t in enumerate(replay[idx])]
        j, valid, cell = neighbors_from_pos(pos)
        drop_p = random.uniform(0, TRAIN_DROP_MAX) if (TRAIN_DROP_MAX > 0 and random.random() < 0.5) else 0.0
        flip_q = random.uniform(0, TRAIN_CORRUPT_MAX) if (TRAIN_CORRUPT_MAX > 0 and random.random() < 0.5) else 0.0
        q, _, rec = compressed_team_forward(comp, h0, m0, j, valid, cell, drop_p, flip_q)
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


def fit_stats():
    path = f"{ENV_DIR}/stats.pt"
    if os.path.exists(path) and not SMOKE:
        S = torch.load(path, map_location="cpu")
        print("loaded message statistics:", path)
    else:
        V = [[], []]
        for e in tqdm(range(STATS_EPISODES), desc="teacher rollouts for statistics", leave=False):
            L, n = STATS_SETTINGS[e % len(STATS_SETTINGS)]
            env, obs, pos = make_env(n, L)
            hid, done = None, False
            while not done and env.steps < 300:
                a, _, xs, post, _ = team_step_layers(teacher, obs, pos, hid); hid = post
                if env.steps % 3 == 0:
                    V[0].append(xs[0].cpu()); V[1].append(xs[1].cpu())
                (obs, pos), _, done, _ = env.step(a)
        S = []
        for l in range(2):
            X = torch.cat(V[l]).float()
            mean = X.mean(0); std = X.std(0).clamp(min=1e-3)
            C = (X - mean).T @ (X - mean) / max(len(X) - 1, 1)
            ev, U = torch.linalg.eigh(C)
            order = ev.argsort(descending=True)
            S.append(dict(mean=mean, std=std, U=U[:, order], ev=ev[order].clamp(min=1e-12)))
        if not SMOKE:
            torch.save(S, path)
    return [{k: v.to(DEV) for k, v in s.items()} for s in S]


def quantize(c, sigma, bits):
    if bits is None or c.numel() == 0:
        return c
    z = (c / sigma).clamp(-3, 3)
    step = 6.0 / (2 ** bits - 1)
    return (torch.round((z + 3) / step) * step - 3) * sigma

class FullScheme:
    levels = None; max_nb = K1
    def __init__(self):
        self.name, self.bits_per_msg = "full comm (scheme path)", 2 * H * 32
    def encode(self, l, x): return x
    def decode(self, l, z_nb, x, m, cell): return z_nb
    def random_like(self, z): return None

class PruneScheme(FullScheme):
    def __init__(self, kk):
        super().__init__(); self.max_nb = kk; self.name = f"prune nearest-{kk}"

class NaiveQuant(FullScheme):
    def __init__(self, b):
        super().__init__(); self.b, self.levels = b, 2 ** b
        self.bits_per_msg, self.name = 2 * H * b, f"naive {b}bit/dim"
    def encode(self, l, x):
        z = ((x - STATS[l]["mean"]) / STATS[l]["std"]).clamp(-3, 3)
        step = 6.0 / (2 ** self.b - 1)
        return torch.round((z + 3) / step) * step - 3
    def decode(self, l, z_nb, x, m, cell): return z_nb * STATS[l]["std"] + STATS[l]["mean"]
    def random_like(self, z):
        step = 6.0 / (2 ** self.b - 1)
        return torch.randint(0, self.levels, z.shape, device=z.device).float() * step - 3

class PCAScheme(FullScheme):
    def __init__(self, k, bits):
        super().__init__(); self.k, self.bits = k, bits
        self.bits_per_msg, self.name = 2 * k * (32 if bits is None else bits), f"pca k={k} {bits}bit"
    def encode(self, l, x):
        S = STATS[l]
        return quantize((x - S["mean"]) @ S["U"][:, :self.k], S["ev"][:self.k].sqrt(), self.bits)
    def decode(self, l, z_nb, x, m, cell):
        S = STATS[l]
        return S["mean"] + z_nb @ S["U"][:, :self.k].T

class LearnedScheme(FullScheme):
    def __init__(self, comp, k, bits, seed, tag=""):
        super().__init__(); self.comp, self.k, self.bits, self.levels = comp, k, bits, 2 ** bits
        self.bits_per_msg = 2 * k * bits
        self.name = f"learned k={k} {bits}bit seed{seed}" + (" noise-trained" if tag else "")
    def encode(self, l, x): return self.comp.encode(l, x)
    def decode(self, l, z_nb, x, m, cell): return self.comp.decode(l, z_nb, x, m, cell)
    def random_like(self, z):
        return torch.randint(0, self.levels, z.shape, device=z.device).float() / (self.levels - 1) * 2 - 1

def team_forward_scheme(scheme, h_pre, m, j, valid, cell, drop_p=0.0, flip_q=0.0, detect=False):
    n = h_pre.size(0)
    if scheme.max_nb < K1:
        valid = valid.clone(); valid[:, scheme.max_nb:] = False
    if drop_p > 0:
        valid = valid & (torch.rand(valid.shape, device=valid.device) >= drop_p)
    links = int(valid.sum().item())
    x = h_pre
    if bool(valid.any()):
        for l in range(teacher.comm.num_layers):
            z = scheme.encode(l, x)
            zj = z[j]
            bad = None
            if flip_q > 0 and scheme.levels is not None:
                bad = torch.rand(zj.shape[:2], device=zj.device) < flip_q
                zj = torch.where(bad.unsqueeze(-1), scheme.random_like(zj), zj)
            vl = valid & ~bad if (detect and bad is not None) else valid
            hl = vl.any(-1)
            tv = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=h_pre.device), vl], 1)
            attn_mask = (~tv).unsqueeze(1).expand(n, 1 + K1, 1 + K1)
            tok = scheme.decode(l, zj, x, m, cell)
            info = teacher.comm.self_attn(torch.cat([x.unsqueeze(1), tok], 1), attn_mask=attn_mask)
            new = teacher.comm.update_cell(info[:, 0], x)
            x = torch.where(hl.unsqueeze(-1), new, x)
    comb = torch.cat((x, m), dim=1)
    adv = teacher.adv(comb) if configs.Advantage_all else teacher.adv(x)
    q = teacher.state(x) + adv - adv.mean(1, keepdim=True)
    return q, x, links

STAT = dict(links=0, agent_steps=0)

def scheme_policy(scheme, drop_p=0.0, flip_q=0.0, detect=False):
    @torch.no_grad()
    def f(obs_np, pos_np, hid):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=DEV)
        lat = teacher.obs_encoder(obs); m = teacher.mlp(lat)
        h_pre = teacher.recurrent(lat) if hid is None else teacher.recurrent(lat, hid)
        pos = torch.as_tensor(pos_np, device=DEV).long()
        j, valid, cell = neighbors_from_pos(pos)
        q, x, links = team_forward_scheme(scheme, h_pre, m, j, valid, cell, drop_p, flip_q, detect)
        STAT["links"] += links; STAT["agent_steps"] += obs.size(0)
        return q.argmax(1).tolist(), x
    return f

def teacher_policy(use_comm):
    def f(obs, pos, hid):
        a, _, _, post, _ = team_step_layers(teacher, obs, pos, hid, use_comm)
        if use_comm:
            _, valid, _ = neighbors_from_pos(torch.as_tensor(pos, device=DEV).long())
            STAT["links"] += int(valid.sum().item())
        STAT["agent_steps"] += len(pos)
        return a, post
    return f

def run_case(case, act_fn):
    env = Environment(); env.load(case[0], case[1], case[2])
    obs, pos = env.observe(); hid, done = None, False
    arr = np.full(len(env.agents_pos), MAXEP, dtype=np.float64)
    arr[(env.agents_pos == env.goals_pos).all(1)] = 0
    while not done and env.steps < MAXEP:
        actions, hid = act_fn(obs, pos, hid)
        (obs, pos), _, done, _ = env.step(actions)
        newly = (env.agents_pos == env.goals_pos).all(1) & (arr == MAXEP)
        arr[newly] = env.steps
    return (np.array_equal(env.agents_pos, env.goals_pos), env.steps,
            float((env.agents_pos == env.goals_pos).all(1).mean()), float(arr.mean()))

def wilson(p_pct, n, z=1.96):
    p = p_pct / 100.0
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(centre - half, 0), 100 * min(centre + half, 1)

def evaluate(policies, bits_per_msg, plan, tag):
    rows = []
    for (L, N), n_cases in plan.items():
        path = f"{TEST_DIR}/{L}length_{N}agents_0.3density.pth"
        if not os.path.exists(path):
            print("missing test set (skipped):", path); continue
        with open(path, "rb") as fh:
            tests = pickle.load(fh)[:n_cases]
        for name, fn in policies.items():
            STAT["links"] = 0; STAT["agent_steps"] = 0
            res = [run_case(c, fn) for c in tqdm(tests, desc=f"{tag} {N:>3} | {name}", leave=False)]
            succ = 100 * np.mean([r[0] for r in res])
            lo, hi = wilson(succ, len(tests))
            lps = STAT["links"] / max(STAT["agent_steps"], 1)
            rows.append(dict(env=ENV_NAME, agents=N, policy=name, success_pct=succ, ci_lo=lo, ci_hi=hi,
                             makespan=np.mean([r[1] for r in res]), mean_arrival=np.mean([r[3] for r in res]),
                             msgs_per_agent_step=lps, bits_per_msg=bits_per_msg.get(name, 0),
                             bits_per_agent_step=lps * bits_per_msg.get(name, 0), n_cases=len(tests)))
            print(f"{tag} {N:>3} | {name:<34} success {succ:5.1f}% [{lo:.0f}-{hi:.0f}] | makespan {rows[-1]['makespan']:6.1f} "
                  f"| arrival {rows[-1]['mean_arrival']:6.1f} | bits/agent-step {rows[-1]['bits_per_agent_step']:9.1f} | n={len(tests)}")
    return pd.DataFrame(rows).round(2)

def load_learned(k, bits, seed, tag=""):
    path = f"{ENV_DIR}/learned_k{k}_b{bits}_seed{seed}{tag}.pt"
    if not os.path.exists(path):
        print("missing compressor (skipped):", path); return None
    comp = Compressor(k, bits).to(DEV)
    comp.load_state_dict(torch.load(path, map_location=DEV)); comp.eval()
    return LearnedScheme(comp, k, bits, seed, tag)


def set_env(name):
    global teacher, STATS, DATA_SETTINGS, ENV_NAME, CFG, TEST_DIR, ENV_DIR
    ENV_NAME, CFG = name, ENVS[name]
    configs.test_senario = CFG["scenario"]
    TEST_DIR = CFG["test_dir"]
    ENV_DIR = f"{OUT_DIR}/{name}"; os.makedirs(ENV_DIR, exist_ok=True)
    DATA_SETTINGS = CFG["data"]
    teacher = load_teacher(CFG["ckpt"])
    STATS = None
    print(f"\n######## environment: {name} ########")

def ensure_compressors(seeds):
    todo = [(k, b, s) for s in seeds for (k, b) in CONFIGS
            if not os.path.exists(f"{ENV_DIR}/learned_k{k}_b{b}_seed{s}{TAG}.pt")]
    if not todo:
        print("all compressors present:", len(seeds) * len(CONFIGS)); return
    print(f"training {len(todo)} missing compressors for {ENV_NAME} ...")
    teacher_replay = collect([], TEACHER_EPS, "teacher")
    print(f"teacher data: {len(teacher_replay)} team steps")
    for (k, bits, seed) in todo:
        name = f"{ENV_NAME} k={k} {bits}bit seed{seed}{TAG}"
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        comp = Compressor(k, bits).to(DEV)
        print(f"\n===== training {name} =====")
        replay = list(teacher_replay)
        train(comp, replay, UPD_STAGE1, LR, f"{name} stage1")
        for r in range(DAGGER_ROUNDS):
            collect(replay, DAGGER_EPS, "student", comp)
            print(f"DAgger round {r+1}: {len(replay)} team steps")
            train(comp, replay, UPD_DAGGER, LR * 0.5, f"{name} dagger{r+1}")
        torch.save(comp.state_dict(), f"{ENV_DIR}/learned_k{k}_b{bits}_seed{seed}{TAG}.pt")
        del replay, comp; gc.collect(); torch.cuda.empty_cache()
    del teacher_replay; gc.collect()

def run_pareto(plan, seeds, baselines, references, tag):
    global STATS
    csv = f"{ENV_DIR}/results_{tag}.csv"
    if os.path.exists(csv) and not FORCE:
        print("reusing finished result:", csv); return pd.read_csv(csv)
    policies, bpm = {}, {}
    if references:
        policies["REF full_comm"] = teacher_policy(True); bpm["REF full_comm"] = 2 * H * 32
        policies["REF no_comm"] = teacher_policy(False); bpm["REF no_comm"] = 0
    schemes = []
    if baselines:
        if STATS is None:
            STATS = fit_stats()
        schemes += [PruneScheme(kk) for kk in PRUNE_KS] + [NaiveQuant(b) for b in NAIVE_BITS] + \
                   [PCAScheme(k, b) for (k, b) in PCA_CONFIGS]
    for seed in seeds:
        for (k, b) in CONFIGS:
            s_ = load_learned(k, b, seed)
            if s_ is not None:
                schemes.append(s_)
    for s_ in schemes:
        policies[s_.name] = scheme_policy(s_); bpm[s_.name] = s_.bits_per_msg
    t0 = time.time()
    df = evaluate(policies, bpm, plan, tag)
    print(f"{tag}: elapsed {time.time()-t0:.0f}s")
    df.to_csv(csv, index=False)
    return df

def plot_pareto(df, path):
    fam = lambda n: ("full/no comm" if n.startswith("REF") else n.split()[0].replace("learned", "learned (ours)"))
    colors = {"full/no comm": "black", "prune": "tab:orange", "naive": "tab:blue", "pca": "tab:green", "learned (ours)": "tab:red"}
    Ns = sorted(df.agents.unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(5.2 * len(Ns), 4.2), squeeze=False)
    for ax, N in zip(axes[0], Ns):
        g = df[df.agents == N]
        for f_, gg in g.groupby(g.policy.map(fam)):
            ax.errorbar(np.maximum(gg.bits_per_agent_step, 1), gg.success_pct,
                        yerr=[gg.success_pct - gg.ci_lo, gg.ci_hi - gg.success_pct], fmt="o", capsize=2,
                        color=colors.get(f_, "gray"), label=f_)
        ax.set_xscale("log"); ax.set_xlabel("bits received per agent-step (log; 1 = none)"); ax.set_ylabel("success rate (%)")
        ax.set_title(f"{ENV_NAME}, {N} agents"); ax.set_ylim(-3, 103); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); print("saved figure ->", path)

def run_robust(plan, tag):
    csv = f"{ENV_DIR}/results_{tag}.csv"
    if os.path.exists(csv) and not FORCE:
        print("reusing finished result:", csv); return pd.read_csv(csv)
    base = [FullScheme()]
    for (k, b) in ROBUST_LEARNED:
        s_ = load_learned(k, b, 0)
        if s_ is not None:
            base.append(s_)
    policies, bpm = {}, {}
    for s_ in base:
        policies[f"{s_.name} | clean"] = scheme_policy(s_); bpm[f"{s_.name} | clean"] = s_.bits_per_msg
        for p in DROP_PS:
            nm = f"{s_.name} | drop {p}"; policies[nm] = scheme_policy(s_, drop_p=p); bpm[nm] = s_.bits_per_msg
        if s_.levels is not None:
            for q in FLIP_QS:
                nm = f"{s_.name} | corrupt {q}"; policies[nm] = scheme_policy(s_, flip_q=q); bpm[nm] = s_.bits_per_msg
                nm = f"{s_.name} | corrupt {q} + CRC"
                policies[nm] = scheme_policy(s_, flip_q=q, detect=True); bpm[nm] = s_.bits_per_msg + CRC_BITS
    df = evaluate(policies, bpm, plan, tag)
    df.to_csv(csv, index=False)
    df["scheme"] = df.policy.str.split(" \\| ").str[0]; df["perturbation"] = df.policy.str.split(" \\| ").str[1]
    for N in sorted(df.agents.unique()):
        print(f"\nrobustness, success (%), {N} agents:")
        print(df[df.agents == N].pivot(index="scheme", columns="perturbation", values="success_pct").round(1).to_string())
    return df

SHOW = ["agents", "policy", "success_pct", "ci_lo", "ci_hi", "makespan", "mean_arrival", "msgs_per_agent_step",
        "bits_per_msg", "bits_per_agent_step", "n_cases"]
RESULTS = {}

if "A" in PHASES:
    set_env("house"); ensure_compressors(SEEDS)
    RESULTS["A: house, seed 1, 32/64 agents"] = run_pareto(PLAN_A, [1] if 1 in SEEDS else SEEDS, False, False, "phaseA_seed1")

if "B" in PHASES:
    set_env("house"); ensure_compressors(SEEDS)
    RESULTS["B: house, 128 agents"] = run_pareto(PLAN_B, SEEDS, False, True, "phaseB_128agents")

if "C" in PHASES:
    set_env("maze"); ensure_compressors(SEEDS)
    dfc = run_pareto(PLAN_C, [0], True, True, "phaseC_maze")
    RESULTS["C: maze, rate vs performance"] = dfc
    try:
        plot_pareto(dfc, f"{ENV_DIR}/pareto_maze.png")
    except Exception as e:
        print("figure skipped:", e)
    if MAZE_ROBUST:
        RESULTS["C: maze, robustness"] = run_robust({(40, 64): 60} if not SMOKE else PLAN_C, "phaseC_maze_robust")

if "D" in PHASES:
    for env_name in D_ENVS:
        set_env(env_name)
        _saved_configs, CONFIGS = CONFIGS, HEADLINE_CONFIGS
        ensure_compressors(SEEDS)
        RESULTS[f"D: {env_name}, headline rows, full test sets"] = run_pareto(PLAN_D, SEEDS, False, True, f"phaseD_{env_name}_200cases")
        CONFIGS = _saved_configs

print("\n================================ SUMMARY ================================")
for name, df in RESULTS.items():
    cols = [c for c in SHOW if c in df.columns]
    print(f"\n--- {name} ---")
    print(df.sort_values(["agents", "bits_per_agent_step"])[cols].to_string(index=False))
print("\nAll result files are in", OUT_DIR, "(results_*.csv, pareto_maze.png). Send me the SUMMARY tables.")
print("Notes: bits = measured bits received per agent per step (messages x bits per message, 2 exchanges); "
      "no entropy coding; full precision = 16,384 bits per message.")
