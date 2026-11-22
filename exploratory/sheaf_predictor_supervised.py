import gc, os, sys, glob, time, types, pickle, random, zipfile, subprocess, warnings, math, copy
warnings.filterwarnings("ignore")

SMOKE          = False
PRED_KINDS     = ["sheaf", "mlp"]
D_S, D_E, K_DIFF = 64, 32, 3
DATA_SETTINGS  = [(20, 8), (30, 8), (40, 8), (30, 16), (40, 16), (40, 32), (40, 48), (40, 64)]
N_TEACHER_EPS  = 48
DATA_MAX_STEPS = 300
TRAIN_UPDATES  = 6000
BATCH          = 128
LR             = 1e-3
W_SEC, W_AE    = 0.1, 0.5
DAGGER_ROUNDS  = 6
DAGGER_EPS     = 40
DAGGER_UPDATES = 3000
STORE_EVERY_DAGGER = 3
RESUME         = True
RESUME_DAGGER  = True
DAGGER_ROUNDS_BY_KIND = {"sheaf": 2, "mlp": 6}
DAGGER_SETTINGS = [(30, 16), (40, 16), (40, 32), (40, 32), (40, 48),
                   (40, 64), (40, 64), (40, 64), (40, 96), (40, 128)]
DAGGER_MAX_STEPS = 400
ROUND_EVAL_SETTINGS = [(40, 32), (40, 64)]
ROUND_EVAL_CASES = 10
EVAL_SETTINGS  = [(40, 8), (40, 32), (40, 64), (40, 128)]
N_CASES        = 30
SEED           = 0
RUN_MAIN       = True
ORACLE_SWEEP   = False
ORACLE_KINDS   = ["sheaf_dagger"]
ORACLE_LAMBDAS = [0.0, 0.5, 1.0]
ORACLE_SETTINGS = [(40, 32), (40, 64)]
ORACLE_CASES   = 30
DEBUG_SYNC     = False

REPO_URL = "https://github.com/marmotlab/SIGMA.git"
ROOT     = "/content/SIGMA"
CKPT     = "models/save_model/model_house/84000_house.pth"
TEST_DIR = "house_test_set"
OUT_DIR  = "/content/sigma_predictors"

if DEBUG_SYNC:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if SMOKE:
    N_TEACHER_EPS, TRAIN_UPDATES, DAGGER_ROUNDS, DAGGER_EPS, DAGGER_UPDATES = 8, 30, 1, 2, 10
    DAGGER_SETTINGS, ROUND_EVAL_SETTINGS, ROUND_EVAL_CASES = [(40, 8)], [(40, 8)], 1
    DAGGER_ROUNDS_BY_KIND, RESUME, RESUME_DAGGER = {}, False, False
    EVAL_SETTINGS, N_CASES, BATCH = [(40, 8)], 2, 32

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

os.chdir(ROOT); sys.path.insert(0, ROOT); os.makedirs(OUT_DIR, exist_ok=True)
import configs
from model import Network
from environment import Environment
assert configs.test_senario == "house"

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("device:", DEV, "| torch", torch.__version__, "| numpy", np.__version__)

R = configs.obs_radius; G = 2 * R + 1; NC = G * G; NCR = (4 * R + 1) ** 2
H = configs.hidden_dim; KCOMM = configs.max_comm_agents; K1 = KCOMM - 1
CELL_D = torch.tensor([[math.hypot(r - R, c - R) for c in range(G)] for r in range(G)],
                      dtype=torch.float32, device=DEV).view(-1)

teacher = Network()
teacher.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
teacher = teacher.eval().to(DEV)
for p_ in teacher.parameters():
    p_.requires_grad_(False)
assert teacher.comm.num_layers == 2, "script assumes 2 attention layers"

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

def nb_from_pos(pos_t):
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
    return idx, valid

def grid_neighbors(agent_grid):
    g = agent_grid.reshape(*agent_grid.shape[:-2], NC) > 0.5
    score = torch.where(g, -CELL_D, torch.full_like(CELL_D, -1e9))
    val, idx = score.topk(K1, dim=-1)
    return idx, val > -1e8

class SheafExtension(nn.Module):
    kind = "sheaf"
    def __init__(self, d=D_S, d_e=D_E, K=K_DIFF):
        super().__init__()
        self.d, self.de, self.K = d, d_e, K
        self.enc = nn.Sequential(nn.Linear(H + 1, 128), nn.ReLU(), nn.Linear(128, d))
        self.prior = nn.Parameter(torch.randn(NC, d) * 0.1)
        self.U = nn.Linear(d, d)
        self.Phi = nn.Parameter(torch.randn(NCR, d_e, d) / d ** 0.5)
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(K)])
        self.alpha = nn.Parameter(torch.full((K,), 0.5))
        self.dec0 = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, H))
        self.dec1 = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, H))

    def graph(self, cell, valid):
        B, k1 = cell.shape
        offs = torch.stack([cell // G - R, cell % G - R], -1)
        offs = torch.cat([torch.zeros(B, 1, 2, dtype=offs.dtype, device=offs.device), offs], 1)
        vm = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=cell.device), valid], 1)
        V = k1 + 1
        rel = offs.unsqueeze(1) - offs.unsqueeze(2)
        pc = ((rel[..., 0] + 2 * R) * (4 * R + 1) + (rel[..., 1] + 2 * R)).clamp(0, NCR - 1)
        A = vm.unsqueeze(1) & vm.unsqueeze(2) & ~torch.eye(V, dtype=torch.bool, device=cell.device)[None]
        Gm = F.embedding(pc, self.Phi.reshape(NCR, -1)).view(B, V, V, self.de, self.d)
        return Gm, A, vm

    def laplacian(self, x, Gm, A):
        Gt = Gm.transpose(1, 2)
        Fx = torch.einsum('bvued,bvd->bvue', Gm, x)
        Gx = torch.einsum('bvued,bud->bvue', Gt, x)
        diff = (Fx - Gx) * A.unsqueeze(-1)
        msg = torch.einsum('bvued,bvue->bvd', Gm, diff)
        return msg / A.sum(-1, keepdim=True).clamp(min=1).float()

    def forward(self, h, m, cell, valid):
        x0 = self.enc(torch.cat([h, m], -1))
        Gm, A, vm = self.graph(cell, valid)
        xn = F.embedding(cell, self.prior) + self.U(x0).unsqueeze(1)
        mask = vm.unsqueeze(-1).float()
        x = torch.cat([x0.unsqueeze(1), xn], 1) * mask
        for k in range(self.K):
            upd = F.elu(self.layers[k](self.laplacian(x, Gm, A)))
            x = x - self.alpha[k] * upd
            x = torch.cat([x0.unsqueeze(1), x[:, 1:]], 1) * mask
        xs = x[:, 1:]
        return self.dec0(xs), self.dec1(xs)

    def true_losses(self, b):
        nh = torch.cat([b["ego_t0"].unsqueeze(1), b["t0"]], 1)
        nm = torch.cat([b["ego_m"].unsqueeze(1), b["nbm"]], 1)
        st = self.enc(torch.cat([nh, nm], -1))
        Gm, A, vm = self.graph(b["cell"], b["valid"])
        Fx = torch.einsum('bvued,bvd->bvue', Gm, st)
        Gx = torch.einsum('bvued,bud->bvue', Gm.transpose(1, 2), st)
        energy = (((Fx - Gx) ** 2) * A.unsqueeze(-1)).sum() / (A.sum().clamp(min=1) * self.de)
        rec = F.mse_loss(self.dec0(st[vm]), nh[vm])
        return energy, rec

class MLPPredictor(nn.Module):
    kind = "mlp"
    def __init__(self, d=D_S):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(H + 1, 128), nn.ReLU(), nn.Linear(128, d))
        self.cell = nn.Embedding(NC, d)
        self.body = nn.Sequential(nn.Linear(2 * d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.o0, self.o1 = nn.Linear(128, H), nn.Linear(128, H)
    def forward(self, h, m, cell, valid):
        x0 = self.enc(torch.cat([h, m], -1)).unsqueeze(1).expand(-1, cell.size(1), -1)
        y = self.body(torch.cat([x0, self.cell(cell)], -1))
        return self.o0(y), self.o1(y)

class MeanPredictor(nn.Module):
    kind = "mean"
    def __init__(self, m0, m1):
        super().__init__()
        self.register_buffer("m0", m0.view(1, 1, -1)); self.register_buffer("m1", m1.view(1, 1, -1))
    def forward(self, h, m, cell, valid):
        B, k1 = cell.shape
        return self.m0.expand(B, k1, -1), self.m1.expand(B, k1, -1)

class NBData:
    def __init__(self):
        self.parts, self.base, self.next_ep = [], 0, 0
        self.built = False
    def clone(self):
        c = NBData(); c.parts, c.base, c.next_ep = list(self.parts), self.base, self.next_ep
        return c
    def add_step(self, in_h, m, t0, t1, pos, nb_local, valid, ep):
        n = in_h.size(0)
        nb = torch.where(valid, nb_local + self.base, torch.full_like(nb_local, -1))
        self.parts.append((in_h.half().cpu(), m.half().cpu(), t0.half().cpu(), t1.half().cpu(),
                           pos.to(torch.int16).cpu(), nb.int().cpu(),
                           torch.full((n,), ep, dtype=torch.int32)))
        self.base += n
    def build(self):
        cat = lambda i: torch.cat([p[i] for p in self.parts], 0)
        merged = [cat(i) for i in range(7)]
        self.parts = [tuple(merged)]
        self.IN, self.M, self.T0, self.T1, self.POS, self.NB, self.EP = merged
        has = (self.NB >= 0).any(1)
        hold = (self.EP % 10 == 0)
        self.rows_train = (has & ~hold).nonzero().squeeze(1)
        self.rows_hold = (has & hold).nonzero().squeeze(1)
        if len(self.rows_hold) == 0:
            self.rows_hold = self.rows_train
        gb = sum(t.numel() * t.element_size() for t in merged) / 1e9
        print(f"dataset: {self.base} agent-steps ({gb:.2f} GB CPU) | train egos {len(self.rows_train)} | holdout egos {len(self.rows_hold)}")

def get_batch(d, rows):
    nb = d.NB[rows].long(); valid = nb >= 0; nbc = nb.clamp(min=0)
    off = d.POS[nbc].long() - d.POS[rows].long().unsqueeze(1)
    cell = ((off[..., 0] + R) * G + (off[..., 1] + R)).clamp(0, NC - 1)
    out = dict(h=d.IN[rows].float(), m=d.M[rows].float(), cell=cell, valid=valid,
               t0=d.T0[nbc].float(), t1=d.T1[nbc].float(), nbm=d.M[nbc].float(),
               ego_t0=d.T0[rows].float(), ego_m=d.M[rows].float())
    return {k: v.to(DEV) for k, v in out.items()}

class Hybrid:
    def __init__(self, P):
        self.P = P
    @torch.no_grad()
    def step(self, obs_np, hidden):
        net = teacher
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=DEV)
        n = obs.size(0)
        latent = net.obs_encoder(obs); m = net.mlp(latent)
        h_pre = net.recurrent(latent) if hidden is None else net.recurrent(latent, hidden)
        cell, valid = grid_neighbors(obs[:, 0])
        has = valid.any(-1)
        x = h_pre
        if bool(has.any()):
            preds = self.P(h_pre, m, cell, valid)
            tv = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=DEV), valid], 1)
            attn_mask = (~tv).unsqueeze(1).expand(n, 1 + K1, 1 + K1)
            for l in range(net.comm.num_layers):
                tokens = torch.cat([x.unsqueeze(1), preds[l]], 1)
                info = net.comm.self_attn(tokens, attn_mask=attn_mask)
                new = net.comm.update_cell(info[:, 0], x)
                x = torch.where(has.unsqueeze(-1), new, x)
        comb = torch.cat((x, m), dim=1)
        adv = net.adv(comb) if configs.Advantage_all else net.adv(x)
        q = net.state(x) + adv - adv.mean(1, keepdim=True)
        return q.argmax(1).tolist(), x, h_pre, m

def collect(data, n_eps, acting, P=None, settings=None, max_steps=None, stride=1):
    settings = settings or DATA_SETTINGS; max_steps = max_steps or DATA_MAX_STEPS
    hyb = Hybrid(P) if acting == "hybrid" else None
    for e in tqdm(range(n_eps), desc=f"collect ({acting})", leave=False):
        L, n = settings[e % len(settings)]
        env = Environment(); obs, pos = env.reset(num_agents=n, map_length=L)
        t_hid = h_hid = None; done = False
        while not done and env.steps < max_steps:
            a_t, _, xs, post_t, m_t = team_step_layers(teacher, obs, pos, t_hid)
            t_hid = post_t
            if acting == "teacher":
                actions, in_h = a_t, xs[0]
            else:
                actions, h_hid, in_h, _ = hyb.step(obs, h_hid)
            pos_t = torch.as_tensor(pos, device=DEV).long()
            idx, valid = nb_from_pos(pos_t)
            if env.steps % stride == 0:
                data.add_step(in_h, m_t, xs[0], xs[1], pos_t, idx, valid, data.next_ep)
            (obs, pos), _, done, _ = env.step(actions)
        data.next_ep += 1
    data.build()

def sup_loss(P, b):
    p0, p1 = P(b["h"], b["m"], b["cell"], b["valid"])
    v = b["valid"]
    l0, l1 = F.mse_loss(p0[v], b["t0"][v]), F.mse_loss(p1[v], b["t1"][v])
    loss, parts = l0 + l1, dict(l0=l0.item(), l1=l1.item())
    if P.kind == "sheaf":
        en, rec = P.true_losses(b)
        loss = loss + W_SEC * en + W_AE * rec
        parts.update(sec=en.item(), ae=rec.item())
    return loss, parts

@torch.no_grad()
def holdout_metrics(P, data, n=1024, chunk=128):
    rows = data.rows_hold[torch.randperm(len(data.rows_hold))[:n]]
    P0, P1, T0, T1, CD = [], [], [], [], []
    for i in range(0, len(rows), chunk):
        b = get_batch(data, rows[i:i + chunk]); v = b["valid"]
        p0, p1 = P(b["h"], b["m"], b["cell"], v)
        P0.append(p0[v]); P1.append(p1[v]); T0.append(b["t0"][v]); T1.append(b["t1"][v]); CD.append(CELL_D[b["cell"]][v])
    P0, P1, T0, T1, CD = [torch.cat(x) for x in (P0, P1, T0, T1, CD)]
    r2 = lambda p, t: 1 - F.mse_loss(p, t).item() / max(t.var(0).mean().item(), 1e-8)
    near = CD <= 2.5
    out = dict(r2_l0=r2(P0, T0), r2_l1=r2(P1, T1), cos_l0=F.cosine_similarity(P0, T0, dim=-1).mean().item())
    if near.any() and (~near).any():
        out.update(r2_l0_near=r2(P0[near], T0[near]), r2_l0_far=r2(P0[~near], T0[~near]))
    return {k: round(v, 3) for k, v in out.items()}

def train_predictor(P, data, updates, lr, tag):
    opt = torch.optim.Adam(P.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates)
    t0 = time.time()
    for u in range(1, updates + 1):
        rows = data.rows_train[torch.randint(len(data.rows_train), (BATCH,))]
        loss, parts = sup_loss(P, get_batch(data, rows))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(P.parameters(), 5); opt.step(); sched.step()
        if u % 500 == 0 or u == updates:
            print(f"[{tag} {u}/{updates}] " + " ".join(f"{k} {v:.4f}" for k, v in parts.items()) +
                  f" | holdout {holdout_metrics(P, data)} | {time.time()-t0:.0f}s")
    return P

def run_case(case, act_fn):
    env = Environment(); env.load(case[0], case[1], case[2])
    obs, pos = env.observe(); hid, done = None, False
    while not done and env.steps < configs.max_episode_length:
        actions, hid = act_fn(obs, pos, hid)
        (obs, pos), _, done, _ = env.step(actions)
    return (np.array_equal(env.agents_pos, env.goals_pos), env.steps,
            float((env.agents_pos == env.goals_pos).all(1).mean()))

def teacher_policy(use_comm):
    def f(obs, pos, hid):
        a, _, _, post, _ = team_step_layers(teacher, obs, pos, hid, use_comm); return a, post
    return f

def hybrid_policy(P):
    hyb = Hybrid(P)
    def f(obs, pos, hid):
        a, x, _, _ = hyb.step(obs, hid); return a, x
    return f

def evaluate(policies, settings=None, n_cases=None):
    rows = []
    for (L, N) in (settings or EVAL_SETTINGS):
        with open(f"{TEST_DIR}/{L}length_{N}agents_0.3density.pth", "rb") as fh:
            tests = pickle.load(fh)[:(n_cases or N_CASES)]
        for name, fn in policies.items():
            res = [run_case(c, fn) for c in tqdm(tests, desc=f"{L}x{L} {N:>3} | {name}", leave=False)]
            rows.append(dict(map=L, agents=N, policy=name, success_pct=100 * np.mean([r[0] for r in res]),
                             avg_steps=np.mean([r[1] for r in res]), arrival_pct=100 * np.mean([r[2] for r in res]),
                             n_cases=len(tests)))
            print(f"{L}x{L} {N:>3} | {name:<24} success {rows[-1]['success_pct']:5.1f}% | steps {rows[-1]['avg_steps']:6.1f}")
    return pd.DataFrame(rows).round(2)

class OracleBlend:
    def __init__(self, P, lam):
        self.P, self.lam = P, lam
    @torch.no_grad()
    def step(self, obs_np, pos_np, hidden, t_hidden):
        net = teacher
        _, _, xs, post_t, _ = team_step_layers(net, obs_np, pos_np, t_hidden)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=DEV); n = obs.size(0)
        latent = net.obs_encoder(obs); m = net.mlp(latent)
        h_pre = net.recurrent(latent) if hidden is None else net.recurrent(latent, hidden)
        cell, valid = grid_neighbors(obs[:, 0])
        has = valid.any(-1)
        x = h_pre
        if bool(has.any()):
            preds = self.P(h_pre, m, cell, valid)
            pos = torch.as_tensor(pos_np, device=DEV).long()
            off = pos.unsqueeze(0) - pos.unsqueeze(1)
            inside = (off.abs() <= R).all(-1)
            cell_ij = torch.where(inside, (off[..., 0] + R) * G + (off[..., 1] + R), torch.full_like(off[..., 0], -1))
            j = (cell_ij.unsqueeze(1) == cell.unsqueeze(2)).float().argmax(-1)
            tokens_true = [xs[0][j], xs[1][j]]
            tv = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=DEV), valid], 1)
            attn_mask = (~tv).unsqueeze(1).expand(n, 1 + K1, 1 + K1)
            for l in range(net.comm.num_layers):
                tok = self.lam * tokens_true[l] + (1 - self.lam) * preds[l]
                info = net.comm.self_attn(torch.cat([x.unsqueeze(1), tok], 1), attn_mask=attn_mask)
                new = net.comm.update_cell(info[:, 0], x)
                x = torch.where(has.unsqueeze(-1), new, x)
        comb = torch.cat((x, m), dim=1)
        adv = net.adv(comb) if configs.Advantage_all else net.adv(x)
        q = net.state(x) + adv - adv.mean(1, keepdim=True)
        return q.argmax(1).tolist(), x, post_t

def oracle_policy(P, lam):
    ob = OracleBlend(P, lam)
    def f(obs, pos, hid):
        h, t = (None, None) if hid is None else hid
        a, x, t_new = ob.step(obs, pos, h, t)
        return a, (x, t_new)
    return f

def run_oracle_sweep():
    classes = {"sheaf_dagger": (SheafExtension, "sheaf_dagger.pt"), "mlp_dagger": (MLPPredictor, "mlp_dagger.pt")}
    policies = {}
    for name in ORACLE_KINDS:
        cls, fn = classes[name]
        path = f"{OUT_DIR}/{fn}"
        if not os.path.exists(path):
            print("missing", path); continue
        P = cls().to(DEV); P.load_state_dict(torch.load(path, map_location=DEV)); P.eval()
        for lam in ORACLE_LAMBDAS:
            policies[f"{name}_lam{lam}"] = oracle_policy(P, lam)
    df = evaluate(policies, ORACLE_SETTINGS, ORACLE_CASES)
    print("\nOracle interpolation (lam = share of REAL neighbour vectors; lam=0 is the deployable predictor):")
    print(df.to_string(index=False))
    df.to_csv("/content/sigma_oracle_sweep.csv", index=False)
    return df

if RUN_MAIN:
    data = NBData()
    collect(data, N_TEACHER_EPS, "teacher")

    Ps, metrics = {}, {}
    Ps["mean"] = MeanPredictor(data.T0[data.rows_train].float().mean(0), data.T1[data.rows_train].float().mean(0)).to(DEV)
    for kind in PRED_KINDS:
        P = (SheafExtension() if kind == "sheaf" else MLPPredictor()).to(DEV)
        path = f"{OUT_DIR}/{kind}.pt"
        if RESUME and os.path.exists(path):
            P.load_state_dict(torch.load(path, map_location=DEV)); print("reusing base predictor:", path)
        else:
            print(f"\n===== supervised training: {kind} ({sum(p.numel() for p in P.parameters())/1e3:.0f}k params) =====")
            train_predictor(P, data, TRAIN_UPDATES, LR, kind)
            torch.save(P.state_dict(), path)
        Ps[kind] = P
        metrics[kind] = holdout_metrics(P, data)

    curve = []
    for kind in PRED_KINDS:
        n_rounds = DAGGER_ROUNDS_BY_KIND.get(kind, DAGGER_ROUNDS)
        if n_rounds <= 0:
            continue
        d, P2 = data.clone(), copy.deepcopy(Ps[kind])
        dpath = f"{OUT_DIR}/{kind}_dagger.pt"
        if RESUME_DAGGER and os.path.exists(dpath):
            P2.load_state_dict(torch.load(dpath, map_location=DEV)); print("continuing DAgger from", dpath)
        for r in range(n_rounds):
            print(f"\n===== DAgger {kind} round {r+1}/{n_rounds} =====")
            collect(d, DAGGER_EPS, "hybrid", P2, DAGGER_SETTINGS, DAGGER_MAX_STEPS, STORE_EVERY_DAGGER)
            train_predictor(P2, d, DAGGER_UPDATES, LR * 0.5, f"{kind}-dagger{r+1}")
            curve.append(evaluate({f"{kind}_dagger_round{r+1}": hybrid_policy(P2.eval())},
                                  ROUND_EVAL_SETTINGS, ROUND_EVAL_CASES))
            torch.save(P2.state_dict(), f"{OUT_DIR}/{kind}_dagger.pt")
        Ps[f"{kind}_dagger"] = P2
        metrics[f"{kind}_dagger (on-policy holdout)"] = holdout_metrics(P2, d)
        metrics[f"{kind}_dagger (teacher holdout)"] = holdout_metrics(P2, data)
        del d; gc.collect(); torch.cuda.empty_cache()

    if curve:
        cdf = pd.concat(curve)
        print("\nDAgger learning curve (quick eval, %d cases/setting):" % ROUND_EVAL_CASES)
        print(cdf.to_string(index=False))
    print("\nSupervised hold-out metrics (R2 = 1 - MSE/Var of the real neighbour vectors; 0 = predicting the mean)")
    print(pd.DataFrame(metrics).T.to_string())

    policies = {"teacher_full_comm": teacher_policy(True), "teacher_no_comm": teacher_policy(False)}
    for name, P in Ps.items():
        policies[f"hybrid_{name}"] = hybrid_policy(P.eval())
    df = evaluate(policies)
    df.to_csv("/content/sigma_predictor_results.csv", index=False)
    print("\n" + df.to_string(index=False))
    print("\nsaved -> /content/sigma_predictor_results.csv ; predictors in", OUT_DIR)
    print("Read: hybrid_sheaf vs hybrid_mlp (graph structure) vs hybrid_mean (presence only) vs teacher_no_comm (floor) vs full comm (ceiling).")

if ORACLE_SWEEP:
    run_oracle_sweep()
