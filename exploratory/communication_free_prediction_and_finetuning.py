import os, sys, re, glob, time, types, pickle, random, zipfile, subprocess, warnings, math, copy, collections
warnings.filterwarnings("ignore")

SMOKE          = False
TRAIN_MODES    = ["nocomm", "nocomm_distill", "predictor"]
UPDATES        = 4000
PRE_UPDATES    = 1500
START_EPISODES = 24
N_ENVS         = 8
BATCH          = 128
LR, LR_PRED, LR_PRE = 5e-5, 3e-4, 1e-3
TGT_FREQ       = 500
EP_CAP         = 300
MAP_SIZES      = [10, 15, 20, 25, 30, 35, 40]
W_DQ, W_H      = 1.0, 1.0
W_PRED, W_AE, W_SHEAF = 1.0, 0.5, 0.1
EVAL_SETTINGS  = [(40, 8), (40, 32), (40, 64), (40, 128)]
N_CASES        = 20
EVAL_TEACHER_ROWS = True
SEED           = 0
FORCE_RETRAIN  = False
CKPT_EVERY     = 500
DEBUG_SYNC     = False

REPO_URL = "https://github.com/marmotlab/SIGMA.git"
ROOT     = "/content/SIGMA"
CKPT     = "models/save_model/model_house/84000_house.pth"
TEST_DIR = "house_test_set"
OUT_DIR  = "/content/sigma_ckpts"

if DEBUG_SYNC:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if SMOKE:
    TRAIN_MODES, UPDATES, PRE_UPDATES, START_EPISODES = ["nocomm", "predictor"], 20, 10, 4
    EVAL_SETTINGS, N_CASES, N_ENVS, BATCH = [(40, 8)], 2, 4, 32

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

R = configs.obs_radius; G = 2 * R + 1; NC = G * G
H = configs.hidden_dim; NMAX = configs.max_num_agents; KCOMM = configs.max_comm_agents
SEQ = configs.seq_len; FS = configs.forward_steps; W = SEQ + FS
GAMMA = 0.99; MAXEP = configs.max_episode_length
CELL_D = torch.tensor([[math.hypot(r - R, c - R) for c in range(G)] for r in range(G)],
                      dtype=torch.float32, device=DEV).view(-1)

def load_base():
    n = Network()
    n.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
    return n

teacher = load_base().eval().to(DEV)
for p_ in teacher.parameters():
    p_.requires_grad_(False)

@torch.no_grad()
def team_step(net, obs_np, pos_np, hidden, use_comm=True, k=KCOMM):
    obs = torch.as_tensor(obs_np.astype(np.float32), device=DEV)
    pos = torch.as_tensor(pos_np.astype(np.float32), device=DEV)
    n = obs.size(0)
    latent = net.obs_encoder(obs)
    mlp_out = net.mlp(latent)
    h_pre = net.recurrent(latent) if hidden is None else net.recurrent(latent, hidden)
    h = h_pre
    pos_mat = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
    dist = (pos_mat[..., 0] ** 2 + pos_mat[..., 1] ** 2).sqrt()
    in_obs = (pos_mat <= R).all(2)
    _, rank = dist.topk(min(k, n), dim=1, largest=False)
    near = torch.zeros((n, n), dtype=torch.bool, device=DEV)
    near.scatter_(1, rank, True)
    comm_mask = in_obs & near
    if use_comm:
        upd = comm_mask.sum(-1) > 1
        if bool(upd.any()):
            x = h.unsqueeze(0)
            attn_mask = ~comm_mask
            for _ in range(net.comm.num_layers):
                info = net.comm.self_attn(x, attn_mask=attn_mask)
                new = net.comm.update_cell(info[0], x[0])
                x = torch.where(upd[None, :, None], new.unsqueeze(0), x)
            h = x[0]
    comb = torch.cat((h, mlp_out), dim=1)
    adv = net.adv(comb) if configs.Advantage_all else net.adv(h)
    q = net.state(h) + adv - adv.mean(1, keepdim=True)
    return q.argmax(1).tolist(), q, h_pre, h, mlp_out

class SheafPredictor(nn.Module):
    def __init__(self, h_dim=H, m_dim=1, d_s=64, d_e=32, n_cell=NC):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(h_dim + m_dim, 128), nn.ReLU(), nn.Linear(128, d_s))
        self.dec = nn.Sequential(nn.Linear(d_s, 128), nn.ReLU(), nn.Linear(128, h_dim))
        self.Rs = nn.Parameter(torch.randn(n_cell, d_e, d_s) / d_s ** 0.5)
        self.Rd = nn.Parameter(torch.randn(n_cell, d_e, d_s) / d_s ** 0.5)
        self.G = nn.Parameter(torch.randn(n_cell, d_s, d_e) / d_e ** 0.5)
        self.b = nn.Parameter(torch.zeros(n_cell, d_s))
    def stalk(self, h, m):
        return self.enc(torch.cat([h, m], -1))
    @staticmethod
    def _mats(P, idx):
        idx = idx.clamp(0, P.size(0) - 1)
        return F.embedding(idx, P.reshape(P.size(0), -1)).view(idx.numel(), P.size(1), P.size(2))
    def edge_src(self, sx, idx):
        return torch.bmm(self._mats(self.Rs, idx), sx.unsqueeze(-1)).squeeze(-1)
    def edge_dst(self, sx, idx):
        return torch.bmm(self._mats(self.Rd, idx), sx.unsqueeze(-1)).squeeze(-1)
    def predict_h(self, sx, idx):
        e = self.edge_src(sx, idx)
        sp = torch.bmm(self._mats(self.G, idx), e.unsqueeze(-1)).squeeze(-1) \
             + F.embedding(idx.clamp(0, NC - 1), self.b)
        return self.dec(sp)

def grid_neighbors(agent_grid):
    g = agent_grid.reshape(*agent_grid.shape[:-2], NC) > 0.5
    score = torch.where(g, -CELL_D, torch.full_like(CELL_D, -1e9))
    val, idx = score.topk(KCOMM - 1, dim=-1)
    return idx, val > -1e8

class Student(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.net = load_base()
        self.pred = SheafPredictor() if kind == "predictor" else None

    def core(self, lat, h_prev, idx, valid):
        m = self.net.mlp(lat)
        h_pre = self.net.recurrent(lat) if h_prev is None else self.net.recurrent(lat, h_prev)
        h = h_pre
        if self.kind == "predictor":
            h = self.comm_pred(h_pre, m, idx, valid)
        return h_pre, h, m

    def comm_pred(self, h, m, idx, valid):
        has = valid.any(-1)
        if not bool(has.any()):
            return h
        B, K1 = idx.shape
        comm = self.net.comm
        sx = self.pred.stalk(h, m)
        sx_r = sx.unsqueeze(1).expand(B, K1, sx.size(-1)).reshape(B * K1, -1)
        hp = self.pred.predict_h(sx_r, idx.reshape(-1)).view(B, K1, -1)
        tv = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=h.device), valid], 1)
        attn_mask = (~tv).unsqueeze(1).expand(B, 1 + K1, 1 + K1)
        x = h
        for _ in range(comm.num_layers):
            tokens = torch.cat([x.unsqueeze(1), hp], 1)
            info = comm.self_attn(tokens, attn_mask=attn_mask)
            new = comm.update_cell(info[:, 0], x)
            x = torch.where(has.unsqueeze(-1), new, x)
        return x

    def q_from(self, h, m):
        comb = torch.cat([h, m], 1)
        adv = self.net.adv(comb) if configs.Advantage_all else self.net.adv(h)
        return self.net.state(h) + adv - adv.mean(1, keepdim=True)

    @torch.no_grad()
    def act(self, obs_np, hidden):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=DEV)
        lat = self.net.obs_encoder(obs)
        idx = valid = None
        if self.kind == "predictor":
            idx, valid = grid_neighbors(obs[:, 0])
        _, h, m = self.core(lat, hidden, idx, valid)
        q = self.q_from(h, m)
        return q.argmax(1).tolist(), q, h

    def seq_forward(self, obs, h0, gather_idx):
        B, T = obs.shape[:2]
        lat = self.net.obs_encoder(obs.reshape(B * T, *obs.shape[2:])).view(B, T, -1)
        idx = valid = None
        if self.kind == "predictor":
            idx, valid = grid_neighbors(obs[:, :, 0])
        h, hs, ms = h0, [], []
        for t in range(T):
            _, h, m = self.core(lat[:, t], h, None if idx is None else idx[:, t],
                                None if valid is None else valid[:, t])
            hs.append(h); ms.append(m)
        Hall, Mall = torch.stack(hs, 1), torch.stack(ms, 1)
        ar = torch.arange(B, device=obs.device)
        q = self.q_from(Hall[ar, gather_idx], Mall[ar, gather_idx])
        return q, Hall

    def pred_losses(self, b):
        P = self.pred
        t_pre, t_m, pos = b["t_pre"], b["t_m"], b["pos"].long()
        step_ok = torch.arange(SEQ, device=DEV)[None, :] < b["seq_len"][:, None]
        d = pos[:, :, 1:] - pos[:, :, :1]
        valid = (d.abs() <= R).all(-1) & step_ok[:, :, None]
        idx = ((d[..., 0] + R) * G + (d[..., 1] + R)).clamp(0, NC - 1)
        hi = t_pre[:, :, 0:1].expand(-1, -1, NMAX - 1, -1)[valid]
        mi = t_m[:, :, 0:1].expand(-1, -1, NMAX - 1, -1)[valid]
        hj, mj, ix = t_pre[:, :, 1:][valid], t_m[:, :, 1:][valid], idx[valid]
        if hi.shape[0] == 0:
            z = sum(p.sum() for p in P.parameters()) * 0
            return z, z, z, 0.0
        sx_i, sx_j = P.stalk(hi, mi), P.stalk(hj, mj)
        l_pred = F.mse_loss(P.predict_h(sx_i, ix), hj)
        l_ae = 0.5 * (F.mse_loss(P.dec(sx_j), hj) + F.mse_loss(P.dec(sx_i), hi))
        l_sheaf = F.mse_loss(P.edge_src(sx_i, ix), P.edge_dst(sx_j.detach(), ix))
        r2 = 1.0 - l_pred.item() / max(hj.var(0).mean().item(), 1e-8)
        return l_pred, l_ae, l_sheaf, r2

class Replay:
    def __init__(self, cap):
        self.cap, self.eps = cap, []
    def add(self, ep):
        self.eps.append(ep)
        if len(self.eps) > self.cap:
            self.eps.pop(0)
    def n_trans(self):
        return sum(e["T"] for e in self.eps)
    def sample(self, B):
        sizes = np.array([e["T"] for e in self.eps], dtype=np.float64)
        ids = np.random.choice(len(self.eps), B, p=sizes / sizes.sum())
        keys = ["obs", "h0", "seq_len", "steps", "act", "rew", "done", "pos", "t_pre", "t_m", "t_post0", "t_q0"]
        out = {k: [] for k in keys}
        has_t = "t_pre" in self.eps[0]
        for ei in ids:
            ep = self.eps[ei]; T, N = ep["T"], ep["N"]
            t = np.random.randint(T)
            steps = min(FS, T - t); seq_len = min(t + 1, SEQ)
            start = 0 if t <= SEQ - 1 else t - SEQ + 1
            h0 = np.zeros(H, np.float16) if t <= SEQ - 1 else ep["hid"][t - SEQ]
            Lw = seq_len + steps
            ow = np.zeros((W, 6, G, G), bool); ow[:Lw] = ep["obs0"][start:start + Lw]
            pw = np.full((SEQ, NMAX, 2), -1000, np.int16); pw[:seq_len, :N] = ep["pos"][start:start + seq_len]
            out["obs"].append(ow); out["h0"].append(h0); out["seq_len"].append(seq_len); out["steps"].append(steps)
            out["act"].append(ep["act"][t])
            out["rew"].append(sum(float(ep["rew"][t + k]) * GAMMA ** k for k in range(steps)))
            out["done"].append(float(ep["done"] and t >= T - FS)); out["pos"].append(pw)
            if has_t:
                tp = np.zeros((SEQ, NMAX, H), np.float16); tp[:seq_len, :N] = ep["t_pre"][start:start + seq_len]
                tm = np.zeros((SEQ, NMAX, 1), np.float16); tm[:seq_len, :N] = ep["t_m"][start:start + seq_len]
                tpo = np.zeros((SEQ, H), np.float16); tpo[:seq_len] = ep["t_post0"][start:start + seq_len]
                out["t_pre"].append(tp); out["t_m"].append(tm); out["t_post0"].append(tpo)
                out["t_q0"].append(ep["t_q0"][t])
        f = lambda k: torch.from_numpy(np.stack(out[k])).to(DEV).float()
        b = dict(obs=f("obs"), h0=f("h0"), pos=torch.from_numpy(np.stack(out["pos"])).to(DEV),
                 seq_len=torch.tensor(out["seq_len"], device=DEV), steps=torch.tensor(out["steps"], device=DEV),
                 act=torch.tensor(out["act"], device=DEV).long(),
                 rew=torch.tensor(out["rew"], device=DEV).float(), done=torch.tensor(out["done"], device=DEV).float())
        if has_t:
            for k in ("t_pre", "t_m", "t_post0", "t_q0"):
                b[k] = f(k)
        return b

class Collector:
    def __init__(self, n_envs, store_teacher, replay):
        self.store_teacher, self.replay = store_teacher, replay
        self.slots, self.recent, self.total = [], collections.deque(maxlen=100), 0
        for i in range(n_envs):
            sl = dict(env=Environment(), eps=0.4 ** (1 + (i / max(n_envs - 1, 1)) * 7))
            self._reset(sl); self.slots.append(sl)
    def _reset(self, sl):
        n, L = random.randint(3, NMAX), random.choice(MAP_SIZES)
        obs, pos = sl["env"].reset(num_agents=n, map_length=L)
        sl.update(obs=obs, pos=pos, s_hid=None, t_hid=None, obs0=[obs[0].copy()], pos_l=[pos.copy()],
                  act=[], rew=[], hid=[], t_pre=[], t_m=[], t_post0=[], t_q0=[])
    def step_round(self, policy, S):
        for sl in self.slots:
            obs, pos = sl["obs"], sl["pos"]
            if self.store_teacher:
                a_t, q_t, pre_t, post_t, m_t = team_step(teacher, obs, pos, sl["t_hid"])
                sl["t_hid"] = post_t
                sl["t_pre"].append(pre_t.cpu().numpy().astype(np.float16))
                sl["t_m"].append(m_t.cpu().numpy().astype(np.float16))
                sl["t_post0"].append(post_t[0].cpu().numpy().astype(np.float16))
                sl["t_q0"].append(q_t[0].cpu().numpy().astype(np.float32))
            if policy == "teacher":
                actions, h0 = list(a_t), post_t[0]
            else:
                actions, _, h_new = S.act(obs, sl["s_hid"]); sl["s_hid"] = h_new; h0 = h_new[0]
            if random.random() < sl["eps"]:
                actions[0] = random.randint(0, 4)
            (nobs, npos), rewards, done, _ = sl["env"].step(actions)
            sl["act"].append(actions[0]); sl["rew"].append(float(rewards[0]))
            sl["hid"].append(h0.cpu().numpy().astype(np.float16))
            sl["obs0"].append(nobs[0].copy()); sl["pos_l"].append(npos.copy())
            sl["obs"], sl["pos"] = nobs, npos
            self.total += 1
            if done or sl["env"].steps >= MAXEP:
                T = len(sl["act"])
                ep = dict(T=T, N=nobs.shape[0], obs0=np.stack(sl["obs0"]), pos=np.stack(sl["pos_l"]).astype(np.int16),
                          act=np.array(sl["act"], np.int64), rew=np.array(sl["rew"], np.float32),
                          hid=np.stack(sl["hid"]), done=bool(done))
                if self.store_teacher:
                    ep.update(t_pre=np.stack(sl["t_pre"]), t_m=np.stack(sl["t_m"]),
                              t_post0=np.stack(sl["t_post0"]), t_q0=np.stack(sl["t_q0"]))
                self.replay.add(ep); self.recent.append((float(done), T)); self._reset(sl)

MODE_CFG = {
    "nocomm":         dict(kind="nocomm",    distill=False, pretrain=False),
    "nocomm_distill": dict(kind="nocomm",    distill=True,  pretrain=False),
    "predictor":      dict(kind="predictor", distill=True,  pretrain=True),
}

def train_mode(mode):
    cfg = MODE_CFG[mode]
    print(f"\n===== training: {mode} {cfg} =====")
    done_path = f"{OUT_DIR}/{mode}.pt"
    if os.path.exists(done_path) and not FORCE_RETRAIN:
        S = Student(cfg["kind"]).to(DEV); S.load_state_dict(torch.load(done_path, map_location=DEV))
        print("reusing finished checkpoint:", done_path); return S.eval()
    S = Student(cfg["kind"]).to(DEV); Tgt = copy.deepcopy(S).eval()
    for p_ in Tgt.parameters():
        p_.requires_grad_(False)
    replay = Replay(EP_CAP)
    col = Collector(N_ENVS, cfg["distill"], replay)
    policy = "teacher" if cfg["pretrain"] else "student"
    t0 = time.time()
    while len(replay.eps) < START_EPISODES:
        col.step_round(policy, S)
    print(f"warm-up: {len(replay.eps)} episodes / {replay.n_trans()} transitions ({time.time()-t0:.0f}s)")

    pre_path = f"{OUT_DIR}/{mode}_pred_pretrained.pt"
    if cfg["pretrain"] and os.path.exists(pre_path) and not FORCE_RETRAIN:
        S.pred.load_state_dict(torch.load(pre_path, map_location=DEV))
        print("reusing pretrained predictor:", pre_path); policy = "student"
    elif cfg["pretrain"]:
        opt_p = torch.optim.Adam(S.pred.parameters(), lr=LR_PRE)
        for u in range(1, PRE_UPDATES + 1):
            col.step_round("teacher", S)
            b = replay.sample(BATCH)
            l_pred, l_ae, l_sh, r2 = S.pred_losses(b)
            loss = W_PRED * l_pred + W_AE * l_ae + W_SHEAF * l_sh
            opt_p.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(S.pred.parameters(), 10); opt_p.step()
            if u % 250 == 0 or u == PRE_UPDATES:
                print(f"[pre {u}] pred {l_pred.item():.4f} ae {l_ae.item():.4f} sheaf {l_sh.item():.4f} "
                      f"R2(neighbour state) {r2:.3f}")
        torch.save(S.pred.state_dict(), pre_path)
        policy = "student"

    other = [p for n_, p in S.named_parameters() if not n_.startswith("pred.")]
    groups = [dict(params=other, lr=LR)]
    if S.pred is not None:
        groups.append(dict(params=list(S.pred.parameters()), lr=LR_PRED))
    opt = torch.optim.Adam(groups)
    for u in range(1, UPDATES + 1):
        col.step_round(policy, S)
        b = replay.sample(BATCH)
        q_on, Hall = S.seq_forward(b["obs"][:, :SEQ], b["h0"], b["seq_len"] - 1)
        with torch.no_grad():
            q_nx, _ = Tgt.seq_forward(b["obs"], b["h0"], b["seq_len"] + b["steps"] - 1)
            tgt = b["rew"].unsqueeze(1) + (GAMMA ** b["steps"].float()).unsqueeze(1) * \
                  (1 - b["done"].unsqueeze(1)) * q_nx.max(1, keepdim=True)[0]
        l_td = F.smooth_l1_loss(q_on.gather(1, b["act"].unsqueeze(1)), tgt)
        loss, logs = l_td, dict(td=l_td.item())
        if cfg["distill"]:
            step_ok = torch.arange(SEQ, device=DEV)[None, :] < b["seq_len"][:, None]
            l_dq = F.smooth_l1_loss(q_on, b["t_q0"])
            l_dh = ((Hall - b["t_post0"]) ** 2).mean(-1)[step_ok].mean()
            loss = loss + W_DQ * l_dq + W_H * l_dh
            logs.update(dq=l_dq.item(), dh=l_dh.item())
        if cfg["kind"] == "predictor":
            l_pred, l_ae, l_sh, r2 = S.pred_losses(b)
            loss = loss + W_PRED * l_pred + W_AE * l_ae + W_SHEAF * l_sh
            logs.update(pred=l_pred.item(), r2=r2)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(S.parameters(), 40); opt.step()
        if u % TGT_FREQ == 0:
            Tgt.load_state_dict(S.state_dict())
        if u % CKPT_EVERY == 0:
            torch.save(S.state_dict(), f"{OUT_DIR}/{mode}_partial.pt")
        if u % 200 == 0 or u == UPDATES:
            rec = col.recent
            sr = np.mean([r[0] for r in rec]) if rec else float("nan")
            ln = np.mean([r[1] for r in rec]) if rec else float("nan")
            print(f"[{mode} {u}/{UPDATES}] " + " ".join(f"{k} {v:.4f}" for k, v in logs.items()) +
                  f" | train-episodes success {sr:.2f} len {ln:.0f} | {time.time()-t0:.0f}s")
    torch.save(S.state_dict(), f"{OUT_DIR}/{mode}.pt")
    return S.eval()

def run_case(case, act_fn):
    env = Environment(); env.load(case[0], case[1], case[2])
    obs, pos = env.observe(); hid, steps, done = None, 0, False
    while not done and env.steps < configs.max_episode_length:
        actions, hid = act_fn(obs, pos, hid)
        (obs, pos), _, done, _ = env.step(actions); steps += 1
    return (np.array_equal(env.agents_pos, env.goals_pos), steps,
            float((env.agents_pos == env.goals_pos).all(1).mean()))

def teacher_policy(use_comm, k):
    def f(obs, pos, hid):
        a, _, _, post, _ = team_step(teacher, obs, pos, hid, use_comm, k); return a, post
    return f

def student_policy(S):
    def f(obs, pos, hid):
        a, _, h = S.act(obs, hid); return a, h
    return f

def evaluate(policies):
    rows = []
    for (L, N) in EVAL_SETTINGS:
        with open(f"{TEST_DIR}/{L}length_{N}agents_0.3density.pth", "rb") as fh:
            tests = pickle.load(fh)[:N_CASES]
        for name, fn in policies.items():
            res = [run_case(c, fn) for c in tqdm(tests, desc=f"{L}x{L} {N:>3} | {name}", leave=False)]
            rows.append(dict(map=L, agents=N, policy=name, success_pct=100 * np.mean([r[0] for r in res]),
                             avg_steps=np.mean([r[1] for r in res]), arrival_pct=100 * np.mean([r[2] for r in res]),
                             n_cases=len(tests)))
            print(f"{L}x{L} {N:>3} | {name:<22} success {rows[-1]['success_pct']:5.1f}% | steps {rows[-1]['avg_steps']:6.1f}")
    return pd.DataFrame(rows).round(2)

students = {m: train_mode(m) for m in TRAIN_MODES}
for m in MODE_CFG:
    if m not in students and os.path.exists(f"{OUT_DIR}/{m}.pt"):
        S_ = Student(MODE_CFG[m]["kind"]).to(DEV); S_.load_state_dict(torch.load(f"{OUT_DIR}/{m}.pt", map_location=DEV))
        students[m] = S_.eval(); print("loaded finished checkpoint for evaluation:", m)
policies = {}
if EVAL_TEACHER_ROWS:
    policies["teacher_full_comm"] = teacher_policy(True, KCOMM)
    policies["teacher_zero_shot_no_comm"] = teacher_policy(False, 1)
for m, S in students.items():
    policies[f"finetuned_{m}"] = student_policy(S)
df = evaluate(policies)
df.to_csv("/content/sigma_nocomm_predictor_results.csv", index=False)
print("\n" + df.to_string(index=False))
print("\nsaved -> /content/sigma_nocomm_predictor_results.csv ; checkpoints in", OUT_DIR)
print("Compare: finetuned_nocomm (floor)  vs  finetuned_nocomm_distill (distillation only)  vs  finetuned_predictor.")
print("The predictor only helps if it beats nocomm_distill, not just nocomm.")
