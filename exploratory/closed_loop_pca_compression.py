import os, sys, glob, time, types, pickle, random, zipfile, subprocess, warnings, math
warnings.filterwarnings("ignore")

SMOKE           = False
KS              = [0, 1, 2, 4, 8, 16, 32, 256]
QUANT           = [(1, 8), (2, 8), (4, 8), (2, 4), (32, 8)]
METHODS         = ["pca"]
PRED_KIND       = "mlp_dagger"
SETTINGS        = [(40, 32), (40, 64)]
N_CASES         = 30
INCLUDE_REFERENCE = True
PCA_EPISODES    = 20
PCA_SETTINGS    = [(30, 16), (40, 32), (40, 64), (40, 64), (40, 96)]
SEED            = 0

REPO_URL = "https://github.com/marmotlab/SIGMA.git"
ROOT     = "/content/SIGMA"
CKPT     = "models/save_model/model_house/84000_house.pth"
TEST_DIR = "house_test_set"
OUT_DIR  = "/content/sigma_predictors"
D_S, D_E, K_DIFF = 64, 32, 3
if SMOKE:
    KS, QUANT, SETTINGS, N_CASES, PCA_EPISODES, PCA_SETTINGS = [0, 4, 256], [(4, 8)], [(40, 8)], 2, 3, [(40, 8)]
_pred_file = {"mlp_dagger": "mlp_dagger.pt", "sheaf_dagger": "sheaf_dagger.pt",
              "mlp": "mlp.pt", "sheaf": "sheaf.pt"}[PRED_KIND]
if "pred_pca" in METHODS and not os.path.exists(f"{OUT_DIR}/{_pred_file}"):
    print(f"WARNING: {OUT_DIR}/{_pred_file} not found -> running without pred_pca.")
    METHODS = [m for m in METHODS if m != "pred_pca"]

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

R = configs.obs_radius; G = 2 * R + 1; NC = G * G; NCR = (4 * R + 1) ** 2
H = configs.hidden_dim; KCOMM = configs.max_comm_agents; K1 = KCOMM - 1
CELL_D = torch.tensor([[math.hypot(r - R, c - R) for c in range(G)] for r in range(G)],
                      dtype=torch.float32, device=DEV).view(-1)

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

CLASSES = {"mlp_dagger": MLPPredictor, "sheaf_dagger": SheafExtension, "mlp": MLPPredictor, "sheaf": SheafExtension}

def collect_vectors(n_eps, settings, max_steps=300, stride=3):
    V0, V1 = [], []
    for e in tqdm(range(n_eps), desc="teacher rollouts for PCA", leave=False):
        L, n = settings[e % len(settings)]
        env = Environment(); obs, pos = env.reset(num_agents=n, map_length=L)
        hid, done = None, False
        while not done and env.steps < max_steps:
            a, _, xs, post, _ = team_step_layers(teacher, obs, pos, hid); hid = post
            if env.steps % stride == 0:
                V0.append(xs[0].cpu()); V1.append(xs[1].cpu())
            (obs, pos), _, done, _ = env.step(a)
    return torch.cat(V0).float(), torch.cat(V1).float()

def fit_pca(V):
    mean = V.mean(0)
    X = V - mean
    cov = X.T @ X / max(len(X) - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    order = evals.argsort(descending=True)
    return mean.to(DEV), evecs[:, order].to(DEV), evals[order].clamp(min=1e-12).to(DEV)

bases_path = f"{OUT_DIR}/pca_bases.pt"
if os.path.exists(bases_path) and not SMOKE:
    BASES = [tuple(t.to(DEV) for t in b) for b in torch.load(bases_path, map_location="cpu")]
    print("loaded PCA bases:", bases_path)
else:
    V0, V1 = collect_vectors(PCA_EPISODES, PCA_SETTINGS)
    BASES = [fit_pca(V0), fit_pca(V1)]
    print(f"PCA fitted on {len(V0)} real vectors per layer")
    if not SMOKE:
        torch.save([tuple(t.cpu() for t in b) for b in BASES], bases_path)
    del V0, V1
for l, (_, _, ev) in enumerate(BASES):
    cum = (ev.cumsum(0) / ev.sum()).cpu().numpy()
    print(f"layer {l} cumulative variance explained: " +
          ", ".join(f"k={k}: {cum[min(k, len(cum))-1]*100:.0f}%" for k in KS if k > 0))

def quantize(c, sigma, bits):
    if bits is None or c.numel() == 0:
        return c
    z = (c / sigma).clamp(-3, 3)
    step = 6.0 / (2 ** bits - 1)
    return (torch.round((z + 3) / step) * step - 3) * sigma

class ClosedLoopCompressed:
    def __init__(self, P, method, k, bits):
        self.P, self.method, self.k, self.bits = P, method, k, bits
    @torch.no_grad()
    def step(self, obs_np, pos_np, hidden):
        net = teacher
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=DEV); n = obs.size(0)
        latent = net.obs_encoder(obs); m = net.mlp(latent)
        h_pre = net.recurrent(latent) if hidden is None else net.recurrent(latent, hidden)
        cell, valid = grid_neighbors(obs[:, 0])
        has = valid.any(-1)
        x = h_pre
        if bool(has.any()):
            pos = torch.as_tensor(pos_np, device=DEV).long()
            off = pos.unsqueeze(0) - pos.unsqueeze(1)
            inside = (off.abs() <= R).all(-1)
            cell_ij = torch.where(inside, (off[..., 0] + R) * G + (off[..., 1] + R), torch.full_like(off[..., 0], -1))
            j = (cell_ij.unsqueeze(1) == cell.unsqueeze(2)).float().argmax(-1)
            preds = self.P(h_pre, m, cell, valid) if self.method == "pred_pca" else None
            tv = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=DEV), valid], 1)
            attn_mask = (~tv).unsqueeze(1).expand(n, 1 + K1, 1 + K1)
            for l in range(net.comm.num_layers):
                real = x[j]
                mean, U, ev = BASES[l]
                Uk = U[:, :self.k]
                c = quantize((real - mean) @ Uk, ev[:self.k].sqrt(), self.bits)
                if self.method == "pca":
                    tok = mean + c @ Uk.T
                else:
                    p = preds[l]
                    tok = p + (c - (p - mean) @ Uk) @ Uk.T
                info = net.comm.self_attn(torch.cat([x.unsqueeze(1), tok], 1), attn_mask=attn_mask)
                new = net.comm.update_cell(info[:, 0], x)
                x = torch.where(has.unsqueeze(-1), new, x)
        comb = torch.cat((x, m), dim=1)
        adv = net.adv(comb) if configs.Advantage_all else net.adv(x)
        q = net.state(x) + adv - adv.mean(1, keepdim=True)
        return q.argmax(1).tolist(), x

def compressed_policy(P, method, k, bits):
    ob = ClosedLoopCompressed(P, method, k, bits)
    def f(obs, pos, hid):
        return ob.step(obs, pos, hid)
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

def evaluate(policies, settings, n_cases):
    rows = []
    for (L, N) in settings:
        with open(f"{TEST_DIR}/{L}length_{N}agents_0.3density.pth", "rb") as fh:
            tests = pickle.load(fh)[:n_cases]
        for name, fn in policies.items():
            res = [run_case(c, fn) for c in tqdm(tests, desc=f"{L}x{L} {N:>3} | {name}", leave=False)]
            rows.append(dict(agents=N, policy=name, success_pct=100 * np.mean([r[0] for r in res]),
                             avg_steps=np.mean([r[1] for r in res]), arrival_pct=100 * np.mean([r[2] for r in res]),
                             n_cases=len(tests)))
            print(f"{L}x{L} {N:>3} | {name:<26} success {rows[-1]['success_pct']:5.1f}% | steps {rows[-1]['avg_steps']:6.1f} "
                  f"| arrival {rows[-1]['arrival_pct']:5.1f}%")
    return pd.DataFrame(rows).round(2)

P = None
if "pred_pca" in METHODS:
    P = CLASSES[PRED_KIND]().to(DEV)
    P.load_state_dict(torch.load(f"{OUT_DIR}/{_pred_file}", map_location=DEV)); P.eval()

policies, meta = {}, {}
if INCLUDE_REFERENCE:
    policies["REF full_comm"], meta["REF full_comm"] = teacher_policy(True), ("reference", None, None)
    policies["REF no_comm"], meta["REF no_comm"] = teacher_policy(False), ("reference", None, None)
for method in METHODS:
    for k in KS:
        name = f"{method} k={k}"
        policies[name] = compressed_policy(P, method, k, None); meta[name] = (method, k, None)
    for (k, bits) in QUANT:
        name = f"{method} k={k} {bits}bit"
        policies[name] = compressed_policy(P, method, k, bits); meta[name] = (method, k, bits)

t0 = time.time()
df = evaluate(policies, SETTINGS, N_CASES)
print(f"\nelapsed {time.time()-t0:.0f}s")

agents = sorted(df.agents.unique())
rows = []
for name, (method, k, bits) in meta.items():
    r = dict(policy=name, method=method, k=k, bits=bits)
    if k is not None:
        r["numbers_per_neighbour"] = 2 * k
        r["bits_per_neighbour"] = 2 * k * (32 if bits is None else bits)
    for N in agents:
        g = df[(df.agents == N) & (df.policy == name)].iloc[0]
        r[f"success@{N}"] = g.success_pct
        if INCLUDE_REFERENCE:
            full = df[(df.agents == N) & (df.policy == "REF full_comm")].success_pct.iloc[0]
            none = df[(df.agents == N) & (df.policy == "REF no_comm")].success_pct.iloc[0]
            r[f"gap_recovered%@{N}"] = round((g.success_pct - none) / (full - none) * 100, 1) if full > none else float("nan")
    rows.append(r)
summary = pd.DataFrame(rows)
print("\nClosed-loop success vs message size (2 exchanges per step, k numbers per visible neighbour per exchange):")
print(summary.to_string(index=False))
df.to_csv("/content/sigma_closed_loop_results.csv", index=False)
summary.to_csv("/content/sigma_closed_loop_summary.csv", index=False)
print("\nsaved -> /content/sigma_closed_loop_results.csv, sigma_closed_loop_summary.csv")
print("Sanity: 'k=256' should match REF full_comm. Bits count raw numbers only (no entropy coding).")
