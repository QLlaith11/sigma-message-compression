import os, sys, glob, time, types, pickle, random, zipfile, subprocess, warnings, math
warnings.filterwarnings("ignore")

SMOKE           = False
KINDS           = ["mlp_dagger"]
LAMBDAS         = [0.0, 0.5, 1.0]
NOISE_R2        = [0.8, 0.6, 0.4]
SETTINGS        = [(40, 32), (40, 64)]
N_CASES         = 30
INCLUDE_REFERENCE = True
SEED            = 0

REPO_URL = "https://github.com/marmotlab/SIGMA.git"
ROOT     = "/content/SIGMA"
CKPT     = "models/save_model/model_house/84000_house.pth"
TEST_DIR = "house_test_set"
OUT_DIR  = "/content/sigma_predictors"

D_S, D_E, K_DIFF = 64, 32, 3
if SMOKE:
    LAMBDAS, NOISE_R2, SETTINGS, N_CASES = [0.0, 1.0], [0.6], [(40, 8)], 2

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

CLASSES = {"mlp_dagger": (MLPPredictor, "mlp_dagger.pt"), "sheaf_dagger": (SheafExtension, "sheaf_dagger.pt"),
           "mlp": (MLPPredictor, "mlp.pt"), "sheaf": (SheafExtension, "sheaf.pt")}

def load_predictor(name):
    cls, fn = CLASSES[name]
    path = f"{OUT_DIR}/{fn}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run sigma_sheaf_predictor_supervised_colab.py first "
                                f"(it saves the predictors to {OUT_DIR}), or set OUT_DIR to where you copied them.")
    P = cls().to(DEV)
    P.load_state_dict(torch.load(path, map_location=DEV))
    return P.eval()

class OracleHybrid:
    def __init__(self, P, mode, value):
        self.P, self.mode, self.value = P, mode, value
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
            pos = torch.as_tensor(pos_np, device=DEV).long()
            off = pos.unsqueeze(0) - pos.unsqueeze(1)
            inside = (off.abs() <= R).all(-1)
            cell_ij = torch.where(inside, (off[..., 0] + R) * G + (off[..., 1] + R), torch.full_like(off[..., 0], -1))
            j = (cell_ij.unsqueeze(1) == cell.unsqueeze(2)).float().argmax(-1)
            real = [xs[0][j], xs[1][j]]
            if self.mode == "blend":
                preds = self.P(h_pre, m, cell, valid)
                toks = [self.value * real[l] + (1 - self.value) * preds[l] for l in range(2)]
            else:
                toks = []
                for l in range(2):
                    sigma = torch.sqrt((1 - self.value) * xs[l].var(0).mean())
                    toks.append(real[l] + sigma * torch.randn_like(real[l]))
            tv = torch.cat([torch.ones(n, 1, dtype=torch.bool, device=DEV), valid], 1)
            attn_mask = (~tv).unsqueeze(1).expand(n, 1 + K1, 1 + K1)
            for l in range(net.comm.num_layers):
                info = net.comm.self_attn(torch.cat([x.unsqueeze(1), toks[l]], 1), attn_mask=attn_mask)
                new = net.comm.update_cell(info[:, 0], x)
                x = torch.where(has.unsqueeze(-1), new, x)
        comb = torch.cat((x, m), dim=1)
        adv = net.adv(comb) if configs.Advantage_all else net.adv(x)
        q = net.state(x) + adv - adv.mean(1, keepdim=True)
        return q.argmax(1).tolist(), x, post_t

def oracle_policy(P, mode, value):
    ob = OracleHybrid(P, mode, value)
    def f(obs, pos, hid):
        h, t = (None, None) if hid is None else hid
        a, x, t_new = ob.step(obs, pos, h, t)
        return a, (x, t_new)
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
            print(f"{L}x{L} {N:>3} | {name:<34} success {rows[-1]['success_pct']:5.1f}% | steps {rows[-1]['avg_steps']:6.1f} "
                  f"| arrival {rows[-1]['arrival_pct']:5.1f}%")
    return pd.DataFrame(rows).round(2)

policies = {}
if INCLUDE_REFERENCE:
    policies["REF full_comm"] = teacher_policy(True)
    policies["REF no_comm"] = teacher_policy(False)
for kind in KINDS:
    P = load_predictor(kind)
    for lam in LAMBDAS:
        policies[f"{kind} blend lam={lam}"] = oracle_policy(P, "blend", lam)
for r2 in NOISE_R2:
    policies[f"noisy oracle R2={r2}"] = oracle_policy(None, "noise", r2)

t0 = time.time()
df = evaluate(policies, SETTINGS, N_CASES)
print(f"\nelapsed {time.time()-t0:.0f}s")
print("\n" + df.to_string(index=False))

if INCLUDE_REFERENCE:
    out = []
    for N, g in df.groupby("agents"):
        full = g.loc[g.policy == "REF full_comm", "success_pct"].iloc[0]
        none = g.loc[g.policy == "REF no_comm", "success_pct"].iloc[0]
        for _, r in g.iterrows():
            rec = (r.success_pct - none) / (full - none) * 100 if full > none else float("nan")
            out.append(dict(agents=N, policy=r.policy, success_pct=r.success_pct, gap_recovered_pct=round(rec, 1)))
    gap = pd.DataFrame(out)
    print("\nShare of the (no comm -> full comm) success gap recovered:")
    print(gap.to_string(index=False))
    gap.to_csv("/content/sigma_oracle_gap.csv", index=False)
df.to_csv("/content/sigma_oracle_results.csv", index=False)
print("\nsaved -> /content/sigma_oracle_results.csv (and sigma_oracle_gap.csv)")
print("lam=0 reproduces the deployable predictor. Noise rows use independent isotropic noise, real predictor errors are structured.")
