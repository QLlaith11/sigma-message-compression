import gc, os, sys, glob, time, types, pickle, random, zipfile, subprocess, warnings, math
warnings.filterwarnings("ignore")

SMOKE          = False
ENV_NAMES      = ["house"]
CONFIGS        = [(1, 1), (1, 2), (2, 4), (4, 4)]
SEEDS          = [0, 1]
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
RESUME         = True
BASE_SEED      = 0
TRAIN_DROP_MAX    = 0.0
TRAIN_CORRUPT_MAX = 0.0
TAG            = ""

ENVS = {
    "house": dict(scenario="house", ckpt="models/save_model/model_house/84000_house.pth",
                  data=[(30, 16), (40, 16), (40, 32), (40, 32), (40, 48), (40, 64), (40, 64), (40, 96), (40, 128)]),
    "maze":  dict(scenario="maze", ckpt="models/save_model/model_maze/82000.pth",
                  data=[(40, 16), (40, 32), (40, 32), (40, 48), (40, 64), (40, 64), (40, 96), (40, 128)]),
}
REPO_URL = "https://github.com/marmotlab/SIGMA.git"
ROOT     = "/content/SIGMA"
OUT_DIR  = "/content/sigma_paper"
if SMOKE:
    ENV_NAMES, CONFIGS, SEEDS, TEACHER_EPS, UPD_STAGE1, DAGGER_ROUNDS, DAGGER_EPS, UPD_DAGGER, B_STEPS, RESUME = \
        ["house"], [(2, 4)], [0], 4, 20, 1, 2, 10, 4, False
    ENVS["house"]["data"] = [(40, 8)]

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
from tqdm.auto import tqdm

os.chdir(ROOT); sys.path.insert(0, ROOT)
os.makedirs(OUT_DIR, exist_ok=True)
import configs
from model import Network
from environment import Environment

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(BASE_SEED); np.random.seed(BASE_SEED); torch.manual_seed(BASE_SEED)
print("device:", DEV, "| torch", torch.__version__, "| numpy", np.__version__, "| out:", OUT_DIR)

R = configs.obs_radius; G = 2 * R + 1; NC = G * G
H = configs.hidden_dim; KCOMM = configs.max_comm_agents; K1 = KCOMM - 1

def load_teacher(path):
    net = Network()
    net.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    net = net.eval().to(DEV)
    for p_ in net.parameters():
        p_.requires_grad_(False)
    assert net.comm.num_layers == 2
    return net

teacher = None
DATA_SETTINGS = None

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

for env_name in ENV_NAMES:
    cfg = ENVS[env_name]
    configs.test_senario = cfg["scenario"]
    teacher = load_teacher(cfg["ckpt"])
    DATA_SETTINGS = cfg["data"]
    env_dir = f"{OUT_DIR}/{env_name}"; os.makedirs(env_dir, exist_ok=True)
    todo = [(k, b, s) for s in SEEDS for (k, b) in CONFIGS
            if not (RESUME and os.path.exists(f"{env_dir}/learned_k{k}_b{b}_seed{s}{TAG}.pt"))]
    print(f"\n######## environment: {env_name} | {len(todo)} compressors to train ########")
    if not todo:
        continue
    teacher_replay = collect([], TEACHER_EPS, "teacher")
    print(f"teacher data: {len(teacher_replay)} team steps")
    for (k, bits, seed) in todo:
        name = f"{env_name} k={k} {bits}bit seed{seed}{TAG}"
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        comp = Compressor(k, bits).to(DEV)
        print(f"\n===== training {name} =====")
        replay = list(teacher_replay)
        train(comp, replay, UPD_STAGE1, LR, f"{name} stage1")
        for r in range(DAGGER_ROUNDS):
            collect(replay, DAGGER_EPS, "student", comp)
            print(f"DAgger round {r+1}: {len(replay)} team steps")
            train(comp, replay, UPD_DAGGER, LR * 0.5, f"{name} dagger{r+1}")
        torch.save(comp.state_dict(), f"{env_dir}/learned_k{k}_b{bits}_seed{seed}{TAG}.pt")
        del replay, comp; gc.collect(); torch.cuda.empty_cache()
    del teacher_replay; gc.collect()
print("\nDone. Compressors are in", OUT_DIR, "- now run paper_2_eval_suite_colab.py")
