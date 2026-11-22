import os, sys, re, glob, time, types, pickle, random, zipfile, subprocess, warnings
warnings.filterwarnings("ignore")

REPO_URL = "https://github.com/marmotlab/SIGMA.git"
ROOT     = "/content/SIGMA"
CKPT     = "models/save_model/model_house/84000_house.pth"
TEST_DIR = "house_test_set"
SETTINGS = [(40, 8), (40, 32), (40, 64), (40, 128)]
N_CASES  = 20
MODES = {
    "full_comm (k=10)": (True, 10),
    "nearest_3 (k=3)":  (True, 3),
    "no_comm":          (False, 1),
}
SEED = 0

def find_root(base):
    for p in glob.glob(os.path.join(base, "**", "model.py"), recursive=True):
        d = os.path.dirname(p)
        if os.path.exists(os.path.join(d, "configs.py")):
            return d
    return None

if not (os.path.exists(os.path.join(ROOT, "model.py"))):
    r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, ROOT],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(os.path.join(ROOT, "model.py")):
        print("git clone failed -> upload the repo zip (e.g. SIGMA-main_1.zip)")
        from google.colab import files
        up = files.upload()
        zname = next(iter(up))
        out = "/content/_sigma_zip"
        with zipfile.ZipFile(zname) as z:
            z.extractall(out)
        ROOT = find_root(out)
        assert ROOT, "could not find model.py/configs.py in the uploaded zip"
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

import torch
import pandas as pd
from tqdm.auto import tqdm

os.chdir(ROOT)
sys.path.insert(0, ROOT)
import configs
from model import Network
from environment import Environment

assert configs.test_senario == "house", "this script expects the room-like ('house') scenario"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEV, "| torch", torch.__version__, "| numpy", np.__version__)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

net = Network()
net.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
net.eval().to(DEV)
print("loaded:", CKPT)

@torch.no_grad()
def policy_step(obs_np, pos_np, hidden, use_comm, k):
    obs = torch.as_tensor(obs_np.astype(np.float32), device=DEV)
    pos = torch.as_tensor(pos_np.astype(np.float32), device=DEV)
    n = obs.size(0)

    latent = net.obs_encoder(obs)
    mlp_out = net.mlp(latent)
    h = net.recurrent(latent) if hidden is None else net.recurrent(latent, hidden)

    pos_mat = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()
    dist = (pos_mat[..., 0] ** 2 + pos_mat[..., 1] ** 2).sqrt()
    in_obs = (pos_mat <= configs.obs_radius).all(2)
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
    state = net.state(h)
    q = state + adv - adv.mean(1, keepdim=True)

    diff = ((mlp_out.unsqueeze(1) - mlp_out.unsqueeze(0)) ** 2).sum(-1)
    m = comm_mask.float()
    sec = (diff * m).sum(1) / m.sum(1).clamp(min=1)
    q_sec = q - net.lambdas * sec.unsqueeze(1)

    a = q.argmax(1)
    mismatch = int((a != q_sec.argmax(1)).sum().item())
    return a.tolist(), h, mismatch

def run_case(case, use_comm, k):
    env = Environment()
    env.load(case[0], case[1], case[2])
    obs, pos = env.observe()
    hid, steps, mism, done = None, 0, 0, False
    while not done and env.steps < configs.max_episode_length:
        actions, hid, mm = policy_step(obs, pos, hid, use_comm, k)
        (obs, pos), _, done, _ = env.step(actions)
        steps += 1
        mism += mm
    success = np.array_equal(env.agents_pos, env.goals_pos)
    arrived = float((env.agents_pos == env.goals_pos).all(1).mean())
    return success, steps, arrived, mism

reported = {}
txt = os.path.join(os.path.dirname(CKPT), "84000_house.txt")
if os.path.exists(txt):
    cur = None
    for line in open(txt):
        m1 = re.match(r"test set: (\d+) length (\d+) agents", line)
        if m1:
            cur = (int(m1.group(1)), int(m1.group(2)))
        m2 = re.match(r"success rate: ([\d.]+)%", line)
        if m2 and cur:
            reported[cur] = float(m2.group(1))

rows, t0 = [], time.time()
for (L, N) in SETTINGS:
    path = f"{TEST_DIR}/{L}length_{N}agents_0.3density.pth"
    with open(path, "rb") as f:
        tests = pickle.load(f)[:N_CASES]
    for name, (use_comm, k) in MODES.items():
        res = [run_case(c, use_comm, k) for c in tqdm(tests, desc=f"{L}x{L} {N:>3} agents | {name}", leave=False)]
        rows.append(dict(
            map=L, agents=N, mode=name,
            success_pct=100 * np.mean([r[0] for r in res]),
            avg_steps=np.mean([r[1] for r in res]),
            arrival_pct=100 * np.mean([r[2] for r in res]),
            sec_argmax_flips=sum(r[3] for r in res),
            reported_full_pct=reported.get((L, N), np.nan) if name.startswith("full") else np.nan,
            n_cases=len(tests),
        ))
        print(f"{L}x{L} {N:>3} agents | {name:<18} success {rows[-1]['success_pct']:5.1f}% "
              f"| steps {rows[-1]['avg_steps']:6.1f} | elapsed {time.time()-t0:5.0f}s")

df = pd.DataFrame(rows).round(2)
df.to_csv("/content/sigma_comm_ablation.csv", index=False)
print()
print(df.to_string(index=False))
print("\nsaved -> /content/sigma_comm_ablation.csv")
print("sec_argmax_flips should be 0 (or ~0 from float ties): the section term shifts all 5 actions equally.")
