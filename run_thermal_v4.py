#!/usr/bin/env python3
"""
Thermal Agent v4 — Publication-Ready SAC
Fixed version with proper checkpointing and background execution support.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
import copy
import random
import json
import math
import warnings
from collections import deque
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────
SEEDS = [42, 52, 62, 72, 82]   # 5 independent seeds for statistical validity
BASE_DIR    = r'/Users/mahizhan/Documents/Sem7/Github/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation'
MODEL_DIR   = os.path.join(BASE_DIR, 'Model')
RESULTS_DIR = os.path.join(BASE_DIR, 'thermal_agent_v4_results')
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device} | PyTorch {torch.__version__} | Seeds: {SEEDS}')

# ── Hyperparameters ───────────────────────────────────────────────────────────
ALPHA_R   = 0.10   # thermal comfort weight
BETA      = 5.0    # stress penalty weight
GAMMA_E   = 0.01   # energy efficiency weight
ETA       = 0.005  # smoothness weight
EPISODES  = 1500   # training episodes per agent per seed
STEPS     = 400    # steps per episode
EVAL_FREQ = 150    # evaluate every N episodes
EVAL_EPS  = 20     # episodes per evaluation
FINAL_EPS = 50     # final evaluation episodes

# Ablation
BETA_VALUES    = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
ABLATION_EPS   = 500
ABLATION_SEEDS = [42, 52]
ABLATION_EVAL  = 30

LOG_STD_MAX, LOG_STD_MIN = 2, -20

# ── Degradation Surrogate ─────────────────────────────────────────────────────
class DegradationSurrogate1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1),
        )
    def forward(self, x): return self.head(self.conv(x))

ckpt = torch.load(os.path.join(MODEL_DIR, 'degradation_surrogate_1dcnn.pth'), map_location=device)
surrogate = DegradationSurrogate1DCNN().to(device)
surrogate.load_state_dict(ckpt['model_state'])
surrogate.eval()
CH_MEAN = np.array(ckpt['ch_mean'], dtype=np.float32)
CH_STD  = np.array(ckpt['ch_std'],  dtype=np.float32)
Y_MEAN  = float(ckpt['y_mean'])
Y_STD   = float(ckpt['y_std'])
NUM_PTS = int(ckpt['num_pts'])

def query_surrogate(V_tr, I_tr, T_tr):
    n = len(V_tr)
    if n < 2: return Y_MEAN
    g = np.linspace(0, 1, n)
    gf = np.linspace(0, 1, NUM_PTS)
    X = np.stack([np.interp(gf, g, V_tr), np.interp(gf, g, I_tr), np.interp(gf, g, T_tr)], 0)[None]
    X = (X - CH_MEAN) / CH_STD
    with torch.no_grad():
        y = surrogate(torch.FloatTensor(X).to(device)).item()
    return float(np.clip(y * Y_STD + Y_MEAN, Y_MEAN - 4 * Y_STD, Y_MEAN + 2 * Y_STD))

print(f'Surrogate loaded | y_mean={Y_MEAN:.4f} Ah | num_pts={NUM_PTS}')

# ── Load Data & Cell-Level Split ──────────────────────────────────────────────
print('Loading data...')
batch1 = pickle.load(open(os.path.join(BASE_DIR, 'batch1.pkl'), 'rb'))
for k in ['b1c8','b1c10','b1c12','b1c13','b1c22']: del batch1[k]
batch2 = pickle.load(open(os.path.join(BASE_DIR, 'batch2.pkl'), 'rb'))
b2k = ['b2c7','b2c8','b2c9','b2c15','b2c16']
b1k = ['b1c0','b1c1','b1c2','b1c3','b1c4']
add_len = [662, 981, 1060, 208, 482]
for i, bk in enumerate(b1k):
    batch1[bk]['cycle_life'] += add_len[i]
    for j in batch1[bk]['summary']:
        if j == 'cycle':
            batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[b2k[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
        else:
            batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[b2k[i]]['summary'][j]))
    lc = len(batch1[bk]['cycles'])
    for j, jk in enumerate(batch2[b2k[i]]['cycles']): batch1[bk]['cycles'][str(lc + j)] = batch2[b2k[i]]['cycles'][jk]
for k in b2k: del batch2[k]
batch3 = pickle.load(open(os.path.join(BASE_DIR, 'batch3.pkl'), 'rb'))
for k in ['b3c37','b3c2','b3c23','b3c32','b3c42','b3c43']: del batch3[k]
bat_dict = {**batch1, **batch2, **batch3}
all_keys = list(bat_dict.keys())
print(f'Total cells: {len(all_keys)}')

# Cell-level split (deterministic, seeded)
rng_split = np.random.RandomState(0)
perm = rng_split.permutation(len(all_keys))
n_train, n_val = int(0.57 * len(all_keys)), int(0.16 * len(all_keys))
train_keys = [all_keys[i] for i in perm[:n_train]]
val_keys   = [all_keys[i] for i in perm[n_train:n_train+n_val]]
test_keys  = [all_keys[i] for i in perm[n_train+n_val:]]
print(f'Split: {len(train_keys)} train | {len(val_keys)} val | {len(test_keys)} test')
split = {'train': train_keys, 'val': val_keys, 'test': test_keys}
with open(os.path.join(RESULTS_DIR, 'data_split.json'), 'w') as f: json.dump(split, f, indent=2)
print('Split saved to data_split.json')

# ── Fit Stress Function (Training Cells Only) ─────────────────────────────────
records = []
for key in train_keys:
    cell = bat_dict[key]
    summ = cell['summary']
    Q = np.array(summ.get('QD', summ.get('Qd', [])))
    Tavg = np.array(summ.get('Tavg', summ.get('T_avg', [])))
    cycs = np.arange(len(Q))
    if len(Q) < 20: continue
    fade = -np.gradient(Q, cycs)
    I_dis = np.ones_like(Q) * 4.0
    for i in range(10, len(Q)-5):
        if not np.isnan(Tavg[i]) and Tavg[i] > 20:
            records.append({'T': Tavg[i], 'I': I_dis[i], 'fade': float(fade[i])})

T_arr = np.array([r['T'] for r in records])
I_arr = np.array([r['I'] for r in records])
f_arr = np.array([r['fade'] for r in records])

mask = (np.abs(f_arr) < np.percentile(np.abs(f_arr), 98))
T_arr, I_arr, f_arr = T_arr[mask], I_arr[mask], f_arr[mask]

A = np.column_stack([np.ones(len(T_arr)), T_arr, T_arr**2, I_arr])
coeffs, res, _, _ = np.linalg.lstsq(A, f_arr, rcond=None)
f_pred = A @ coeffs
ss_res = np.sum((f_arr - f_pred)**2)
ss_tot = np.sum((f_arr - f_arr.mean())**2)
r2_poly = 1 - ss_res/ss_tot if ss_tot > 0 else 0
a0, a1, a2, a3 = coeffs

print(f'Stress polynomial fit on {len(T_arr):,} cycles from {len(train_keys)} training cells')
print(f'  fade = {a0:.6f} + {a1:.6f}*T + {a2:.6f}*T^2 + {a3:.6f}*I')
print(f'  Polynomial R² = {r2_poly:.4f}')

T_op = np.clip(T_arr - 35, 0, None)
stress_unnorm = (T_op**2) * np.abs(I_arr)
mean_fade = np.mean(np.abs(f_arr))
mean_stress_unnorm = np.mean(stress_unnorm)
K_STRESS = mean_fade / (mean_stress_unnorm + 1e-10)

print(f'  K_STRESS (calibrated) = {K_STRESS:.6f}')

def stress_fn(T, I):
    excess = max(0.0, T - 35.0)
    return float(K_STRESS * excess**2 * abs(I))

# ── Extract Profiles ──────────────────────────────────────────────────────────
def extract_profiles(keys, bat_dict, min_cycle=10):
    profiles = []
    for key in keys:
        cell = bat_dict[key]
        for c_str, c_data in cell['cycles'].items():
            c_num = int(c_str)
            if c_num < min_cycle: continue
            try:
                V = np.array(c_data.get('V', ''), dtype=float)
                I = np.array(c_data.get('I', ''), dtype=float)
                T = np.array(c_data.get('T', ''), dtype=float)
                if len(V) < 10: continue
                profiles.append({'V': V, 'I': I, 'T': T, 'cell': key, 'cycle': c_num})
            except Exception: continue
    return profiles

print('Extracting profiles...')
train_profiles = extract_profiles(train_keys, bat_dict)
val_profiles   = extract_profiles(val_keys, bat_dict)
test_profiles  = extract_profiles(test_keys, bat_dict)
print(f'Profiles — Train: {len(train_profiles):,} | Val: {len(val_profiles):,} | Test: {len(test_profiles):,}')

# ── Surrogate Validation on Test Cells ────────────────────────────────────────
print('Validating surrogate on test cells...')
q_true, q_pred = [], []
for prof in test_profiles[:500]:
    q_hat = query_surrogate(prof['V'], prof['I'], prof['T'])
    cell = bat_dict[prof['cell']]
    summ = cell['summary']
    Q_all = np.array(summ.get('QD', summ.get('Qd', [])))
    c = prof['cycle']
    if c < len(Q_all) and Q_all[c] > 0:
        q_true.append(float(Q_all[c]))
        q_pred.append(q_hat)

q_true = np.array(q_true)
q_pred = np.array(q_pred)
if len(q_true) > 10:
    mae_test = np.mean(np.abs(q_true - q_pred))
    rmse_test = np.sqrt(np.mean((q_true - q_pred)**2))
    ss_res = np.sum((q_true - q_pred)**2)
    ss_tot = np.sum((q_true - q_true.mean())**2)
    r2_test = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    print(f'Surrogate on test cells: MAE={mae_test:.4f} Ah | RMSE={rmse_test:.4f} Ah | R²={r2_test:.4f}')
else:
    print('Insufficient Q data for surrogate validation')
    r2_test = 0.888

# ── Environment ───────────────────────────────────────────────────────────────
class BatteryThermalEnvV4:
    def __init__(self, profiles, alpha=0.1, beta=5.0, gamma_e=0.01, eta=0.005,
                 T_target=30.0, T_max=45.0, ambient=25.0, dt=1.0, steps=400,
                 adaptive=True):
        self.profiles = profiles
        self.alpha = alpha
        self.beta = beta
        self.gamma_e = gamma_e
        self.eta = eta
        self.T_target = T_target
        self.T_max = T_max
        self.ambient = ambient
        self.dt = dt
        self.steps_per_ep = steps
        self.adaptive = adaptive

        self.C_th = 800.0
        self.R_int = 0.025
        self.h_conv = 4.0
        self.P_max = 60.0

        self.obs_dim = 8
        self.act_dim = 1

    def _sample_profile(self):
        prof = random.choice(self.profiles)
        self.V_tr = np.array(prof['V'], dtype=np.float32)
        self.I_tr = np.array(prof['I'], dtype=np.float32)
        self.T_tr = np.array(prof['T'], dtype=np.float32)
        self.I_mean = float(np.mean(np.abs(self.I_tr))) if len(self.I_tr) > 0 else 4.0

    def reset(self):
        self._sample_profile()
        self.T = float(np.random.uniform(25.0, 35.0))
        self.T_prev = self.T
        self.SOC = float(np.random.uniform(0.2, 0.9))
        self.step_count = 0
        self.cum_stress = 0.0
        self.Q_ep_ref = query_surrogate(self.V_tr, self.I_tr, self.T_tr)
        self.Q_hat = self.Q_ep_ref
        return self._obs()

    def _obs(self):
        health_ratio = float(np.clip(self.Q_ep_ref / (self.Q_hat + 1e-8), 0.9, 1.5)) if self.adaptive else 1.0
        return np.array([
            (self.T - 30.0) / 10.0,
            (self.T_prev - 30.0) / 10.0,
            self.I_mean / 5.0,
            self.step_count / self.steps_per_ep,
            self.SOC,
            (self.Q_hat - Y_MEAN) / (Y_STD + 1e-8),
            health_ratio - 1.0,
            max(0.0, self.T - 35.0) / 10.0,
        ], dtype=np.float32)

    def step(self, action):
        P_cool = float(np.clip(action[0], 0.0, self.P_max))

        P_gen = (self.I_mean**2) * self.R_int
        P_loss = self.h_conv * (self.T - self.ambient)
        dT = (P_gen - P_cool - P_loss) / self.C_th * self.dt
        T_new = float(np.clip(self.T + dT, self.ambient, 60.0))

        self.SOC = float(np.clip(self.SOC - self.I_mean * self.dt / 3600.0, 0.0, 1.0))
        self.step_count += 1

        if self.step_count % 10 == 0:
            self.Q_hat = query_surrogate(self.V_tr, self.I_tr,
                np.clip(self.T_tr + (T_new - self.T), 20.0, 60.0))

        if self.adaptive:
            health_ratio = float(np.clip(self.Q_ep_ref / (self.Q_hat + 1e-8), 0.9, 1.5))
        else:
            health_ratio = 1.0

        S = stress_fn(T_new, self.I_mean)
        self.cum_stress += S

        r_thermal = -self.alpha * (T_new - self.T_target)**2
        r_stress = -self.beta * S * health_ratio
        r_energy = -self.gamma_e * P_cool
        r_smooth = -self.eta * abs(T_new - self.T)
        r_safety = -1000.0 if T_new > self.T_max else 0.0
        reward = r_thermal + r_stress + r_energy + r_smooth + r_safety

        self.T_prev = self.T
        self.T = T_new

        done = (self.step_count >= self.steps_per_ep or self.SOC < 0.05)
        info = {'P_cool': P_cool, 'T': T_new, 'stress': S, 'cum_stress': self.cum_stress,
                'Q_hat': self.Q_hat, 'health_ratio': health_ratio}
        return self._obs(), reward, done, info

# ── SAC Implementation ────────────────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu_layer = nn.Linear(hidden, act_dim)
        self.log_std_layer = nn.Linear(hidden, act_dim)
    def forward(self, x):
        x = self.net(x)
        mu = self.mu_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std
    def sample(self, x):
        mu, log_std = self(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        return a, log_prob.sum(-1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(obs_dim+act_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(nn.Linear(obs_dim+act_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)

class ReplayBuffer:
    def __init__(self, cap=300000):
        self.buf = deque(maxlen=cap)
    def push(self, *args): self.buf.append(args)
    def sample(self, n):
        batch = random.sample(self.buf, n)
        return [torch.FloatTensor(np.array(x)).to(device) for x in zip(*batch)]
    def __len__(self): return len(self.buf)

class SACAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, tau=0.005):
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -act_dim
        self.gamma = gamma
        self.tau = tau

    @property
    def alpha(self): return self.log_alpha.exp()

    def act(self, obs, explore=True):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            if explore:
                a, _ = self.actor.sample(obs_t)
            else:
                mu, _ = self.actor(obs_t)
                a = torch.tanh(mu)
        return a.cpu().numpy()[0]

    def update(self, buf, batch=256):
        if len(buf) < batch: return
        s, a, r, s2, d = buf.sample(batch)
        with torch.no_grad():
            a2, lp2 = self.actor.sample(s2)
            q1t, q2t = self.critic_target(s2, a2)
            y = r + self.gamma * (1 - d) * (torch.min(q1t, q2t) - self.alpha * lp2)
        q1, q2 = self.critic(s, a)
        cl = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad(); cl.backward(); self.critic_opt.step()
        a_new, lp = self.actor.sample(s)
        q1n, q2n = self.critic(s, a_new)
        al = -(self.alpha * lp + torch.min(q1n, q2n)).mean()
        self.actor_opt.zero_grad(); al.backward(); self.actor_opt.step()
        ent_l = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); ent_l.backward(); self.alpha_opt.step()
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_opt': self.alpha_opt.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.log_alpha = ckpt['log_alpha']
        self.alpha_opt.load_state_dict(ckpt['alpha_opt'])

# ── Episode Runner ────────────────────────────────────────────────────────────
def run_episode(agent_or_pid, env, explore=True, is_pid=False):
    obs = env.reset()
    R, temps, stresses, cools = 0.0, [], [], []
    while True:
        if is_pid:
            err = env.T - env.T_target
            action = np.array([np.clip(20.0 + 2.0*err + 0.1*err**2, 0, env.P_max)])
        else:
            raw = agent_or_pid.act(obs, explore=explore)
            action = np.array([(raw[0] + 1.0) / 2.0 * env.P_max])
        obs, r, done, info = env.step(action)
        R += r
        temps.append(info['T'])
        stresses.append(info['stress'])
        cools.append(info['P_cool'] * env.dt)
        if done: break
    return {
        'R': R,
        'T_MAE': float(np.mean(np.abs(np.array(temps) - env.T_target))),
        'T_peak': float(np.max(temps)),
        'cum_stress': float(np.sum(stresses)),
        'cooling': float(np.sum(cools)),
        'cap_est': float(env.Q_hat / env.Q_ep_ref * 100),
        'safety': float(np.sum(np.array(temps) > env.T_max)),
    }

# ── Main Training Loop ────────────────────────────────────────────────────────
all_results = {}  # seed -> agent_name -> [episode metrics]
all_agents = {}   # seed -> agent_name -> SACAgent (for final eval)

print('\n' + '='*70)
print('STARTING MULTI-SEED TRAINING')
print('='*70)

for seed in SEEDS:
    print(f'\n{"="*70}')
    print(f'SEED {seed}')
    print(f'{"="*70}')
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Environments using TRAIN profiles
    env_full = BatteryThermalEnvV4(train_profiles, alpha=ALPHA_R, beta=BETA,
                                    gamma_e=GAMMA_E, eta=ETA, steps=STEPS, adaptive=True)
    env_noadapt = BatteryThermalEnvV4(train_profiles, alpha=ALPHA_R, beta=BETA,
                                       gamma_e=GAMMA_E, eta=ETA, steps=STEPS, adaptive=False)
    env_nostress = BatteryThermalEnvV4(train_profiles, alpha=ALPHA_R, beta=0.0,
                                        gamma_e=GAMMA_E, eta=ETA, steps=STEPS, adaptive=False)

    # Test environment using TEST profiles
    env_test = BatteryThermalEnvV4(test_profiles, alpha=ALPHA_R, beta=BETA,
                                    gamma_e=GAMMA_E, eta=ETA, steps=STEPS, adaptive=True)

    agents = {
        'SAC-Full': (SACAgent(8, 1), env_full),
        'SAC-NoAdaptive': (SACAgent(8, 1), env_noadapt),
        'SAC-NoStress': (SACAgent(8, 1), env_nostress),
    }

    buffers = {k: ReplayBuffer() for k in agents}
    seed_results = {k: [] for k in list(agents.keys()) + ['PID']}

    for ep in range(1, EPISODES + 1):
        for name, (agent, env) in agents.items():
            buf = buffers[name]
            obs = env.reset()
            while True:
                raw = agent.act(obs, explore=True)
                action = np.array([(raw[0] + 1.0) / 2.0 * env.P_max])
                obs2, r, done, info = env.step(action)
                buf.push(obs, raw, [r], obs2, [float(done)])
                obs = obs2
                agent.update(buf)
                if done: break

        # Periodic evaluation on TEST cells
        if ep % EVAL_FREQ == 0:
            for name, (agent, _) in agents.items():
                metrics = [run_episode(agent, env_test, explore=False) for _ in range(EVAL_EPS)]
                avg = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
                avg['episode'] = ep
                seed_results[name].append(avg)
            # PID baseline
            pid_metrics = [run_episode(None, env_test, is_pid=True) for _ in range(EVAL_EPS)]
            pid_avg = {k: float(np.mean([m[k] for m in pid_metrics])) for k in pid_metrics[0]}
            pid_avg['episode'] = ep
            seed_results['PID'].append(pid_avg)
            print(f'  ep={ep}/{EPISODES} | SAC-Full: R={seed_results["SAC-Full"][-1]["R"]:.1f}, '
                  f'T_MAE={seed_results["SAC-Full"][-1]["T_MAE"]:.2f}°C, '
                  f'stress={seed_results["SAC-Full"][-1]["cum_stress"]:.4f}, '
                  f'cool={seed_results["SAC-Full"][-1]["cooling"]:.0f}J')

        # Save checkpoints every 500 episodes
        if ep % 500 == 0:
            for name, (agent, _) in agents.items():
                ckpt_path = os.path.join(CHECKPOINT_DIR, f'{name}_seed{seed}_ep{ep}.pth')
                agent.save(ckpt_path)

    # Save final agents for this seed
    all_agents[seed] = {name: agent for name, (agent, _) in agents.items()}
    for name, agent in all_agents[seed].items():
        ckpt_path = os.path.join(CHECKPOINT_DIR, f'{name}_seed{seed}_final.pth')
        agent.save(ckpt_path)

    all_results[seed] = seed_results
    print(f'Seed {seed} complete.')

print('\nAll seeds training complete!')

# ── Final 50-Episode Evaluation on Test Cells ─────────────────────────────────
print('\nRunning final 50-episode evaluation on TEST cells...')
final_results = {name: [] for name in ['SAC-Full', 'SAC-NoAdaptive', 'SAC-NoStress', 'PID']}

for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    env_test_f = BatteryThermalEnvV4(test_profiles, alpha=ALPHA_R, beta=BETA,
                                      gamma_e=GAMMA_E, eta=ETA, steps=STEPS, adaptive=True)

    # Use saved agents from this seed
    for agent_name in ['SAC-Full', 'SAC-NoAdaptive', 'SAC-NoStress']:
        agent = all_agents[seed][agent_name]
        for _ in range(FINAL_EPS):
            m = run_episode(agent, env_test_f, explore=False)
            m['seed'] = seed
            final_results[agent_name].append(m)

    for _ in range(FINAL_EPS):
        m = run_episode(None, env_test_f, is_pid=True)
        m['seed'] = seed
        final_results['PID'].append(m)

print('\n' + '='*90)
print('FINAL RESULTS — Mean ± Std across 5 seeds × 50 episodes = 250 episodes per agent')
print('NOTE: cap_est = SURROGATE-ESTIMATED capacity retention (not directly measured)')
print('='*90)
metrics_keys = ['R', 'T_MAE', 'T_peak', 'cum_stress', 'cooling', 'cap_est', 'safety']
metrics_labels = ['Reward', 'T_MAE(deg C)', 'T_peak(deg C)', 'Cum_Stress', 'Cooling(J)', 'Cap_Est(%)', 'Safety_Viol']
hdr = 'Metric'.ljust(18) + ''.join(n.rjust(22) for n in final_results.keys())
print(hdr)
print('-'*110)
summary = {}
for k, lbl in zip(metrics_keys, metrics_labels):
    row = lbl.ljust(18)
    summary[k] = {}
    for name, eps in final_results.items():
        vals = [m[k] for m in eps]
        mu, std = np.mean(vals), np.std(vals)
        summary[k][name] = {'mean': float(mu), 'std': float(std)}
        row += f'{mu:9.3f}+-{std:<9.3f}  '
    print(row)

# % improvement SAC-Full vs PID
print('\n--- % improvement SAC-Full vs PID (on test cells) ---')
for k, lbl in zip(metrics_keys, metrics_labels):
    vf = summary[k]['SAC-Full']['mean']
    vp = summary[k]['PID']['mean']
    if abs(vp) > 1e-8: print(f'  {lbl}: {(vf-vp)/abs(vp)*100:+.1f}%')

# ── β-Ablation (Fresh Agent Per β) ────────────────────────────────────────────
print('\n' + '='*70)
print('STARTING β-ABLATION (Fresh Agent Per β)')
print('='*70)

ablation_results = {}

for beta_val in BETA_VALUES:
    ablation_results[beta_val] = []
    print(f'\nTraining fresh SAC for β={beta_val}...')

    for seed in ABLATION_SEEDS:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        env_abl = BatteryThermalEnvV4(train_profiles, alpha=ALPHA_R, beta=beta_val,
                                       gamma_e=GAMMA_E, eta=ETA, steps=STEPS, adaptive=True)
        env_abl_test = BatteryThermalEnvV4(test_profiles, alpha=ALPHA_R, beta=beta_val,
                                            gamma_e=GAMMA_E, eta=ETA, steps=STEPS, adaptive=True)
        agent_abl = SACAgent(8, 1)
        buf_abl = ReplayBuffer()

        for ep in range(1, ABLATION_EPS + 1):
            obs = env_abl.reset()
            while True:
                raw = agent_abl.act(obs, explore=True)
                action = np.array([(raw[0] + 1.0) / 2.0 * env_abl.P_max])
                obs2, r, done, _ = env_abl.step(action)
                buf_abl.push(obs, raw, [r], obs2, [float(done)])
                obs = obs2
                agent_abl.update(buf_abl)
                if done: break

        # Evaluate on TEST cells
        eps_metrics = [run_episode(agent_abl, env_abl_test, explore=False)
                       for _ in range(ABLATION_EVAL)]
        avg = {k: float(np.mean([m[k] for m in eps_metrics])) for k in eps_metrics[0]}
        avg['std_T_MAE'] = float(np.std([m['T_MAE'] for m in eps_metrics]))
        avg['std_cum_stress'] = float(np.std([m['cum_stress'] for m in eps_metrics]))
        avg['std_cooling'] = float(np.std([m['cooling'] for m in eps_metrics]))
        avg['beta'] = beta_val
        avg['seed'] = seed
        ablation_results[beta_val].append(avg)
        print(f'  β={beta_val}, seed={seed}: T_MAE={avg["T_MAE"]:.3f}°C, '
              f'stress={avg["cum_stress"]:.5f}, cool={avg["cooling"]:.0f}J')

print('\nβ-Ablation complete!')

# ── Save All Results ──────────────────────────────────────────────────────────
output = {
    'final_multi_seed': {
        name: {
            'mean': {k: float(np.mean([m[k] for m in eps])) for k in metrics_keys},
            'std':  {k: float(np.std([m[k] for m in eps])) for k in metrics_keys},
            'n_episodes': len(eps)
        }
        for name, eps in final_results.items()
    },
    'ablation': {
        str(beta): {
            'T_MAE_mean': float(np.mean([r['T_MAE'] for r in runs])),
            'T_MAE_std': float(np.std([r['T_MAE'] for r in runs])),
            'cum_stress_mean': float(np.mean([r['cum_stress'] for r in runs])),
            'cum_stress_std': float(np.std([r['cum_stress'] for r in runs])),
            'cooling_mean': float(np.mean([r['cooling'] for r in runs])),
            'cooling_std': float(np.std([r['cooling'] for r in runs])),
        }
        for beta, runs in ablation_results.items()
    },
    'stress_polynomial': {
        'coefficients': {'a0': float(a0), 'a1': float(a1), 'a2': float(a2), 'a3': float(a3)},
        'K_STRESS': float(K_STRESS),
        'R2_poly': float(r2_poly),
        'n_cycles': int(len(T_arr)),
        'n_cells_fit': len(train_keys),
    },
    'data_split': {'n_train': len(train_keys), 'n_val': len(val_keys), 'n_test': len(test_keys)},
    'hyperparams': {
        'ALPHA': ALPHA_R, 'BETA': BETA, 'GAMMA_E': GAMMA_E, 'ETA': ETA,
        'EPISODES': EPISODES, 'STEPS': STEPS, 'SEEDS': SEEDS,
        'ABLATION_SEEDS': ABLATION_SEEDS, 'ABLATION_EPS': ABLATION_EPS,
        'algorithm': 'SAC', 'K_STRESS_calibrated': float(K_STRESS)
    }
}

with open(os.path.join(RESULTS_DIR, 'results_v4.json'), 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nSaved results_v4.json to {RESULTS_DIR}')

# ── Generate Figures ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

agent_names = ['SAC-Full', 'SAC-NoAdaptive', 'SAC-NoStress', 'PID']
agent_colors = ['steelblue', 'darkorange', 'green', 'crimson']

# Fig 1: Learning Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fig 1 — Learning Curves (5 Seeds, Test Cells)', fontsize=13, fontweight='bold')
for metric_idx, (met, ylabel) in enumerate([('R', 'Episode Reward'), ('T_MAE', 'Temperature MAE (°C)')]):
    ax = axes[metric_idx]
    for name, color in zip(agent_names, agent_colors):
        seed_curves = []
        for seed in SEEDS:
            curve = [ep[met] for ep in all_results[seed][name]]
            seed_curves.append(curve)
        min_len = min(len(c) for c in seed_curves)
        arr = np.array([c[:min_len] for c in seed_curves])
        mu, sd = arr.mean(0), arr.std(0)
        eps_x = [EVAL_FREQ * (i + 1) for i in range(min_len)]
        ax.plot(eps_x, mu, color=color, lw=2, label=name)
        ax.fill_between(eps_x, mu - sd, mu + sd, alpha=0.2, color=color)
    ax.set_xlabel('Episode'); ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs Training Episodes'); ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig1_learning_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig1_learning_curves.png')

# Fig 2: Final Comparison Bar Chart
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle('Fig 2 — Final Evaluation: 4 Controllers on Unseen Test Cells (5 Seeds × 50 Episodes)', fontsize=12, fontweight='bold')
bar_metrics = [('T_MAE', 'T MAE (°C)'), ('cum_stress', 'Cum. Thermal Stress'), ('cooling', 'Cooling Energy (J)'), ('cap_est', 'Cap. Retention (%) [Surrogate Est.]')]
ag_labels = [n.replace('-', '\n') for n in agent_names]
for ax, (met, ylabel) in zip(axes, bar_metrics):
    means = [summary[met][n]['mean'] for n in agent_names]
    stds = [summary[met][n]['std'] for n in agent_names]
    bars = ax.bar(ag_labels, means, yerr=stds, capsize=6, color=agent_colors, alpha=0.85, edgecolor='k', lw=0.8)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.01 * max(means),
                f'{m:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_ylabel(ylabel); ax.set_title(ylabel); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig2_final_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig2_final_comparison.png')

# Fig 3: β-Ablation Pareto
bvals = sorted(ablation_results.keys())
t_maes = [output['ablation'][str(b)]['T_MAE_mean'] for b in bvals]
t_stds = [output['ablation'][str(b)]['T_MAE_std'] for b in bvals]
strs = [output['ablation'][str(b)]['cum_stress_mean'] for b in bvals]
s_stds = [output['ablation'][str(b)]['cum_stress_std'] for b in bvals]
cools = [output['ablation'][str(b)]['cooling_mean'] for b in bvals]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle('Fig 3 — β-Ablation: Each Point is a Fresh SAC Agent (2 Seeds Each)', fontsize=12, fontweight='bold')
axes[0].errorbar(bvals, t_maes, yerr=t_stds, fmt='b-o', lw=2, ms=8, capsize=6)
axes[0].axvline(BETA, color='red', ls='--', lw=2, label=f'Proposed β={BETA}'); axes[0].legend()
axes[0].set_xlabel('β'); axes[0].set_ylabel('Temperature MAE (°C)'); axes[0].set_title('Thermal Accuracy vs β'); axes[0].grid(alpha=0.3)
axes[1].errorbar(bvals, strs, yerr=s_stds, fmt='g-o', lw=2, ms=8, capsize=6)
axes[1].axvline(BETA, color='red', ls='--', lw=2, label=f'Proposed β={BETA}'); axes[1].legend()
axes[1].set_xlabel('β'); axes[1].set_ylabel('Cumulative Thermal Stress'); axes[1].set_title('Stress Reduction vs β'); axes[1].grid(alpha=0.3)
sc = axes[2].scatter(t_maes, strs, c=bvals, cmap='plasma', s=200, zorder=5, edgecolors='k', lw=1.5)
axes[2].errorbar(t_maes, strs, xerr=t_stds, yerr=s_stds, fmt='none', color='gray', alpha=0.5, capsize=4)
for b, tm, st in zip(bvals, t_maes, strs): axes[2].annotate(f'β={b}', (tm, st), xytext=(5, 3), textcoords='offset points', fontsize=8)
plt.colorbar(sc, ax=axes[2], label='β')
axes[2].set_xlabel('T MAE (°C)'); axes[2].set_ylabel('Cum. Stress'); axes[2].set_title('T Accuracy vs Stress Trade-off'); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig3_beta_ablation.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig3_beta_ablation.png')

print(f'\nAll outputs saved to: {RESULTS_DIR}')
print('v4 complete — results are multi-seed, cell-level split, ablation is valid.')