"""Generate thermal_agent.ipynb — DDPG Thermal Agent with Surrogate-Coupled Reward."""
import json, os

BASE = "/Users/mahizhan/Documents/Sem7/Github/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation"
_id = [0]

def uid():
    _id[0] += 1
    return f"ta-{_id[0]:04d}"

def md(src):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": src}

def code(lines):
    if isinstance(lines, str):
        lines = [lines]
    return {"cell_type": "code", "execution_count": None,
            "id": uid(), "metadata": {}, "outputs": [], "source": lines}

cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md(
    "# Thermal Agent — DDPG with Degradation-Surrogate-Coupled Reward\n\n"
    "**Novel contribution:** A data-validated 1D-CNN degradation surrogate (trained on 124 real\n"
    "LFP/graphite cells, R²=0.888) is embedded into the DDPG reward and state at every timestep.\n"
    "The agent learns cooling policies that jointly minimise temperature deviation *and*\n"
    "predicted capacity loss — a trade-off no classical PID controller can achieve.\n\n"
    "| Component | Detail |\n"
    "|---|---|\n"
    "| Algorithm | DDPG + Ornstein-Uhlenbeck exploration + soft target updates |\n"
    "| Environment | Lumped-capacitance thermal model (LCTM) |\n"
    "| Surrogate | DegradationSurrogate1DCNN — queried at every RL timestep |\n"
    "| State dim | 6: [T/50, T_prev/50, I/6, cycle_norm, SOC, Q_pred/Q_ref] |\n"
    "| Action dim | 1: cooling_power ∈ [-1,1] → [0, 50 W] |\n"
    "| Reward | -α(T-30)² - β·max(0, Q_ref-Q_pred) - γ·P_cool - 1000·[T>45°C] |\n"
    "| Baseline | PID controller (classical setpoint control) |\n"
))

# ── Cell 1: Imports ────────────────────────────────────────────────────────────
cells.append(md("## 1 · Imports & Setup"))
cells.append(code([
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "import pickle, os, copy, random, json\n",
    "import matplotlib\n",
    "matplotlib.use('Agg')\n",
    "import matplotlib.pyplot as plt\n",
    "from collections import deque\n",
    "import warnings; warnings.filterwarnings('ignore')\n",
    "\n",
    "# ── Reproducibility ──────────────────────────────────────────────────────────\n",
    "SEED = 42\n",
    "torch.manual_seed(SEED)\n",
    "np.random.seed(SEED)\n",
    "random.seed(SEED)\n",
    "torch.backends.cudnn.deterministic = True\n",
    "\n",
    "BASE_DIR    = r'/Users/mahizhan/Documents/Sem7/Github/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation'\n",
    "MODEL_DIR   = os.path.join(BASE_DIR, 'Model')\n",
    "RESULTS_DIR = os.path.join(BASE_DIR, 'thermal_agent_results')\n",
    "os.makedirs(RESULTS_DIR, exist_ok=True)\n",
    "\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f'Device: {device} | PyTorch {torch.__version__}')\n",
    "print(f'Results: {RESULTS_DIR}')\n",
]))

# ── Cell 2: Load surrogate ─────────────────────────────────────────────────────
cells.append(md(
    "## 2 · Load Degradation Surrogate (DegradationSurrogate1DCNN)\n\n"
    "Loaded from the full checkpoint which contains the model weights **and** the\n"
    "per-channel normalisation statistics (`ch_mean`, `ch_std`, `y_mean`, `y_std`).\n"
    "These are essential — the surrogate must receive normalised inputs.\n"
))
cells.append(code([
    "class DegradationSurrogate1DCNN(nn.Module):\n",
    "    \"\"\"3-channel 1D-CNN: (batch,3,60) -> (batch,1) capacity prediction (normalised).\"\"\"\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.conv = nn.Sequential(\n",
    "            nn.Conv1d(3, 32, kernel_size=5, padding=2),\n",
    "            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),\n",
    "            nn.Conv1d(32, 64, kernel_size=3, padding=1),\n",
    "            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),\n",
    "            nn.Conv1d(64, 128, kernel_size=3, padding=1),\n",
    "            nn.BatchNorm1d(128), nn.ReLU(),\n",
    "            nn.AdaptiveAvgPool1d(4),\n",
    "        )\n",
    "        self.head = nn.Sequential(\n",
    "            nn.Flatten(),\n",
    "            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),\n",
    "            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),\n",
    "            nn.Linear(128, 1),\n",
    "        )\n",
    "    def forward(self, x): return self.head(self.conv(x))\n",
    "\n",
    "\n",
    "# Load checkpoint (weights + normalisation stats)\n",
    "ckpt_path = os.path.join(MODEL_DIR, 'degradation_surrogate_1dcnn.pth')\n",
    "ckpt      = torch.load(ckpt_path, map_location=device)\n",
    "\n",
    "surrogate = DegradationSurrogate1DCNN().to(device)\n",
    "surrogate.load_state_dict(ckpt['model_state'])\n",
    "surrogate.eval()\n",
    "\n",
    "# Normalisation statistics\n",
    "CH_MEAN  = np.array(ckpt['ch_mean'], dtype=np.float32)   # (1,3,1)\n",
    "CH_STD   = np.array(ckpt['ch_std'],  dtype=np.float32)   # (1,3,1)\n",
    "Y_MEAN   = float(ckpt['y_mean'])\n",
    "Y_STD    = float(ckpt['y_std'])\n",
    "NUM_PTS  = int(ckpt['num_pts'])   # 60\n",
    "\n",
    "print(f'Surrogate loaded from: {ckpt_path}')\n",
    "print(f'Normalisation: y_mean={Y_MEAN:.4f} Ah  y_std={Y_STD:.4f} Ah')\n",
    "print(f'CH_MEAN: V={CH_MEAN[0,0,0]:.3f}  I={CH_MEAN[0,1,0]:.3f}  T={CH_MEAN[0,2,0]:.3f}')\n",
]))

# ── Cell 3: query_surrogate ────────────────────────────────────────────────────
cells.append(md(
    "## 3 · Surrogate Query Function\n\n"
    "At every RL timestep the agent accumulates a partial V/I/T trace.\n"
    "This function interpolates it to 60 pts, normalises with training statistics,\n"
    "and returns a predicted next-cycle capacity in Ah.\n"
))
cells.append(code([
    "def query_surrogate(V_tr, I_tr, T_tr):\n",
    "    \"\"\"\n",
    "    Args: V_tr, I_tr, T_tr — lists of floats (partial or full cycle trace)\n",
    "    Returns: predicted next-cycle capacity in Ah (denormalised)\n",
    "    \"\"\"\n",
    "    n = len(V_tr)\n",
    "    if n < 2:\n",
    "        return Y_MEAN   # default = average capacity\n",
    "    g_raw = np.linspace(0, 1, n)\n",
    "    g_fix = np.linspace(0, 1, NUM_PTS)\n",
    "    V_f = np.interp(g_fix, g_raw, V_tr)\n",
    "    I_f = np.interp(g_fix, g_raw, I_tr)\n",
    "    T_f = np.interp(g_fix, g_raw, T_tr)\n",
    "    X = np.stack([V_f, I_f, T_f], axis=0)[np.newaxis]   # (1,3,60)\n",
    "    X = (X - CH_MEAN) / CH_STD                           # normalise\n",
    "    with torch.no_grad():\n",
    "        y_norm = surrogate(torch.FloatTensor(X).to(device)).item()\n",
    "    q = y_norm * Y_STD + Y_MEAN                          # denormalise\n",
    "    return float(np.clip(q, Y_MEAN - 4*Y_STD, Y_MEAN + 2*Y_STD))\n",
    "\n",
    "\n",
    "Q_REF = Y_MEAN   # reference capacity (Ah)\n",
    "print(f'Q_REF = {Q_REF:.4f} Ah')\n",
]))

# ── Cell 4: Load data ──────────────────────────────────────────────────────────
cells.append(md("## 4 · Load Battery Dataset"))
cells.append(code([
    "print('Loading batch1...')\n",
    "batch1 = pickle.load(open(os.path.join(BASE_DIR,'batch1.pkl'),'rb'))\n",
    "for k in ['b1c8','b1c10','b1c12','b1c13','b1c22']: del batch1[k]\n",
    "\n",
    "print('Loading batch2...')\n",
    "batch2 = pickle.load(open(os.path.join(BASE_DIR,'batch2.pkl'),'rb'))\n",
    "b2k=['b2c7','b2c8','b2c9','b2c15','b2c16']\n",
    "b1k=['b1c0','b1c1','b1c2','b1c3','b1c4']\n",
    "add_len=[662,981,1060,208,482]\n",
    "for i,bk in enumerate(b1k):\n",
    "    batch1[bk]['cycle_life']+=add_len[i]\n",
    "    for j in batch1[bk]['summary']:\n",
    "        if j=='cycle':\n",
    "            batch1[bk]['summary'][j]=np.hstack((batch1[bk]['summary'][j],\n",
    "                batch2[b2k[i]]['summary'][j]+len(batch1[bk]['summary'][j])))\n",
    "        else:\n",
    "            batch1[bk]['summary'][j]=np.hstack((batch1[bk]['summary'][j],\n",
    "                batch2[b2k[i]]['summary'][j]))\n",
    "    lc=len(batch1[bk]['cycles'])\n",
    "    for j,jk in enumerate(batch2[b2k[i]]['cycles']):\n",
    "        batch1[bk]['cycles'][str(lc+j)]=batch2[b2k[i]]['cycles'][jk]\n",
    "for k in b2k: del batch2[k]\n",
    "\n",
    "print('Loading batch3...')\n",
    "batch3=pickle.load(open(os.path.join(BASE_DIR,'batch3.pkl'),'rb'))\n",
    "for k in ['b3c37','b3c2','b3c23','b3c32','b3c42','b3c43']: del batch3[k]\n",
    "\n",
    "numBat1,numBat2,numBat3=len(batch1),len(batch2),len(batch3)\n",
    "numBat=numBat1+numBat2+numBat3\n",
    "bat_dict={**batch1,**batch2,**batch3}\n",
    "all_keys=list(bat_dict.keys())\n",
    "\n",
    "train_ind = np.arange(1, numBat1+numBat2-1, 2)\n",
    "train_keys=[all_keys[i] for i in train_ind if i<len(all_keys)]\n",
    "print(f'Total cells: {numBat} | Training cells for profiles: {len(train_keys)}')\n",
]))

# ── Cell 5: Build profiles ────────────────────────────────────────────────────
cells.append(md(
    "## 5 · Build Cycle Profiles\n\n"
    "Each profile = one real charge cycle's I(t) and V(t) trace.\n"
    "The environment will sample from these to drive current through the thermal model.\n"
))
cells.append(code([
    "profiles = []\n",
    "for key in train_keys:\n",
    "    cell = bat_dict[key]\n",
    "    cl   = int(cell['cycle_life'])\n",
    "    for c_str, c_data in cell['cycles'].items():\n",
    "        c_num = int(c_str)\n",
    "        if c_num < 10: continue\n",
    "        try:\n",
    "            I = np.array(c_data['I'], dtype=np.float32)\n",
    "            V = np.array(c_data['V'], dtype=np.float32)\n",
    "            if len(I) >= 10:\n",
    "                profiles.append({'I':I,'V':V,'cycle_num':c_num,'cycle_life':cl})\n",
    "        except Exception:\n",
    "            pass\n",
    "print(f'Profiles available: {len(profiles):,}')\n",
    "print(f'Example profile — cycle {profiles[0][\"cycle_num\"]}/{profiles[0][\"cycle_life\"]}'\n",
    "      f', {len(profiles[0][\"I\"])} timesteps')\n",
]))

# ── Cell 6: Environment ────────────────────────────────────────────────────────
cells.append(md(
    "## 6 · Battery Thermal Environment (Lumped-Capacitance Model)\n\n"
    "```\n"
    "Physics:  C_th · dT/dt = I²·R_int  −  P_cool  −  h·(T − T_amb)\n"
    "          C_th=800 J/°C   R_int=0.02 Ω   h=5 W/°C   P_max=50 W\n"
    "```\n\n"
    "**State (dim=6):** `[T/50, T_prev/50, I/6, cycle_norm, SOC, Q_pred/Q_ref]`\n\n"
    "**Action (dim=1):** `a ∈ [-1,1] → P_cool = (a+1)/2 × 50 W`\n\n"
    "**Reward:**\n"
    "```\n"
    "r = -α·(T − 30)²                     temperature comfort\n"
    "  − β·max(0, Q_ref − Q_pred)          degradation penalty  ← NOVEL TERM\n"
    "  − γ·(P_cool / P_max)               cooling energy\n"
    "  − 1000·[T > 45°C]                  hard safety penalty\n"
    "```\n"
))
cells.append(code([
    "class BatteryThermalEnv:\n",
    "    STATE_DIM  = 6\n",
    "    ACTION_DIM = 1\n",
    "\n",
    "    def __init__(self, profiles, alpha=0.1, beta=5.0, gamma_e=0.01,\n",
    "                 T_target=30.0, T_max=45.0, ambient=25.0, dt=1.0, steps=200):\n",
    "        self.profiles     = profiles\n",
    "        self.alpha        = alpha\n",
    "        self.beta         = beta\n",
    "        self.gamma_e      = gamma_e\n",
    "        self.T_target     = T_target\n",
    "        self.T_max        = T_max\n",
    "        self.ambient      = ambient\n",
    "        self.C_th         = 800.0    # J/°C thermal mass\n",
    "        self.R_int        = 0.02     # Ω   internal resistance\n",
    "        self.h_conv       = 5.0      # W/°C convective cooling\n",
    "        self.P_max        = 50.0     # W   max active cooling\n",
    "        self.dt           = dt\n",
    "        self.steps_per_ep = steps\n",
    "        self._rst()\n",
    "\n",
    "    def _rst(self):\n",
    "        self.T = self.T_prev = self.ambient\n",
    "        self.step_idx = 0; self.SOC = 1.0\n",
    "        self.V_tr=[]; self.I_tr=[]; self.T_tr=[]\n",
    "        self.I_profile=self.V_profile=None\n",
    "        self.cycle_num=0; self.cycle_life=1\n",
    "\n",
    "    def reset(self):\n",
    "        self._rst()\n",
    "        self.T=self.T_prev=self.ambient+np.random.uniform(-2,2)\n",
    "        p=random.choice(self.profiles)\n",
    "        self.I_profile=p['I']; self.V_profile=p['V']\n",
    "        self.cycle_num=p['cycle_num']; self.cycle_life=max(p['cycle_life'],1)\n",
    "        return self._obs()\n",
    "\n",
    "    def _I(self):\n",
    "        return float(abs(self.I_profile[min(self.step_idx,len(self.I_profile)-1)]))\n",
    "\n",
    "    def _obs(self):\n",
    "        I  = self._I()\n",
    "        q  = query_surrogate(self.V_tr, self.I_tr, self.T_tr)\n",
    "        return np.array([\n",
    "            self.T / 50.0,\n",
    "            self.T_prev / 50.0,\n",
    "            I / 6.0,\n",
    "            self.cycle_num / self.cycle_life,\n",
    "            self.SOC,\n",
    "            q / Q_REF,\n",
    "        ], dtype=np.float32)\n",
    "\n",
    "    def step(self, action):\n",
    "        action  = float(np.clip(action, -1, 1))\n",
    "        P_cool  = (action + 1) / 2 * self.P_max\n",
    "        I       = self._I()\n",
    "        Q_gen   = I**2 * self.R_int\n",
    "        Q_loss  = self.h_conv * (self.T - self.ambient)\n",
    "        dT      = (Q_gen - P_cool - Q_loss) / self.C_th * self.dt\n",
    "        self.T_prev = self.T\n",
    "        self.T  = float(np.clip(self.T + dT, self.ambient - 5, 80))\n",
    "        vi      = min(self.step_idx, len(self.V_profile)-1)\n",
    "        self.V_tr.append(float(self.V_profile[vi]))\n",
    "        self.I_tr.append(I)\n",
    "        self.T_tr.append(self.T)\n",
    "        self.SOC = float(np.clip(self.SOC - I*self.dt/3600/Q_REF, 0, 1))\n",
    "        q_pred   = query_surrogate(self.V_tr, self.I_tr, self.T_tr)\n",
    "        safety   = self.T > self.T_max\n",
    "        reward   = (-self.alpha*(self.T-self.T_target)**2\n",
    "                    - self.beta*max(0.0, Q_REF-q_pred)\n",
    "                    - self.gamma_e*(P_cool/self.P_max)\n",
    "                    + (-1000.0 if safety else 0.0))\n",
    "        self.step_idx += 1\n",
    "        done = self.step_idx >= self.steps_per_ep or safety\n",
    "        info = dict(T=self.T, P_cool=P_cool, q_pred=q_pred,\n",
    "                    safety=safety, r_temp=-self.alpha*(self.T-self.T_target)**2,\n",
    "                    r_degrad=-self.beta*max(0.0,Q_REF-q_pred))\n",
    "        return self._obs(), reward, done, info\n",
    "\n",
    "print('BatteryThermalEnv defined. State dim:', BatteryThermalEnv.STATE_DIM)\n",
]))

# ── Cell 7: Networks ──────────────────────────────────────────────────────────
cells.append(md(
    "## 7 · DDPG Networks — Actor, Critic, OUNoise, ReplayBuffer\n\n"
    "**Actor:** `s(6) → LayerNorm(256) → LayerNorm(128) → Tanh → a(1)`  \n"
    "**Critic:** `[s,a](7) → 256 → 128 → Q(1)`  \n"
    "LayerNorm in the actor is important here — the surrogate output (Q_pred/Q_ref) "
    "can vary more than the other state dimensions, so normalisation stabilises training.\n"
))
cells.append(code([
    "class Actor(nn.Module):\n",
    "    def __init__(self, sd=6, ad=1):\n",
    "        super().__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(sd, 256), nn.LayerNorm(256), nn.ReLU(),\n",
    "            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),\n",
    "            nn.Linear(128, ad), nn.Tanh()\n",
    "        )\n",
    "        nn.init.uniform_(self.net[-2].weight, -3e-3, 3e-3)\n",
    "        nn.init.uniform_(self.net[-2].bias,   -3e-3, 3e-3)\n",
    "    def forward(self, s): return self.net(s)\n",
    "\n",
    "\n",
    "class Critic(nn.Module):\n",
    "    def __init__(self, sd=6, ad=1):\n",
    "        super().__init__()\n",
    "        self.net = nn.Sequential(\n",
    "            nn.Linear(sd+ad, 256), nn.ReLU(),\n",
    "            nn.Linear(256, 128),   nn.ReLU(),\n",
    "            nn.Linear(128, 1)\n",
    "        )\n",
    "        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)\n",
    "        nn.init.uniform_(self.net[-1].bias,   -3e-3, 3e-3)\n",
    "    def forward(self, s, a): return self.net(torch.cat([s, a], dim=-1))\n",
    "\n",
    "\n",
    "class OUNoise:\n",
    "    \"\"\"Ornstein-Uhlenbeck noise — temporally correlated exploration.\"\"\"\n",
    "    def __init__(self, dim=1, theta=0.15, sigma=0.2, dt=1e-2):\n",
    "        self.theta=theta; self.sigma=sigma; self.dt=dt; self.dim=dim\n",
    "        self.reset()\n",
    "    def reset(self): self.x=np.zeros(self.dim)\n",
    "    def sample(self):\n",
    "        dx=self.theta*(-self.x)*self.dt+self.sigma*np.sqrt(self.dt)*np.random.randn(self.dim)\n",
    "        self.x+=dx; return self.x.copy()\n",
    "\n",
    "\n",
    "class ReplayBuffer:\n",
    "    def __init__(self, cap=100_000): self.buf=deque(maxlen=cap)\n",
    "    def push(self,s,a,r,ns,d): self.buf.append((s,float(a),float(r),ns,float(d)))\n",
    "    def sample(self,bs):\n",
    "        b=random.sample(self.buf,bs)\n",
    "        s,a,r,ns,d=zip(*b)\n",
    "        return (torch.FloatTensor(np.array(s)).to(device),\n",
    "                torch.FloatTensor(np.array(a)).unsqueeze(-1).to(device),\n",
    "                torch.FloatTensor(np.array(r)).unsqueeze(-1).to(device),\n",
    "                torch.FloatTensor(np.array(ns)).to(device),\n",
    "                torch.FloatTensor(np.array(d)).unsqueeze(-1).to(device))\n",
    "    def __len__(self): return len(self.buf)\n",
    "\n",
    "print('Actor, Critic, OUNoise, ReplayBuffer defined.')\n",
]))

# ── Cell 8: DDPGAgent + PID ────────────────────────────────────────────────────
cells.append(md("## 8 · DDPG Agent & PID Baseline"))
cells.append(code([
    "class DDPGAgent:\n",
    "    def __init__(self, sd=6, ad=1, alr=1e-4, clr=3e-4,\n",
    "                 gamma=0.99, tau=0.005, buf_cap=100_000, bs=256):\n",
    "        self.actor  = Actor(sd,ad).to(device)\n",
    "        self.critic = Critic(sd,ad).to(device)\n",
    "        self.at     = copy.deepcopy(self.actor)\n",
    "        self.ct     = copy.deepcopy(self.critic)\n",
    "        for p in self.at.parameters():  p.requires_grad_(False)\n",
    "        for p in self.ct.parameters(): p.requires_grad_(False)\n",
    "        self.ao=optim.Adam(self.actor.parameters(),  lr=alr)\n",
    "        self.co=optim.Adam(self.critic.parameters(), lr=clr)\n",
    "        self.buf=ReplayBuffer(buf_cap)\n",
    "        self.bs,self.gamma,self.tau=bs,gamma,tau\n",
    "        self.noise=OUNoise(ad)\n",
    "\n",
    "    def act(self, state, explore=True):\n",
    "        s=torch.FloatTensor(state).unsqueeze(0).to(device)\n",
    "        with torch.no_grad(): a=self.actor(s).cpu().numpy()[0]\n",
    "        if explore: a+=self.noise.sample()\n",
    "        return float(np.clip(a,-1,1))\n",
    "\n",
    "    def update(self):\n",
    "        if len(self.buf)<self.bs: return\n",
    "        s,a,r,ns,d=self.buf.sample(self.bs)\n",
    "        with torch.no_grad():\n",
    "            tQ=r+self.gamma*(1-d)*self.ct(ns,self.at(ns))\n",
    "        cL=F.mse_loss(self.critic(s,a),tQ)\n",
    "        self.co.zero_grad(); cL.backward()\n",
    "        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),1.0)\n",
    "        self.co.step()\n",
    "        aL=-self.critic(s,self.actor(s)).mean()\n",
    "        self.ao.zero_grad(); aL.backward()\n",
    "        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1.0)\n",
    "        self.ao.step()\n",
    "        for p,tp in zip(self.actor.parameters(),self.at.parameters()):\n",
    "            tp.data.copy_(self.tau*p.data+(1-self.tau)*tp.data)\n",
    "        for p,tp in zip(self.critic.parameters(),self.ct.parameters()):\n",
    "            tp.data.copy_(self.tau*p.data+(1-self.tau)*tp.data)\n",
    "\n",
    "    def save(self,path):\n",
    "        torch.save({'actor':self.actor.state_dict(),'critic':self.critic.state_dict()},path)\n",
    "        print(f'Saved: {path}')\n",
    "    def load(self,path):\n",
    "        ck=torch.load(path,map_location=device)\n",
    "        self.actor.load_state_dict(ck['actor'])\n",
    "        self.critic.load_state_dict(ck['critic'])\n",
    "        self.at=copy.deepcopy(self.actor)\n",
    "        self.ct=copy.deepcopy(self.critic)\n",
    "\n",
    "\n",
    "class PIDController:\n",
    "    \"\"\"Classical PID baseline (no learning, no degradation awareness).\"\"\"\n",
    "    def __init__(self, T_target=30.0, Kp=0.05, Ki=0.005, Kd=0.001):\n",
    "        self.T_target=T_target\n",
    "        self.Kp,self.Ki,self.Kd=Kp,Ki,Kd\n",
    "        self.integral=self.prev_err=0.0\n",
    "    def reset(self): self.integral=self.prev_err=0.0\n",
    "    def act(self, state, explore=None):\n",
    "        T=state[0]*50.0; e=T-self.T_target\n",
    "        self.integral=float(np.clip(self.integral+e,-100,100))\n",
    "        d=e-self.prev_err; self.prev_err=e\n",
    "        return float(np.clip(self.Kp*e+self.Ki*self.integral+self.Kd*d,-1,1))\n",
    "\n",
    "print('DDPGAgent and PIDController ready.')\n",
]))

# ── Cell 9: Training ───────────────────────────────────────────────────────────
cells.append(md(
    "## 9 · Training Loop (500 Episodes)\n\n"
    "`BETA = 5.0` is the key coupling parameter — it controls how strongly\n"
    "predicted capacity loss penalises the agent. The ablation study in Cell 10\n"
    "sweeps this to produce the Pareto trade-off curve.\n"
))
cells.append(code([
    "# ── Hyperparameters ─────────────────────────────────────────────────────────\n",
    "EPISODES     = 500\n",
    "STEPS_PER_EP = 200\n",
    "BATCH_SIZE   = 256\n",
    "ALPHA        = 0.1    # temperature comfort weight\n",
    "BETA         = 5.0    # degradation penalty weight  ← KEY PARAMETER\n",
    "GAMMA_E      = 0.01   # energy efficiency weight\n",
    "EVAL_EVERY   = 50\n",
    "\n",
    "env   = BatteryThermalEnv(profiles, alpha=ALPHA, beta=BETA,\n",
    "                          gamma_e=GAMMA_E, steps=STEPS_PER_EP)\n",
    "agent = DDPGAgent(bs=BATCH_SIZE)\n",
    "pid   = PIDController()\n",
    "\n",
    "def run_ep(ag, env_obj, explore=True, is_pid=False):\n",
    "    state = env_obj.reset()\n",
    "    if is_pid: ag.reset()\n",
    "    tot_r=0.0; t_errs=[]; q_preds=[]; p_cools=[]; safes=0\n",
    "    for _ in range(env_obj.steps_per_ep):\n",
    "        a  = ag.act(state, explore=explore) if not is_pid else ag.act(state)\n",
    "        ns,r,done,info = env_obj.step(a)\n",
    "        if not is_pid and explore:\n",
    "            ag.buf.push(state,a,r,ns,float(done))\n",
    "            ag.update()\n",
    "        tot_r+=r\n",
    "        t_errs.append(abs(info['T']-env_obj.T_target))\n",
    "        q_preds.append(info['q_pred'])\n",
    "        p_cools.append(info['P_cool'])\n",
    "        if info['safety']: safes+=1\n",
    "        state=ns\n",
    "        if done: break\n",
    "    return {'R':tot_r,\n",
    "            'T_MAE':float(np.mean(t_errs)),\n",
    "            'cap_pct':float(np.mean(q_preds)/Q_REF*100),\n",
    "            'cooling':float(np.sum(p_cools)),\n",
    "            'safety':safes}\n",
    "\n",
    "train_R=[]; eval_ddpg=[]; eval_pid=[]\n",
    "print('='*65)\n",
    "print(f'Training DDPG | Episodes={EPISODES} | beta={BETA}')\n",
    "print('='*65)\n",
    "\n",
    "for ep in range(EPISODES):\n",
    "    agent.noise.reset()\n",
    "    m=run_ep(agent,env,explore=True)\n",
    "    train_R.append(m['R'])\n",
    "    if (ep+1)%EVAL_EVERY==0:\n",
    "        md_=run_ep(agent,env,explore=False)\n",
    "        mp_=run_ep(pid,env,explore=False,is_pid=True)\n",
    "        eval_ddpg.append(md_); eval_pid.append(mp_)\n",
    "        print(f\"Ep{ep+1:4d} | R={m['R']:7.1f} | \"\n",
    "              f\"DDPG T_MAE={md_['T_MAE']:.2f}C Cap={md_['cap_pct']:.1f}% Saf={md_['safety']} | \"\n",
    "              f\"PID T_MAE={mp_['T_MAE']:.2f}C Cap={mp_['cap_pct']:.1f}%\")\n",
    "\n",
    "agent.save(os.path.join(MODEL_DIR,'thermal_agent_ddpg.pth'))\n",
    "print('Training complete.')\n",
]))

# ── Cell 10: Beta ablation ────────────────────────────────────────────────────
cells.append(md(
    "## 10 · β Ablation Study — The Pareto Trade-off\n\n"
    "Train 6 independent agents with β ∈ {0, 1, 2, 5, 10, 20} (100 episodes each).\n"
    "β=0 is thermal-only DDPG with NO degradation coupling — the ablation baseline.\n\n"
    "This produces the key analytical figure: thermal accuracy vs capacity preservation Pareto curve.\n"
    "This plot is what distinguishes this paper from prior work [P5, P7, P8] which all use β=fixed.\n"
))
cells.append(code([
    "BETA_VALS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]\n",
    "ablation  = {}\n",
    "print('Running beta ablation (100 eps each)...')\n",
    "\n",
    "for bv in BETA_VALS:\n",
    "    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)\n",
    "    env_ab = BatteryThermalEnv(profiles, alpha=ALPHA, beta=bv,\n",
    "                               gamma_e=GAMMA_E, steps=STEPS_PER_EP)\n",
    "    ag_ab  = DDPGAgent(bs=BATCH_SIZE)\n",
    "    for _ in range(100):\n",
    "        ag_ab.noise.reset()\n",
    "        run_ep(ag_ab, env_ab, explore=True)\n",
    "    tm_s, cp_s = [], []\n",
    "    for _ in range(10):\n",
    "        m=run_ep(ag_ab,env_ab,explore=False)\n",
    "        tm_s.append(m['T_MAE']); cp_s.append(m['cap_pct'])\n",
    "    ablation[bv]={'T_MAE':float(np.mean(tm_s)),'cap_pct':float(np.mean(cp_s))}\n",
    "    print(f'  beta={bv:5.1f} -> T_MAE={ablation[bv][\"T_MAE\"]:.2f}C  Cap={ablation[bv][\"cap_pct\"]:.1f}%')\n",
]))

# ── Cell 11: Trajectory collection ───────────────────────────────────────────
cells.append(md(
    "## 11 · Collect Fixed-Profile Trajectories\n\n"
    "Fix one cycle profile so DDPG and PID face **identical conditions** — "
    "only the controller differs. This is the fair comparison for the paper figures.\n"
))
cells.append(code([
    "random.seed(SEED)\n",
    "fixed_profile = random.choice(profiles)\n",
    "\n",
    "def run_fixed(ag, profile, is_pid=False):\n",
    "    env_f = BatteryThermalEnv([profile], alpha=ALPHA, beta=BETA,\n",
    "                              gamma_e=GAMMA_E, steps=STEPS_PER_EP)\n",
    "    st=env_f.reset()\n",
    "    if is_pid: ag.reset()\n",
    "    Ts=[]; Ps=[]; Qs=[]; Rs=[]\n",
    "    for _ in range(env_f.steps_per_ep):\n",
    "        a=ag.act(st,explore=False) if not is_pid else ag.act(st)\n",
    "        ns,r,done,info=env_f.step(a)\n",
    "        Ts.append(info['T']); Ps.append(info['P_cool'])\n",
    "        Qs.append(info['q_pred']); Rs.append(r)\n",
    "        st=ns\n",
    "        if done: break\n",
    "    return {'T':Ts,'P':Ps,'Q':Qs,'R':Rs}\n",
    "\n",
    "traj_d=run_fixed(agent, fixed_profile, is_pid=False)\n",
    "traj_p=run_fixed(pid,   fixed_profile, is_pid=True)\n",
    "print(f'Trajectories collected: {len(traj_d[\"T\"])} steps')\n",
]))

# ── Cell 12: Plots ─────────────────────────────────────────────────────────────
cells.append(md("## 12 · Publication-Quality Figures"))
cells.append(code([
    "eval_eps = list(range(EVAL_EVERY, EPISODES+1, EVAL_EVERY))\n",
    "t  = np.arange(len(traj_d['T']))\n",
    "n  = min(len(eval_eps), len(eval_ddpg))\n",
    "\n",
    "# ── Fig 1: Training curves ────────────────────────────────────────────────────\n",
    "fig, axes = plt.subplots(1,2,figsize=(14,5))\n",
    "fig.suptitle('Fig 1 — DDPG Training Curves', fontsize=13, fontweight='bold')\n",
    "ax=axes[0]\n",
    "ax.plot(range(1,len(train_R)+1), train_R, color='steelblue', alpha=0.35, lw=0.8)\n",
    "w=20\n",
    "if len(train_R)>=w:\n",
    "    ax.plot(range(w,len(train_R)+1),\n",
    "            np.convolve(train_R,np.ones(w)/w,'valid'),'b-',lw=2,label=f'{w}-ep MA')\n",
    "ax.set_xlabel('Episode'); ax.set_ylabel('Total Reward')\n",
    "ax.set_title('Training Reward'); ax.legend(); ax.grid(alpha=0.3)\n",
    "ax=axes[1]\n",
    "ax.plot(eval_eps[:n],[m['R'] for m in eval_ddpg[:n]],'b-o',lw=2,ms=5,label='DDPG-Surrogate')\n",
    "if eval_pid:\n",
    "    ax.axhline(np.mean([m['R'] for m in eval_pid]),color='orange',ls='--',lw=2,label='PID')\n",
    "ax.set_xlabel('Episode'); ax.set_ylabel('Eval Reward')\n",
    "ax.set_title('Eval Reward vs PID Baseline'); ax.legend(); ax.grid(alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(RESULTS_DIR,'fig1_training.png'),dpi=150,bbox_inches='tight')\n",
    "plt.show(); print('Saved fig1_training.png')\n",
    "\n",
    "# ── Fig 2: Trajectory comparison ─────────────────────────────────────────────\n",
    "fig, axes = plt.subplots(2,2,figsize=(14,10))\n",
    "fig.suptitle('Fig 2 — Trajectories: DDPG-Surrogate vs PID (identical current profile)',\n",
    "             fontsize=13,fontweight='bold')\n",
    "ax=axes[0,0]\n",
    "ax.plot(t,traj_p['T'],color='orange',lw=2,label='PID')\n",
    "ax.plot(t,traj_d['T'],color='steelblue',lw=2,label='DDPG-Surrogate')\n",
    "ax.axhline(30,color='green',ls='--',lw=1.5,label='T_target=30C')\n",
    "ax.axhline(45,color='red',ls='--',lw=1.5,label='Safety=45C')\n",
    "ax.fill_between(t,28,32,alpha=0.08,color='green')\n",
    "ax.set_xlabel('Timestep (s)'); ax.set_ylabel('Temp (C)')\n",
    "ax.set_title('Temperature Trajectory'); ax.legend(fontsize=8); ax.grid(alpha=0.3)\n",
    "ax=axes[0,1]\n",
    "ax.plot(t,traj_p['P'],color='orange',lw=2,label='PID')\n",
    "ax.plot(t,traj_d['P'],color='steelblue',lw=2,label='DDPG-Surrogate')\n",
    "ax.set_xlabel('Timestep (s)'); ax.set_ylabel('Cooling Power (W)')\n",
    "ax.set_title('Cooling Action'); ax.legend(); ax.grid(alpha=0.3)\n",
    "ax=axes[1,0]\n",
    "ax.plot(t,np.array(traj_p['Q'])/Q_REF*100,color='orange',lw=2,label='PID')\n",
    "ax.plot(t,np.array(traj_d['Q'])/Q_REF*100,color='steelblue',lw=2,label='DDPG-Surrogate')\n",
    "ax.axhline(100,color='green',ls='--',lw=1)\n",
    "ax.set_xlabel('Timestep (s)'); ax.set_ylabel('Predicted Capacity (%)')\n",
    "ax.set_title('Surrogate-Predicted Capacity'); ax.legend(fontsize=8); ax.grid(alpha=0.3)\n",
    "ax=axes[1,1]\n",
    "ax.plot(t,np.cumsum(traj_p['R']),color='orange',lw=2,label='PID')\n",
    "ax.plot(t,np.cumsum(traj_d['R']),color='steelblue',lw=2,label='DDPG-Surrogate')\n",
    "ax.set_xlabel('Timestep (s)'); ax.set_ylabel('Cumulative Reward')\n",
    "ax.set_title('Cumulative Reward'); ax.legend(); ax.grid(alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(RESULTS_DIR,'fig2_trajectories.png'),dpi=150,bbox_inches='tight')\n",
    "plt.show(); print('Saved fig2_trajectories.png')\n",
    "\n",
    "# ── Fig 3: Beta ablation (Pareto curve) ──────────────────────────────────────\n",
    "bvals =[b for b in sorted(ablation.keys())]\n",
    "t_maes=[ablation[b]['T_MAE']   for b in bvals]\n",
    "caps  =[ablation[b]['cap_pct'] for b in bvals]\n",
    "fig, axes = plt.subplots(1,3,figsize=(16,5))\n",
    "fig.suptitle('Fig 3 — beta Ablation: Degradation Penalty Trade-off',fontsize=13,fontweight='bold')\n",
    "axes[0].plot(bvals,t_maes,'b-o',lw=2,ms=8)\n",
    "axes[0].axvline(5,color='red',ls='--',lw=1.5,label='beta=5 (proposed)')\n",
    "axes[0].set_xlabel('beta'); axes[0].set_ylabel('T MAE (C)')\n",
    "axes[0].set_title('Temp Error vs beta'); axes[0].legend(); axes[0].grid(alpha=0.3)\n",
    "axes[1].plot(bvals,caps,'g-o',lw=2,ms=8)\n",
    "axes[1].axvline(5,color='red',ls='--',lw=1.5,label='beta=5 (proposed)')\n",
    "axes[1].set_xlabel('beta'); axes[1].set_ylabel('Cap Preserved (%)')\n",
    "axes[1].set_title('Capacity vs beta'); axes[1].legend(); axes[1].grid(alpha=0.3)\n",
    "sc=axes[2].scatter(t_maes,caps,c=bvals,cmap='viridis',s=140,zorder=5)\n",
    "for b,tm,c in zip(bvals,t_maes,caps):\n",
    "    axes[2].annotate(f'b={b}',(tm,c),xytext=(5,5),textcoords='offset points',fontsize=9)\n",
    "plt.colorbar(sc,ax=axes[2],label='beta')\n",
    "axes[2].set_xlabel('T MAE (C)'); axes[2].set_ylabel('Cap (%)')\n",
    "axes[2].set_title('Pareto: Thermal Accuracy vs Capacity')\n",
    "axes[2].grid(alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(RESULTS_DIR,'fig3_ablation.png'),dpi=150,bbox_inches='tight')\n",
    "plt.show(); print('Saved fig3_ablation.png')\n",
    "\n",
    "# ── Fig 4: Eval metrics over training ────────────────────────────────────────\n",
    "fig, axes = plt.subplots(1,3,figsize=(16,5))\n",
    "fig.suptitle('Fig 4 — Evaluation Metrics Over Training',fontsize=13,fontweight='bold')\n",
    "axes[0].plot(eval_eps[:n],[m['T_MAE']   for m in eval_ddpg[:n]],'b-o',lw=2,label='DDPG')\n",
    "if eval_pid:\n",
    "    axes[0].axhline(np.mean([m['T_MAE'] for m in eval_pid]),color='orange',ls='--',lw=2,label='PID')\n",
    "axes[0].set_xlabel('Episode'); axes[0].set_ylabel('T MAE (C)')\n",
    "axes[0].set_title('Temperature Error'); axes[0].legend(); axes[0].grid(alpha=0.3)\n",
    "axes[1].plot(eval_eps[:n],[m['cap_pct'] for m in eval_ddpg[:n]],'b-o',lw=2,label='DDPG')\n",
    "if eval_pid:\n",
    "    axes[1].axhline(np.mean([m['cap_pct'] for m in eval_pid]),color='orange',ls='--',lw=2,label='PID')\n",
    "axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Cap (%)')\n",
    "axes[1].set_title('Capacity Preservation'); axes[1].legend(); axes[1].grid(alpha=0.3)\n",
    "axes[2].plot(eval_eps[:n],[m['safety'] for m in eval_ddpg[:n]],'r-o',lw=2)\n",
    "axes[2].set_xlabel('Episode'); axes[2].set_ylabel('Safety Violations')\n",
    "axes[2].set_title('Safety Violations (target=0)'); axes[2].grid(alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.savefig(os.path.join(RESULTS_DIR,'fig4_eval_metrics.png'),dpi=150,bbox_inches='tight')\n",
    "plt.show(); print('Saved fig4_eval_metrics.png')\n",
]))

# ── Cell 13: Final evaluation + save ─────────────────────────────────────────
cells.append(md("## 13 · Final Evaluation Summary"))
cells.append(code([
    "print('Running 20-episode final evaluation...')\n",
    "d_ms=[]; p_ms=[]\n",
    "for _ in range(20):\n",
    "    d_ms.append(run_ep(agent,env,explore=False))\n",
    "    p_ms.append(run_ep(pid,env,explore=False,is_pid=True))\n",
    "\n",
    "def tbl(ms, name):\n",
    "    print(f'\\n  {name}')\n",
    "    print(f'  {\"-\"*55}')\n",
    "    for k,lbl in [('T_MAE','Temp MAE (C)'),('cap_pct','Capacity Preserved (%)'),\n",
    "                  ('cooling','Cooling Energy (J)'),('safety','Safety Violations')]:\n",
    "        v=[m[k] for m in ms]\n",
    "        print(f'  {lbl:28s}: {np.mean(v):8.3f} +/- {np.std(v):.3f}')\n",
    "\n",
    "print('='*60)\n",
    "print('FINAL EVALUATION (20-episode mean +/- std)')\n",
    "print('='*60)\n",
    "tbl(d_ms,'DDPG-Surrogate [beta=5]  <- OURS')\n",
    "tbl(p_ms,'PID Baseline')\n",
    "\n",
    "# ── Save JSON results ─────────────────────────────────────────────────────────\n",
    "results = {\n",
    "    'ddpg': {k:float(np.mean([m[k] for m in d_ms])) for k in d_ms[0]},\n",
    "    'pid':  {k:float(np.mean([m[k] for m in p_ms])) for k in p_ms[0]},\n",
    "    'ablation': {str(b):v for b,v in ablation.items()},\n",
    "    'hyperparams': {'ALPHA':ALPHA,'BETA':BETA,'GAMMA_E':GAMMA_E,\n",
    "                    'EPISODES':EPISODES,'STEPS_PER_EP':STEPS_PER_EP}\n",
    "}\n",
    "with open(os.path.join(RESULTS_DIR,'results_summary.json'),'w') as f:\n",
    "    json.dump(results,f,indent=2)\n",
    "\n",
    "print(f'\\nAll figures saved to: {RESULTS_DIR}')\n",
    "print(f'Model saved to:       {MODEL_DIR}/thermal_agent_ddpg.pth')\n",
    "print(f'Results JSON:         {RESULTS_DIR}/results_summary.json')\n",
]))

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "battery", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.21"},
    },
    "cells": cells,
}

out = os.path.join(BASE, "thermal_agent.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Written: {out}  ({len(cells)} cells)")
