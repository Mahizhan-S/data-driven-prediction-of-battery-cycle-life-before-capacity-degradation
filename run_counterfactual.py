"""
Counterfactual Temperature Sensitivity Experiment
Holds V(t), I(t) fixed; perturbs T(t) by +2, +5, +10°C.
Measures ΔQ̂ to test surrogate temperature sensitivity.
"""
import pickle, json, numpy as np, os, sys, torch, torch.nn as nn

BASE = '/Users/mahizhan/Documents/Sem7/Github/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation'

# ── 1. Load checkpoint ─────────────────────────────────────────────────────────
ckpt = torch.load(os.path.join(BASE, 'Model', 'degradation_surrogate_1dcnn.pth'),
                  map_location='cpu', weights_only=False)

print(f"arch: {ckpt['arch']}  input_shape: {ckpt['input_shape']}")
mu  = torch.tensor(ckpt['ch_mean'], dtype=torch.float32)   # (1,3,1)
std = torch.tensor(ckpt['ch_std'],  dtype=torch.float32)
y_mean = float(ckpt['y_mean']); y_std = float(ckpt['y_std'])

# ── 2. Rebuild architecture ────────────────────────────────────────────────────
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
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )
    def forward(self, x): return self.head(self.conv(x)).squeeze(-1)

model = DegradationSurrogate1DCNN()
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

# ── 3. Load batch data with continuation merge ─────────────────────────────────
batch1 = pickle.load(open(os.path.join(BASE, 'batch1.pkl'), 'rb'))
for k in ['b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22']: del batch1[k]

batch2 = pickle.load(open(os.path.join(BASE, 'batch2.pkl'), 'rb'))
b2k = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
b1k = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
add_len = [662, 981, 1060, 208, 482]
for i, bk in enumerate(b1k):
    batch1[bk]['cycle_life'] += add_len[i]
    for j in batch1[bk]['summary']:
        if j == 'cycle':
            batch1[bk]['summary'][j] = np.hstack((
                batch1[bk]['summary'][j],
                batch2[b2k[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
        else:
            batch1[bk]['summary'][j] = np.hstack((
                batch1[bk]['summary'][j], batch2[b2k[i]]['summary'][j]))
    lc = len(batch1[bk]['cycles'])
    for j2, jk in enumerate(batch2[b2k[i]]['cycles']):
        batch1[bk]['cycles'][str(lc + j2)] = batch2[b2k[i]]['cycles'][jk]
for k in b2k: del batch2[k]

batch3 = pickle.load(open(os.path.join(BASE, 'batch3.pkl'), 'rb'))
for k in ['b3c37', 'b3c2', 'b3c23', 'b3c32', 'b3c42', 'b3c43']: del batch3[k]
bat_dict = {**batch1, **batch2, **batch3}

# ── 4. Get test keys ───────────────────────────────────────────────────────────
split_p = os.path.join(BASE, 'data_split.json')
if os.path.exists(split_p):
    test_keys = json.load(open(split_p)).get('test', [])
    print(f"Test split: {len(test_keys)} cells from data_split.json")
else:
    test_keys = [k for k in bat_dict if k.startswith('b3')]
    print(f"Fallback: {len(test_keys)} b3 cells")

# ── 5. Extract 500 cycle profiles ─────────────────────────────────────────────
NUM_PTS = 60; profiles = []
for key in test_keys:
    if key not in bat_dict: continue
    cell = bat_dict[key]
    QD = cell['summary'].get('QD', cell['summary'].get('QDischarge', None))
    if QD is None: continue
    for ci in range(5, min(len(QD) - 1, 300)):
        cs = str(ci)
        if cs not in cell['cycles']: continue
        c = cell['cycles'][cs]
        try:
            Vr = np.array(c['V'], float); Ir = np.array(c['I'], float); Tr = np.array(c['T'], float)
        except: continue
        if len(Vr) < 4 or len(Tr) < 4: continue
        g = np.linspace(0, 1, NUM_PTS)
        Vf = np.interp(g, np.linspace(0, 1, len(Vr)), Vr).astype(np.float32)
        If = np.interp(g, np.linspace(0, 1, len(Ir)), Ir).astype(np.float32)
        Tf = np.interp(g, np.linspace(0, 1, len(Tr)), Tr).astype(np.float32)
        profiles.append({'V': Vf, 'I': If, 'T': Tf})
    if len(profiles) >= 500: break
print(f"Extracted {len(profiles)} profiles")

# ── 6. Predict function ────────────────────────────────────────────────────────
def predict(V, I, T):
    x = torch.tensor(np.stack([V, I, T], 0), dtype=torch.float32).unsqueeze(0)  # (1,3,60)
    xn = (x - mu) / (std + 1e-8)
    with torch.no_grad():
        out = float(model(xn).item())
    return out * y_std + y_mean  # denorm to Ah

# ── 7. Counterfactual experiment ───────────────────────────────────────────────
print("\n" + "="*70)
print("COUNTERFACTUAL TEMPERATURE SENSITIVITY")
print("V(t) and I(t) held fixed; T(t) uniformly perturbed by ΔT")
print("="*70)
print(f"\n{'ΔT':>6} | {'Mean ΔQ̂ (Ah)':>13} | {'±Std':>8} | {'95% CI':>24} | {'% of Q̄':>7}")
print("-"*75)

Q_base = np.array([predict(p['V'], p['I'], p['T']) for p in profiles])
Q_mean = float(Q_base.mean())
results = {}

for dT in [2, 5, 10]:
    Q_pert = np.array([predict(p['V'], p['I'], p['T'] + dT) for p in profiles])
    deltas = Q_pert - Q_base
    ci = 1.96 * deltas.std() / np.sqrt(len(deltas))
    pct = abs(deltas.mean()) / Q_mean * 100
    results[dT] = {
        'mean_dQ': float(deltas.mean()),
        'std': float(deltas.std()),
        'ci95_half': float(ci),
        'pct_of_Qmean': float(pct)
    }
    print(f"{'+'+str(dT)+'°C':>6} | {deltas.mean():>13.5f} | {deltas.std():>8.5f} | "
          f"[{deltas.mean()-ci:+.5f}, {deltas.mean()+ci:+.5f}] | {pct:>6.3f}%")

print(f"\nBaseline mean Q̂ = {Q_mean:.4f} Ah  (n={len(profiles)} profiles)")

max_pct = max(r['pct_of_Qmean'] for r in results.values())
print()
if max_pct < 1.0:
    print(f"CONCLUSION: INSENSITIVE — max shift (+10°C) = {results[10]['pct_of_Qmean']:.3f}% of Q̄.")
    print("The surrogate does not detect moderate temperature perturbations.")
    print("→ Physics-calibrated stress term is the primary thermal-degradation signal in the reward.")
    sensitivity_verdict = "insensitive"
else:
    print(f"CONCLUSION: SENSITIVE — max shift (+10°C) = {results[10]['pct_of_Qmean']:.3f}% of Q̄.")
    print("→ Surrogate has non-negligible temperature sensitivity; revise Section 3.6.")
    sensitivity_verdict = "sensitive"

# ── 8. Save results ────────────────────────────────────────────────────────────
os.makedirs(os.path.join(BASE, 'thermal_agent_v4_results'), exist_ok=True)
out = {
    'experiment': 'counterfactual_T_sensitivity',
    'n_profiles': len(profiles),
    'Q_mean_Ah': Q_mean,
    'sensitivity_verdict': sensitivity_verdict,
    'results': {f'+{k}C': v for k, v in results.items()}
}
out_path = os.path.join(BASE, 'thermal_agent_v4_results', 'counterfactual_T_sensitivity.json')
json.dump(out, open(out_path, 'w'), indent=2)
print(f"\nSaved → {out_path}")
