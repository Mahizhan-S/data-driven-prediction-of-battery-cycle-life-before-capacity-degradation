# Dataset Migration Plan
## From Severson 2019 → KIT NMC 228-Cell (Primary) + Severson (Surrogate) + PyBaMM (RL Env)

---

## ⚠️ IMPORTANT: Read This First

**You do NOT need to delete your current work.** The migration is additive — your existing
Severson-trained surrogate (R²=0.888) remains valid and useful. The KIT dataset
becomes the **primary dataset** for the multi-agent system, while Severson becomes
the **surrogate training source** only.

---

## 1. The Three Datasets You Now Need

### Dataset 1 — KIT NMC 228-Cell (Primary, New)

| Property | Value |
|:---|:---|
| **Name** | Comprehensive Battery Aging Dataset |
| **Institution** | Karlsruhe Institute of Technology (KIT), Germany |
| **DOI** | 10.35097/1969 (result/capacity data) · 10.35097/1947 (log/raw data) |
| **Paper** | Nature Scientific Data, DOI: 10.1038/s41597-024-03831-x |
| **Download** | https://doi.org/10.35097/1947 (69.4 GB) |
| **License** | CC BY 4.0 (free for academic use, citation required) |
| **Python scripts** | https://github.com/energystatusdata/bat-age-data-scripts/ |
| **Size** | 228 NMC/C+SiO cells, 3 billion data points, 600+ days |
| **Temperatures** | Multiple (calendar + cyclic + driving cycles) |
| **Signals** | V(t), I(t), T(t), capacity fade, EIS impedance, SoH per checkup |

**Why this is the right dataset:**
- NMC chemistry is the dominant EV battery (Tesla, Nissan Leaf, Chevy Bolt all use NMC)
- Multi-temperature cycling → feeds the thermal agent
- Multi-condition aging → feeds the health agent (GNN)
- EIS impedance data → enables safety thresholds derived from data
- 228 cells >> 124 Severson cells → more robust training

### Dataset 2 — Severson 2019 (Keep for Surrogate Only)

| Property | Value |
|:---|:---|
| **You already have it** | batch1.pkl, batch2.pkl, batch3.pkl |
| **Role** | Trains the 1D-CNN degradation surrogate ONLY |
| **Do NOT remove** | Your surrogate R²=0.888 is already trained and saved |

### Dataset 3 — PyBaMM (Simulator, New)

| Property | Value |
|:---|:---|
| **Name** | Python Battery Mathematical Modelling |
| **Install** | `pip install pybamm` |
| **Purpose** | Interactive RL training environment for Charging Agent + QMIX |
| **Why needed** | Static datasets cannot train RL agents — need interactive simulation |

---

## 2. What You Need to Change vs. Keep

```
CURRENT STATE (Severson only)          →   NEW STATE (KIT + Severson + PyBaMM)
────────────────────────────────────      ──────────────────────────────────────
Surrogate (1D-CNN)                        KEEP AS IS ✅
  └─ Trained on Severson                  └─ Severson → still used for this

Thermal Agent (SAC)                       MODIFY ⚠️
  └─ Current profiles from Severson       └─ Replace with KIT I(t) profiles
  └─ T calibration from Severson          └─ Use KIT multi-temperature T data

Health Agent                              REBUILD on KIT 🔄
  └─ Not yet implemented                  └─ Use KIT capacity + EIS for GNN

Charging Agent (PPO)                      NEW — needs PyBaMM 🆕
  └─ Not yet implemented                  └─ PyBaMM SPMe as environment

Safety Agent (Rule-based)                 KEEP AS IS ✅
  └─ Physics hard limits                  └─ No dataset needed

QMIX Coordinator                          NEW — needs all agents 🆕
  └─ Not yet implemented                  └─ Trains on joint agent trajectories
```

---

## 3. Step-by-Step Migration Instructions

### Step 1 — Download the KIT Dataset

The full dataset is 69.4 GB. You have two options:

**Option A — Download result data only (small, ~few GB)**
This gives you capacity fade per checkup, impedance per checkup — enough for the Health Agent.
```
URL: https://doi.org/10.35097/1969
Size: Much smaller than 69.4 GB (result data only)
```

**Option B — Download log data (full, 69.4 GB)**
This gives you raw V(t), I(t), T(t) at 2-second resolution — needed for Thermal Agent current profiles.
```
URL: https://doi.org/10.35097/1947
Size: 69.4 GB — requires ~100 GB free disk space
```

**Recommended for now:** Download Option A first. You can add Option B later for the thermal agent.

### Step 2 — Install KIT Dataset Python Scripts

```bash
# Clone the official data loading scripts from KIT
git clone https://github.com/energystatusdata/bat-age-data-scripts.git
cd bat-age-data-scripts
pip install -r requirements.txt

# The scripts provide:
# - load_cell_data()   → loads V(t), I(t), T(t) per cell
# - load_result_data() → loads capacity fade, impedance per checkup
# - plot_aging_curves()
```

### Step 3 — Install PyBaMM

```bash
pip install pybamm

# Verify installation
python -c "import pybamm; print(pybamm.__version__)"

# PyBaMM models you'll use:
# - pybamm.lithium_ion.SPM()    → Single Particle Model (fast)
# - pybamm.lithium_ion.SPMe()   → SPM with Electrolyte (recommended)
# - pybamm.lithium_ion.DFN()    → Doyle-Fuller-Newman (most accurate, slow)
```

### Step 4 — Update Data Loading in Your Notebooks

**Currently (Severson):**
```python
import pickle
batch1 = pickle.load(open('batch1.pkl', 'rb'))
# Access: batch1['b1c0']['summary']['QD']  → capacity per cycle
# Access: batch1['b1c0']['cycles']['0']['T']  → temperature
```

**New (KIT NMC):**
```python
# Using KIT official scripts
import sys
sys.path.append('path/to/bat-age-data-scripts')
from data_loader import load_cell_data, load_result_data

# Load capacity fade data (result data)
result_data = load_result_data('path/to/kit_result_data/')
# Returns: {cell_id: {'capacity': [...], 'impedance': [...], 'temperature_condition': X, 'crate': Y}}

# Load raw cycling data (log data — if downloaded)
cell_data = load_cell_data('path/to/kit_log_data/', cell_id='P001_1')
# Returns: DataFrame with columns: time, V, I, T, Q_charge, Q_discharge, cycle
```

### Step 5 — Understand KIT Cell Naming Convention

KIT cells are named: `P{group}_{replicate}`
- Groups encode: temperature × C-rate × aging type
- Example: P001_1, P001_2, P001_3 → 3 replicates of same condition
- The metadata CSV maps cell IDs to (temperature, C-rate, aging type)

```python
# Read the metadata to understand cell conditions
import pandas as pd
metadata = pd.read_csv('path/to/cell_metadata.csv')
# Columns: cell_id, temperature_degC, charge_crate, discharge_crate, aging_type
# aging_type: 'calendar', 'cyclic', 'driving_cycle'
```

### Step 6 — Update the Surrogate (Minimal Changes)

Your existing surrogate is trained on Severson (LFP chemistry). For the KIT dataset (NMC), you have two options:

**Option A (Easiest): Keep Severson surrogate, add a domain adaptation layer**
```python
# Add a simple linear calibration layer
# KIT capacity is in different range (NMC ~3-4 Ah vs Severson LFP ~1.1 Ah)
# Just re-scale the output: Q_kit = Q_severson * (Q_kit_nominal / Q_sev_nominal)
Q_SEVERSON_NOMINAL = 1.1  # Ah
Q_KIT_NOMINAL = 3.5       # Ah (check datasheet for your specific cell)
```

**Option B (Better for paper): Retrain surrogate on KIT data**
```python
# Same 1D-CNN architecture, same training code
# Just replace: batch1.pkl loading → KIT CSV loading
# Input signals: V(t), I(t), T(t) per cycle (same structure)
# Target: capacity at next checkup (same concept)
# Expected R² on NMC: 0.90-0.95 (NMC is more predictable than LFP)
```

### Step 7 — Update the Thermal Agent Environment

The LCTM simulator does NOT change — it's a physics model independent of dataset.
What changes is where you get current profiles from:

**Currently:**
```python
# Draw current profile from Severson training cells
def sample_current_from_training_cell():
    cell_key = random.choice(train_keys)
    I_profile = batch1[cell_key]['cycles']['5']['I']
    return I_profile
```

**Updated (KIT):**
```python
# Draw current profile from KIT cycling data
def sample_current_from_kit_cell(kit_data, metadata):
    # Filter to cyclic aging cells (not calendar)
    cyclic_cells = metadata[metadata['aging_type'] == 'cyclic']['cell_id'].tolist()
    cell_id = random.choice(cyclic_cells)
    cell_df = kit_data[cell_id]
    # Get cycle 10 (after formation)
    cycle_data = cell_df[cell_df['cycle number'] == 10]
    return cycle_data['<I>/mA'].values / 1000  # mA → A
```

### Step 8 — Build Health Agent on KIT Data (New Component)

This is the biggest addition. KIT has what you need:

```python
# Health Agent training data from KIT
# Inputs (per checkup/every ~50 cycles):
#   - Capacity fade trajectory: Q(cycle_0), Q(cycle_50), Q(cycle_100)...
#   - EIS impedance spectrum (frequency → real/imaginary parts)
#   - Temperature at which checkup was done
# 
# Output (GNN target):
#   - Remaining Useful Life (RUL) or SoH at next checkup

# Build graph: cells as nodes, similar aging conditions as edges
# Node features: [capacity_fade_rate, impedance_real, impedance_imag, T_condition]
# Edge features: [similarity in C-rate, similarity in temperature]
```

---

## 4. What Your Notebooks Will Look Like After Migration

```
Project Structure (After Migration)
├── batch1.pkl, batch2.pkl, batch3.pkl          ← KEEP (Severson, surrogate only)
├── kit_data/                                   ← NEW (KIT 228-cell)
│   ├── result_data/                            ← Capacity + EIS per checkup
│   └── log_data/                               ← Raw V,I,T at 2s resolution (optional)
├── bat-age-data-scripts/                       ← KIT official scripts
│
├── degradation_surrogate_1dcnn.ipynb           ← KEEP (retrain optional on KIT)
├── Model/degradation_surrogate_1dcnn.pth       ← KEEP (use as-is OR retrain)
│
├── thermal_agent_v4.ipynb                      ← MODIFY (update current profile source)
├── thermal_agent_v5.ipynb                      ← NEW (KIT-sourced current profiles)
│
├── health_agent_gnn.ipynb                      ← NEW (build on KIT EIS + capacity)
├── charging_agent_ppo.ipynb                    ← NEW (train with PyBaMM)
├── safety_agent.py                             ← NEW (rule-based, no dataset needed)
├── qmix_coordinator.ipynb                      ← NEW (train after all agents work)
│
└── PAPER_IMPLEMENTATION_PLAN.md               ← UPDATE (already done)
```

---

## 5. Effort Estimate Per Change

| Component | Effort | What Changes |
|:---|:---:|:---|
| Download KIT data | 1 hour (+ download time) | Download + run kit scripts |
| Install PyBaMM | 30 min | pip install + verify |
| Update Thermal Agent current profiles | 2–3 hours | Change `sample_current_from_*` function |
| Retrain surrogate on KIT (optional) | 4–6 hours | Replace data loader, same architecture |
| Build Health Agent (GNN) on KIT | 2–3 days | New notebook from scratch |
| Build Charging Agent (PPO + PyBaMM) | 3–5 days | New notebook from scratch |
| Build Safety Agent (rule-based) | 1 day | Simple threshold logic |
| Build QMIX Coordinator | 1–2 weeks | After all agents work individually |

**Total to get full multi-agent system working: ~3–4 weeks**

---

## 6. What NOT to Change (Save Yourself Time)

| Component | Why Keep |
|:---|:---|
| `Model/degradation_surrogate_1dcnn.pth` | R²=0.888 is good enough to start with |
| `Model/thermal_agent_sac_v3.pth` | Thermal agent already works — just update current profiles |
| LCTM simulator code | Physics model, dataset-independent |
| Reward function structure | Same formula works for KIT data |
| `thermal_agent_v1–v4` notebooks | Historical baseline results, don't delete |

---

## 7. Paper Claim After Migration

Once done, your paper can say:

> *"We train our multi-agent BMS framework on the KIT Comprehensive Battery Aging Dataset
> (228 NMC/C+SiO cells, Waldmann et al., Nature Scientific Data 2024), which provides
> multi-temperature and multi-C-rate cycling data essential for thermal and charging agent
> training. The degradation surrogate is additionally validated on the Severson 2019 LFP
> dataset (Nature Energy, 124 cells) to demonstrate cross-chemistry generalization.
> The charging agent is trained using the PyBaMM SPMe electrochemical simulator."*

This is a **stronger paper** because:
1. NMC > LFP for real-world relevance (EVs use NMC)
2. 228 cells > 124 cells
3. Multi-temperature data answers the reviewer question about thermal conditions
4. Cross-chemistry validation (NMC + LFP) = bonus novelty claim

---

## 8. Quick Start Commands

```bash
# 1. Download KIT result data (smaller, start here)
# Go to: https://doi.org/10.35097/1969
# Click Download → save to: kit_data/result_data/

# 2. Get KIT Python scripts
git clone https://github.com/energystatusdata/bat-age-data-scripts.git
pip install -r bat-age-data-scripts/requirements.txt

# 3. Install PyBaMM
conda activate battery
pip install pybamm

# 4. Test KIT data loading
python -c "
import sys
sys.path.append('./bat-age-data-scripts')
# Follow README in bat-age-data-scripts for exact function calls
print('KIT scripts loaded successfully')
"

# 5. Test PyBaMM
python -c "
import pybamm
model = pybamm.lithium_ion.SPMe()
print(f'PyBaMM {pybamm.__version__} — SPMe model ready')
"
```

---

*Last updated: September 2026*
*Repository: https://github.com/Mahizhan-S/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation*
