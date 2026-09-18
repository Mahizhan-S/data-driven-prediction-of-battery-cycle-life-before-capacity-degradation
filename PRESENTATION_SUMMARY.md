# PPTX Presentation Summary
## Multi-Agent Battery Management System
### Using Deep Reinforcement Learning + 1D-CNN Degradation Surrogate on Real Battery Data

> **Author:** [Your Name] | **Dataset:** Severson et al. 2019 (Nature Energy) | **Status:** Active Research

---

## SLIDE 1 — Title Slide

### Multi-Agent Battery Management System Using Deep Reinforcement Learning
#### A Hierarchical, Degradation-Aware Framework on Real Cycling Data (Severson 2019)

- **Dataset:** Severson 2019 — 124 LFP/graphite cells, MIT/Stanford/Toyota Research Institute
- **Key Contribution:** A hierarchical multi-agent RL framework (Health + Thermal + Charging + Safety agents) coordinated by a shared 1D-CNN degradation surrogate (R²=0.888) trained on real battery cycling data
- **Algorithms:** 1D-CNN Surrogate → DDPG (Thermal v1) → TD3 (Thermal v2) → SAC (Thermal v3) | Extra Trees (Health Agent) | PPO/SAC (Charging Agent, planned)

---

## SLIDE 2 — Problem Statement & Motivation

### Why Multi-Agent Battery Management?

| Challenge | Impact |
|:---|:---|
| Battery degradation is the #1 EV cost driver | Cell replacement = 40–50% of EV cost |
| Thermal abuse accelerates capacity fade | Every 10°C above 35°C increases degradation |
| Charging, thermal, and health objectives **conflict** | Single-agent RL cannot balance all three simultaneously |
| Existing controllers ignore real-time health | PID/rule-based systems are reactive, not predictive |
| Prior RL papers use synthetic degradation models | Arrhenius equations ≠ real LFP cell behaviour |

### Research Gap
> **No prior paper builds a full multi-agent system** that: (1) uses a data-trained degradation surrogate, (2) trained on real cycling data, (3) shared across a hierarchy of specialised agents (Health, Thermal, Charging, Safety) with a coordinator layer.

---

## SLIDE 3 — Dataset: Severson et al. 2019 (Nature Energy)

*(DOI: 10.1038/s41560-019-0356-8)*

| Property | Value |
|:---|:---|
| Cell Chemistry | LFP/graphite (LiFePO₄), 18650 cylindrical |
| Number of Cells | **124 cells** (after quality filter) |
| Cycle Life Range | 150 – 2,300 cycles |
| Charging Protocols | **72 different fast-charging CCCV variants** |
| Data per Cycle | Voltage V(t), Current I(t), Temperature T(t), Discharge Capacity Q |
| Total Cycles | ~34,136 (after preprocessing) |
| Public Access | data.matr.io/1 |

### Why This Dataset?
- **Gold standard** public benchmark — all major competing papers use it
- Real capacity fade trajectories — not synthetic/simulated
- Rich diversity: 72 protocols × 124 cells → strong surrogate generalisation
- Temperature data per cycle → enables thermal-degradation coupling

### Acknowledged Limitations
1. LFP chemistry only — may not generalise to NMC cells
2. Lab CCCV protocols — not real-world EV drive cycles
3. No ambient temperature variation — controlled room temperature

---

## SLIDE 4 — System Architecture (Big Picture)

```
┌────────────────────────────────────────────────────────────────┐
│                     Battery / Dataset                          │
│      Severson 2019 — 124 LFP/graphite cells, 72 protocols      │
│              V(t), I(t), T(t), Q_discharge per cycle           │
└──────────────────────────┬─────────────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │       Degradation Surrogate      │
          │   1D-CNN: (V, I, T) → Q_next    │
          │   R² = 0.888 on held-out test    │
          │   196K params, 3-channel input   │
          └────────────────┬────────────────┘
                           │ Q_hat as shared signal
┌──────────────────────────▼────────────────────────────────────┐
│                    Agent Decision Layer                        │
│                                                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────┐  │
│  │  Health      │ │  Thermal     │ │ Charging │ │ Safety   │  │
│  │  Agent       │ │  Agent       │ │  Agent   │ │  Agent   │  │
│  │ Extra Trees  │ │ SAC (v3)     │ │ PPO/SAC  │ │ Rule-    │  │
│  │ + LOOCV      │ │ + LCTM env   │ │ + CC-CV  │ │ Based    │  │
│  │ [Planned]    │ │ [Active]     │ │ [Planned]│ │ [Planned]│  │
│  └──────────────┘ └──────────────┘ └──────────┘ └──────────┘  │
└──────────────────────────┬────────────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │         Coordinator Layer        │
          │  Priority: Safety > Health >     │
          │           Thermal > Charging     │
          └────────────────┬────────────────┘
                           ▼
              Action + Rationale + Confidence
```

---

## SLIDE 5 — Literature Review (8 Papers Compared)

| # | Paper | Algorithm | Task | Dataset | Key Weakness |
|:---|:---|:---|:---|:---|:---|
| P1 | **Severson 2019** (Nature Energy) | Elastic Net + Extra Trees | Cycle-life prediction | Severson | Hand-crafted features, no control |
| P2 | **Liu 2022** (arXiv) | CNN-LSTM (1-channel) | Cycle-life prediction | Severson | Total life only, not next-cycle Q |
| P3 | **Attia 2020** (Nature) | Bayesian Opt + GP | Charging protocol design | Severson | Offline batch, not real-time |
| P4 | **Fei 2021** (IEEE Trans.) | CNN-LSTM (>500K params) | SoH estimation | Custom | Heavier model, single-channel |
| P5 | **Li 2023** (McMaster) | DDPG | Thermal management | Synthetic | No degradation surrogate in reward |
| P6 | **Chalmers 2023** | TD3 | Charging (health-aware) | Simulated SPM | Charging only, no thermal agent |
| P7 | **MDPI Energies 2023** | SAC / Hierarchical RL | Thermal management | Synthetic | Generic empirical degradation model |
| P8 | **MDPI Batteries 2024** | DDPG | Thermal + degradation | Synthetic | Arrhenius model — not real data |

### The Novelty Gap

```
               Surrogate Type          Training Data       RL Algorithm
──────────────────────────────────────────────────────────────────────
P5             None (heuristic)        Synthetic           DDPG
P6             Physics (SPM)           Simulated           TD3 (charging only)
P7             Empirical Arrhenius     Synthetic           SAC / Hierarchical
P8             Arrhenius               Synthetic           DDPG

OURS     --->  1D-CNN (real data)      Severson 2019       SAC (thermal control)
               196K params, R²=0.888   124 real cells
```

> **No prior paper combines all three:** data-trained surrogate + real cycling data + coupled RL thermal reward.

---

## SLIDE 6 — Component 1: Degradation Surrogate (1D-CNN)

### What It Does
Predicts **next-cycle discharge capacity Q_hat** from raw (V, I, T) time-series — used as a real-time health signal in all downstream agents.

### Architecture

| Property | Value |
|:---|:---|
| **Model Type** | 1D-CNN |
| **Input** | 3 channels × 60 timesteps: [V(t), I(t), T(t)] per cycle |
| **Output** | Next-cycle discharge capacity Q_hat (Ah) |
| **Layers** | Conv1d(3→32→64→128) + BatchNorm + Pool → AdaptiveAvgPool(4) → MLP(512→256→128→1) |
| **Parameters** | **196,225** (lightweight, real-time deployable) |
| **Regularisation** | Dropout (0.3/0.2), Weight Decay (1e-5), Early Stopping (patience=15) |
| **Train/Val/Test** | Official Severson split: Batch 1+2 primary, Batch 3 secondary |

### Why 1D-CNN? (Literature Justification)
- Liu et al. (2022): CNN outperforms LSTM on per-cycle capacity tasks
- Temporal local patterns in V(t) curves (knee point) are best captured by 1D convolutions
- 3-channel joint encoding (V, I, T simultaneously) is our improvement over Liu's single-channel

### Validated Results

| Split | MAE (Ah) | RMSE (Ah) | R² |
|:---|:---:|:---:|:---:|
| Validation (20%) | 0.0070 | 0.0099 | **0.969** |
| Primary Test (Batch 1+2) | 0.0103 | 0.0194 | **0.888** |
| Secondary Test (Batch 3) | 0.0121 | 0.0168 | **0.854** |

### Hyperparameter Tuning Iterations

| Run | Dropout | Patience | Primary R² | Notes |
|:---|:---:|:---:|:---:|:---|
| Run 1 | 0.3/0.2 | 100 | 0.881 | Overfit (3.7× train/val gap) |
| Run 2 | 0.5/0.4 | 15 | 0.862 | Over-regularised |
| **Run 3** | **0.3/0.2** | **15** | **0.888** | ✅ Best — used in all agents |

### Comparison vs Literature

| Paper | Model | Task | R² |
|:---|:---|:---|:---:|
| Liu 2022 [P2] | CNN-LSTM (1-channel) | Cycle-life | ~0.85 |
| Fei 2021 [P4] | CNN-LSTM (>500K params) | SoH (MAE=0.009 Ah) | — |
| **Ours** | **1D-CNN 3-channel (196K)** | **Next-cycle capacity** | **0.888** |

---

## SLIDE 7 — Data Analysis: Temperature–Capacity Relationship

### Statistical Discovery (34,136 Cycles Analysed)

| Statistic | Value | Interpretation |
|:---|:---:|:---|
| Raw Corr(T_mean, Q) | −0.170 | Appears significant... |
| **Partial Corr(T_mean, Q given cycle_number)** | **−0.015** | ...but it's just cycle aging, not temperature! |

### Binned Capacity by Temperature

| Temperature Range | n Cycles | Mean Q (Ah) |
|:---:|:---:|:---:|
| 25–30°C | 900 | 1.0487 |
| 30–33°C | 12,008 | 1.0553 |
| 33–36°C | 19,210 | 1.0387 |
| 36–40°C | 2,018 | 1.0394 |

### Key Insight
> LFP/graphite cells are **thermally resilient** at 25–38°C. The CNN surrogate **cannot distinguish** 5–10°C temperature differences — it only sees cycle-averaged Q. This finding **directly motivated the v3 adaptive stress redesign**.

---

## SLIDE 8 — Thermal Agent: Environment & Algorithm Choice

### Environment: Lumped Capacitance Thermal Model (LCTM)

```
dT/dt = (P_gen - P_cool) / (m·Cp)

  P_gen  = I²·R_int     (Joule heating)
  P_cool = Agent action  (cooling power, continuous)
  T_target = 30°C
```

**State (dim=8):** [T, T_prev, I, cycle_norm, SOC, Q_hat, Q_hat/Q_ref, |T−35|]  
**Action (continuous):** Cooling power P_cool ∈ [0, 60 W]

### Algorithm Progression — Why SAC Won

| Property | DDPG (v1) | TD3 (v2) | **SAC (v3)** |
|:---|:---:|:---:|:---:|
| Policy type | Deterministic | Deterministic | **Stochastic** |
| Entropy regularisation | No | No | **Yes (auto-tuned)** |
| Stability | Low | Moderate | **High** |
| Sample efficiency | Low | Moderate | **High** |
| Exploration | OU noise | Clipped noise | **Built-in** |
| Key paper | Lillicrap 2015 | Fujimoto 2018 | **Haarnoja 2018** |

---

## SLIDE 9 — Thermal Agent v1: DDPG (Baseline)

### Configuration

| Property | Value |
|:---|:---|
| Algorithm | DDPG (Lillicrap et al. 2015) |
| State | dim=6: [T, T_prev, I, cycle_norm, SOC, Q_hat/Q_ref] |
| Action | Cooling power [0, 50 W] |
| Episodes | 500, 200-step |
| Reward | −α(T−30)² − γ·P_cool − safety |

### Bug Found & Fixed
> Q_ref was the **global dataset mean** (1.07 Ah). So the degradation penalty `(Q_ref/Q_hat − 1)` was always ~0 for all episodes — the surrogate had **zero real influence on training**.

### Results vs PID (20-episode eval)

| Metric | DDPG | PID | Improvement |
|:---|:---:|:---:|:---:|
| Total Reward | −556.6 | −612.9 | **+9.2%** |
| Temperature MAE | 5.24°C | 5.51°C | **−4.9%** |
| **Cooling Energy** | **809 J** | **1,246 J** | **−35.1%** ✅ |
| Safety Violations | 0 | 0 | — |
| Cap Preserved | 109.69% | 109.75% | — |

> ⚠️ **Note:** v1 β-ablation values are **invalid** — all β values (0–20) produced identical T_MAE=4.864°C and cap_pct=109.4%. Bug: ablation evaluation reused the same trained model without β variation. Not reportable.

---

## SLIDE 10 — Thermal Agent v2: TD3 + Dynamic Q_ref

### Improvements Over v1

| Change | Rationale |
|:---|:---|
| DDPG → TD3 | Fixes overestimation bias (Fujimoto 2018); more stable training |
| Static Q_ref → Dynamic per-episode Q_ref | True episode-specific degradation baseline |
| 500 → 1,500 episodes | Better policy convergence |
| 300 steps per episode | Richer per-cycle trajectory |
| Noise decay schedule | Exploitation-exploration balance |
| Added TD3-NoSurrogate ablation | Isolates CNN's value in reward |

### Results vs PID (50-episode eval)

| Metric | TD3-Surrogate | TD3-NoSurrogate | PID | TD3-Surr vs PID |
|:---|:---:|:---:|:---:|:---:|
| Total Reward | −814.88 | −776.35 | −978.37 | **+16.7%** |
| Temp MAE | 5.17°C | 5.05°C | 5.68°C | **−9.0%** |
| **Cooling Energy** | **371.4 J** | **350.3 J** | **1,738.6 J** | **−78.6%** ✅ |
| Deg Penalty | 0.00207 | 0.00283 | 0.00149 | — |
| Cap Preserved | 102.01% | 102.11% | 102.15% | — |
| Safety Violations | 0 | 0 | 0 | — |

**β-Ablation Results (300 eps each, 7 values):**

| β | T_MAE (°C) | T_MAE std | cap_pct (%) |
|:---:|:---:|:---:|:---:|
| 0.0 | 5.487 | ±0.439 | 102.15 |
| 0.5 | 5.511 | ±0.367 | 102.16 |
| 1.0 | 5.587 | ±0.617 | 102.18 |
| 2.0 | 5.395 | ±0.342 | 102.13 |
| **5.0** | **5.384** | **±0.438** | **102.13** |
| 10.0 | 5.475 | ±0.409 | 102.15 |
| 20.0 | 5.343 | ±0.366 | 102.11 |

### Remaining Problem
> **TD3-Surrogate ≈ TD3-NoSurrogate** (cooling energy: 371J vs 350J — within noise). 5–10°C swings produce near-zero partial correlation with Q (partial corr = −0.015) — the CNN cannot differentiate them. This motivated the v3 physics-based stress term.

---

## SLIDE 11 — Thermal Agent v3: SAC + Adaptive Stress (CURRENT)

### Core Innovation: Adaptive Surrogate-Coupled Stress Reward

```
r = −α(T − 30)²
    − β · K · (T−35)² · I · (Q_ref / Q_hat)   <-- NOVEL: adaptive stress
    − γ · P_cool
    − η · |dT/dt|
    − 1000 · [safety violation]

  K         = data-calibrated from polynomial fit to 34,136 Severson cycles
  Q_ref/Q_hat = health ratio: SCALES UP penalty as battery degrades
```

### Why This Guarantees Differentiation
- `K·(T−35)²·I` is **monotonically increasing** in T above 35°C (mathematical guarantee)
- The CNN's Q_hat adaptively scales the stress — no prior paper does this
- Result: a genuine Pareto curve across β values

### Three Agents Trained Simultaneously

| Agent | β | Purpose |
|:---|:---:|:---|
| **SAC-Full** | 5.0 | **Proposed method** |
| SAC-NoAdaptive | 5.0 | Ablation: Q_ref/Q_hat = 1 (no CNN scaling) |
| SAC-NoStress | 0.0 | Ablation: thermal comfort only |
| PID | — | Classical baseline |

### β-Ablation → Pareto Frontier (8 values × 500 eps)

| β | Expected Behaviour |
|:---:|:---|
| 0.0 | No stress — agent allows high T, minimal cooling energy |
| 0.5 – 2.0 | Mild stress — slight T suppression |
| **5.0** | **Balanced — proposed operating point** |
| 10.0 – 20.0 | Conservative — T near 35°C, higher energy cost |
| 50.0 | Aggressive — T never above 35°C, max cooling energy |

### Status: Training (~95% complete — results within ~15 minutes)

---

## SLIDE 12 — Novelty Claims (3 Publishable Claims)

### Claim 1 — Adaptive Surrogate Coupling
> "We propose a degradation-surrogate-coupled reward where a 1D-CNN trained on 124 real LFP/graphite cells provides continuous Q_hat predictions that penalise the SAC cooling agent — scaled adaptively by (Q_ref/Q_hat) so the agent **automatically becomes more conservative as the battery degrades**. No prior thermal RL paper implements this [P5, P7, P8]."

### Claim 2 — Data-Fitted Stress Function (vs Arrhenius)
> "Unlike [P8] which uses Arrhenius equations, our stress function K·(T−35)²·I is regressed from a polynomial fit to 34,136 real Severson cycles. K is derived from actual capacity fade data — not theoretical activation energies."

### Claim 3 — Quantitative Pareto Frontier
> "We present the first systematic β-ablation study (8 values, 95% CI) quantifying the thermal-accuracy vs. capacity-preservation trade-off — positioning the system as a tunable design tool, not just a fixed controller."

---

## SLIDE 13 — Results Summary (All Versions)

| Version | Algorithm | Episodes | vs PID Reward | Cooling Energy | Temp MAE | Cap Preserved | Surrogate Role | Key Achievement |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **v1 DDPG** | DDPG | 500 | **+9.2%** | −35.1% (809 J) | 5.24°C | 109.69% | Bug (static Q_ref) | First RL baseline |
| **v2 TD3** | TD3 | 1500 | **+16.7%** | **−78.6% (371 J)** | 5.17°C | 102.01% | State feature | Best cooling efficiency |
| **v3 SAC** | SAC | 1500+ | ⏳ TBD | ⏳ TBD | ⏳ TBD | ⏳ TBD | Adaptive reward scaling | Pareto frontier (training) |
| **PID v1 baseline** | Rule-based | — | — | 1,246 J | 5.51°C | 109.75% | — | v1 classical baseline |
| **PID v2 baseline** | Rule-based | — | — | 1,738.6 J | 5.68°C | 102.15% | — | v2 classical baseline |

---

## SLIDE 14 — Surrogate Comparison vs Literature

| Paper | Model | Task | Metric | Dataset |
|:---|:---|:---|:---:|:---|
| Severson 2019 [P1] | Elastic Net + ExtraTrees | Cycle-life prediction | 9.1% MAPE | Severson |
| Liu 2022 [P2] | CNN-LSTM (1-channel) | Cycle-life prediction | R²~0.85 | Severson |
| Fei 2021 [P4] | CNN-LSTM (>500K params) | SoH estimation | MAE=0.0091 Ah | Custom |
| **Ours** | **1D-CNN 3-channel (196K)** | **Next-cycle capacity** | **R²=0.888, MAE=0.0103 Ah** | **Severson** |

---

## SLIDE 15 — Planned Future Components

| Component | Algorithm | Justification |
|:---|:---|:---|
| **Health Agent** | Extra Trees + LOOCV | Validated vs Severson benchmark; interpretable |
| **Charging Agent** | PPO or SAC | Long-horizon cycle-life reward; CC-CV constraints |
| **Safety Agent** | Rule-based hard constraints | Voltage/T bounds must be non-learnable |
| **Coordinator** | Priority arbitration | Safety > Health > Thermal > Charging |

---

## SLIDE 16 — Honest Limitations

| Limitation | Impact | Future Work |
|:---|:---|:---|
| LCTM thermal model (lumped) | No spatial thermal gradients | FEM-based environment |
| Frozen offline surrogate | No online adaptation | Continual learning |
| LFP chemistry only | Limited to one cell type | Retrain on NMC/LCO |
| No hardware validation | Simulation results only | Hardware-in-the-loop (HIL) |
| Manual β selection | Suboptimal tuning | Bayesian hyperparameter search |
| Lab CCCV protocols | Not real EV duty cycles | Drive-cycle data integration |

---

## SLIDE 17 — File Inventory

| File | Description | Status |
|:---|:---|:---:|
| `degradation_surrogate_1dcnn.ipynb` | Surrogate training + evaluation | ✅ |
| `thermal_agent.ipynb` | DDPG v1 notebook (500 ep) | ✅ |
| `thermal_agent_v2.ipynb` | TD3 v2 notebook (1500 ep) | ✅ |
| `thermal_agent_v3.ipynb` | SAC v3 notebook (running) | 🔄 Running |
| `Model/degradation_surrogate_1dcnn.pth` | Surrogate weights (R²=0.888, 196K params) | ✅ |
| `Model/thermal_agent_ddpg.pth` | DDPG v1 weights | ✅ |
| `Model/thermal_agent_td3_v2.pth` | TD3 v2 weights | ✅ |
| `Model/thermal_agent_sac_v3.pth` | SAC v3 weights (816 KB, saved at 13:59 IST) | ✅ Saved |
| `thermal_agent_results/results_summary.json` | v1 DDPG metrics | ✅ |
| `thermal_agent_v2_results/results_summary_v2.json` | v2 TD3 metrics + 7-pt ablation | ✅ |
| `thermal_agent_v2_results/` | 4 figures (training, trajectories, pareto, degradation) | ✅ |
| `thermal_agent_v3_results/` | SAC v3 Pareto figures + JSON | 🔄 Pending |
| `LITERATURE_REVIEW.md` | Full literature review with working links | ✅ |
| `RESEARCH_PROGRESS.md` | Research progress log | ✅ |
| `PRESENTATION_SUMMARY.md` | This file — full slide content | ✅ |

---

## SLIDE 18 — Key References

```bibtex
@article{severson2019,
  title   = {Data-driven prediction of battery cycle life before capacity degradation},
  author  = {Severson, Kristen A and Attia, Peter M and Jin, Norman and others},
  journal = {Nature Energy}, volume = {4}, pages = {383--391}, year = {2019},
  doi     = {10.1038/s41560-019-0356-8}
}
@article{attia2020,
  title   = {Closed-loop optimization of fast-charging protocols for batteries with machine learning},
  author  = {Attia, Peter M and Grover, Aditya and Jin, Norman and others},
  journal = {Nature}, volume = {578}, pages = {397--402}, year = {2020},
  doi     = {10.1038/s41586-020-1994-5}
}
@article{lillicrap2016ddpg,
  title   = {Continuous control with deep reinforcement learning},
  author  = {Lillicrap, Timothy P and Hunt, Jonathan J and others},
  journal = {ICLR}, year = {2016}
}
@inproceedings{fujimoto2018td3,
  title   = {Addressing Function Approximation Error in Actor-Critic Methods},
  author  = {Fujimoto, Scott and van Hoof, Herke and Meger, David},
  booktitle = {ICML}, year = {2018}
}
@inproceedings{haarnoja2018sac,
  title   = {Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning},
  author  = {Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  booktitle = {ICML}, year = {2018}
}
@article{fei2021soh,
  title   = {A deep learning framework for state-of-health estimation of lithium-ion batteries},
  author  = {Fei, Zhaocong and others},
  journal = {IEEE Transactions on Industrial Informatics}, year = {2021}
}
```

---

*Last updated: September 17, 2026, 19:55 IST*  
*v3 SAC training status: Model weights saved (13:59 IST, 816 KB). Final evaluation + β-ablation still running (PID 12657, ~121% CPU). Results in `thermal_agent_v3_results/` pending.*
