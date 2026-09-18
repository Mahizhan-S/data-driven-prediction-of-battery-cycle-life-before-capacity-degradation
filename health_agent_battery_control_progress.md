# Health-Agent-Informed Multi-Agent Battery Control System

## Research Progress Summary — 17 September 2026

**Dataset:** Severson et al. 2019 — MIT/Stanford/Toyota Research Institute

**Objective:** Build a hierarchical multi-agent reinforcement-learning framework for intelligent, degradation-aware battery fast-charging control, going beyond single-objective thermal or charging optimisation.

---

## 1. What Has Been Done So Far

The work has progressed from battery degradation prediction to reinforcement-learning-based thermal control and, most recently, to an adaptive degradation-aware reward design.

The current development path is:

```text
Severson battery data
        ↓
1D-CNN degradation surrogate
        ↓
Thermal RL controller
        ↓
DDPG v1
        ↓
TD3 v2 + dynamic Q_ref
        ↓
TD3 surrogate/no-surrogate ablation
        ↓
SAC v3 + adaptive Q_ref / Q_hat coupling
        ↓
Future integration of Health + Charging + Safety + Coordinator
```

### Current component status

| Component | Status | Current State |
|---|---|---|
| Degradation Surrogate | ✅ Complete | 1D-CNN trained and validated; primary test R² = 0.888 |
| Thermal Agent v1 | ✅ Complete | DDPG; initial RL thermal controller |
| Thermal Agent v2 | ✅ Complete | TD3; dynamic Q_ref and ablation study |
| Thermal Agent v3 | 🔄 Training | SAC with adaptive surrogate-coupled stress reward |
| Health Agent | 🟡 Partially developed | Extra Trees + LOOCV + ΔQ specified in research plan; later XGBoost and Elastic Net experiments also tested |
| Charging Agent | ⏳ Planned | PPO or SAC for continuous C-rate control |
| Safety Agent | ⏳ Planned | Rule-based hard safety constraints |
| Coordinator | ⏳ Planned | Priority arbitration across agents |

---

# 2. Degradation Surrogate

The degradation surrogate is the shared backbone of the planned multi-agent system. Instead of repeatedly relying on a slower simulator, a neural surrogate predicts next-cycle discharge capacity from measured battery signals.

### Model

- **Architecture:** 1D-CNN
- **Input:** Voltage, Current, Temperature
- **Input shape:** 3 channels × 60 time steps per cycle
- **Output:** Next-cycle discharge capacity Q_hat (Ah)
- **Architecture:** Conv1D 3→32→64→128 + BatchNorm + Pooling + AdaptiveAvgPool + MLP 512→256→128→1
- **Parameters:** 196,225
- **Regularisation:** Dropout, weight decay, early stopping

### Surrogate tuning runs

| Run | Dropout | Patience | Primary Test R² | Observation | Decision |
|---|---:|---:|---:|---|---|
| Run 1 | 0.3 / 0.2 | 100 | 0.881 | Overfit; 3.7× train/validation gap | Rejected |
| Run 2 | 0.5 / 0.4 | 15 | 0.862 | Over-regularised | Rejected |
| **Run 3** | **0.3 / 0.2** | **15** | **0.888** | Best generalisation | **Selected** |

### Final surrogate results

| Split | MAE (Ah) | RMSE (Ah) | R² |
|---|---:|---:|---:|
| Validation (20%) | 0.0070 | 0.0099 | **0.969** |
| Primary Test (Batch 1 + 2) | 0.0103 | 0.0194 | **0.888** |
| Secondary Test (Batch 3) | 0.0121 | 0.0168 | **0.854** |

**Selected model:** `Model/degradation_surrogate_1dcnn.pth`

---

# 3. Temperature–Capacity Data Analysis

Before redesigning the thermal reward, temperature/capacity behaviour was analysed over **34,136 Severson cycles**.

### Main findings

| Statistic | Value |
|---|---:|
| Raw correlation: T_mean vs Q | −0.170 |
| Partial correlation: T_mean vs Q controlling for cycle number | **−0.015** |

The dataset showed relatively weak direct temperature sensitivity in the observed 25–38°C region. This finding motivated a redesign of the thermal reward so that the learned surrogate would not be expected to distinguish very small temperature differences by itself.

### Capacity by temperature bin

| Temperature Range | Number of Cycles | Mean Q (Ah) |
|---|---:|---:|
| 25–30°C | 900 | 1.0487 |
| 30–33°C | 12,008 | 1.0553 |
| 33–36°C | 19,210 | 1.0387 |
| 36–40°C | 2,018 | 1.0394 |

---

# 4. Thermal Agent Development

The thermal controller is the most developed decision agent so far. The action is continuous cooling power in the range **0–50 W**.

## Thermal Agent Evolution

| Version | Algorithm | Main Change | Training | Surrogate / Health Coupling | Result / Status |
|---|---|---|---:|---|---|
| **v1** | **DDPG** | Initial continuous RL thermal control | 500 episodes | Intended degradation signal | 809 J cooling; 5.24°C MAE |
| **v2** | **TD3** | Dynamic Q_ref, noise decay, longer training | 1500 episodes | Surrogate included | 371 J cooling; 5.17°C MAE |
| **v2 Ablation** | **TD3** | Remove surrogate | 1500 episodes | No surrogate | 350 J cooling; 5.05°C MAE |
| **v3** | **SAC** | Adaptive health-dependent stress reward | 500/configuration | Explicit Q_ref / Q_hat coupling | Training; final metrics pending |
| **v3 Ablation** | **SAC-NoAdaptive** | Remove health scaling | 500/configuration | Stress retained, adaptive scaling removed | Training; metrics pending |
| **v3 Ablation** | **SAC-NoStress** | Remove degradation stress | 500/configuration | No degradation stress | Training; metrics pending |
| **PID** | **PID** | Classical thermal baseline | — | No learned surrogate | Baseline |

---

# 5. Thermal Agent v1 — DDPG

### Configuration

- Algorithm: **DDPG**
- State: `[T, T_prev, I, cycle_norm, SOC, Q_hat / Q_ref]`
- Action: Cooling power `[0, 50 W]`
- Training: 500 episodes

### Result vs PID

| Metric | DDPG v1 | PID | Difference |
|---|---:|---:|---:|
| Total Reward | **−556.6** | −612.9 | **+9.2%** |
| Temperature MAE | **5.24°C** | 5.51°C | **−5%** |
| Cooling Energy | **809 J** | 1246 J | **−35%** |
| Safety Violations | **0** | 0 | — |

### Important issue found

`Q_ref` was initially defined as a global mean. This caused the degradation penalty to become effectively zero, so the health/degradation component was not actually influencing the controller as intended.

**Saved model:** `Model/thermal_agent_ddpg.pth`

---

# 6. Thermal Agent v2 — TD3

### Main improvements over v1

- Replaced DDPG with **TD3**.
- Introduced **dynamic per-episode Q_ref**.
- Added noise decay.
- Increased training to **1500 episodes**.
- Used **300-step episodes**.
- Added an explicit **no-surrogate ablation**.

### Results

| Metric | TD3 + Surrogate | TD3 No-Surrogate | PID |
|---|---:|---:|---:|
| Total Reward | −814.9 | −776.3 | −978.4 |
| Temperature MAE | 5.17°C | 5.05°C | 5.68°C |
| Cooling Energy | **371 J** | 350 J | **1738 J** |
| Safety Violations | **0** | 0 | 0 |

### TD3 + Surrogate vs PID

| Metric | Difference vs PID |
|---|---:|
| Reward | **+16.7%** |
| Temperature MAE | **9.0% lower** |
| Cooling Energy | **78.6% lower** |
| Safety violations | **0** |

### Key research finding

The important issue was that **TD3 + surrogate performed very similarly to TD3 without the surrogate**. This showed that simply providing the degradation prediction to the controller was not creating strong health-aware behaviour.

**Saved model:** `Model/thermal_agent_td3_v2.pth`

---

# 7. Thermal Agent v3 — SAC with Adaptive Degradation Coupling

The current approach changes how the degradation surrogate affects the controller.

Instead of only providing Q_hat as a state feature, the reward explicitly couples thermal stress to predicted battery health:

```text
r = -α(T - 30)^2
    - β K (T - 35)^2 I (Q_ref / Q_hat)
    - γ P_cool
    - η |dT/dt|
    - safety
```

### Main idea

```text
Battery health gets worse
        ↓
Q_hat decreases
        ↓
Q_ref / Q_hat increases
        ↓
Thermal stress penalty increases
        ↓
SAC becomes more conservative
```

This is the main proposed **adaptive surrogate coupling** mechanism.

### v3 experiments

| Model | β | Adaptive Health Scaling | Degradation Stress | Purpose |
|---|---:|---|---|---|
| **SAC-Full** | 5.0 | ✅ Yes | ✅ Yes | Proposed method |
| **SAC-NoAdaptive** | 5.0 | ❌ No | ✅ Yes | Ablation: test adaptive CNN coupling |
| **SAC-NoStress** | 0.0 | ❌ No | ❌ No | Ablation: thermal-only control |
| **PID** | — | — | — | Classical baseline |

### β sweep

β values tested/planned:

`0, 0.5, 1, 2, 5, 10, 20, 50`

The purpose is to quantify the trade-off between thermal control, cooling energy, and degradation protection and construct a Pareto frontier.

**Current status:** SAC v3 was training when the progress report was written; final performance numbers are not yet available in that report.

**Saved artifact:** `Model/thermal_agent_sac_v3.pth` (running/in progress in the report)

---

# 8. Health Agent

The research plan specifies a dedicated Health Agent to estimate battery health/degradation.

### Planned approach in the progress report

| Item | Planned Health Agent |
|---|---|
| Model | Extra Trees |
| Validation | LOOCV |
| Features | ΔQ-based features |
| Dataset | Severson 2019 |
| Output | Battery health / degradation estimate |
| Status in report | Planned / pending implementation |

### Later Health-Agent experiments from the ongoing work

In subsequent work, Elastic Net and XGBoost were also tested:

| Health Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Elastic Net | 150.1979 | 281.3434 | 0.4827 |
| **XGBoost** | **85.8989** | **122.6259** | **0.9017** |

These later Health-Agent results are separate from the specific Extra Trees design listed in the uploaded progress report.

---

# 9. Charging Agent

The Charging Agent has been designed but is not yet reported as implemented in the current research report.

| Item | Planned Design |
|---|---|
| Purpose | Adapt the charging rate for fast charging while considering long-term battery life |
| Algorithms | PPO or SAC |
| Action | Continuous C-rate adaptation |
| Health information | Intended to use degradation information |
| Status | Planned |
| Results | Not yet available |

---

# 10. Safety Agent

The Safety Agent is intentionally rule-based rather than learned.

| Item | Design |
|---|---|
| Type | Rule-based hard constraints |
| Main constraints | Voltage and temperature bounds |
| Output | Safety veto / allow action |
| Learning | None |
| Role | Prevent unsafe actions even if another learned agent proposes them |
| Status | Planned |

---

# 11. Coordinator Layer

The coordinator combines agent outputs and resolves conflicts.

### Priority

```text
Safety > Health > Thermal > Charging
```

### Intended output

- Final charging/control action
- Rationale / explanation
- Confidence

The coordinator is still planned and has not yet been experimentally evaluated in the current report.

---

# 12. Complete Results Table

This is the main consolidated results table for the work completed or evaluated so far.

| Model / Agent | Type | Key Output | MAE | RMSE | R² | Cooling Energy | Temp MAE | Reward | Safety Violations | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1D-CNN Run 1 | Degradation prediction | Next-cycle Q_hat | — | — | **0.881** | — | — | — | — | Rejected |
| 1D-CNN Run 2 | Degradation prediction | Next-cycle Q_hat | — | — | **0.862** | — | — | — | — | Rejected |
| **1D-CNN Run 3** | Degradation prediction | Next-cycle Q_hat | **0.0103 Ah** | **0.0194 Ah** | **0.888** | — | — | — | — | ✅ Selected |
| PID | Thermal baseline | Cooling power | — | — | — | **1738 J** | **5.68°C** | **−978.4** | **0** | ✅ Baseline |
| DDPG v1 | Thermal RL | Cooling power | — | — | — | **809 J** | **5.24°C** | **−556.6** | **0** | ✅ Complete |
| TD3 No-Surrogate | Thermal RL ablation | Cooling power | — | — | — | **350 J** | **5.05°C** | **−776.3** | **0** | ✅ Complete |
| TD3 + Surrogate v2 | Thermal RL | Cooling power | — | — | — | **371 J** | **5.17°C** | **−814.9** | **0** | ✅ Complete |
| SAC-Full v3 | Thermal RL | Cooling power | Pending | Pending | Pending | Pending | Pending | Pending | Pending | 🔄 Training |
| SAC-NoAdaptive | Thermal RL ablation | Cooling power | Pending | Pending | Pending | Pending | Pending | Pending | Pending | 🔄 Training |
| SAC-NoStress | Thermal RL ablation | Cooling power | Pending | Pending | Pending | Pending | Pending | Pending | Pending | 🔄 Training |
| Elastic Net Health Agent | Health estimation | Health/degradation | **150.1979** | **281.3434** | **0.4827** | — | — | — | — | ✅ Tested later |
| **XGBoost Health Agent** | Health estimation | Health/degradation | **85.8989** | **122.6259** | **0.9017** | — | — | — | — | ✅ Tested later |

---

# 13. Agent-Wise Summary

| Agent / Component | Main Question | Model(s) Tried | Output | Current State |
|---|---|---|---|---|
| **Degradation Surrogate** | How will battery capacity change? | 1D-CNN | Q_hat | ✅ Completed |
| **Health Agent** | What is the current health/degradation level? | Extra Trees planned; Elastic Net and XGBoost later tested | Health estimate | 🟡 Partially developed |
| **Thermal Agent** | How much cooling should be applied? | DDPG → TD3 → SAC | Cooling power | 🔄 SAC v3 ongoing |
| **Charging Agent** | How aggressively should the battery charge? | PPO / SAC planned | C-rate | ⏳ Planned |
| **Safety Agent** | Is the proposed action safe? | Rule-based | Veto / allow | ⏳ Planned |
| **Coordinator** | How should conflicting agent recommendations be resolved? | Priority arbitration | Final control decision | ⏳ Planned |

---

# 14. Main Research Findings So Far

### Finding 1 — The degradation surrogate works

The selected 1D-CNN predicts next-cycle capacity with a **primary test R² of 0.888** and a **secondary test R² of 0.854**.

### Finding 2 — RL can reduce thermal-control energy relative to PID

The completed TD3 + surrogate experiment reported **371 J** cooling energy compared with **1738 J** for PID, while both reported zero safety violations.

### Finding 3 — Merely adding the surrogate to the RL state is not enough

TD3 with the surrogate and TD3 without the surrogate produced very similar results. This motivated the move to an explicitly reward-coupled design.

### Finding 4 — The research contribution is moving toward adaptive health-aware control

The SAC v3 formulation makes predicted health directly influence thermal stress through `Q_ref / Q_hat` instead of treating the surrogate only as an ordinary input feature.

### Finding 5 — The full multi-agent system is not finished yet

The Health, Charging, Safety, and Coordinator layers still need to be integrated and evaluated as a single hierarchical control system.

---

# 15. Current Research Position

At the current stage, the project can be accurately described as:

> **A degradation-aware thermal-control framework with a validated 1D-CNN degradation surrogate, multiple generations of continuous-control RL (DDPG and TD3), and an ongoing SAC formulation that explicitly couples predicted battery health to thermal stress. The remaining work is to complete and validate the Health, Charging, Safety, and Coordinator agents and integrate them into the final hierarchical multi-agent battery management system.**

---

# 16. Saved Files Mentioned in the Progress Report

| File | Purpose | Status |
|---|---|---|
| `degradation_surrogate_1dcnn.ipynb` | Surrogate training | ✅ |
| `thermal_agent.ipynb` | DDPG v1 | ✅ |
| `thermal_agent_v2.ipynb` | TD3 v2 | ✅ |
| `thermal_agent_v3.ipynb` | SAC v3 | 🔄 |
| `Model/degradation_surrogate_1dcnn.pth` | Selected surrogate | ✅ |
| `Model/thermal_agent_ddpg.pth` | DDPG v1 weights | ✅ |
| `Model/thermal_agent_td3_v2.pth` | TD3 v2 weights | ✅ |
| `Model/thermal_agent_sac_v3.pth` | SAC v3 weights | 🔄 |
| `thermal_agent_v2_results/` | v2 figures + JSON | ✅ |
| `thermal_agent_v3_results/` | v3 figures + JSON | 🔄 |
| `RELATED_WORKS_AND_NOVELTY.md` | Literature review | ✅ |

---

## Source Note

The main architecture, experiments, thermal-agent metrics, surrogate results, temperature analysis, novelty claims, and saved-artifact status above are based on the uploaded **Research Progress Report**. Later Health-Agent Elastic Net/XGBoost metrics are included separately because they were reported during the ongoing work after that report.
