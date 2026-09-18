# Data-Driven Degradation-Surrogate-Informed Reinforcement Learning for Battery Thermal Management

**[Author Name], [Co-Author Name]**  
*[Department], [Institution], [City, Country]*  
*[email@institution.edu]*

---

> **Manuscript Status — September 2026:**  
> v1 (DDPG) and v2 (TD3): validated and reportable.  
> v3 (SAC): completed; known ablation bugs fixed in v4.  
> **v4 (SAC, 5 seeds, 34-cell external holdout, corrected ablations): currently running.**  
> ⏳ = pending v4 completion.

---

## Abstract

Battery thermal management systems (BTMS) must balance temperature regulation, cooling-energy consumption, and battery degradation. We investigate whether a lightweight degradation surrogate trained on real battery cycling data can provide health information to a reinforcement-learning (RL) thermal controller. A 3-channel 1D convolutional neural network (1D-CNN) with 196,225 parameters is trained using raw voltage V(t), current I(t), and temperature T(t) time-series from the Severson 2019 LFP/graphite battery dataset. The surrogate predicts next-cycle discharge capacity and achieves an R² of 0.888 on the primary evaluation protocol and **R²=0.854 on a fully cell-held-out external test set comprising 34 cells**. The predicted capacity is incorporated into the thermal-control state and used to adaptively scale a temperature–current stress penalty. We progressively develop DDPG, TD3, and SAC controllers, treating the progression as a methodological development history rather than a controlled algorithm benchmark. Under the proposed control-oriented lumped-capacitance thermal model (LCTM), the validated TD3 controller reduced simulated cooling energy by **78.6%** relative to the quadratic proportional heuristic baseline (371 J vs. 1,739 J) while achieving a temperature MAE of 5.17°C vs. 5.68°C and zero safety violations over 50 evaluation episodes. A seven-point reward-reweighting analysis examines the numerical effect of the degradation penalty weight β on the evaluated objective. These results demonstrate the feasibility of integrating a real-data-trained degradation surrogate into health-aware thermal control, while the absence of hardware validation and the use of a simplified thermal simulator limit claims of physical energy savings. Final multi-seed SAC experiments with a tuned PID baseline, valid ablations, and external cell-level evaluation are under way to assess robustness.

**Keywords:** Battery thermal management, reinforcement learning, degradation surrogate, 1D-CNN, SAC, TD3, Severson dataset, LFP batteries, health-aware control

---

## 1. Introduction

Lithium-ion (Li-ion) batteries are the cornerstone energy storage technology for electric vehicles, grid storage, and consumer electronics. Battery thermal management is a critical determinant of both safety and longevity: elevated temperatures accelerate degradation mechanisms and increase thermal-safety risk. In this study, **45°C is adopted as a conservative hard temperature constraint** in the control environment — consistent with published BTMS RL studies [4, 5] — and 35°C is used as the thermal-stress onset threshold based on temperature-dependent ageing literature [1]. Classical BTMS — typically PID controllers or rule-based heuristics — operate reactively, adjusting cooling power based on instantaneous temperature error without awareness of battery health state.

Reinforcement learning (RL) offers an adaptive alternative: data-driven policies can learn to balance multiple objectives through interaction with a thermal environment [2, 3]. Recent work has applied RL to BTMS and health-aware charging with PID and MPC baselines [4, 5, 11]. However, a recurring limitation is that the degradation model is physics-derived (e.g., Arrhenius) or generic empirical — not trained on real cell cycling data. The **Severson 2019 dataset** [6] provides cycling data for 124 commercial LFP/graphite 18650 cells, offering a unique opportunity to train a data-driven surrogate directly on real LFP cells and embed it into an RL thermal reward.

To our knowledge, existing health-aware RL thermal management studies have predominantly employed physics-based electrochemical models or health constraints derived from simulated environments. This work investigates the use of a lightweight degradation surrogate trained on real LFP cycling data to provide health information within a thermal-control RL framework — differing from Yuan & Zou [11] (health-aware TD3 charging via PyBaMM, not thermal management) and from physics-model BTMS RL [4, 5].

### 1.1 Contributions

**C1:** A **3-channel 1D-CNN** (196K parameters) achieving **R²=0.854 on 34 fully cell-held-out external test cells** and R²=0.888 on the primary evaluation protocol. The surrogate predicts next-cycle discharge capacity from raw (V, I, T) signals — a short-horizon health estimate embeddable into RL reward computation.

**C2:** A **health-adaptive thermal stress reward** where Q̂ dynamically scales the stress penalty: as Q̂ falls with ageing, per-unit stress cost rises automatically, making the controller more conservative at higher degradation states.

**C3:** A **progressive development framework** (DDPG → TD3 → SAC) documenting and correcting methodological limitations at each stage — a reproducible research methodology.

**C4:** In the proposed LCTM simulation environment, the TD3 controller reduces simulated cooling energy by **78.6%** vs. Heuristic-P (371 J vs. 1,739 J), with 5.17°C temperature MAE and zero safety violations — qualified explicitly as a simulator result.

**C5:** A **β reward-reweighting analysis** and planned **T_onset sensitivity study** characterising the temperature–degradation stress trade-off.

---

## 2. Related Work

### 2.1 Data-Driven Battery Lifetime and Health Prediction

**Severson et al. (2019)** [6] demonstrated elastic net regression achieving 9.1% MAPE for cycle-life prediction on 43 primary-test cells. Their task — long-horizon cycle-life prediction from early-cycle data — differs from ours: **next-cycle discharge capacity estimation from current-cycle (V, I, T) signals**.

**Fei et al. (2021)** [8] applied elastic net, GPR, SVM, RF, GBRT, and NN for early battery lifetime prediction on the Severson dataset; best SVM achieves R²=0.90 and RMSE=115 cycles for cycle-life prediction — a different task.

**Fan et al. (2020)** [9] proposed a GRU-CNN hybrid for SoH estimation on NASA and Oxford datasets (max error ≤4.3%). Dataset and task differ from ours.

**Liu et al. (2022)** [7] applied 1D-CNN and CNN-LSTM to the Severson dataset for cycle-life prediction (R²≈0.85, single-channel voltage). Different task and evaluation protocol.

### 2.2 RL for Battery Thermal Management

**Abbasi et al. (2024)** [4] applied RL for coupled charging and thermal management of an EV battery pack, comparing against MPC and CC-CV. Their work establishes MPC as a competitive baseline for BTMS RL and demonstrates RL advantages in energy efficiency.

**Zhang et al. (2024)** [5] applied RL for battery and occupant compartment thermal management in EVs, explicitly comparing against a tuned PID controller and demonstrating the importance of proper baseline definition.

**Yuan & Zou (2025)** [11] applied TD3 with a PyBaMM SPMe electrochemical model for health-aware fast charging. Task: charging control; health model: physics-based PyBaMM, not data-trained; different from our thermal management focus.

**Ghalkhani & Habibi (2023)** [12] reviewed AI-based BMS covering DDPG, TD3, PPO, SAC, MPC, confirming RL BTMS is active with PID/MPC as expected baselines.

### 2.3 Gap Analysis

| Work | Health Model | Training Data | Task | Baselines |
| :--- | :---: | :---: | :---: | :---: |
| Abbasi 2024 [4] | Thermal model | Simulated | Thermal+charging | MPC, CC-CV |
| Zhang 2024 [5] | None | Simulated | BTMS | PID |
| Yuan & Zou 2025 [11] | Physics (PyBaMM) | Simulated | Charging | CC-CV |
| **This work (v4 SAC)** | **1D-CNN (real LFP)** | **Severson 2019** | **Thermal mgmt** | **PID + MPC** |

**Our differentiation:** real LFP cycling data → learned short-horizon capacity surrogate → adaptive health-scaled thermal stress reward → continuous RL cooling control.

---

## 3. Dataset and Surrogate

### 3.1 Severson 2019 Dataset

| Property | Value |
| :--- | :--- |
| Source | Severson et al., Nature Energy, 2019 [6] |
| Public access | [data.matr.io/1](https://data.matr.io/1) |
| Chemistry | LFP/graphite 18650 cylindrical |
| Charging protocols | 72 distinct CCCV fast-charging variants |
| Cycle life range | 150 – 2,300 cycles |

#### Definitive Cell Accounting (verified from preprocessing code)

The following counts are verified from `degradation_surrogate_1dcnn.ipynb` (the executed preprocessing code) and the raw pkl files:

| Step | Count | Notes |
| :--- | :---: | :--- |
| Raw pkl records | **140** | b1: 46, b2: 48, b3: 46 |
| Batch-2 continuation records merged into B1 | **−5** | b2c7→b1c0, b2c8→b1c1, b2c9→b1c2, b2c15→b1c3, b2c16→b1c4 |
| **Physical cell records after merging** | **135** | Unique physical cells |
| Failed B1 cells (capacity/sensor artefacts) | **−5** | b1c8, b1c10, b1c12, b1c13, b1c22 |
| Noisy B3 cells | **−6** | b3c2, b3c23, b3c32, b3c37, b3c42, b3c43 |
| **Final usable physical cells** | **124** | All have valid V, I, T time-series |
| Cell-level split | 70 train / 20 val / **34 external test** | Fixed random seed, saved to `data_split.json` |

> Although the raw pkl files contain 140 records, five Batch-2 records represent continuation experiments from five Batch-1 cells. The preprocessing code (verified in `degradation_surrogate_1dcnn.ipynb`) appends the Batch-2 continuation cycles to the corresponding Batch-1 cell records and deletes the five Batch-2 entries. After this merge, 135 unique physical cell records remain. Removing 5 failed B1 cells and 6 noisy B3 cells yields **124 usable physical cells**, consistent with the 70+20+34=124 split.

> **Channel data:** All 124 usable cells contain valid `V` (voltage), `I` (current), and `T` (temperature) raw time-series fields. These three channels are the CNN inputs; `Qdlin` and `Tdlin` are also available in the pkl files but are not used as CNN input channels.

#### Explaining the 34,136 Figure

The merged dataset contains cycle data across 124 usable physical cells. The preprocessing code produces **34,136 valid 60-timestep cycle windows** from these cells:

| Stage | Records | Notes |
| :--- | :---: | :--- |
| Total cycle records (post-merge, 124 cells) | ~97K | Includes continuation cycles appended from B2 |
| Less: early burn-in cycles (idx < 5) | ~95K | Below minimum cycle threshold |
| Less: cycles with V/T/I length < 4 | ~92K | Too few measurement points for resampling |
| Resample to 60 pts: V(t), I(t), T(t) | ~92K | Linear interpolation on uniform [0,1] grid |
| Less: NaN/Inf in any channel | ~68K | Channel quality filter |
| Less: outlier cycles (V range < 0.1V) | **34,136** | Anomalous charge artefacts removed |

> Exact thresholds are implemented in `degradation_surrogate_1dcnn.ipynb`, Cell 6, and will be documented in supplementary material for reproducibility.

#### Cell-Level Data Split

All 60-timestep windows from a given cell are assigned **exclusively to one split**:

| Split | Cells | Windows (approx.) | Purpose |
| :--- | :---: | :---: | :--- |
| Train | 70 | ~19,500 | CNN training, stress K calibration, RL training |
| Validation | 20 | ~5,600 | Hyperparameter tuning, RL val curves |
| **External test** | **34** | **~9,000** | **Final evaluation — never accessed during development** |

Normalisation statistics computed from training cells only, applied to val/test. Batch 3 cells preferentially assigned to external test.

### 3.2 Degradation Surrogate: 1D-CNN

**Task:** Predict next-cycle discharge capacity Q̂(n+1) (Ah) from cycle n's raw time-series (V(t), I(t), T(t)) — a between-episode health estimate, distinct from long-horizon cycle-life prediction in [6, 8].

**Causal timing:** The surrogate is applied once per completed charge–discharge cycle using that cycle's V, I, T measurements. The resulting Q̂(n+1) is then used in the RL controller's state and reward for episode n+1 — not the current cycle. This avoids any within-cycle causality violation:

```
Cycle n:  V(t), I(t), T(t) → 1D-CNN → Q̂(n+1)
                                              ↓
                          RL Episode n+1: state & stress reward scaled by Q_ref/Q̂(n+1)
```

#### Architecture

| Component | Configuration |
| :--- | :--- |
| **Input** | **3 channels × 60 timesteps: [V(t), I(t), T(t)]** — raw voltage, current, temperature resampled to 60 pts via linear interpolation on a uniform grid |
| Conv block 1 | Conv1d(3→32, k=5) + BN + ReLU + MaxPool(2) |
| Conv block 2 | Conv1d(32→64, k=3) + BN + ReLU + MaxPool(2) |
| Conv block 3 | Conv1d(64→128, k=3) + BN + ReLU + AdaptiveAvgPool(4) |
| MLP head | FC(512→256, drop=0.3) → FC(256→128, drop=0.2) → FC(128→1) |
| **Total parameters** | **196,225** |
| Optimiser | Adam, lr=1e-3, ReduceLROnPlateau (factor=0.5, patience=10) |
| Early stopping | Patience=15 on validation MAE |

> **Feature clarification (verified from executed notebook cell 6):** The CNN reads `c['V']`, `c['I']`, `c['T']` — the raw per-cycle measurement arrays stored in the pkl files. Each is resampled from the native measurement density (~1,100 pts/cycle) to exactly 60 points via `np.interp` on a uniform [0,1] grid, then stacked as `np.stack([V_f, I_f, T_f], axis=0)`. The pkl files also contain `Qdlin` and `Tdlin` (1,000-pt discharge-indexed interpolations) — these are **not** used as CNN inputs. Per-channel z-score normalisation (mean/std from training cells only) is applied before the network.

### 3.3 Surrogate Results (Table 1)

| Evaluation Set | Cells | Role | MAE (Ah) | RMSE (Ah) | R² |
| :--- | :---: | :--- | :---: | :---: | :---: |
| Validation | 20 | Model selection | 0.0070 | 0.0099 | 0.969 |
| Primary evaluation (all 90 train+val cells) | 70+20 | Comparison with published protocols | 0.0103 | 0.0194 | 0.888 |
| **External cell-held-out test** | **34** | **True generalisation — principal result** | **0.0121** | **0.0168** | **0.854** |

> **R²=0.854 is the principal surrogate result.** The primary evaluation (R²=0.888) includes training cells and is provided for secondary comparison with prior Severson-based works. All contribution claims use the 0.854 external test figure.

### 3.4 Surrogate Comparison (Table 2)

| Paper | Model | Dataset | Prediction Task | Horizon | Principal Metric |
| :--- | :--- | :--- | :--- | :---: | :---: |
| Severson 2019 [6] | Elastic net | Severson | Cycle-life | Long | 9.1% MAPE |
| Fei 2021 [8] | SVM/ensemble | Severson | Cycle-life | Long | R²=0.90, RMSE=115 cyc |
| Liu 2022 [7] | 1D-CNN | Severson | Cycle-life | Long | R²≈0.85 |
| Fan 2020 [9] | GRU-CNN | NASA/Oxford | SoH estimation | Medium | MAE ≤4.3% |
| **Ours** | **1D-CNN** | **Severson** | **Next-cycle Q** | **Short** | **R²=0.854 (ext. 34 cells)** |

> Different tasks, datasets, and horizons. Direct metric comparison is not meaningful; table provides research context only.

### 3.5 Surrogate Temperature Sensitivity (Table 3)

To provide experimental evidence for the partial correlation finding, we performed a counterfactual test using the trained CNN checkpoint on 590 external test cell profiles:

- V(t) and I(t) held fixed; T(t) uniformly perturbed by ΔT.
- Q̂ predicted at baseline and perturbed temperature.
- ΔQ̂ = Q̂(T+ΔT) − Q̂(T) reported per profile.

| ΔT added | Mean ΔQ̂ (Ah) | ±Std | 95% CI | % of Q̂̄ |
| :---: | :---: | :---: | :---: | :---: |
| +2°C | −0.00505 | 0.00131 | [−0.00516, −0.00495] | 0.47% |
| +5°C | −0.01339 | 0.00899 | [−0.01412, −0.01267] | **1.25%** |
| +10°C | −0.01236 | 0.00832 | [−0.01303, −0.01169] | 1.16% |

*Baseline mean Q̂̄ = 1.069 Ah, n = 590 cycle profiles from 34 external test cells (profile-level CI; profiles are nested within cells — cell-level bootstrap CI is recommended for peer review).*

> **Interpretation:** Temperature shifts of 2–10°C produce statistically significant but small capacity estimate changes (max 1.25% of Q̂̄, 95% profile-level CI excludes zero). The surrogate is **not fully insensitive** to temperature, but the effect is modest — consistent with the near-zero partial correlation (r = −0.015 after controlling for cycle index). The non-monotonic response (+10°C < +5°C) suggests a local nonlinearity in the voltage plateau region captured by the CNN. This **supports** the use of the physics-calibrated stress term $S_t = K\max(0, T - T_{onset})^2|I|$ as a **complementary** temperature–degradation signal — one that is not redundant with the surrogate's modest intrinsic temperature sensitivity.

### 3.6 Temperature–Capacity Partial Correlation

| Statistic | Value |
| :--- | :---: |
| Pearson corr(T_mean, Q) | −0.170 |
| **Partial corr(T_mean, Q \| cycle_index)** | **−0.015** |

This is an empirical observation within the Severson operating regime (25–38°C, CCCV protocols). The near-zero partial correlation, combined with the counterfactual experiment (Table 3), **supports** the conclusion that the surrogate's capacity estimates shift by only 0.5–1.3% per 2–10°C temperature perturbation. This small but non-zero sensitivity means the surrogate captures **both** temperature-ageing effects (weakly, through the voltage plateau shape) **and** cycle-index degradation (strongly). The physics-calibrated stress term therefore provides an **additional, complementary** temperature-degradation signal that is not fully redundant with the surrogate's Q̂ estimates. The v4 NoStress ablation will quantify the stress term's contribution to control performance.

---

## 4. Thermal Control Environment

### 4.1 Lumped Capacitance Thermal Model (LCTM)

$$\frac{dT}{dt} = \frac{I^2 R_{int} - P_{cool} - h(T - T_{amb})}{C_{th}}$$

| Parameter | Value |
| :--- | :---: |
| $C_{th}$ | 800 J/K |
| $R_{int}$ | 0.025 Ω |
| $h$ | 4.0 W/K |
| $P_{max}$ | 60 W |
| $T_{amb}$ | 25°C |
| $T_{target}$ | 30°C |
| $T_{max}$ (hard constraint) | **45°C** |

> **On the 45°C constraint:** This value is adopted as a conservative safety limit in the simulator, consistent with published BTMS RL studies [4, 5]. It is a **simulator design parameter**, not a claim that 45°C universally triggers thermal runaway. LFP thermal runaway onset is chemistry- and cell-design-dependent and typically reported at much higher temperatures (>150°C); the 45°C limit represents an operational temperature bound consistent with battery manufacturer guidelines.

> **LCTM scope:** This model does not capture spatial temperature gradients, SOC-dependent resistance, or electrochemical dynamics. It is used as a control-oriented proxy because RL training requires hundreds of thousands of steps, making full electrochemical simulation infeasible for active training on CPU. All results are **simulator comparisons under fixed LCTM conditions**.

### 4.2 Thermal Stress Formulation

The instantaneous thermal stress at timestep $t$ is:

$$S_t = K \cdot \max(0,\, T_t - T_{onset})^2 \cdot |I_t|$$

with fixed timestep $\Delta t$ (same for all experiments). The cumulative stress over an episode of $N$ steps is:

$$S_{cum} = \sum_{t=1}^{N} S_t \cdot \Delta t$$

The per-step reward penalty uses $S_t$ (not $S_t \Delta t$):

$$r_{\beta\text{-term}} = -\beta \cdot S_t \cdot \frac{Q_{ref}}{Q̂}$$

> **β and timestep:** The timestep $\Delta t$ is fixed and identical across all agents and baselines in every reported experiment, so β operates under a consistent numerical scaling throughout all comparisons.

**K calibration (training cells only — cycle-integrated OLS):**

For each training-cell cycle where T exceeds T_onset on any timestep, compute the unnormalised cycle stress:

$$S^0_{\text{cycle}}(n) = \sum_{t} \max(0,\, T_t - T_{onset})^2 \cdot |I_t| \cdot \Delta t$$

where $|I_t|$ is current in **amperes**. K is fitted via ordinary least squares across all stress-active training cycles:

$$K = \frac{\sum_n \text{fade}(n) \cdot S^0_{\text{cycle}}(n)}{\sum_n \left[S^0_{\text{cycle}}(n)\right]^2}$$

where $\text{fade}(n) = Q_d(n-1) - Q_d(n)$ (Ah/cycle). This approach uses only T > T_onset cycles (avoiding the zero-stress problem), is units-consistent (K has units Ah·(A·°C²·s)⁻¹), and grounds the stress term in empirical LFP fade observations from Severson training cells only (no leakage).

**T_onset source:** Adopted from Waldmann et al. [1] (NMC/LMO chemistry) as a reasonable onset. A sensitivity study (Section 7.5) tests T_onset ∈ {30, 32.5, 35, 37.5, 40}°C.

---

## 5. Health-Aware Reinforcement Learning

### 5.1 Algorithm Progression (Development History — Not Controlled Benchmark)

The DDPG → TD3 → SAC progression is a **methodological development history**. Versions differ in algorithm, training budget, episode length, and reward formulation — scientifically invalid as an algorithm comparison. The final **controlled comparison** in v4 uses identical evaluation conditions for all agents.

| Property | DDPG (v1) | TD3 (v2) | SAC (v3) | SAC (v4) |
| :--- | :---: | :---: | :---: | :---: |
| Total env steps | 100K | 450K | 400K | 600K |
| Q_ref handling | Static global mean ⚠️ | Dynamic per-episode ✓ | Dynamic ✓ | Dynamic ✓ |
| Seeds | 1 | 1 | 1 | **5** |
| Cell split | None | None | None | **70/20/34** |
| NoAdaptive ablation | ✗ | N/A | ✗ Bug | ✅ `adaptive=False` |
| β-ablation | ✗ | Reward-reweighting | ✗ | ✅ Fresh agent/β |

### 5.2 Reward Function

**v1 (DDPG):** $r = -\alpha(T - T_{tgt})^2 - \gamma P_{cool} - \text{safety}$

**v2 (TD3) — Dynamic Q_ref:** $r = -\alpha(T - T_{tgt})^2 - \beta \cdot \frac{Q_{ref}}{Q̂} \cdot P_{deg} - \gamma P_{cool} - \text{safety}$

where $P_{deg} = S_t = K\max(0,\, T_t - T_{onset})^2|I_t|$ is the instantaneous thermal stress (identical to the $S_t$ used in v3/v4; the notation $P_{deg}$ was used in the v2 implementation).

**v3/v4 (SAC) — Adaptive stress:** $r = -\alpha(T - T_{tgt})^2 - \beta \cdot S_t \cdot \frac{Q_{ref}}{Q̂} - \gamma P_{cool} - \eta|\dot{T}| - \text{safety}$

where α=0.1, β=5.0, γ=0.01, η=0.005, T_tgt=30°C, T_max=45°C.

### 5.3 Baselines

| Baseline | Description | Status |
| :--- | :--- | :---: |
| **Heuristic-P** | $P_{cool} = \text{clip}(20 + 2e + 0.1e^2,\, 0,\, P_{max})$ — quadratic proportional, **no integral or derivative term** | v1–v3 |
| **Tuned PID** | $u = K_p e + K_i \int e\,dt + K_d \dot{e}$ + anti-windup; gains tuned on val cells | ⏳ v4 |
| **MPC** | Receding-horizon optimal on LCTM, 10-step horizon | ⏳ v4 |
| **SAC-NoStress** | β=0 | ⏳ v4 |
| **SAC-NoAdaptive** | `adaptive=False` — health_ratio=1.0 | ⏳ v4 |
| **SAC-Full** | Full proposed method | ⏳ v4 |

---

## 6. Experimental Setup

**Evaluation protocol (v4):** All agents evaluated on the same 34 external test cells, same current profiles, same initial T/SOC, same episode length, same LCTM parameters.

**Statistics:** Mean ± std across **5 independently trained seeds** (the primary statistical unit). Episode-level metrics aggregated over 34 external test cells per seed. Note: this is a nested design (episodes within seeds within cells) — "5 seeds × 50 episodes" does not yield 250 statistically independent observations.

### Primary Metrics

| Metric | Symbol | Unit | Direction |
| :--- | :--- | :---: | :---: |
| Temperature MAE | T_MAE | °C | ↓ |
| Peak temperature | T_peak | °C | ↓ |
| Cumulative thermal stress | S_cum | (°C)²·A·s | ↓ |
| Cooling energy | E_cool | J | ↓ |
| Safety violations | N_safe | steps | ↓ (target: 0) |
| Action smoothness | σ_a | W | ↓ |

> **Surrogate-derived diagnostic (supplementary only):** Normalised Predicted Capacity Ratio (NPCR) = Q̂/Q_ref × 100%. Values above 100% are artefacts of surrogate estimation relative to the episode reference. NPCR is **not a measure of physical capacity** and is not used as a primary performance metric.

---

## 7. Results

### 7.1 v1: DDPG Baseline (Table 3) — Preliminary

*20-episode evaluation, Heuristic-P baseline, single seed 42.*

| Metric | DDPG | Heuristic-P | Δ |
| :--- | :---: | :---: | :---: |
| T_MAE | 5.24°C | 5.51°C | −4.9% |
| **E_cool** | **809 J** | **1,246 J** | **−35.1%** |
| Safety violations | 0 | 0 | — |

> ⚠️ Static Q_ref bug → surrogate had no training influence. β-ablation invalid (all β return identical metrics). Baseline feasibility only.

### 7.2 v2: TD3 + Dynamic Q_ref (Table 4) ★ Best Validated Result

*50-episode evaluation, Heuristic-P baseline, single seed 42.*

| Metric | TD3-Surrogate | TD3-NoSurrogate | Heuristic-P | TD3 vs H-P |
| :--- | :---: | :---: | :---: | :---: |
| T_MAE | 5.17°C | 5.05°C | 5.68°C | **−9.0%** |
| **E_cool** | **371 J** | **350 J** | **1,739 J** | **−78.6%**† |
| Deg. penalty | 0.00207 | 0.00283 | 0.00149 | — |
| Safety violations | 0 | 0 | 0 | — |

†: *Within the proposed LCTM simulator, TD3 reduced simulated cooling energy by 78.6% vs. Heuristic-P — 50-episode evaluation, single seed 42, fixed LCTM parameters, Heuristic-P baseline (not a tuned PID).*

**TD3-Surrogate vs. TD3-NoSurrogate:** TD3-Surrogate (371 J) and TD3-NoSurrogate (350 J) produced similar cooling-energy values in this single-seed evaluation. Whether this ~6% difference is within statistical noise cannot be determined from a single seed — v4's 5-seed evaluation will clarify this. The surrogate's role is to provide an adaptive health-state multiplier for the stress penalty, not to serve as a direct temperature feedback signal.

### 7.3 v2: β Reward-Reweighting Analysis (Table 5)

> **Important clarification on methodology:** This analysis uses a **single trained TD3 agent** (trained at β=5.0) and re-evaluates it under different reward weight values. Because the policy is frozen, changing β during evaluation changes only the **reported reward calculation** — it does not change the agent's actions or temperature trajectories. Therefore, T_MAE, E_cool, and S_cum reflect the fixed policy's behaviour, while the reported reward values reflect different penalty weightings. The variation in T_MAE across β values (Table 5) may reflect episode-to-episode stochasticity across 300 episodes per β, not policy-level β sensitivity.

> **The v4 β-ablation corrects this:** each β value trains a fresh SAC agent from scratch, producing true policy-level sensitivity.

| β | Reported T_MAE (°C) | ±std | Deg. Penalty | Interpretation |
| :---: | :---: | :---: | :---: | :--- |
| 0.0 | 5.487 | 0.439 | 0.002397 | Policy trained at β=5.0, reward reweighted |
| 0.5 | 5.511 | 0.367 | 0.002317 | ← |
| 1.0 | 5.587 | 0.617 | 0.002287 | ← |
| 2.0 | 5.395 | 0.342 | 0.002409 | ← |
| **5.0** (training β) | **5.384** | **0.438** | **0.002458** | Native evaluation |
| 10.0 | 5.475 | 0.409 | 0.002409 | ← |
| 20.0 | 5.343 | 0.366 | 0.002494 | ← |

The T_MAE variation (~0.2°C range) across β is attributed to episode-level stochasticity (300 episodes per β). The policy itself does not change. This table characterises reward-level β sensitivity; policy-level sensitivity requires the v4 fresh-agent ablation.

### 7.4 v3: SAC Preliminary (Table 6) — Known Bugs

*50-episode evaluation, Heuristic-P baseline, single seed 42.*

| Metric | SAC-Full | SAC-NoAdaptive | SAC-NoStress | Heuristic-P |
| :--- | :---: | :---: | :---: | :---: |
| T_MAE | 5.40°C | 5.44°C | **5.29°C** | 5.68°C |
| **E_cool** | 1,486 J | 1,636 J | **1,037 J** | 2,771 J |
| S_cum | 0.000 ⚠️ | 0.000 ⚠️ | 0.000 ⚠️ | 0.000 ⚠️ |
| Safety violations | 0 | 0 | 0 | 0 |

> ⚠️ S_cum = 0.0 everywhere (K_STRESS too small — fixed in v4 with cycle-integrated OLS calibration on training cells). β-ablation invalid. NoAdaptive bug. Single seed. Directional only.

### 7.5 T_onset Sensitivity (Planned — v4, Table 7)

| T_onset (°C) | T_MAE | T_peak | S_cum | E_cool | Safety |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 30 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 32.5 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| **35** | — | — | — | — | — |
| 37.5 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 40 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

### 7.6 v4: SAC Final Results (Planned — Table 8) ⏳

*5 seeds × 50 episodes on 34-cell external test. All agents: identical evaluation conditions.*

| Metric | Heuristic-P | Tuned PID | MPC | SAC-NoStress | SAC-NoAdaptive | **SAC-Full** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| T_MAE (°C) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ ± ⏳ |
| T_peak (°C) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ ± ⏳ |
| S_cum | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ ± ⏳ |
| E_cool (J) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ ± ⏳ |
| Safety violations | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

*Statistics: mean ± std across 5 independently trained seeds.*

**Planned ablation sub-table:**

| Ablation | Health scaling | Stress term | Expected primary effect |
| :--- | :---: | :---: | :--- |
| SAC-NoStress | — | ✗ | Higher E_cool allowed |
| SAC-NoAdaptive | ✗ (fixed ratio=1) | ✓ | No age-adaptive behaviour |
| **SAC-Full** | ✓ (Q̂-scaled) | ✓ | Full proposed method |

### 7.7 Stress Model Validation on External Test Cells (Table 9) ⏳

*Calibrated on training cells only; evaluated on 34 external test cells.*

| Stress Model | Pearson r | Spearman ρ | R² | MAE (Ah/cyc) |
| :--- | :---: | :---: | :---: | :---: |
| Temperature-only: (T−35)² | ⏳ | ⏳ | ⏳ | ⏳ |
| Current-only: \|I\| | ⏳ | ⏳ | ⏳ | ⏳ |
| **Full: K(T−35)²\|I\|** ← proposed | ⏳ | ⏳ | ⏳ | ⏳ |
| Polynomial baseline | ⏳ | ⏳ | ⏳ | ⏳ |

> Spearman ρ is reported alongside Pearson r because the stress model needs only to **rank** degradation severity correctly — not predict absolute fade values. Both metrics are therefore informative.

### 7.8 Robustness Analysis (Planned — Table 10)

| Perturbation | Values tested | Primary metrics |
| :--- | :--- | :---: |
| Ambient temperature | 25, 30, 35, 40°C | T_MAE, E_cool |
| Internal resistance | −20%, nominal, +20% | T_MAE, S_cum |
| Convective coefficient h | −20%, nominal, +20% | E_cool, S_cum |
| Current profile | Nominal, high-load, low-load, stochastic | All |

---

## 8. Discussion

### 8.1 Source of the 78.6% Energy Reduction

The 35.1% → 78.6% jump from v1 (DDPG) to v2 (TD3) reflects two **simultaneous** changes: (i) the Q_ref bug fix (static global → dynamic per-episode) and (ii) the TD3 algorithm. Because both changed together, the energy reduction cannot be attributed to either alone. The v4 ablation will partially disentangle these.

### 8.2 The Surrogate's Role — A Clarification

TD3-Surrogate and TD3-NoSurrogate produced similar cooling energy in the single-seed evaluation (371 J vs. 350 J). The counterfactual experiment (Section 3.5, Table 3) indicates that temperature perturbations of 2–10°C produce only modest changes in Q̂ (max 1.25%) — meaning the surrogate cannot serve as a strong direct temperature feedback signal. The surrogate's primary role is to provide a health-state multiplier ($Q_{ref}/Q̂$) that scales the stress penalty, making the controller's degradation response proportional to battery age. The physics-calibrated stress term $S_t$ provides the complementary direct temperature signal. Whether this two-pathway architecture meaningfully improves control over a single-pathway design will be established by the v4 SAC-NoStress ablation.

### 8.3 Simulator Scope and External Validity

All quantitative results are bounded by the LCTM assumptions. The robustness analysis (Section 7.8) characterises sensitivity to model parameters. HIL validation on a physical test bench is the required next step before any claim of practical energy savings.

---

## 9. Limitations

| Limitation | Severity | Mitigation |
| :--- | :---: | :--- |
| LCTM — no spatial gradients, no electrochemical coupling | High | Explicitly scoped; future: PyBaMM environment |
| No hardware validation | High | HIL as future work |
| Single seed v1–v3 | High | v4: 5 seeds |
| No cell-level split v1–v3 | High | v4: 70/20/34 external test |
| Heuristic-P ≠ true PID (v1–v3 comparison potentially inflated) | Moderate | Tuned PID + MPC in v4 |
| v2 β table is reward-reweighting, not policy sensitivity | Moderate | Labelled correctly; v4 fresh-agent ablation |
| 35°C T_onset from NMC literature, not LFP-specific | Low | T_onset sensitivity study planned |
| LFP chemistry only | Moderate | NMC generalisation as future work |
| Lab CCCV profiles only | Moderate | Drive-cycle integration as future work |
| NPCR metric > 100% is surrogate artefact | Low | Demoted to supplementary diagnostic |

---

## 10. Conclusion

1. A **3-channel 1D-CNN** achieves **R²=0.854** on 34 fully cell-held-out external test cells — the principal generalisation result — and R²=0.888 on the primary evaluation protocol.

2. **TD3 with dynamic surrogate coupling** reduces simulated cooling energy by **78.6%** vs. Heuristic-P (371 J vs. 1,739 J) in the LCTM environment, with 5.17°C temperature MAE and zero safety violations — a simulator result, single-seed, Heuristic-P baseline.

3. **TD3-Surrogate and TD3-NoSurrogate produced similar cooling energy** in single-seed evaluation; multi-seed v4 evaluation will determine whether the difference is statistically meaningful. The surrogate's primary role is an adaptive health-state multiplier, not a direct temperature signal.

4. The **β reward-reweighting analysis** shows ~0.2°C T_MAE variation across β in single-seed evaluation; true policy-level sensitivity requires v4's fresh-agent-per-β ablation.

5. **Cell accounting is definitively resolved:** 140 raw pkl records − 5 continuation merges − 11 exclusions = 124 usable cells, split 70/20/34. **K calibration is corrected:** cycle-integrated OLS over stress-active training-cell cycles only (T > T_onset = 35°C); calibration current expressed in amperes.

6. **SAC v4** (5 seeds, 34-cell external test, tuned PID + MPC baselines, corrected ablations) is running and will provide the statistically rigorous results required for peer review.

---

## 11. Future Work

1. Complete v4 with multi-seed external-test results, PID/MPC baselines, T_onset sensitivity, valid ablations
2. ~~Surrogate counterfactual temperature sensitivity test~~ **✅ DONE** (Table 3): temperature perturbations of +2/+5/+10°C produced maximum 1.25% shift in predicted capacity across 590 profiles from 34 external test cells.
3. Hardware-in-the-loop (HIL) validation on physical 18650 test bench
4. Drive-cycle integration (WLTP, UDDS) replacing CCCV profiles
5. Continual surrogate learning as cells age beyond training distribution
6. NMC/LCO generalisation by retraining surrogate on different chemistry data
7. Full multi-agent BMS integrating thermal, health, and charging agents with coordinator
8. Physics-informed surrogate with monotone capacity constraint

---

## Appendix A: Hyperparameters

| Parameter | Symbol | Value |
| :--- | :---: | :---: |
| Thermal comfort weight | α | 0.1 |
| Stress penalty weight | β | 5.0 |
| Energy efficiency weight | γ | 0.01 |
| Smoothness weight | η | 0.005 |
| Stress onset | T_onset | 35°C |
| K calibration method | — | Cycle-integrated OLS over stress-active (T > T_onset) training cycles; current in amperes |
| SAC discount | γ_RL | 0.99 |
| Soft update | τ | 0.005 |
| Learning rate | lr | 3×10⁻⁴ |
| Replay buffer | — | 300,000 |
| Batch size | — | 256 |
| Hidden dim | — | 256 |
| v4 Seeds | — | 42, 52, 62, 72, 82 |
| Timestep | Δt | fixed (same all experiments) |

---

## References

[1] Waldmann, T. et al. (2014). Temperature dependent ageing mechanisms in Lithium-ion batteries. *J. Power Sources*, 262, 129–137. [DOI:10.1016/j.jpowsour.2014.03.112](https://doi.org/10.1016/j.jpowsour.2014.03.112)

[2] Hu, X., Xu, L., Lin, X., & Pecht, M. (2020). Battery Lifetime Prognostics. *Joule*, 4(2), 310–346. [DOI:10.1016/j.joule.2019.11.018](https://doi.org/10.1016/j.joule.2019.11.018)

[3] Qian, C. et al. (2022). Battery Management System: Review of Current Issues. *IEEE Trans. Ind. Informatics*. [DOI:10.1109/TII.2021.3091199](https://doi.org/10.1109/TII.2021.3091199)

[4] **Abbasi, R. et al.** (2024). Deep reinforcement learning based fast charging and thermal management optimization of an electric vehicle battery pack. *Journal of Energy Storage*, 95, 112466. [DOI:10.1016/j.est.2024.112466](https://doi.org/10.1016/j.est.2024.112466)

[5] **Zhang, Y. et al.** (2024). Reinforcement learning-based control for the thermal management of the battery and occupant compartments of electric vehicles. *Sustainable Energy & Fuels*, 8, 588–603. [DOI:10.1039/D3SE01403G](https://doi.org/10.1039/D3SE01403G)

[6] Severson, K.A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391. [DOI:10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8) | Data: [data.matr.io/1](https://data.matr.io/1)

[7] Liu, K. et al. (2022). Development of a Lithium-Ion Battery Lifetime Prediction Model Using Deep Learning. *SCITEPRESS*. [PDF](https://www.scitepress.org/Papers/2022/108972/108972.pdf)

[8] Fei, Z. et al. (2021). Early prediction of battery lifetime via a machine learning based framework. *Energy*, 225, 120205. [DOI:10.1016/j.energy.2021.120205](https://doi.org/10.1016/j.energy.2021.120205) *(SVM/ensemble for cycle-life prediction, Severson dataset)*

[9] Fan, Y. et al. (2020). A novel deep learning framework for state of health estimation of lithium-ion battery. *J. Energy Storage*, 32, 101746. [DOI:10.1016/j.est.2020.101746](https://doi.org/10.1016/j.est.2020.101746) *(GRU-CNN on NASA/Oxford; max error ≤4.3%)*

[10] Attia, P.M. et al. (2020). Closed-loop optimization of fast-charging protocols with machine learning. *Nature*, 578, 397–402. [DOI:10.1038/s41586-020-1994-5](https://doi.org/10.1038/s41586-020-1994-5)

[11] Yuan, M., & Zou, C. (2025). Lifelong Reinforcement Learning for Health-Aware Fast Charging of Lithium-Ion Batteries. *IEEE Trans. Transportation Electrification*, 12(1). [DOI:10.1109/TTE.2025.3625421](https://doi.org/10.1109/TTE.2025.3625421) | arXiv: [2505.11061](https://arxiv.org/abs/2505.11061)

[12] Ghalkhani, M., & Habibi, S. (2023). Review of the Li-Ion Battery, Thermal Management, and AI-Based BMS for EV. *Energies*, 16(1), 185. [DOI:10.3390/en16010185](https://doi.org/10.3390/en16010185)

[13] Lillicrap, T.P. et al. (2016). Continuous control with deep reinforcement learning. *ICLR 2016*. [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)

[14] Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *ICML 2018*. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)

[15] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic. *ICML 2018*. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)

[16] Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

[17] **Navidi, T., Thelen, A., Li, W., & Hu, C.** (2024). Physics-Informed Machine Learning for Battery Degradation Diagnostics: A Comparison of State-of-the-Art Methods. *Energy Storage Materials*, 68, 103343. [DOI:10.1016/j.ensm.2024.103343](https://doi.org/10.1016/j.ensm.2024.103343)

---

## Submission Gate Checklist

| Requirement | Status |
| :--- | :---: |
| ✅ Cell accounting: 140→16 excl.→124 usable→70/20/34, from pkl files | ✅ **RESOLVED** |
| ✅ K calibration fixed: cycle-integrated OLS, T > T_onset, current in amperes | ✅ **RESOLVED** |
| ✅ β table labelled as reward-reweighting, not policy sensitivity | ✅ **RESOLVED** |
| ✅ "Within noise" removed → "similar in single-seed evaluation" | ✅ **RESOLVED** |
| ✅ 45°C is simulator constraint, not universal thermal-runaway threshold | ✅ **RESOLVED** |
| ✅ Reference [4] = Abbasi et al. 2024 (real verified DOI) | ✅ **RESOLVED** |
| ✅ Reference [5] = Zhang et al. 2024 (real verified DOI) | ✅ **RESOLVED** |
| ✅ Reference [18]/[17] = Navidi et al. 2024 ENSM (correct paper) | ✅ **RESOLVED** |
| ✅ β/Δt relationship corrected: Δt fixed, β timestep-independent across comparisons | ✅ **RESOLVED** |
| ✅ "34,136 valid cycle windows" (not "training windows") | ✅ **RESOLVED** |
| ✅ R²=0.854 as principal surrogate result throughout | ✅ |
| ✅ Heuristic-P correctly named (no I or D term) | ✅ |
| ✅ 78.6% simulator qualifier on every mention | ✅ |
| ✅ NPCR demoted to supplementary diagnostic | ✅ |
| ✅ DDPG→TD3→SAC = development history, not algorithm benchmark | ✅ |
| ✅ No unsupported "first/no prior work" claim | ✅ |
| ✅ Nested statistics correctly described | ✅ |
| ⏳ v4: 5 seeds, 34-cell external test | **Running now** |
| ⏳ Tuned PID baseline | After v4 |
| ⏳ MPC baseline | After v4 |
| ⏳ Valid SAC ablations (NoAdaptive, NoStress, fresh-agent β) | After v4 |
| ⏳ Calibrated K_STRESS (non-zero S_cum) | After v4 |
| ⏳ T_onset sensitivity (30–40°C sweep) | After v4 |
| ⏳ Stress model validation on external cells (r, R², MAE, Spearman) | After v4 |
| ⏳ Robustness: T_amb, R_int, h, current profile | After v4 |
| ✅ Surrogate counterfactual T sensitivity (Table 3): max shift 1.25% across 34 external test cells | ✅ **DONE** |

---

*Paper version: September 18, 2026 — Fourth revision, addressing all remaining critical issues.*  
*Target venues: Journal of Energy Storage · IEEE Trans. Transportation Electrification · Applied Energy*
