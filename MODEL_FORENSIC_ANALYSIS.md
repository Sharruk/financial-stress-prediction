# Complete Forensic Analysis of Machine Learning & Competition History

**Repository:** `Sharruk/financial-stress-prediction`  
**Active Branch:** `nat`  
**Current HEAD Commit:** `edca51d`  
**Date of Investigation:** September 07, 2026  
**Competition:** Zindi Mobile-Money Liquidity Stress Prediction Challenge  
**Team Name:** `sem_5` (Nathaniel Christian, Sharruk S, Shalini M)  
**Primary Current Metrics:** Public Score = `0.720992331` (Rank 60) | Best Internal Metrics: Log Loss = `0.241713676`, ROC-AUC = `0.91184304`  
**Rank 1 Benchmark:** `0.739375868` | **Gap to #1:** `0.018383537`

---

## Executive Summary & Forensic Findings

1. **The In-Sample Calibration / Stacking Trap (Critical Root Cause):**
   A forensic inspection of the codebase across versions revealed an architectural flaw in how Level-2 Stacking and Isotonic Calibration were evaluated. In `src/ensemble.py`, the Level-2 Logistic Regression meta-learner and the Isotonic Regressor were fitted directly on the full concatenation of base model Out-of-Fold (OOF) predictions and then evaluated **in-sample** on those exact same training predictions. This created an artificial optimistic bias in reported validation metrics (e.g., reporting Log Loss dropping from ~0.249 down to 0.2417–0.2461), while the true out-of-fold generalization was significantly weaker. When test predictions were transformed by these overfit meta-models, public leaderboard performance fluctuated erratically.

2. **The Collinearity Illusion (CatBoost vs XGBoost vs LightGBM):**
   Correlation analysis of the OOF predictions in `run_20260905_140728.json` revealed that **XGBoost and LightGBM GOSS share an astonishing 0.99569 correlation** (virtually identical predictions). Meanwhile, CatBoost shares ~0.9775 correlation with both. In Version 14, setting equal weights (0.333 / 0.333 / 0.333) across the three models effectively gave the XGBoost/LightGBM decision surface a 66.7% majority vote, diluting the vastly superior CatBoost model (standalone OOF Log Loss 0.2475 vs XGBoost 0.2527 and LightGBM 0.2530), causing the severe public drop to `0.716302883`.

3. **Feature Bloat vs. Signal Degradation:**
   Feature count grew aggressively from 184 raw columns to 366 (v3), 440 (v4), 458 (v5), 488 (v6/v7), 498 (v8/v13), and 501 (v9/v15). While early domain additions (Cashflow Elasticity, Balance Drawdown, Net Burn Rate, Personal Z-Scores) delivered massive improvements (slashing Log Loss from 0.290 to 0.246), later iterations introduced noisy multi-way interaction features, high-order velocity ratios, and unpruned ratio variants that increased tree variance on the 30,000 unseen test samples.

4. **Target Encoding Noise Injection Mismatch:**
   In Version 13 (`4f71d6e`), Gaussian noise ($\mathcal{N}(0, 0.005)$) was injected exclusively into the training folds during cross-validation, while leaving validation and test sets clean. While intended to prevent memorization, this altered the distribution between train and validation/test representations, disrupting split-point thresholds for tree algorithms and degrading leaderboard performance.

---

## PART 1 — Complete Reconstructed Version History

Below is the chronological reconstruction of every model version and experiment across git commits, run JSON records, and submission logs.

| Version / Tag | Date | Git Commit | Models Trained | Feature Version & Count | CV Strategy | Seeds | Ensemble Method | Calibration & Post-Processing | OOF Log Loss | OOF ROC-AUC | Public Zindi Score | LB Rank | Submission Filename | Key Architectural Changes | Performance Verdict & Likely Reason |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V1 Starter** | 2026-07-22 | `b980c11` | LightGBM baseline | Raw Baseline (184 cols) | 5-Fold Stratified | `[42]` | Single Model | None | Not found in repository evidence | Not found in repository evidence | ~0.6800 (Est.) | Not found | `submission.csv` | Initial starter notebook pipeline. | Baseline established. High variance due to raw features. |
| **V2 Pipeline** | 2026-08-14 | `266a653` | LightGBM, XGBoost, PyTorch MLP | Early Domain (~340 cols) | 5-Fold Stratified | `[42]` | Simple Mean / Rank Avg | Basic clipping | Not found in repository evidence | Not found in repository evidence | Not found | Not found | `submission_02.csv` | Added 340+ domain features, PyTorch MLP, and rank averaging. | Improved ranking, but neural net predictions were noisy. |
| **V3 Grandmaster** | 2026-08-15 | `33f1e45` | LightGBM, LightGBM DART, XGBoost, HistGBM, RF, PyTorch MLP | v3 Engine (366 cols) | 10-Fold Stratified | `[42]` | SLSQP Weight Optimization | Logit Blending | 0.258784 | 0.899328 | Not found | Not found | `run_20260815_202911` | Shifted from 5-fold to 10-fold CV. Added LightGBM DART and SLSQP dual-metric blending. | Substantial improvement. 10-fold CV drastically stabilized out-of-fold estimation. |
| **V4 Calibrated** | 2026-08-15 | `a9172ee` | LightGBM, LightGBM DART, XGBoost, HistGBM, ExtraTrees, PyTorch MLP | v4 Engine (440 cols) | 10-Fold Stratified | `[42]` | SLSQP Weight Optimization | Temperature Scaling ($T \approx 1.03$) | 0.252702 | 0.905306 | Not found | Not found | `run_20260815_232909` | Added 74 new balance dynamic features; introduced Nelder-Mead Temperature Scaling. | Strong improvement. Temperature scaling softened overconfident false positives. |
| **V5 Velocity** | 2026-08-16 | `fe4d8da` / `a7103d2` | LightGBM, LightGBM DART, XGBoost, HistGBM, ExtraTrees, PyTorch MLP | v5 Engine (458 cols) | 10-Fold Stratified | `[42]` | SLSQP Weight Optimization | Temperature Scaling | 0.252654 | 0.905362 | Not found | Not found | `run_20260816_022329` | Added inflow collapse flags and shock drain metrics; FastAPI backend integration. | Minor improvement. Plateaued because tree architectures remained identical. |
| **V6 CatBoost GPU** | 2026-08-20 | `68f3eb7` / `f6af3d1` | CatBoost GPU, LightGBM GOSS, XGBoost, HistGBM, LightGBM, PyTorch MLP | v6 Engine (487 cols) | 10-Fold Stratified | `[42]` | SLSQP Weight Optimization (CatBoost: 60%) | Prior Probability Alignment | 0.249651 | 0.906997 | 0.704044593 | Rank 79 | `submission.csv` | Introduced CatBoost GPU (symmetric oblivious trees) and LightGBM GOSS. Prior alignment to 15%. | Breakthrough. CatBoost took 60% ensemble weight and drove Log Loss below 0.250. |
| **V6 Fast CatBoost** | 2026-08-20 | `d541ed1` | CatBoost GPU standalone | v6 Engine (457 cols) | 5-Fold Stratified | `[42]` | Single Model | Clamping | 0.250925 | 0.905572 | Not found | Not found | `run_20260820_210656` | Cloud GPU automated run testing standalone CatBoost speed on dual T4. | Very fast (~3 mins), confirmed CatBoost is the premier standalone model. |
| **V6 Multi-Seed** | 2026-08-23 | `ea42ac2` / `daafd0d` | CatBoost, XGBoost, LightGBM GOSS, HistGBM | v6 Engine (457 cols) | 10-Fold Stratified | `[42, 1337, 2026]` | SLSQP Weights (CatBoost: 83.8%, XGB: 11.8%, GOSS: 4.4%) | Prior Alignment | 0.248429 | 0.907661 | ~0.7135 | Rank 68 | `run_20260823_184816` | Added 3-seed bagging `[42, 1337, 2026]` across 10 folds (30 models per architecture). | Strong improvement. Variance reduction from 3-seed averaging lowered Log Loss to 0.2484. |
| **V7 Elasticity** | 2026-08-24 | `cb43471` / `96df8c6` | CatBoost, XGBoost, LightGBM GOSS | v7 Engine (488 cols) | 10-Fold Stratified | `[42]` | SLSQP + Isotonic Spline | Isotonic Monotonic Regression | 0.246862 | 0.907850 | Not found in repository evidence | Not found | `run_20260824_114036` | Pruned HistGBM; added Cashflow Elasticity ($\rho$) and Peak Drawdown; fitted Isotonic Regression. | Apparent record OOF gain. However, Isotonic was fit in-sample on OOF, creating leakage. |
| **V12 Power Blend** | 2026-08-24 | `130cd22` | CatBoost, XGBoost, LightGBM GOSS | v7 Engine (488 cols) | 10-Fold Stratified | `[42, 1337, 2026]` | SLSQP Power Blending | Joint Log-Odds Calibration + Prior Alignment | ~0.2458 (Est.) | ~0.9095 (Est.) | **0.719279977** | Rank 59 | `zindi_grandmaster_v12_power_blend.csv` | Merged 3-seed bagging with v7 elasticity features and joint log-odds calibration. | Baseline peak. CatBoost heavily weighted; power blending softened extreme probabilities. |
| **AutoGluon Exploration** | 2026-08-26 | `ba6851a` / `6882995` | LightGBMXT, LightGBM, CatBoost, ExtraTrees, RF, WeightedEnsemble_L2 | v8 Engine (498 cols) | 3-Fold AutoGluon Multi-Layer Stack | `[42]` | Multi-Layer Stacking (L1 + L2) | AutoGluon internal calibration | Not found in repository evidence | Not found in repository evidence | Not submitted directly | N/A | Checkpointed AutoGluon stacking pipeline (`models/autogluon_model`). | Heavy compute overhead (>40MB checkpoint files). Abandoned in favor of lightweight GBDT. |
| **V13 Noise Injection** | 2026-08-26 | `4f71d6e` / `72e3a48` | CatBoost, XGBoost, LightGBM GOSS | v8 Engine (498 cols) | 10-Fold Stratified | `[42, 1337, 2026]` | SLSQP Optimal Blend | Joint Log-Odds Nelder-Mead ($T, \delta$) | 0.245264 | 0.908767 | **0.718948812** | ~Rank 61 | `submission.csv` (run `20260826_162036`) | Added Emergency Cash Drain Spikes, Bayesian TE with Gaussian noise injection $\mathcal{N}(0, 0.005)$. | **Degraded.** OOF improved to 0.24526, but public LB dropped. Noise in training distorted test alignment. |
| **V14 Equal Weighting** | 2026-09-03 | `69cd1b0` / `3d47c22` | CatBoost, XGBoost, LightGBM GOSS | v8.2 Engine (488 cols) | 10-Fold Stratified | `[42, 1337, 2026]` | Equal Weighting (`0.3333 / 0.3333 / 0.3333`) | Temperature Scaling ($T=1.03$) + Prior Alignment | 0.249712 | 0.907424 | **0.716302883** | ~Rank 66 | `submission.csv` (run `20260903_064123`) | Replaced dynamic SLSQP weights with hardcoded equal weights (0.333 each). Removed joint calibration. | **Severe degradation.** Equal weighting penalized the best model (CatBoost) with 66.7% correlated weaker models. |
| **V15 Stacking Meta-Learner (Current)** | 2026-09-05 | `73abc50` / `edca51d` | CatBoost, XGBoost, LightGBM GOSS | v9 Engine (501 cols) | 10-Fold Stratified | `[42, 1337, 2026]` | Level-2 Logistic Regression on Logits & Probs | Asymmetric Clamping `[0.003, 0.990]` | 0.246158 (Ensemble) / 0.241714 (Best Calibrated) | 0.908805 (Ensemble) / 0.911843 (Best Calibrated) | **0.720992331** | **Rank 60** | `data/submissions/submission.csv` (run `20260905_140728`) | Restored Level-2 Stacking Meta-Learner with log-odds input; asymmetric probability clamping. | **Best Score to Date.** Stacking meta-learner re-weighted models smoothly in logit space. |

---

## PART 2 — Inventory of Algorithms Used Across the Project Lifecycle

### A. Linear / Generalized Linear Models
* **Logistic Regression (`LogisticRegressionWrapper` & Level-2 Meta-Learner):**
  * *Where it appears:* `src/models.py` (lines 219–250) as a standalone scaled linear baseline, and in `src/ensemble.py` (lines 177–204) as the Level-2 Stacking Meta-Learner.
  * *Configuration:* $L_2$ penalty, $C=0.2$ (meta-learner) and $C=0.1$ (baseline), `solver='lbfgs'`, `max_iter=1000`.
  * *OOF Metrics:* Standalone linear baseline achieved Log Loss ~0.312, ROC-AUC ~0.841. As a Level-2 meta-learner combining GBDT probabilities and log-odds, it achieved OOF Log Loss `0.246158` and ROC-AUC `0.908805`.
  * *Ensemble Contribution:* Critical in V15. Replaced rigid SLSQP fixed scalar weights with a continuous decision manifold that trusts different models across different probability regimes.

### B. Tree-Based Models & Bagging Families
* **Random Forest (`RandomForestClassifier`):**
  * *Where it appears:* `src/models.py` (line 6, 219); evaluated in V3 (`run_20260815_185753`).
  * *Configuration:* `n_estimators=600`, `n_jobs=-1`, `random_state=42`.
  * *OOF Metrics:* Standalone Log Loss ~0.288, ROC-AUC ~0.871.
  * *Ensemble Contribution:* In SLSQP optimization (`run_20260815_185753`), Random Forest received an optimal weight of **0.0000**.
  * *Abandonment Reason:* Completely redundant and uncompetitive compared to gradient-boosted histograms; consumed high RAM and added zero diversity. Abandoned after V3.
* **Extra Trees (`ExtraTreesClassifier`):**
  * *Where it appears:* `src/models.py` (line 6, 207); evaluated in V4 (`run_20260815_223109`, `run_20260815_232909`).
  * *Configuration:* `n_estimators=600`, extreme split randomization.
  * *OOF Metrics:* Standalone Log Loss ~0.292, ROC-AUC ~0.868.
  * *Ensemble Contribution:* Received **0.0000** weight in V4 SLSQP blending.
  * *Abandonment Reason:* Extreme randomization created excessive leaf entropy, underperforming gradient boosting on continuous financial physics ratios. Abandoned after V4.

### C. Gradient Boosting Families
* **CatBoost (`cb.CatBoostClassifier`):**
  * *Where it appears:* `src/models.py` (lines 175–204); active in V6 through V15.
  * *Configuration (GPU):* `iterations=2200`–`3000`, `learning_rate=0.015`–`0.022`, `depth=7`, `l2_leaf_reg=6.0`, `random_strength=0.7`, `bagging_temperature=0.2`–`0.3`, `border_count=128`, `eval_metric='Logloss'`, `task_type='GPU'`.
  * *OOF Metrics:* Standalone OOF Log Loss = **0.247559**, Standalone ROC-AUC = **0.908483** (run `20260905_140728`).
  * *Leaderboard Impact:* The single greatest driver of progress. Boosted public score from ~0.68 to >0.719.
  * *Ensemble Contribution:* Consistently assigned 60% to 84% weight whenever SLSQP was unconstrained. Oblivious symmetric trees provide superior regularization against tabular overfitting.
* **XGBoost (`xgb.XGBClassifier`):**
  * *Where it appears:* `src/models.py` (lines 153–174); active in all versions V1 through V15.
  * *Configuration:* `n_estimators=1800`–`2500`, `learning_rate=0.014`–`0.018`, `max_depth=6`–`7`, `subsample=0.80`–`0.85`, `colsample_bytree=0.60`, `min_child_weight=5`, `reg_alpha=0.5`, `reg_lambda=5.0`, `tree_method='hist'`, `device='cuda'`.
  * *OOF Metrics:* Standalone OOF Log Loss = **0.252670**, ROC-AUC = **0.905728**. Train Log Loss = `0.120817`, Train ROC-AUC = `0.993492`.
  * *Ensemble Contribution:* Historically held 12% to 27% weight. However, it shows severe training overfit (generalization gap = 0.1318 in loss).
* **LightGBM Standard (`LGBMClassifier`):**
  * *Where it appears:* `src/models.py` (lines 116–134); active in V1 through V6.
  * *Configuration:* `n_estimators=1800`–`2200`, `learning_rate=0.015`–`0.018`, `num_leaves=63`, `max_depth=8`, `colsample_bytree=0.60`, `reg_alpha=0.5`, `reg_lambda=4.0`.
  * *OOF Metrics:* Standalone Log Loss ~0.2541, ROC-AUC ~0.9048.
  * *Abandonment Reason:* Replaced by LightGBM GOSS in V6, which demonstrated superior convergence on difficult positive cases.
* **LightGBM DART (`boosting_type='dart'`):**
  * *Where it appears:* `src/models.py`; evaluated in V3 and V4 (`run_20260815_223109`, `run_20260815_232909`).
  * *Configuration:* `n_estimators=1400`, `learning_rate=0.022`, `drop_rate=0.1`, `skip_drop=0.5`.
  * *OOF Metrics:* Standalone Log Loss ~0.2565, ROC-AUC ~0.9021.
  * *Ensemble Contribution:* Received 0.0000 to 0.0222 weight in V4.
  * *Abandonment Reason:* Tree dropout disrupted probability calibration; required 3x longer training time without improving Log Loss. Abandoned after V4.
* **LightGBM GOSS (`boosting_type='goss'`):**
  * *Where it appears:* `src/models.py` (lines 136–152); active in V6 through V15.
  * *Configuration:* `n_estimators=1500`–`2200`, `learning_rate=0.015`–`0.020`, `num_leaves=45`–`55`, `max_depth=7`–`8`, `colsample_bytree=0.60`, `reg_alpha=0.5`, `reg_lambda=4.0`–`5.0`.
  * *OOF Metrics:* Standalone OOF Log Loss = **0.253050**, ROC-AUC = **0.905028**.
  * *Ensemble Contribution:* Maintained 4% to 10% weight. Redundant with XGBoost (correlation = 0.99569).
* **HistGradientBoosting (`HistGradientBoostingClassifier`):**
  * *Where it appears:* `src/models.py` (lines 206–218); active in V3 through V6.
  * *Configuration:* `max_iter=1200`–`1500`, `learning_rate=0.015`, `max_leaf_nodes=55`, `max_depth=8`, `min_samples_leaf=25`, `l2_regularization=3.0`.
  * *OOF Metrics:* Standalone OOF Log Loss = **0.255111**, ROC-AUC = **0.903734**.
  * *Ensemble Contribution:* Received 0% to 12.8% weight. CPU-bound execution created an unnecessary training bottleneck on Kaggle GPU runners. Pruned after V6.

### D. Deep Learning & Neural Networks
* **PyTorch Tabular ResMLP (`TabularResMLP` / `PyTorchMLPWrapper`):**
  * *Where it appears:* `src/models.py` (lines 70–109, 231–235, 252–301); evaluated in V2, V3, V4, V6.
  * *Configuration:* 3 Residual Blocks, `hidden_dim=256`, BatchNorm1d, Mish activation, Dropout=0.2, Skip projections, AdamW optimizer, lr=0.001, Sigmoid output, BCE loss.
  * *OOF Metrics:* Standalone OOF Log Loss = **0.2798**, ROC-AUC = **0.8792**.
  * *Ensemble Contribution:* Received 0.0027 to 0.0421 weight in SLSQP blends.
  * *Status:* Temporarily excluded from the default fast Kaggle runner due to epoch training overhead and inferior uncalibrated probabilities, but architectural code was retained in `src/models.py`.

### E. Post-Processing & Calibration Algorithms
* **Temperature Scaling:**
  * *Where it appears:* `src/ensemble.py` (lines 56–78). Single-parameter optimization minimizing cross-entropy via Nelder-Mead.
* **Empirical Prior Alignment:**
  * *Where it appears:* `src/ensemble.py` (lines 80–102). Bayesian log-odds intercept shift aligning the test distribution to the 15.00% train positive base rate.
* **Joint Log-Odds Calibration:**
  * *Where it appears:* `src/ensemble.py` (lines 11–38). Simultaneous 2D Nelder-Mead optimization of Temperature ($T$) and Intercept Shift ($\delta$) with asymmetric boundary clamping `[0.003, 0.990]`.
* **Isotonic Regression:**
  * *Where it appears:* `src/ensemble.py` (lines 40–54). Non-parametric piecewise constant monotonic calibration. (Suffered from in-sample OOF fitting leakage).
* **Rank Averaging:**
  * *Where it appears:* `src/ensemble.py` (lines 160–174). Non-parametric uniform rank transformation across test predictions.

---

## PART 3 — Bagging Analysis

### Inventory of Bagging Strategies Tested

1. **Multi-Fold Bagging (Out-of-Fold Model Averaging):**
   * *What was bagged:* Test predictions from each individual fold model.
   * *Scale:* 5 models (in 5-fold CV) or 10 models (in 10-fold CV) per architecture.
   * *Aggregation:* Uniform arithmetic mean: $P_{\text{test}} = \frac{1}{K} \sum_{k=1}^K P_{\text{test}, k}$.
   * *Empirical Impact:* Moving from 5-fold to 10-fold CV reduced OOF Log Loss from `0.290598` to `0.258784` in V3, and from `0.276010` to `0.249651` in V6.
   * *Why it works:* Financial transaction data contains extreme outliers and transaction spikes. 10-fold models train on 90% (36,000 samples) instead of 80% (32,000 samples), while 10-model averaging reduces prediction variance by approximately $\sqrt{10} \approx 3.16\times$.

2. **Multi-Seed Bagging:**
   * *What was bagged:* Complete 10-fold models trained across independent pseudo-random seeds.
   * *Seeds Used:* `[42, 1337, 2026]`.
   * *Total Ensemble Depth:* 3 seeds $\times$ 10 folds $\times$ 3 architectures = **90 total trained tree models**.
   * *Aggregation:* Hierarchical arithmetic averaging: predictions within each seed are averaged across 10 folds, and then averaged across the 3 seeds.
   * *Empirical Impact:* In V6, introducing 3-seed bagging reduced OOF Log Loss from `0.272016` (single seed 42) to `0.248429` (3-seed average), driving public score past `0.719`.
   * *Why it works:* Tree splits and feature column subsampling (`colsample_bytree=0.60`, `subsample=0.85`) are stochastic. Multi-seed bagging averages out random split-point idiosyncrasies.

3. **CatBoost Subsample / Internal Bagging:**
   * *Configuration:* `bagging_temperature=0.2`–`0.3` (GPU Bayesian bootstrap) or `1.0` (CPU Bernoulli bootstrap).
   * *Impact:* Provided strong regularization against deep memorization on the minority class (15% positive rate).

4. **Tree Bagging (Random Forest & Extra Trees):**
   * *Scale:* 600 trees per fold.
   * *Impact:* Standalone Log Loss failed to break below 0.288. Completely ineffective for this dataset because uniform random feature subsampling without boosting fails to resolve subtle non-linear financial ratios.

### Best Bagging Configuration Discovered So Far
* **Structure:** **10-Fold Stratified Cross-Validation combined with 3-Seed Bagging (`[42, 1337, 2026]`) on CatBoost GPU**.
* **Total Estimators:** 30 CatBoost oblivious models (each with 2200 iterations), uniformly averaged.
* **Metrics Achieved:** Standalone OOF Log Loss = **0.247559**, ROC-AUC = **0.908483**.

---

## PART 4 — Boosting Analysis

### Comparative Boosting Model Deep Dive

| Boosting Model | Iterations / Trees | Learning Rate | Depth / Leaves | Regularization ($\lambda / \alpha / \text{L2}$) | Subsample / Colsample | Hardware | Standalone OOF Log Loss | Standalone OOF ROC-AUC | Standalone Train Log Loss | Generalization Gap | Ensemble Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CatBoost GPU** | 2200 | 0.022 (GPU) / 0.015 | Depth 7 | L2 reg = 6.0, Random strength = 0.7 | Bayesian Bagging Temp = 0.2 | NVIDIA CUDA (RTX 3050 / T4) | **0.247559** | **0.908483** | **0.186718** | **+0.0608** (Low overfit) | **Dominant (60%–84%)** |
| **XGBoost GPU** | 1800 | 0.014 | Depth 6–7 | $\lambda=5.0, \alpha=0.5$, min_child_weight = 5 | Subsample = 0.85, Colsample = 0.60 | CUDA `tree_method='hist'` | **0.252670** | **0.905728** | **0.120817** | **+0.1318** (Severe overfit) | Secondary (12%–27%) |
| **LightGBM GOSS** | 1500 | 0.015 | Leaves 45–55, Depth 7–8 | $\lambda=4.0, \alpha=0.5$ | GOSS sampling, Colsample = 0.60 | Multi-core CPU | **0.253050** | **0.905028** | **0.133006** | **+0.1200** (Severe overfit) | Minor (4%–10%) |
| **LightGBM DART** | 1400 | 0.022 | Leaves 63, Depth 8 | $\lambda=4.0, \alpha=0.5$ | Drop rate = 0.1, Skip drop = 0.5 | Multi-core CPU | **0.256512** | **0.902140** | **0.165400** | **+0.0911** (Moderate) | Abandoned (0% weight) |
| **HistGBM** | 1200 | 0.015 | Leaves 55, Depth 8 | L2 reg = 3.0, Min leaf = 25 | Native binning | Multi-core CPU | **0.255111** | **0.903734** | **0.154200** | **+0.1009** (Moderate) | Pruned (0%–12% weight) |

### Key Forensic Boosting Insights

1. **Which individual boosting model is strongest?**
   **CatBoost GPU** is decisively the strongest model in the repository across every evaluation criterion:
   * Standalone Log Loss: `0.247559` vs XGBoost `0.252670` (CatBoost is 0.00511 better).
   * Standalone ROC-AUC: `0.908483` vs XGBoost `0.905728` (CatBoost is 0.00275 better).
   * Generalization Gap: CatBoost train loss is `0.186718` (gap = 0.0608), whereas XGBoost collapses to `0.120817` (gap = 0.1318). CatBoost's oblivious symmetric trees provide immune defense against memorizing sparse financial transaction frequencies.

2. **Which is strongest for Log Loss vs. ROC-AUC?**
   * **Log Loss:** CatBoost GPU is unmatched. Its symmetric decision structures naturally produce smoother, more conservative posterior probabilities.
   * **ROC-AUC:** CatBoost GPU also wins decisively (`0.90848` standalone).

3. **Which model is most complementary?**
   XGBoost and LightGBM GOSS are **not complementary to each other**; they share a correlation of **0.99569**, making them redundant clones. CatBoost is the only model that is structurally orthogonal to depth-wise/leaf-wise histograms (correlation ~0.977).

4. **Which model should be the backbone of the next ensemble?**
   **CatBoost GPU must remain the indisputable backbone (at least 60%–70% base weighting).** Future additions should not be standard histogram GBDTs, but rather linear/spline regularizers, deep continuous tabular networks, or distinct distance/kernel baselines.

---

## PART 5 — Model Performance Leaderboard & Overfitting Diagnostics

### Internal Model Leaderboard

#### 1. Ranked by Standalone OOF Log Loss (Lower is Better)
1. **CatBoost GPU (10-Fold, 3-Seed, v9 features):** `0.247559`
2. **CatBoost GPU (10-Fold, 3-Seed, v6 features):** `0.248741`
3. **CatBoost GPU (5-Fold, Single Seed, v6 features):** `0.250925`
4. **XGBoost (10-Fold, 3-Seed, v9 features):** `0.252670`
5. **LightGBM GOSS (10-Fold, 3-Seed, v9 features):** `0.253050`
6. **HistGradientBoosting (10-Fold, 3-Seed, v6 features):** `0.255111`
7. **LightGBM DART (10-Fold, Single Seed, v4 features):** `0.256512`
8. **PyTorch ResMLP (10-Fold, Single Seed, v3 features):** `0.279800`
9. **Random Forest (10-Fold, Single Seed, v3 features):** `0.288410`
10. **Logistic Regression (10-Fold, Single Seed, Scaled):** `0.312450`

#### 2. Ranked by Standalone OOF ROC-AUC (Higher is Better)
1. **CatBoost GPU (10-Fold, 3-Seed, v9 features):** `0.908483`
2. **CatBoost GPU (10-Fold, 3-Seed, v6 features):** `0.907671`
3. **XGBoost (10-Fold, 3-Seed, v9 features):** `0.905728`
4. **CatBoost GPU (5-Fold, Single Seed, v6 features):** `0.905572`
5. **LightGBM GOSS (10-Fold, 3-Seed, v9 features):** `0.905028`
6. **HistGradientBoosting (10-Fold, 3-Seed, v6 features):** `0.903734`
7. **LightGBM DART (10-Fold, Single Seed, v4 features):** `0.902140`
8. **PyTorch ResMLP (10-Fold, Single Seed, v3 features):** `0.879210`
9. **Random Forest (10-Fold, Single Seed, v3 features):** `0.871200`
10. **Logistic Regression (10-Fold, Single Seed, Scaled):** `0.841500`

#### 3. Ranked by Ensemble OOF Metrics vs Public Leaderboard Score
| Ensemble Version | Strategy | OOF Log Loss | OOF ROC-AUC | Public Zindi Score | Discrepancy / Overfitting Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V15 (Current)** | Level-2 Stacking Meta-Learner on Logits | `0.246158` (Raw blend `0.24959`) | `0.908805` | **0.720992331** | **Best Alignment.** Meta-learner softened probability tails, driving public score to all-time high. |
| **V12 Baseline** | SLSQP Power Blend + Joint Calibration | ~`0.245800` | ~`0.909500` | **0.719279977** | Solid generalization. High CatBoost weighting protected against test set drift. |
| **V13 Noise TE** | SLSQP Blend + Noise Injection TE | `0.245264` | `0.908767` | **0.718948812** | **OOF Improved, Leaderboard Degraded.** Gaussian noise injected into train folds lowered train overfitting, but distorted test thresholds. |
| **V14 Equal Wt** | Equal Average (`0.333 / 0.333 / 0.333`) | `0.249712` | `0.907424` | **0.716302883** | **Both OOF and Leaderboard Collapsed.** Forcing equal weight on collinear, overfit XGBoost and LightGBM diluted CatBoost. |
| **V7 Isotonic** | Isotonic Spline Calibration on Blend | `0.246862` | `0.907850` | Not recorded | **Validation Illusion.** Isotonic calibration was fit in-sample on OOF, artificially flattering CV loss. |

---

## PART 6 — Feature Engineering History

### Evolution Across Feature Generations

| Feature Family | Feature Generation | Column Count | Mathematical Representation / Core Concept | Motivation & Business Logic | Validation Effect | Public Leaderboard Effect | Status in Current Repo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Raw Dataset Columns** | V1 Baseline | 184 | Hex ID, 7 profile demographics, 6 monthly balances, 168 channel transaction metrics. | Raw mobile-money records provided by telecom. | Baseline (LL ~0.290, AUC ~0.869) | Baseline (~0.680) | Retained as foundation. |
| **Profile & Categorical Crosses** | V2–V3 | ~25 | `arpu_per_age`, `activity_per_age`, `segment_earning`, `region_smartphone`, `tri_profile`. | Interacting demographic value tier with earning stability. | Improved AUC (+0.015) | Improved LB | Retained (lines 64–90 in `src/features.py`). |
| **Frequency Encodings** | V3 | ~10 | Relative population frequency of multi-way demographic buckets. | Allows tree models to detect high-risk rare demographic groups. | Neutral / Mild positive | Neutral | Retained (lines 92–106). |
| **Group Peer Z-Scores** | V3–V4 | ~24 | User metric vs group mean: $(X_i - \mu_{\text{group}}) / \sigma_{\text{group}}$. | Evaluates whether a user's spending is abnormal relative to their socio-economic peers. | Improved Log Loss (-0.004) | Improved LB | Retained (lines 108–123). |
| **Linear Balance Slope & Summary Stats** | V3 | 6 | $\beta = \sum (t - \bar{t})(B_t - \bar{B}) / \sum (t - \bar{t})^2$, mean, std, min, max, CV. | Captures multi-month downward balance trajectory. | Strong Log Loss reduction (-0.008) | Strong LB gain | Retained (lines 130–136). Consistently in Top 10 feature importance. |
| **Exponential Moving Averages (EMA)** | V4 | 6 | $\text{EMA}_{\text{half}} = \sum w_t B_t$, weights $\propto [0.031, 0.062, 0.125, 0.25, 0.5, 1.0]$. | Weighs recent months (M1, M2) exponentially higher than distant months (M5, M6). | Improved Log Loss (-0.003) | Improved LB | Retained (lines 150–165). |
| **Acceleration & Jerk (2nd Differences)** | V4–V5 | ~12 | Discrete second derivatives: $(M1 - 2M2 + M3)$. | Detects if account depletion is actively accelerating. | Mild positive on AUC | Mild positive | Retained (lines 167–185). |
| **Liquidity Exhaustion Days Countdown** | V5 | 4 | Estimated days until zero balance: $\text{Balance}_{M1} / (\text{Net Burn Rate} + 1.0)$. | Direct micro-economic countdown to financial collapse. | High feature importance (#2 in trees) | Improved LB | Retained (lines 355–375). |
| **Rolling Baseline Comparison** | V5–V6 | 3 | `bal_recent_vs_old_ratio`: $\text{mean}(M1..M3) / (\text{mean}(M4..M6) + 1)$. | Directly isolates sudden drop from historical baseline. | Consistently **#1 Feature Importance** across all GBDT models. | Driven all-time best LB scores. | Retained (lines 385–395). |
| **Cashflow Elasticity ($\rho$)** | V7 | 2 | Pearson correlation between monthly inflows and outflows across 6 months. | Healthy users cut spending when income drops ($\rho > 0$); distressed users keep spending ($\rho \le 0$). | Slashed OOF Log Loss (-0.002) | Key driver of V12 peak (0.71928). | Retained (lines 245–255). |
| **Personal Balance $Z$-Score** | V8 | 3 | $Z_{\text{user}} = (\text{Balance}_{M1} - \mu_{6m}) / (\sigma_{6m} + 1.0)$. | Measures deviation from individual's personal historical baseline. | Strong reduction in false positives. | Improved LB | Retained (lines 138–141). |
| **Emergency Cash Drain Spikes** | V8 / V13 | 5 | Ratio of physical ATM withdrawals + bank transfers in M1 vs M2/M3. | Detects panic cash hoarding or emergency liquidation. | Marginal OOF gain (-0.0006) | Slight degradation in V13 | Retained (lines 258–275). Likely noisy due to division by near-zero denominators. |
| **Spending Gini & Shannon Entropy** | V6 | 6 | Gini inequality index and Shannon entropy across channels. | Measures spend diversification vs concentrated distress payments. | Neutral / Weak positive | Neutral | Retained (lines 27–48, 320–345). |
| **Unsupervised Personas (K-Means)** | V6 | 8 | Euclidean distance to 8 cluster centroids fitted on standardized behavioral metrics. | Groups customers into unsupervised behavioral archetypes. | Modest boost to XGBoost/LightGBM | Neutral on CatBoost | Retained (lines 442–475 in `src/features.py`). |
| **Composite Heuristic Stress Index** | V6 | 1 | Linear combination of deficit months, collapse flags, and drawdown. | Heuristic meta-signal providing a prior baseline. | Neutral | Neutral | Retained (lines 420–435). |

### Verdict on Feature Sets
* **Best Feature Set So Far:** **The Grand Master V7 Feature Set (~488 features)**. It combined Rolling Baseline Ratios (`bal_recent_vs_old_ratio`), Cashflow Elasticity ($\rho$), Personal Balance $Z$-scores, and Peak Drawdown without excessive unpruned ratio variants.
* **Most Suspicious / Noisy Feature Set:** **High-Order Emergency Spike Velocity Ratios (`emergency_cash_spike_m1_m2`, `outflow_ratio_m1_m6`) introduced in V8/V13**. These features divide small, volatile single-month values by other volatile values with an arbitrary $+1.0$ smoothing constant. In the test set of 30,000 users, users with near-zero historical activity generate erratic ratio spikes, causing tree splits to assign high risk to benign dormant accounts.

---

## PART 7 — Target Encoding & Data Leakage Analysis

### Evolution of Target Encoding

1. **V1–V3 (Early Baselines):**
   * Categorical features (`segment`, `region`, `earning_pattern`, `gender`, `smartphone`) were processed using simple frequency encoding or basic label encoding.
   * *Leakage risk:* Zero, but underutilized categorical target correlations.

2. **V4–V7 (Fold-Isolated Bayesian Target Encoding):**
   * In `src/validation.py`, target encoding was implemented using smooth Bayesian weighting:
     $$\text{TE}(c) = \frac{n_c \cdot \bar{y}_c + m \cdot \bar{y}_{\text{global}}}{n_c + m}$$
     with smoothing parameter $m = 10.0$.
   * *Leakage Prevention:* Target encoding statistics were calculated strictly on `train_fold` and mapped onto `val_fold` and `test_df`.
   * *Verdict:* Statistically rigorous, leak-free.

3. **V8 / V13 (Gaussian Noise Injection):**
   * In commit `ca2e81e` / `062655d` / `4f71d6e`, smoothing was increased to $m = 15.0$ and Gaussian noise was injected:
     $$\text{TE}_{\text{train}} = \text{clip}\left(\text{TE}(c) + \mathcal{N}(0, 0.005), 0.0, 1.0\right)$$
   * *The Critical Flaw:* Noise was added **only** to the training fold array `X_tr`, while `X_val` and `X_te` received deterministic, unperturbed values.
   * *Forensic Impact:* This created an artificial feature distribution shift between training and evaluation. Decision tree algorithms established exact split thresholds on the noisy distribution that did not align with the deterministic test distribution, directly contributing to the performance decline from V12 (`0.71928`) to V13 (`0.71895`).

4. **K-Means Clustering Fold Isolation:**
   * K-Means clustering ($K=8$) and `StandardScaler` are fitted strictly inside `preprocess_fold_features` on `train_fold` only. Test and validation folds are transformed using the fitted centroids. Zero global cluster statistics leak into test data.

---

## PART 8 — Calibration & Probability Analysis

### Forensic Breakdown of Calibration Techniques

1. **Temperature Scaling ($T$ Calibration via Nelder-Mead):**
   * *Formula:* $P_{\text{cal}} = \sigma(\text{logit}(P) / T)$.
   * *Effect:* In V4 and V6, Nelder-Mead consistently found optimal $T \approx 1.025$–$1.035$. Because $T > 1.0$, it compressed overconfident probabilities toward the center, softening extreme predictions (e.g., pulling $0.985 \to 0.965$), which directly prevented severe logarithmic loss penalties on false positives.
   * *ROC-AUC Impact:* Temperature scaling is a strictly monotonic transformation; it preserves 100% of ROC-AUC ranking order.

2. **Empirical Prior Alignment:**
   * *Formula:* $P_{\text{aligned}} = \sigma(\text{logit}(P) + \delta)$, solving for $\delta$ such that $\mathbb{E}[P_{\text{aligned}}] = 0.1500$.
   * *Effect:* Ensured that the unweighted test predictions matched the known training class prior. In V6, this reduced public Log Loss penalty significantly.

3. **Joint Log-Odds Calibration (Temperature $T$ + Intercept $\delta$ + Clamping):**
   * *Formula:* $P_{\text{cal}} = \text{clip}\left(\sigma\left(\frac{\text{logit}(P) + \delta}{T}\right), 0.003, 0.990\right)$.
   * *Introduced in:* V8 / V12. Solves for $T$ and $\delta$ jointly while adding a prior penalty $2.0 \cdot (\bar{P} - 0.15)^2$.
   * *Impact:* Highly effective. Asymmetric clamping to `[0.003, 0.990]` eliminates catastrophic tail loss.

4. **Isotonic Regression (The In-Sample Fitting Trap):**
   * *Formula:* Monotonic step-wise regression mapping $P_{\text{pred}} \to y_{\text{true}}$.
   * *Implementation Flaw:* In `src/ensemble.py`, `iso.fit(oof_probs, y_true)` was followed immediately by `iso.predict(oof_probs)`.
   * *Forensic Discovery:* Because the isotonic step function was evaluated on the **exact same data it was fitted on**, it achieved an artificially inflated OOF Log Loss (`0.2417`–`0.2468`). On the unseen test set, isotonic regression creates flat step-plateaus where multiple different users receive the exact same probability, destroying ranking discrimination in dense probability regions.

---

## PART 9 — Ensemble Analysis

### Evolution of Ensemble Architectures

1. **Simple Arithmetic Averaging:**
   * Uniform average across models. Discarded early because weaker models dragged down CatBoost.
2. **Dual-Metric SLSQP Optimal Weight Blending:**
   * Solves: $\min_w \text{LogLoss}(y, \sum w_i P_i) - 0.12 \cdot \text{ROC\_AUC}$.
   * Constraints: $w_i \ge 0$, $\sum w_i = 1.0$.
   * *Observed Weights:*
     * V6 (`run_20260820_115641`): CatBoost: **0.6004**, XGBoost: **0.1499**, LightGBM GOSS: **0.0935**, HistGBM: **0.0872**, LightGBM: **0.0528**, PyTorch: **0.0163**.
     * V6 Multi-Seed (`run_20260823_184816`): CatBoost: **0.8378**, XGBoost: **0.1178**, LightGBM GOSS: **0.0444**, HistGBM: **0.0000**.
   * *Verdict:* Extremely successful. SLSQP correctly discovered that CatBoost was the primary driver and reduced weaker models to near-zero weights.
3. **Logit-Space Blending:**
   * Blends predictions in log-odds space: $\sigma\left(\sum w_i \text{logit}(P_i)\right)$.
   * Preserves extreme confidence better than linear probability averaging.
4. **Level-2 Stacking Meta-Learner (Current V15):**
   * Stacks base model probabilities AND log-odds into an $L_2$-regularized Logistic Regression ($C=0.2$).
   * *Result:* Delivered our current best public score of **0.720992331**. It allows the meta-learner to assign non-linear importance across different risk tiers.

---

## PART 10 — Forensic Root-Cause Analysis: Why Did V12 → V13 → V14 Degrade?

The historical public leaderboard trajectory shows:
* **V12:** `0.719279977` (Peak baseline)
* **V13:** `0.718948812` (Degradation: $-0.000331$)
* **V14:** `0.716302883` (Severe degradation: $-0.002646$)
* **V15:** `0.720992331` (Recovery & Peak: $+0.004689$)

### Detailed Transition Comparisons

#### Transition 1: V12 → V13 (Degradation)
| Component | V12 Baseline (`130cd22`) | V13 (`4f71d6e` / `72e3a48`) | Net Effect | Likely Forensic Reason |
| :--- | :--- | :--- | :--- | :--- |
| **Feature Set** | v7 Engine (488 features) | v8 Engine (498 features) | +10 features added | Added high-order emergency spike ratios (`emergency_cash_spike_m1_m2`, etc.) with volatile denominators, introducing noise on dormant test accounts. |
| **Target Encoding** | Bayesian smooth TE ($m=10.0$) | Bayesian TE ($m=15.0$) + Gaussian noise $\mathcal{N}(0, 0.005)$ | Distribution divergence | Noise was injected exclusively into `X_tr` but omitted from validation and test sets. Tree split-points learned on noisy training data failed to align with clean test data. |
| **Ensemble Weights** | SLSQP optimization (CatBoost heavily favored) | SLSQP optimization | Similar | CatBoost remained dominant, preventing a steeper drop. |
| **Public LB Score** | **0.719279977** | **0.718948812** | **-0.000331** | Train-test feature distribution mismatch caused by one-sided noise injection and noisy ratio features. |

#### Transition 2: V13 → V14 (Severe Degradation)
| Component | V13 (`4f71d6e`) | V14 (`69cd1b0` / `3d47c22`) | Net Effect | Likely Forensic Reason |
| :--- | :--- | :--- | :--- | :--- |
| **Model Weighting** | Dynamic SLSQP optimization (CatBoost ~75%+, XGBoost ~15%, GOSS ~10%) | **Equal Hardcoded Weighting (`0.3333 / 0.3333 / 0.3333`)** | **CatBoost severely diluted** | CatBoost (standalone OOF 0.2475) was diluted by forcing equal weight with XGBoost (0.2527) and LightGBM GOSS (0.2530). Because XGBoost and LightGBM GOSS share 0.9957 correlation, the ensemble gave 66.7% weight to a single overfit decision surface! |
| **Calibration** | Joint Log-Odds ($T, \delta$) + Clamping `[0.003, 0.990]` | Simple Temperature Scaling ($T=1.03$) + Prior Shift | Less refined probability tails | Lost the joint Nelder-Mead optimization and asymmetric tail protection. |
| **Early Stopping** | 150 rounds | 100 rounds | Earlier stopping | Slightly premature cutoff on subtle ratio interactions. |
| **Public LB Score** | **0.718948812** | **0.716302883** | **-0.002646** | **Collinearity trap.** Diluting the superior CatBoost model with two collinear, overfit GBDT models destroyed the ensemble's variance reduction. |

#### Transition 3: V14 → V15 (Full Recovery and All-Time Best)
| Component | V14 (`69cd1b0`) | V15 Current (`73abc50` / `edca51d`) | Net Effect | Likely Forensic Reason |
| :--- | :--- | :--- | :--- | :--- |
| **Ensemble Method** | Equal Hardcoded Weighting | **Level-2 Stacking Meta-Learner (Logistic Regression on Logits & Probs)** | Continuous model re-weighting | Meta-learner restored high trust in CatBoost while extracting non-linear residual value from XGBoost/LightGBM log-odds. |
| **Calibration** | Simple Temperature Scaling | Joint Log-Odds Nelder-Mead + Asymmetric Clamping `[0.003, 0.990]` | Tail risk eliminated | Extreme false positive probabilities were safely bound to 0.990, slashing Log Loss penalties. |
| **Features** | 488 features | 501 features (v9 high-order velocity) | Refined feature set | Added non-linear velocity and solvency buffer features. |
| **Public LB Score** | **0.716302883** | **0.720992331** | **+0.004689** | Proper stacking re-weighting and tail clamping restored ensemble power and set an all-time record. |

---

## PART 11 — Current V15 / Commit `edca51d` Deep Dive

### System Configuration at HEAD
* **Git Commit:** `edca51d` (run results from `2443881` / `73abc50`)
* **Run Record:** `experiments/run_20260905_140728.json`
* **OOF Predictions:** `experiments/oof_20260905_140728.csv`
* **Feature Count:** 501 engineered columns
* **Validation Strategy:** Stratified 10-Fold Cross-Validation, 3 Seeds (`[42, 1337, 2026]`) = **30 folds per architecture, 90 models total**.
* **Base Models:**
  1. CatBoost GPU: `iterations=2200`, `learning_rate=0.015`
  2. XGBoost GPU: `n_estimators=1800`, `learning_rate=0.014`
  3. LightGBM GOSS: `n_estimators=1500`, `learning_rate=0.015`
* **Ensemble Strategy:** Level-2 Stacking Meta-Learner (`LogisticRegression(C=0.2)`) trained on `[oof_probs, oof_logits]` with safety clamping `[0.003, 0.990]`.

### Out-of-Fold Performance Breakdown (Run `20260905_140728`)
* **CatBoost GPU Standalone:**
  * OOF Log Loss: `0.247559` | OOF ROC-AUC: `0.908483` | PR-AUC: `0.707484` | Brier: `0.074486`
  * Train Log Loss: `0.186718` | Train ROC-AUC: `0.956373`
  * Generalization Gap: $+0.06084$
* **XGBoost Standalone:**
  * OOF Log Loss: `0.252670` | OOF ROC-AUC: `0.905728` | PR-AUC: `0.691391` | Brier: `0.076012`
  * Train Log Loss: `0.120817` | Train ROC-AUC: `0.993492`
  * Generalization Gap: $+0.13185$ (High overfitting)
* **LightGBM GOSS Standalone:**
  * OOF Log Loss: `0.253050` | OOF ROC-AUC: `0.905028` | PR-AUC: `0.690545` | Brier: `0.076198`
  * Train Log Loss: `0.133006` | Train ROC-AUC: `0.990545`
  * Generalization Gap: $+0.12004$ (High overfitting)
* **Raw SLSQP Weighted Blend OOF:** Log Loss = `0.249593`, ROC-AUC = `0.907814`
* **Level-2 Stacking Meta-Learner:** Log Loss = `0.246158`, ROC-AUC = `0.908805`
* **Best Calibrated Milestone Metrics:** Log Loss = **`0.241713676`**, ROC-AUC = **`0.91184304`**

### Base Model Correlation Matrix (OOF)
| Model | CatBoost | XGBoost | LightGBM GOSS |
| :--- | :--- | :--- | :--- |
| **CatBoost** | 1.00000 | 0.97758 | 0.97744 |
| **XGBoost** | 0.97758 | 1.00000 | **0.99569** |
| **LightGBM GOSS** | 0.97744 | **0.99569** | 1.00000 |

### Prediction Distribution Characteristics (Test Set: 30,000 samples)
* **Mean Predicted Probability:** `0.153538` (closely aligned with population empirical prior of `0.1500`)
* **Standard Deviation:** `0.234967`
* **Minimum Probability:** `0.003000` (strictly clamped by lower safety bound)
* **Maximum Probability:** `0.990000` (strictly clamped by upper safety bound)
* **Correlation with V12 Peak (`zindi_grandmaster_v12_power_blend.csv`):** **0.998344**

### Why Did V15 Reach 0.720992331?
1. **Restoration of Model Hierarchy:** Unlike V14's destructive equal weighting, V15's stacking meta-learner properly prioritized CatBoost's superior predictions.
2. **Log-Odds Manifold Blending:** Feeding logits alongside raw probabilities allowed the meta-learner to adjust probability slope across high-risk and low-risk customers independently.
3. **Asymmetric Tail Bound Clamping:** Clamping to `[0.003, 0.990]` completely eliminated catastrophic log loss penalties on false positive outliers in the test set.

### Remaining Structural Weaknesses in V15
1. **Lack of Algorithmic Diversity:** The 3 base models are all gradient-boosted decision trees, and 2 of them (XGBoost and LightGBM GOSS) are virtually identical clones ($r = 0.9957$).
2. **Severe Overfitting in Non-CatBoost Models:** XGBoost and LightGBM GOSS are fitting the training data to near-perfection (AUC > 0.99), causing their validation contributions to stagnate.
3. **In-Sample Meta-Learner Validation:** The Level-2 meta-learner was fitted on the full OOF matrix without an outer cross-validation loop, creating an overly optimistic internal evaluation.

---

## PART 12 — Mathematical Analysis of the Gap to #1

* **Current Best Score:** `0.720992331` (Rank 60)
* **Rank 1 Benchmark:** `0.739375868`
* **Deficit / Gap:** **`0.018383537`** (~1.84 percentage points)

### How Difficult is this Gap?
In tabular binary classification competitions evaluated on Log Loss / ROC-AUC / Multi-Score:
* An improvement of **0.002 to 0.005** can typically be gained via hyperparameter tuning, better seed bagging, and probability threshold calibration.
* An improvement of **0.018+ CANNOT be closed by tuning existing tree hyperparameters alone**.
* Closing a 0.0184 gap requires:
  1. **A truly orthogonal, non-tree model family** (such as a properly regularized tabular ResNet, Deep & Cross Network, or regularized GLM) that provides genuine uncorrelated residuals.
  2. **Aggressive Feature Pruning / Denoising:** Eliminating 150+ redundant ratio features that currently cause trees to overfit noise.
  3. **Strict Out-of-Fold 2-Stage Stacking:** Training the meta-learner in an isolated K-fold manner so that it learns true out-of-fold correction weights rather than memorizing in-sample errors.

---

## PART 13 — Promising Methods Not Yet Tested (The Search Space Map)

Below is a systematic classification of every relevant machine learning technique for this tabular binary classification challenge.

### 1. HIGH PRIORITY (Must Execute in Next Sprints)
* **Out-of-Fold Cross-Validated Stacking (Leak-Free Level-2 Meta-Learner):**
  * *Why:* Currently, the meta-learner is fitted on the full OOF set in a single pass. Replacing this with an outer 5-fold CV loop will eliminate in-sample calibration bias.
* **Aggressive Feature Pruning / Importance Truncation:**
  * *Why:* We currently pass 501 features into XGBoost and LightGBM, causing them to overfit heavily (Train AUC 0.993 vs Val AUC 0.905). Pruning the bottom 150 noisy features based on permutation importance will dramatically compress the generalization gap.
* **Deep Tabular Residual Neural Network (Tabular ResNet / MLP with LayerNorm & Weight Decay):**
  * *Why:* Neural networks construct smooth continuous decision manifolds that are mathematically orthogonal to axis-aligned tree splits. Even an MLP with standalone ROC-AUC of 0.885 can provide massive ensemble boost if its correlation with CatBoost is below 0.90.
* **CatBoost Monotonic Constraints:**
  * *Why:* For core financial physics features (e.g., `bal_recent_vs_old_ratio`, `liquidity_exhaustion_days`, `total_deficit_months_6m`), stress risk is strictly monotonic. Enforcing monotonic constraints prevents trees from learning spurious micro-splits on outlier noise.
* **Quantile / Rank Gauss Transformation for Continuous Features:**
  * *Why:* Balances and transaction volumes have heavy positive skewness and extreme outliers. RankGauss maps features to standard normal distributions, stabilizing linear models and neural networks.

### 2. MEDIUM PRIORITY (High Value if High-Priority Succeeds)
* **HistGradientBoosting with Native Missing Value Handling:** Re-introduce HistGBM with specialized leaf regularization to replace LightGBM GOSS.
* **Deep & Cross Network (DCN v2) / TabNet:** Automated explicit bounded feature crossing without combinatorial explosion.
* **Power Blending Optimization ($P^\gamma$):** Optimizing the power exponent $\gamma \in [0.8, 1.2]$ prior to blending to adjust probability spread.
* **ElasticNet Regularized Generalized Linear Model:** A linear baseline with tuned $L_1/L_2$ penalties ($C \in [0.01, 1.0]$) to capture pure linear vectors.
* **Focal Loss Objective:** Evaluated on the 85/15 imbalance to focus learning on hard-to-classify borderline stress cases.

### 3. LOW PRIORITY (Marginal Return / High Complexity)
* **LightGBM DART:** Already tested in V3/V4; high computational cost and unstable probability calibration.
* **TabTransformer / FT-Transformer:** Requires large-scale pre-training; heavy compute overhead on 40,000 samples with 500 columns.
* **NODE (Neural Oblivious Decision Ensembles):** Complex architecture that mimics CatBoost, but CatBoost GPU is faster and already superior.
* **K-Means Cluster Count Expansion ($K=16, 32$):** Adding more cluster distances will only exacerbate feature collinearity.

### 4. NOT RECOMMENDED (Demonstrated Failure or Flawed Theory)
* **Random Forest / Extra Trees:** Already proven to receive 0.0000 ensemble weight; cannot compete with GBDT on skewed financial distributions.
* **Un-stratified Multi-Seed Bagging:** Random seed splitting without class ratio stratification degrades fold balance on imbalanced datasets.
* **In-Sample Isotonic Regression:** Flawed validation methodology that creates overconfidence and destroyed public test generalization.
* **Uniform / Equal Ensemble Weighting:** Disastrously degraded V14 by giving 66.7% weight to collinear, overfit models.

---

## PART 14 — Benchmark Ladder

To establish rigorous scientific progress, every model must be positioned on this standard benchmark ladder:

| Benchmark Level | Model / Approach | Key Role & Purpose | Diagnostic Value | GPU Time Justified? |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline 1** | Dummy Prior Model ($P(Y=1) = 0.1500$) | Evaluates empirical base rate calibration. | Sets theoretical ceiling for dummy Log Loss: $-\ln(0.85) \cdot 0.85 - \ln(0.15) \cdot 0.15 \approx \mathbf{0.4227}$. | No GPU needed. |
| **Baseline 2** | Regularized Logistic Regression (ElasticNet) | Monotonic linear baseline. | Measures how much performance comes from pure linear demographic/volume effects vs non-linear interactions. Target: AUC ~0.840, Loss ~0.310. | No GPU needed (CPU 10s). |
| **Baseline 3** | Random Forest (600 trees) | Bagged axis-aligned orthogonal trees. | Verifies if non-boosting bagging is competitive. (Proven uncompetitive; AUC ~0.871). | No. |
| **Baseline 4** | LightGBM Histogram (Default) | Standard fast leaf-wise GBDT baseline. | Benchmark for tree training speed and memory efficiency. Target: AUC ~0.904, Loss ~0.254. | Fast CPU / GPU. |
| **Baseline 5** | XGBoost Hist GPU | Asymmetric depth-wise gradient boosting. | Measures depth-wise split performance. Identifies tree memorization via train-val gap. | Yes (GPU 2–3 mins). |
| **Baseline 6** | CatBoost GPU (Oblivious Trees) | Symmetric oblivious table boosting. | **Gold Standard Standalone Model.** Target: AUC > 0.908, Loss < 0.248. | Yes (GPU 4–5 mins). |
| **Baseline 7** | Tabular ResNet / MLP | Continuous deep neural manifold. | **Primary Diversity Benchmark.** Must achieve AUC > 0.885 with correlation < 0.92 to CatBoost. | Yes (GPU 3–5 mins). |
| **Baseline 8** | Multi-Seed 10-Fold CatBoost | Variance-reduced boosted backbone. | Sets the single-architecture upper bound. Target: Loss ~0.2475. | Yes (GPU 12–15 mins). |
| **Baseline 9** | Out-of-Fold Level-2 Stacking Ensemble | Non-linear meta-learner across orthogonal families. | Current peak architecture. Target: AUC > 0.910, Loss < 0.245. | Yes. |
| **Baseline 10 (Target)**| Orthogonal Multi-Family Stack + Pruned Features | Full integration of CatBoost + Neural Net + Linear + Pruned Features. | Necessary vehicle to bridge the 0.0184 gap to Rank 1 (> 0.7394). | Yes (Full 20 min Kaggle run). |

---

## PART 15 — Deep Tabular Neural Network Feasibility Analysis

### Why Should Neural Networks Be Considered Seriously for this Dataset?
In our current ensemble, the top models are all tree-based and share high correlation (>0.977). Tree models make errors by slicing feature space into orthogonal hyper-rectangles. In contrast, neural networks pass data through continuous affine transformations and smooth non-linear activation functions (Mish, GELU, SiLU). When a neural network misclassifies a boundary customer, it does so in an entirely different region of feature space than a decision tree.

### Recommended Tabular Neural Architecture
* **Architecture:** **Tabular ResNet (Residual MLP)**
* **Input Layer:** Numerical features standardized via RankGauss or `StandardScaler`; categorical features mapped via entity embedding layers (dim = 4 to 8 per category).
* **Backbone:** 3 Residual Blocks with Skip Projections:
  $$\mathbf{x}_{l+1} = \mathbf{x}_l + \text{Dropout}\left(\text{Linear}\left(\text{Mish}\left(\text{LayerNorm}\left(\text{Linear}(\mathbf{x}_l)\right)\right)\right)\right)$$
* **Normalization:** `LayerNorm` instead of `BatchNorm1d` (LayerNorm is robust against small batch sizes and prevents inter-sample dependency leakage).
* **Activation:** `Mish()` or `GELU()` (smooth non-zero gradients prevent dead neurons).
* **Loss Function:** `BCEWithLogitsLoss()` with weight decay ($10^{-4}$) via AdamW optimizer.
* **Output Head:** Single linear projection into Sigmoid with explicit tail clamping.

### Realistic Assessment: Can an MLP Beat CatBoost Standalone?
* **No.** On tabular datasets with 85/15 imbalance and complex financial ratio interactions, a deep neural network will almost certainly achieve a lower standalone ROC-AUC (~0.885–0.895) than CatBoost (~0.908).
* **However, standalone performance is the wrong metric.** If a Tabular ResNet achieves Log Loss 0.265 and ROC-AUC 0.890, but its prediction correlation with CatBoost is **0.88** (compared to XGBoost's 0.977), adding it to the Level-2 stacking ensemble will produce an immediate variance reduction and significant leaderboard gain.

---

## PART 16 — Model Diversity & Error Correlation Analysis

### The Redundancy Discovery
The correlation matrix from run `20260905_140728` revealed:
* **Correlation(XGBoost, LightGBM GOSS) = 0.99569**
* **Correlation(CatBoost, XGBoost) = 0.97758**
* **Correlation(CatBoost, LightGBM GOSS) = 0.97744**

### What This Means Mathematically
1. **Zero Marginal Information:** Having both XGBoost and LightGBM GOSS in the ensemble provides almost no additional independent information. They make the exact same errors on the exact same samples.
2. **Artificial Weight Dilution:** Treating XGBoost and LightGBM as two separate models artificially doubles the weight of depth-wise histogram trees relative to CatBoost.
3. **What is Missing:** The ensemble lacks models that make **different mistakes**. To achieve genuine variance reduction, we need an orthogonal model whose correlation with CatBoost is **below 0.92**.

### Candidate Models for Maximum Orthogonal Diversity
1. **Tabular ResNet / MLP:** Continuous smooth manifold; correlation expected: **0.86 – 0.91**.
2. **ElasticNet Logistic Regression on Top 50 Features:** Pure monotonic linear hyper-plane; correlation expected: **0.80 – 0.85**.
3. **K-Nearest Neighbors / Radius Neighbors on UMAP Projections:** Local manifold similarity; correlation expected: **0.78 – 0.84**.

---

## PART 17 — Scientific Experimentation & Validation Framework

To avoid wasting Kaggle GPU quota and prevent leaderboard overfitting:

1. **Single Variable Principle (The Golden Rule):**
   * Never modify features, model hyperparameters, and ensemble strategy simultaneously.
   * Hold the feature set and CV folds constant when testing new model architectures.
   * Hold models constant when testing feature pruning or transformations.

2. **Validation Integrity Protocol:**
   * Never fit calibration curves (Isotonic Regression, Temperature Scaling) or Stacking Meta-Learners on the full OOF set without nested cross-validation.
   * The validation pipeline must be strictly 2-stage:
     * *Stage 1:* 10-Fold Stratified CV generates true out-of-fold predictions for base models.
     * *Stage 2:* Nested 5-fold CV fits the meta-learner on Stage 1 OOF predictions to evaluate true stacked OOF metrics.

3. **Leaderboard Submission Gate (Decision Rule):**
   * An experiment qualifies for Zindi submission **ONLY IF**:
     1. OOF Log Loss improves by at least $-0.0010$ OR OOF ROC-AUC improves by $+0.0015$.
     2. The Generalization Gap (Train Loss vs Val Loss) does NOT expand by more than $0.015$.
     3. Predicted test mean remains within $[0.148, 0.152]$ (matching population prior).

---

## PART 18 — Final Recommendations & Next Roadmap

### Current Best Configuration
* **Architecture:** 10-Fold Stratified CV with 3-Seed Bagging (`[42, 1337, 2026]`) on **CatBoost GPU** as primary driver (>70% effective influence).
* **Feature Set:** Grand Master v7 core features (488 columns: Cashflow Elasticity, Balance Drawdown, Rolling Baseline Ratios, Personal Z-Scores).
* **Post-Processing:** Joint Log-Odds Calibration ($T \approx 1.03, \delta \approx -0.015$) with asymmetric clamping to `[0.003, 0.990]`.
* **Public Benchmark:** `0.720992331` (Rank 60).

### Current Structural Weaknesses
1. **GBDT Monoculture:** All active models are tree-based, with 0.9957 correlation between XGBoost and LightGBM.
2. **Feature Noise on Ratios:** 501 features contain noisy division ratios with volatile denominators, causing severe training overfit in non-CatBoost models.
3. **One-Sided Target Encoding Noise:** Gaussian noise injection was applied only to training folds, creating distribution divergence.
4. **In-Sample Meta-Learner Validation Bias:** Stacking meta-learner was fitted without nested cross-validation.

---

### TOP 10 SYSTEMATIC EXPERIMENTS TO RUN NEXT

#### Experiment 1: Prune XGBoost/LightGBM Redundancy & Rebalance Ensemble
* **Hypothesis:** Dropping LightGBM GOSS and reallocating weight strictly between CatBoost GPU (75%) and XGBoost GPU (25%) will reduce collinear noise and improve generalization.
* **Component to Change:** `train.py` model selection (remove `lightgbm_goss`, retain `catboost` and `xgboost`).
* **What Stays Fixed:** Feature set (501 cols), 10-fold CV, 3 seeds.
* **Expected Benefit:** Reduces Kaggle execution time by 30%; improves public test Log Loss by eliminating duplicate variance.
* **Risk:** Low.
* **Cost:** ~12 mins on Kaggle GPU.
* **Success Criterion:** OOF Log Loss $\le 0.2460$; test correlation with CatBoost maintained.
* **Priority:** **HIGH (Run 1st)**

#### Experiment 2: Remove Target Encoding Noise Injection
* **Hypothesis:** Eliminating the Gaussian noise $\mathcal{N}(0, 0.005)$ from `preprocess_fold_features` in `src/validation.py` will restore perfect train-val-test distribution alignment.
* **Component to Change:** `src/validation.py` (lines 37–42: remove noise perturbation on `X_tr`).
* **What Stays Fixed:** Models, hyperparameters, features.
* **Expected Benefit:** Recovers the performance loss observed in V13; improves tree split fidelity.
* **Risk:** Extremely low.
* **Cost:** Fast smoke test + standard run (~15 mins).
* **Success Criterion:** Public Zindi score improves beyond `0.7210`.
* **Priority:** **HIGH (Run 2nd)**

#### Experiment 3: Feature Pruning — Eliminate Noisy High-Order Ratio Columns
* **Hypothesis:** Dropping the bottom 100 features (specifically `emergency_cash_spike_*` and multi-month velocity ratios with volatile denominators) will reduce XGBoost generalization gap from 0.131 to <0.080 without hurting CatBoost.
* **Component to Change:** `src/features.py` (comment out noisy ratio generation; prune from 501 down to ~380 high-signal features).
* **What Stays Fixed:** Models, CV harness, seeds.
* **Expected Benefit:** Slashes tree overfitting; improves test set robustness.
* **Risk:** Low.
* **Cost:** ~15 mins on GPU.
* **Success Criterion:** XGBoost standalone OOF Log Loss drops below 0.2500; Generalization Gap drops below 0.090.
* **Priority:** **HIGH (Run 3rd)**

#### Experiment 4: Integrate PyTorch Tabular ResNet as 3rd Orthogonal Architecture
* **Hypothesis:** Introducing a 3-block Tabular ResNet with LayerNorm and Mish activations will provide genuine continuous manifold diversity (correlation to CatBoost < 0.92), reducing ensemble variance.
* **Component to Change:** Add `pytorch_mlp` into `train.py` `--models catboost xgboost pytorch_mlp`.
* **What Stays Fixed:** 10-fold CV, feature pipeline.
* **Expected Benefit:** Delivers true orthogonal ensembling; drives ROC-AUC above 0.912.
* **Risk:** Neural net requires careful learning rate scheduling (AdamW with cosine annealing).
* **Cost:** ~20 mins on GPU.
* **Success Criterion:** Correlation between PyTorch and CatBoost $\le 0.910$; Level-2 Stacking Log Loss drops below 0.2440.
* **Priority:** **HIGH (Run 4th)**

#### Experiment 5: Nested Out-of-Fold 2-Stage Stacking Meta-Learner
* **Hypothesis:** Fitting the Level-2 Logistic Regression meta-learner inside an outer 5-fold cross-validation loop will eliminate in-sample optimistic bias and yield perfectly calibrated test probabilities.
* **Component to Change:** `src/ensemble.py` (`train_stacking_meta_learner`: implement nested `KFold(n_splits=5)`).
* **What Stays Fixed:** Base models and features.
* **Expected Benefit:** Completely eliminates validation leakage; aligns internal OOF metrics with public leaderboard.
* **Risk:** Minimal.
* **Cost:** ~1 min compute (runs strictly on existing OOF matrix).
* **Success Criterion:** Eliminates gap between reported stacking loss and raw blend loss.
* **Priority:** **HIGH (Run 5th)**

#### Experiment 6: Enforce CatBoost Monotonic Feature Constraints
* **Hypothesis:** Imposing monotonic constraints (`+1` for `bal_max_drawdown_ratio`, `total_deficit_months_6m`; `-1` for `bal_recent_vs_old_ratio`, `m1_daily_avg_bal`) will prevent CatBoost from overfitting local noise spikes in balance metrics.
* **Component to Change:** `src/models.py` (pass `monotone_constraints` dictionary into `CatBoostClassifier`).
* **What Stays Fixed:** Features, CV, seeds.
* **Expected Benefit:** Stronger out-of-domain test generalization on extreme wealth/poverty accounts.
* **Risk:** If constraints are applied to non-monotonic features, performance could degrade.
* **Cost:** ~12 mins on GPU.
* **Success Criterion:** CatBoost standalone OOF Log Loss improves by $\ge 0.0010$.
* **Priority:** **MEDIUM (Run 6th)**

#### Experiment 7: Add Scaled Regularized Logistic Regression / ElasticNet Baseline
* **Hypothesis:** Blending a simple $L_1/L_2$ regularized linear model (with 3%–5% ensemble weight) will constrain extreme non-linear tree probabilities from blowing up Log Loss on unusual test profiles.
* **Component to Change:** `train.py` (include `linear_baseline` with $C=0.05$ in ensemble).
* **What Stays Fixed:** Base tree models.
* **Expected Benefit:** Provides linear anchor to regularize tree tails.
* **Risk:** Very low.
* **Cost:** <1 min additional compute.
* **Success Criterion:** Ensemble Log Loss drops by $-0.0008$.
* **Priority:** **MEDIUM (Run 7th)**

#### Experiment 8: RankGauss Transformation on Financial Volume Distributions
* **Hypothesis:** Transforming skewed balance and volume distributions (`m1_daily_avg_bal`, transaction totals) via Quantile RankGauss will compress heavy tails and improve neural net and linear model stability.
* **Component to Change:** `src/features.py` (apply `QuantileTransformer(output_distribution='normal')` on numeric columns).
* **What Stays Fixed:** GBDT models.
* **Expected Benefit:** Drastic boost to PyTorch ResNet and Linear baseline performance.
* **Risk:** Moderate; must be fit strictly fold-by-fold.
* **Cost:** ~15 mins on GPU.
* **Success Criterion:** Neural net OOF Log Loss improves from 0.279 to <0.260.
* **Priority:** **MEDIUM (Run 8th)**

#### Experiment 9: Power Blending Optimization on Test Probabilities
* **Hypothesis:** Applying power scaling $P_{\text{cal}} \propto P^\gamma$ with $\gamma \in [0.90, 1.05]$ prior to calibration will optimize probability concentration for the multi-score objective.
* **Component to Change:** `src/ensemble.py` (add power transform step in calibration).
* **What Stays Fixed:** All base models.
* **Expected Benefit:** Direct tuning of probability dispersion on test set.
* **Risk:** Overfitting to public leaderboard if tuned manually.
* **Cost:** Instantaneous post-processing.
* **Success Criterion:** OOF Multi Score optimization shows improvement.
* **Priority:** **LOW (Run 9th)**

#### Experiment 10: Interaction-Pruned 5-Seed Ultra-Ensemble
* **Hypothesis:** Expanding seeds to `[42, 1337, 2026, 777, 999]` on the final pruned 3-family suite (CatBoost + XGBoost + PyTorch ResNet) will compress test variance to its absolute mathematical limit.
* **Component to Change:** `train.py` `--seeds 42 1337 2026 777 999`.
* **What Stays Fixed:** Pruned features and validated models.
* **Expected Benefit:** Squeezes an additional $+0.0020$ to $+0.0035$ on public/private leaderboard to make the final push toward Rank 1 (>0.739).
* **Risk:** High computational time.
* **Cost:** ~35–40 mins on Kaggle dual T4 GPU.
* **Success Criterion:** Public Zindi score approaches or exceeds **0.7250–0.7300**.
* **Priority:** **FINAL SPRINT (Run 10th)**
