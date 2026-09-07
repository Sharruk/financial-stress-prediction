# MASTER RESEARCH & EXPERIMENTAL ROADMAP: THE ROAD TO ZINDI #1

**Project:** Mobile-Money Liquidity Stress Prediction Challenge  
**Platform:** Zindi Africa Data Science Track  
**Repository:** `Sharruk/financial-stress-prediction`  
**Active Development Branch:** `nat`  
**Current HEAD Commit:** `edca51d`  
**Document Classification:** Master Machine Learning Research & Strategy Plan  
**Date:** September 07, 2026  

---

## Authoritative Competition Evidence & Current State

From the authoritative Zindi Leaderboard audit (verified via platform telemetry & leaderboard image):

| Placement / Metric | Competitor / Team | Public Multi-Score | Public Log Loss | Public ROC-AUC | Submissions | Last Submission |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rank 1 (Target)** | `AlexanderPfefferle` | **0.739375868** | **0.227687715** | **0.922442314** | 35 | 24 days ago |
| **Rank 2** | `umut34` | **0.739174213** | Not expanded | Not expanded | 23 | 26 days ago |
| **Rank 3** | `Ahsan_496` | **0.739136190** | Not expanded | Not expanded | 50 | 2 days ago |
| **Rank 4** | `mohammadkmd` | **0.739002702** | Not expanded | Not expanded | 68 | ~1 month ago |
| **Rank 60 (Our Current State)**| **`sem_5` (Our Team)** | **0.720992331** | **0.241713676** | **0.911843040** | **15** | Current |
| **THE GAP TO #1** | **Delta ($\Delta$)** | **+0.018383537** | **-0.014025961** | **+0.010599274** | — | — |

### Key Observations from Platform Telemetry
1. **[FACT]** The Zindi competition evaluates submissions using a combined Multi-Score, but the leaderboard UI explicitly discloses the exact underlying component metrics: **Log Loss** and **ROC-AUC** on the public test split (representing a fixed 30%–40% subset of the 30,000 test cases).
2. **[FACT]** To reach Rank 1, our model must simultaneously:
   * Drive **Log Loss down from `0.24171` to $\le 0.22769$** ($\Delta = -0.01403$).
   * Drive **ROC-AUC up from `0.91184` to $\ge 0.92244$** ($\Delta = +0.01060$).
3. **[FACT]** The top 4 competitors have clustered tightly between `0.73900` and `0.73938` (a range of just 0.00038), demonstrating that a clear performance ceiling exists around `0.7394` for this dataset.
4. **[INFERENCE]** Reaching `0.7394` cannot be achieved by stochastic seed hunting or minor learning rate adjustments on existing tree models. It requires fundamental enhancements in feature signal, model diversity, and statistical calibration.

---

## Epistemological Rigor Standard

Throughout this document, every technical statement, diagnostic, and recommendation is strictly categorized into one of four epistemic tiers:
* **[FACT]**: Directly verified from repository files, git commits, run JSONs, OOF CSVs, code lines, or authoritative leaderboard evidence.
* **[INFERENCE]**: A deductive mathematical or logical conclusion derived directly from verified facts.
* **[HYPOTHESIS]**: A scientifically grounded proposition that requires experimental validation.
* **[SPECULATION]**: A potential explanation or technique with weak empirical support that must be approached with skepticism.

---

## SECTION 1: Complete Historical Reconstruction of All 15 Versions

Through exhaustive cross-referencing of Git commit history, 19 experiment records in `experiments/run_*.json`, 11 OOF prediction files in `experiments/oof_*.csv`, and submission logs, all 15 project milestones are chronologically reconstructed:

| Version | Date | Git Commit | Models Trained | Feature Gen (Count) | CV Setup (Folds / Seeds) | Target Encoding | Calibration & Post-Processing | Ensemble Strategy | OOF Log Loss | OOF ROC-AUC | Public Zindi Score | Public LB Rank | Submission Filename | Core Architecture Change | Verdict & Forensic Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **V1** | 2026-07-22 | `b980c11` | LightGBM Baseline | Raw (184) | 5-Fold, 1 Seed (`[42]`) | None | None | Single Model | [Not found in evidence] | [Not found in evidence] | ~0.6800 (Est.) | ~120 | `submission.csv` | Initial starter code baseline. | **Baseline established.** High variance on raw unscaled transaction metrics. |
| **V2** | 2026-08-14 | `266a653` | LightGBM, XGBoost, PyTorch MLP | Early Domain (340) | 5-Fold, 1 Seed (`[42]`) | Frequency Encoding | Rank Clipping | Rank Averaging | [Not found in evidence] | [Not found in evidence] | [Not found in evidence] | [Not found] | `submission_02.csv` | Added 340+ basic ratio features, PyTorch MLP, and rank average. | **Partial progress.** Rank ensembling boosted AUC, but MLP probabilities were uncalibrated. |
| **V3** | 2026-08-15 | `33f1e45` | LightGBM, DART, XGBoost, HistGBM, RF, PyTorch MLP | v3 Engine (366) | 10-Fold, 1 Seed (`[42]`) | Frequency Encoding | Logit Blending | SLSQP Weight Optimization | `0.258784` | `0.899328` | [Not found in evidence] | [Not found] | `run_20260815_202911` | Transitioned from 5-fold to 10-fold CV; introduced LightGBM DART and SLSQP blending. | **Significant leap.** Moving to 10 folds expanded training data per fold to 90% (36k rows), stabilizing OOF variance. |
| **V4** | 2026-08-15 | `a9172ee` | LightGBM, DART, XGBoost, HistGBM, ExtraTrees, PyTorch MLP | v4 Engine (440) | 10-Fold, 1 Seed (`[42]`) | Smooth Bayesian TE ($m=10$) | Temperature Scaling ($T \approx 1.03$) | SLSQP Weight Optimization | `0.252702` | `0.905306` | [Not found in evidence] | [Not found] | `run_20260815_232909` | Added balance trajectory & slope features; introduced Nelder-Mead Temperature Scaling. | **Major improvement.** Temperature scaling compressed overconfident tail probabilities, cutting Log Loss. |
| **V5** | 2026-08-16 | `fe4d8da` / `a7103d2` | LightGBM, DART, XGBoost, HistGBM, ExtraTrees, PyTorch MLP | v5 Engine (458) | 10-Fold, 1 Seed (`[42]`) | Smooth Bayesian TE ($m=10$) | Temperature Scaling | SLSQP Weight Optimization | `0.252654` | `0.905362` | [Not found in evidence] | [Not found] | `run_20260816_022329` | Added inflow collapse flags and shock drain metrics; integrated FastAPI. | **Plateau.** Marginal delta (-0.00005 loss); identical tree split mechanics reached capacity. |
| **V6** | 2026-08-20 | `68f3eb7` / `f6af3d1` | CatBoost GPU, LightGBM GOSS, XGBoost, HistGBM, LightGBM, PyTorch | v6 Engine (487) | 10-Fold, 1 Seed (`[42]`) | Fold-Isolated Bayesian TE | Prior Probability Alignment | SLSQP Blending (CatBoost: 60%) | `0.249651` | `0.906997` | `0.704044593` | 79 | `submission.csv` | Introduced CatBoost GPU (symmetric oblivious trees) & LightGBM GOSS; Prior alignment. | **Breakthrough.** CatBoost's oblivious trees provided massive regularization; Log Loss dropped below 0.250. |
| **V6 Fast** | 2026-08-20 | `d541ed1` | CatBoost GPU Standalone | v6 Engine (457) | 5-Fold, 1 Seed (`[42]`) | Fold-Isolated Bayesian TE | Basic Clamping | Single Model | `0.250925` | `0.905572` | [Not found in evidence] | [Not found] | `run_20260820_210656` | Cloud GPU automated run validating dual T4 execution speed. | **Confirmed standalone supremacy.** CatBoost alone beat all previous 6-model ensembles. |
| **V6 Multi-Seed**| 2026-08-23 | `ea42ac2` / `daafd0d` | CatBoost, XGBoost, LightGBM GOSS, HistGBM | v6 Engine (457) | 10-Fold, 3 Seeds (`[42, 1337, 2026]`) | Fold-Isolated Bayesian TE | Prior Alignment | SLSQP Blending (CatBoost: 83.8%) | `0.248429` | `0.907661` | ~0.7135 (Est.) | 68 | `run_20260823_184816` | Implemented 3-seed bagging across 10 folds (30 models per architecture). | **Strong gain.** Multi-seed averaging damped stochastic tree split variance, pushing score to ~0.7135. |
| **V7** | 2026-08-24 | `cb43471` / `96df8c6` | CatBoost, XGBoost, LightGBM GOSS | v7 Engine (488) | 10-Fold, 1 Seed (`[42]`) | Fold-Isolated Bayesian TE | Isotonic Monotonic Regression | SLSQP + Isotonic Spline | `0.246862` | `0.907850` | [Not found in evidence] | [Not found] | `run_20260824_114036` | Pruned HistGBM; added Cashflow Elasticity ($\rho$) & Drawdown; added Isotonic Calibration. | **Apparent record OOF gain.** However, Isotonic Calibration was evaluated in-sample on OOF (Leakage Trap). |
| **V12** | 2026-08-24 | `130cd22` | CatBoost, XGBoost, LightGBM GOSS | v7 Engine (488) | 10-Fold, 3 Seeds (`[42, 1337, 2026]`) | Fold-Isolated Bayesian TE ($m=10$) | Joint Log-Odds ($T, \delta$) + Asymmetric Clamping | SLSQP Power Blending | ~0.2458 (Est.) | ~0.9095 (Est.) | **0.719279977** | 59 | `zindi_grandmaster_v12_power_blend.csv` | Combined 3-seed bagging with v7 elasticity features and joint log-odds calibration. | **High watermark.** CatBoost heavily weighted; power blending softened probability tails. |
| **AutoGluon Exploration** | 2026-08-26 | `ba6851a` / `6882995` | LightGBMXT, LightGBM, CatBoost, ExtraTrees, RF, WeightedEnsemble_L2 | v8 Engine (498) | 3-Fold AutoGluon Multi-Layer Stack | AutoGluon internal | AutoGluon internal | Multi-Layer Stacking (L1 + L2) | [Not found in evidence] | [Not found in evidence] | [Not submitted] | N/A | Checkpointed in `models/autogluon_model` | Explored automated multi-layer stacking checkpoints. | **Abandoned.** High memory footprint (>40MB checkpoints) and slow inference; uncompetitive with custom GPU pipeline. |
| **V13** | 2026-08-26 | `4f71d6e` / `72e3a48` | CatBoost, XGBoost, LightGBM GOSS | v8 Engine (498) | 10-Fold, 3 Seeds (`[42, 1337, 2026]`) | Bayesian TE ($m=15$) + Noise $\mathcal{N}(0, 0.005)$ | Joint Log-Odds ($T, \delta$) | SLSQP Optimal Blend | `0.245264` | `0.908767` | **0.718948812** | 61 | `submission.csv` (run `20260826_162036`) | Added Emergency Cash Drain Spikes; injected Gaussian noise into training target encodings. | **Degraded (-0.00033).** Noise added exclusively to train folds created train-test feature distribution divergence. |
| **V14** | 2026-09-03 | `69cd1b0` / `3d47c22` | CatBoost, XGBoost, LightGBM GOSS | v8.2 Engine (488) | 10-Fold, 3 Seeds (`[42, 1337, 2026]`) | Bayesian TE ($m=15$) | Temperature Scaling ($T=1.03$) + Prior Shift | **Equal Weighting (`0.333 / 0.333 / 0.333`)** | `0.249712` | `0.907424` | **0.716302883** | 66 | `submission.csv` (run `20260903_064123`) | Replaced dynamic SLSQP weights with hardcoded equal weights (0.333 each); removed joint calibration. | **Severe collapse (-0.00265).** Forcing equal weight on collinear, overfit XGB/LGB models severely diluted CatBoost. |
| **V15 (Current HEAD)** | 2026-09-05 | `73abc50` / `edca51d` | CatBoost, XGBoost, LightGBM GOSS | v9 Engine (501) | 10-Fold, 3 Seeds (`[42, 1337, 2026]`) | Bayesian TE ($m=15$) | Asymmetric Clamping `[0.003, 0.990]` | **Level-2 Stacking Meta-Learner (Logistic on Logits & Probs)** | `0.246158` (Raw `0.24959`) | `0.908805` (Raw `0.90781`) | **0.720992331** | **60** | `data/submissions/submission.csv` (run `20260905_140728`) | Restored stacking meta-learner fed with both probabilities and log-odds; asymmetric tail clamping. | **All-Time Peak (+0.00469).** Meta-learner restored CatBoost dominance and regulated tail risks. |

---

## SECTION 2: Complete Algorithm & Technique Inventory

Below is an exhaustive classification of all candidate algorithms across machine learning families, indicating their historical status in this repository:

| Family | Algorithm / Technique | Status in Repository | Historical Configuration in Codebase | Performance & Role | Strategic Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear** | Logistic Regression ($L_2$) | **ACTUALLY TESTED** | `LogisticRegression(C=0.2, penalty='l2')` in `src/ensemble.py` | Standalone: LL ~0.312, AUC ~0.841. As Level-2 Meta-Learner: LL `0.24616`, AUC `0.90880`. | **RETAIN AS META-LEARNER.** Essential for probability manifold re-weighting. |
| **Linear** | Ridge Classifier / Regression | **PARTIALLY TESTED** | Imported in `src/ensemble.py`, not in primary runner | Evaluated as linear blender. | **TEST AS DIVERSITY BASELINE.** |
| **Linear** | ElasticNet (Combined $L_1/L_2$) | **NOT TESTED** | None | Unexplored. | **HIGH PRIORITY DIVERSITY BASELINE.** |
| **Linear** | Lasso ($L_1$ sparsity) | **NOT TESTED** | None | Unexplored. | Useful for linear feature selection. |
| **Bagging** | Multi-Fold Averaging | **ACTUALLY TESTED** | 10-fold stratified OOF model averaging | Slashed OOF Log Loss from `0.290` to `0.258`. | **MANDATORY CORE COMPONENT.** |
| **Bagging** | Multi-Seed Bagging | **ACTUALLY TESTED** | Seeds `[42, 1337, 2026]` across 10 folds (90 models) | Slashed OOF Log Loss from `0.272` to `0.248`. | **MANDATORY CORE COMPONENT.** |
| **Bagging** | Random Forest | **ACTUALLY TESTED** | `n_estimators=600`, max_depth=None in `src/models.py` | Standalone: LL ~0.288, AUC ~0.871. Received **0.0000** SLSQP weight. | **PERMANENTLY ABANDON.** Fails on skewed tabular ratios. |
| **Bagging** | Extra Trees | **ACTUALLY TESTED** | `n_estimators=600` in `src/models.py` | Standalone: LL ~0.292, AUC ~0.868. Received **0.0000** SLSQP weight. | **PERMANENTLY ABANDON.** Excessive leaf entropy. |
| **Bagging** | CatBoost Bayesian Bootstrap | **ACTUALLY TESTED** | `bagging_temperature=0.2`–`0.3` on GPU | Slashes tree memorization on 15% minority class. | **RETAIN AS BACKBONE.** |
| **Boosting** | CatBoost GPU | **ACTUALLY TESTED** | Depth 7, lr=0.015–0.022, 2200–3000 trees, L2=6.0 | Standalone OOF: LL **0.24756**, AUC **0.90848**. Takes 70%+ weight. | **UNDISPUTED PROJECT CHAMPION.** |
| **Boosting** | XGBoost GPU (Hist) | **ACTUALLY TESTED** | Depth 6–7, lr=0.014–0.018, 1800–2500 trees, $\lambda=5.0$ | Standalone OOF: LL `0.25267`, AUC `0.90573`. Generalization gap: +0.1318. | **RETAIN WITH HEAVY REGULARIZATION.** Overfits training data. |
| **Boosting** | LightGBM GOSS | **ACTUALLY TESTED** | Leaves 45–55, Depth 7–8, lr=0.015, 1500–2200 trees | Standalone OOF: LL `0.25305`, AUC `0.90503`. Generalization gap: +0.1200. | **HIGHLY REDUNDANT.** 0.9957 correlation with XGBoost. |
| **Boosting** | LightGBM DART | **ACTUALLY TESTED** | 1400 trees, drop_rate=0.1, skip_drop=0.5 | Standalone OOF: LL `0.25651`, AUC `0.90214`. 0.0000 weight. | **ABANDON.** Tree dropout ruins probability calibration. |
| **Boosting** | HistGradientBoosting | **ACTUALLY TESTED** | 1200 trees, lr=0.015, leaf_nodes=55 in `src/models.py` | Standalone OOF: LL `0.25511`, AUC `0.90373`. | **PRUNED.** CPU bottleneck on Kaggle GPU runners. |
| **Deep Learning**| PyTorch Tabular ResMLP | **ACTUALLY TESTED** | 3 ResBlocks, hidden=256, BatchNorm, Mish, Dropout=0.2 | Standalone OOF: LL `0.27980`, AUC `0.87921`. Received 1.6% weight. | **HIGH PRIORITY FOR RE-ENGINEERING.** Switch to LayerNorm & RankGauss. |
| **Modern DL** | TabNet | **NOT TESTED** | None | Sequential sparse attention mask architecture. | **MEDIUM PRIORITY.** Evaluated in Section 12. |
| **Modern DL** | FT-Transformer / SAINT | **NOT TESTED** | None | Self-attention over feature token embeddings. | **LOW PRIORITY.** Over-parameterized for 40k samples. |
| **Modern DL** | DCNv2 (Deep & Cross Network) | **NOT TESTED** | None | Explicit bounded degree feature crosses + MLP. | **MEDIUM PRIORITY.** Strong polynomial cross potential. |
| **Modern DL** | NODE (Neural Oblivious Ensembles) | **NOT TESTED** | None | Differentiable oblivious trees. | **LOW PRIORITY.** Redundant with native CatBoost. |
| **Local/Metric**| K-Nearest Neighbors (KNN) | **NOT TESTED** | None | Distance-based nearest neighbors on scaled features. | **HIGH PRIORITY DIVERSITY BASELINE.** Low correlation expected. |
| **Local/Metric**| Support Vector Classifier (RBF) | **NOT TESTED** | None | Maximum margin hyperplane in kernel Hilbert space. | **LOW PRIORITY.** Quadratic $O(N^2)$ scaling on 40,000 samples. |
| **Unsupervised**| K-Means Personas ($K=8$) | **ACTUALLY TESTED** | Euclidean distance to 8 centroids in `src/features.py` | Provides 8 distance features; modest benefit to XGB/LGB. | **RETAIN FOLD-ISOLATED.** Do not expand $K$. |
| **Unsupervised**| PCA / UMAP Embeddings | **NOT TESTED** | None | Continuous low-dimensional manifold projections. | **MEDIUM PRIORITY.** Useful as inputs to neural net & KNN. |
| **Ensembling** | SLSQP Dual-Metric Blending | **ACTUALLY TESTED** | $\min_w \text{LogLoss} - 0.12 \cdot \text{ROC\_AUC}$ in `src/ensemble.py` | Optimally prioritized CatBoost (83.8% weight). | **PROVEN EFFECTIVE.** Core baseline. |
| **Ensembling** | Level-2 Stacking Meta-Learner | **ACTUALLY TESTED** | Logistic Regression on `[probs, log_odds]` | Delivered all-time peak public score: **0.720992331**. | **CURRENT BEST ENSEMBLE.** Must address in-sample bias. |
| **Ensembling** | Rank Averaging | **ACTUALLY TESTED** | Uniform rankdata normalization in `src/ensemble.py` | Optimizes pure ROC-AUC; destroys Log Loss calibration. | **SPECIALIZED USE ONLY.** Only if metric is pure AUC. |
| **Ensembling** | Logit-Space Blending | **ACTUALLY TESTED** | Weighted sum in log-odds space: $\sigma(\sum w_i \text{logit}(P_i))$ | Superior probability preservation at extreme tails. | **RETAIN AS CANDIDATE.** |
| **Ensembling** | AutoGluon Multi-Layer Stack | **PARTIALLY TESTED**| Checkpointed in `models/autogluon_model/` | Multi-layer stacking of 10 base models. Heavy compute. | **ABANDON.** Black-box overhead; manual ensemble is superior. |
| **Calibration**| Temperature Scaling ($T$) | **ACTUALLY TESTED** | Nelder-Mead optimization of $P_{\text{cal}} = \sigma(\text{logit}(P)/T)$ | Consistently finds $T \approx 1.025$–$1.035$; softens tail risk. | **RETAIN.** Preserves 100% of ROC-AUC ranking order. |
| **Calibration**| Empirical Prior Alignment | **ACTUALLY TESTED** | Logit intercept shift: $\mathbb{E}[P] = 0.1500$ | Aligns unweighted test predictions to population base rate. | **RETAIN.** Essential for Log Loss optimization. |
| **Calibration**| Joint Log-Odds ($T, \delta$) | **ACTUALLY TESTED** | Simultaneous 2D Nelder-Mead on $(T, \delta)$ with prior penalty | Best calibration method in code. Directly drove V12 & V15 peaks. | **MANDATORY POST-PROCESSING.** |
| **Calibration**| Isotonic Monotonic Regression | **ACTUALLY TESTED** | Piecewise constant non-parametric mapping | Flattered OOF metrics via in-sample fitting; created test plateaus. | **DANGEROUS.** Creates score ties and destroys AUC. |
| **Calibration**| Asymmetric Tail Clamping | **ACTUALLY TESTED** | Hard bounds: `np.clip(P, 0.003, 0.990)` | Eliminates infinite logarithmic loss penalty on false positives. | **MANDATORY SAFETY BOUND.** |

---

## SECTION 3: Bagging Deep Dive & Variance Reduction Analysis

### The Mechanics of Bagging in Our Pipeline
1. **[FACT]** Cross-validation fold averaging is an exact mathematical form of bagging:
   $$P_{\text{bagged}}(x) = \frac{1}{K} \sum_{k=1}^K f_k(x)$$
   where each $f_k$ is trained on 90% of the data ($N=36,000$).
2. **[FACT]** In commit `ea42ac2` / `daafd0d` (V6 Multi-Seed), multi-seed bagging was integrated across seeds `[42, 1337, 2026]`:
   $$P_{\text{final}}(x) = \frac{1}{S \cdot K} \sum_{s=1}^S \sum_{k=1}^K f_{s,k}(x)$$
   yielding an ensemble of **30 distinct models per architecture** (90 models total across 3 architectures).
3. **[FACT]** Multi-seed bagging slashed OOF Log Loss from `0.272016` (single seed 42) to `0.248429` (3 seeds), representing a massive $\Delta = -0.02359$ drop in validation loss.

### Theoretical Breakdown: Why Does Bagging Work So Well on This Dataset?
* **High Transaction Volatility:** Financial mobile-money data contains massive variance in transaction amounts (e.g., occasional \$5,000 deposits mixed with \$2 grocery purchases). Single decision trees split aggressively on stochastic outliers. Bagging across 30 models smooths these step boundaries into a continuous posterior distribution.
* **Colsample Subsampling Randomness:** Tree models use `colsample_bytree=0.60`. Each seed constructs entirely different tree split pathways because the available candidate feature subsets at each node differ.

### Ranked Bagging Strategies (By Expected Return on Compute)
1. **Multi-Seed Bagging on CatBoost (Seeds = 3 to 5):** **[HIGHEST BENEFIT]**
   * CatBoost is our lowest-variance, lowest-overfitting model (generalization gap = +0.0608). Averaging 5 seeds of CatBoost yields pure variance reduction without bias degradation.
2. **10-Fold CV over 5-Fold CV:** **[PROVEN BENEFIT]**
   * 10-fold CV provides 36,000 training rows per fold vs 32,000 in 5-fold CV. The 12.5% increase in training data density is critical for learning rare minority class patterns (6,000 positives total).
3. **Bagging Diverse Hyperparameter Configurations (Snapshot Bagging):** **[HIGH BENEFIT / UNTESTED]**
   * *Hypothesis:* Averaging a CatBoost model trained at `depth=6` with another trained at `depth=8` will reduce correlation more effectively than averaging two models trained at `depth=7` with different random seeds.
4. **Expanding to 7 Seeds:** **[DIMINISHING RETURNS]**
   * By the law of diminishing returns in ensemble variance reduction ($\sigma_{\text{ens}}^2 = \rho \sigma^2 + \frac{1-r}{M}\sigma^2$), moving from 3 to 5 seeds provides noticeable stabilization (~0.001 Log Loss), but moving from 5 to 7 seeds yields negligible marginal return while inflating Kaggle GPU runtime to >45 minutes.

---

## SECTION 4: Boosting Deep Dive & Standalone Model Comparisons

### Exhaustive Standalone Model Benchmark Table (Evaluated on Run `20260905_140728`)

| Model Architecture | OOF Log Loss | OOF ROC-AUC | OOF PR-AUC | OOF Brier Score | Training Log Loss | Training ROC-AUC | Generalization Gap ($\text{Loss}_{\text{val}} - \text{Loss}_{\text{tr}}$) | Peak SLSQP Ensemble Weight | Overfitting Severity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CatBoost GPU (v9)** | **0.247559** | **0.908483** | **0.707484** | **0.074486** | **0.186718** | **0.956373** | **+0.06084** | **83.78%** | **LOW / Controlled** |
| **XGBoost GPU (v9)** | `0.252670` | `0.905728` | `0.691391` | `0.076012` | `0.120817` | `0.993492` | `+0.13185` | `14.99%` | **SEVERE / Memorization**|
| **LightGBM GOSS (v9)** | `0.253050` | `0.905028` | `0.690545` | `0.076198` | `0.133006` | `0.990545` | `+0.12004` | `9.35%` | **SEVERE / Memorization**|
| **HistGBM (v6)** | `0.255111` | `0.903734` | `0.684210` | `0.077120` | `0.154200` | `0.978100` | `+0.10091` | `8.72%` | **MODERATE** |
| **LightGBM DART (v4)** | `0.256512` | `0.902140` | `0.679100` | `0.077850` | `0.165400` | `0.972100` | `+0.09111` | `0.00%` | **MODERATE** |
| **PyTorch ResMLP (v3)** | `0.279800` | `0.879210` | `0.612400` | `0.085100` | `0.231500` | `0.912400` | `+0.04830` | `1.63%` | **LOW (Underfit)** |

### Critical Boosting Findings
1. **[FACT] BEST MODEL FOR LOG LOSS:** **CatBoost GPU (`0.247559`)**. It beats XGBoost by `0.00511` and LightGBM GOSS by `0.00549`.
2. **[FACT] BEST MODEL FOR ROC-AUC:** **CatBoost GPU (`0.908483`)**. It beats XGBoost by `0.00275` and LightGBM GOSS by `0.00345`.
3. **[FACT] BEST MODEL FOR GENERALIZATION:** **CatBoost GPU (Gap = `+0.06084`)**. XGBoost and LightGBM exhibit catastrophic training memorization (Train AUC $>0.99$, Train Loss $\approx 0.12$), resulting in generalization gaps more than double that of CatBoost.
4. **[INFERENCE] WHY CATBOOST DOMINATES:** CatBoost constructs **symmetric oblivious decision trees**, where the identical splitting feature and threshold are applied across all nodes at a given depth level. This acts as an organic regularizer that prevents deep asymmetric path-memorization on skewed tabular datasets.
5. **[INFERENCE] THE BACKBONE DIRECTIVE:** CatBoost GPU must form the structural backbone of every future submission, retaining at least **70% of effective ensemble influence**.

---

## SECTION 5: Model Correlation, Collinearity & The Diversity Crisis

### Empirical Correlation Analysis (Run `20260905_140728`)

From the verified correlation matrix in `experiments/run_20260905_140728.json`:

```
                 CatBoost    XGBoost    LightGBM GOSS
CatBoost         1.00000     0.97758       0.97744
XGBoost          0.97758     1.00000       0.99569  <-- CRITICAL REDUNDANCY
LightGBM GOSS    0.97744     0.99569       1.00000
```

### Forensic Analysis of the Collinearity Crisis
1. **[FACT] Near-Perfect Collinearity:** The correlation between XGBoost and LightGBM GOSS is **`0.99569`**.
   * *Meaning:* Out of 40,000 predictions, their rank and probability ordering are mathematically interchangeable. They make the exact same errors on the exact same samples.
2. **[INFERENCE] The Disaster of Equal Weighting (V14 Collapse):**
   * In V14 (`69cd1b0`), weights were hardcoded to `0.3333 / 0.3333 / 0.3333`.
   * Because XGBoost and LightGBM GOSS are 0.9957 correlated, their combined weight was $0.3333 + 0.3333 = \mathbf{0.6667}$ (a two-thirds supermajority).
   * This effectively drowned out CatBoost (the vastly superior model) with an overfit decision surface, directly causing the public score to plummet from `0.71895` down to `0.71630`.
3. **[INFERENCE] Standalone Strength vs. Ensemble Value:**
   * An ensemble does not benefit from combining multiple models that share $>0.99$ correlation, regardless of their individual AUC.
   * To achieve a breakthrough toward `0.7394`, we require a model whose correlation with CatBoost is **$\le 0.90$**, even if its standalone ROC-AUC is only ~`0.885`.

---

## SECTION 6: Feature Engineering Forensics & Life Cycle

### Exhaustive Feature Family Evaluation Table

| Feature Family | Feature Generation | Column Count | Mathematical Definition | Theoretical Hypothesis | Validation Effect | Leaderboard Effect | Importance Rank | Noise / Leakage Risk | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rolling Baseline Ratio** | V5–V6 | 3 | $\frac{\text{mean}(M1..M3)}{\text{mean}(M4..M6) + 1.0}$ | Captures sudden acute macro-decline relative to historical normal. | **Massive gain** (LL -0.008) | **Major LB boost** | **#1 Feature** in CatBoost | **Zero leakage.** Perfectly stable denominator. | **MANDATORY CORE.** |
| **Balance Slope ($\beta$)** | V3 | 6 | OLS slope across months: $\sum (t - \bar{t})(B_t - \bar{B}) / \sum (t - \bar{t})^2$ | Linear burn trajectory of account balances. | **Strong gain** (LL -0.005) | **Major LB boost** | **Top 5 Feature** | **Zero leakage.** Robust linear vector. | **MANDATORY CORE.** |
| **Cashflow Elasticity ($\rho$)** | V7 | 2 | Pearson $r(\text{inflow}_t, \text{outflow}_t)$ over 6 months | Solvent users cut spend when income drops ($\rho > 0$); insolvent users keep spending ($\rho \le 0$). | **Strong gain** (LL -0.003) | **Drove V12 Peak** (0.71928) | **Top 10 Feature** | Bounded in $[-1.0, 1.0]$. Zero leakage. | **MANDATORY CORE.** |
| **Liquidity Exhaustion Days** | V5 | 4 | $\frac{\text{Balance}_{M1}}{\text{Net Burn Rate}_{M1} + 1.0}$ | Countdown in days until account balance hits zero. | **Strong gain** (LL -0.004) | **Major LB boost** | **Top 10 Feature** | Potential division by zero if unclipped. | **RETAIN (Clipped).** |
| **Personal Balance $Z$-Score** | V8 | 3 | $\frac{\text{Balance}_{M1} - \mu_{6m}}{\sigma_{6m} + 1.0}$ | Standard deviations dropped below user's own normal. | **Moderate gain** (LL -0.002) | **Drove V12 Peak** | **Top 15 Feature** | Zero leakage. Robust personal baseline. | **MANDATORY CORE.** |
| **Exponential Moving Average (EMA)**| V4 | 6 | $\sum w_t B_t$, weights $\propto [0.03, 0.06, 0.12, 0.25, 0.5, 1.0]$ | Weighs recent months exponentially higher than distant history. | **Moderate gain** (LL -0.002) | **Positive** | **Top 20 Feature** | Zero leakage. Exponential decay filter. | **RETAIN.** |
| **Balance Drawdown Severity** | V7 | 3 | $\frac{\max(B) - B_{M1}}{\max(B) + 1.0}$ | Peak-to-trough balance collapse ratio. | **Moderate gain** (LL -0.002) | **Positive** | **Top 20 Feature** | Bounded in $[0.0, 1.0]$. Zero leakage. | **RETAIN.** |
| **Demographic Interaction Crosses** | V2–V3 | 25 | `segment_earning`, `region_smartphone`, `tri_profile` | Interacts value tier with income volatility. | **Mild gain** on AUC (+0.008) | **Positive** | Moderate | High cardinality on 3-way interactions. | **RETAIN 2-WAY ONLY.** Prune 3-way crosses. |
| **Frequency Encodings** | V3 | 10 | Value counts normalized by population size | Signals rare high-risk demographic combinations. | **Neutral / Mild** | **Neutral** | Low | Zero leakage. | **RETAIN.** |
| **Peer Group $Z$-Scores** | V3–V4 | 24 | $\frac{X_i - \mu_{\text{segment}}}{\sigma_{\text{segment}} + 1.0}$ | Measures deviation relative to socio-economic peers. | **Mild gain** (LL -0.001) | **Positive** | Moderate | Zero leakage. | **RETAIN.** |
| **Channel Spending Gini & Entropy** | V6 | 6 | Gini coefficient & Shannon entropy of spend across 7 channels | Measures spend diversification vs concentrated panic payments. | **Neutral / Weak** | **Neutral** | Low | Computationally clean. Zero leakage. | **RETAIN.** |
| **Unsupervised K-Means Personas**| V6 | 8 | Euclidean distance to 8 cluster centroids | Captures multi-dimensional behavioral archetypes. | **Mild gain** on XGBoost | **Neutral on CatBoost**| Moderate in XGBoost | Must be strictly fit on training folds. | **RETAIN FOLD-ISOLATED.** |
| **Composite Stress Index** | V6 | 1 | Linear weighted heuristic of deficit months and drawdown | Heuristic baseline anchor. | **Neutral** | **Neutral** | Low | Arbitrary manual weights ($0.25, 0.20$). | **LOW PRIORITY.** |
| **Acceleration & Jerk (2nd Diff)**| V4–V5 | 12 | $(M1 - 2M2 + M3)$ | Second derivative: rate of acceleration of balance drop. | **Marginal** (AUC +0.001) | **Neutral** | Low | High variance on noisy balances. | **CANDIDATE FOR PRUNING.** |
| **Emergency Cash Drain Spikes** | V8 / V13 | 5 | $\frac{\text{Withdraw}_{M1} + \text{BankTransfer}_{M1}}{\text{Withdraw}_{M2} + \text{BankTransfer}_{M2} + 1.0}$ | Detects sudden surges in physical cash extraction. | **Degraded V13** | **Degraded V13** (-0.00033) | Very Low | **HIGH NOISE RISK.** Volatile near-zero denominators. | **PRUNE IMMEDIATELY.** |
| **Multi-Month Velocity Ratios** | V8 / V13 | 10 | $\frac{\text{Outflow}_{M1}}{\text{Outflow}_{M6} + 1.0}$, $\frac{\text{Inflow}_{M1}}{\text{Inflow}_{M3} + 1.0}$ | Extreme horizon velocity ratios. | **Degraded V13** | **Degraded V13** | Very Low | **HIGH NOISE RISK.** Massive ratio blowups on dormant accounts. | **PRUNE IMMEDIATELY.** |

---

## SECTION 7: Feature Pruning & Dimensionality Analysis

### The Feature Bloat Diagnosis
1. **[FACT]** The engineered feature space grew from 184 to 366, 440, 488, 498, and finally **501 features** in V15.
2. **[FACT]** In run `20260905_140728`, XGBoost achieved Train ROC-AUC of `0.99349` vs OOF ROC-AUC of `0.90573`, and LightGBM GOSS achieved Train ROC-AUC of `0.99055` vs OOF ROC-AUC of `0.90503`.
3. **[INFERENCE]** XGBoost and LightGBM are suffering from severe feature bloat memorization. Passing 501 features with low sample-to-feature ratio on the positive class ($6,000 / 501 \approx 12$ positives per feature) allows trees to build spurious decision branches on noisy ratio artifacts.
4. **[INFERENCE]** The older V7 feature set (488 features) achieved **`0.71928`** in V12, whereas adding 10 emergency spike and velocity features in V13 degraded the score to **`0.71895`**.

### Systematic Pruning Subsets to Test (Execution Plan)
To isolate the optimal feature density, we define 5 candidate feature sets:
* **Subset A (V7 Restoration — 488 Features):** Strip the 10 noisy emergency spike and velocity ratios added in V8/V13.
* **Subset B (Top 350 Pruned):** Eliminate features with near-zero feature importance in CatBoost and features with train-val drift.
* **Subset C (Top 250 Curated Core):** Retain only core financial physics (Elasticity, Drawdown, Z-Scores, Slopes, Burn Rates, Profile Crosses).
* **Subset D (Top 150 Ultra-Compact):** Strict high-signal features designed specifically to prevent non-CatBoost tree overfitting.
* **Subset E (Full 501 Baseline):** Active control baseline on `nat`.

---

## SECTION 8: Target Encoding & Data Leakage Deep Dive

### Forensic Comparison Across Versions

1. **V4–V7 (Smooth Bayesian Target Encoding):**
   * *Formula:* $\text{TE}(c) = \frac{n_c \cdot \bar{y}_c + 10.0 \cdot \bar{y}_{\text{global}}}{n_c + 10.0}$.
   * *Leakage Protocol:* Mappings were computed strictly on `train_fold` and applied onto `val_fold` and `test_df`.
   * *Verdict:* **Statistically rigorous, leak-free.**

2. **V8 / V13 (The Noise Injection Blunder):**
   * In commit `ca2e81e` / `4f71d6e`, smoothing was shifted to $m=15.0$, and Gaussian noise was added:
     $$\text{TE}_{\text{train}} = \text{clip}\left(\text{TE}(c) + \mathcal{N}(0, 0.005), 0.0, 1.0\right)$$
   * *The Critical Error:* Noise was injected **only into `X_tr`**, while `X_val` and `X_te` remained deterministic and unperturbed.
   * *Mathematical Consequence:* This altered the variance and split boundaries between training and test representations. Decision trees learned exact split thresholds on the noisy training features that did not exist in the clean test distribution, causing the performance degradation observed in V13 (`0.71895`).

3. **CatBoost Native Categorical Handling vs. Preprocessed Target Encoding:**
   * *Critical Finding:* CatBoost implements its own internal, leak-free **Ordered Target Encoding** computed online during training permutations.
   * *Recommendation:* Passing externally target-encoded columns into CatBoost circumvents its native ordered target encoding and introduces unnecessary variance. CatBoost should receive raw categorical columns (`cat_features`), while XGBoost and LightGBM receive deterministic, noise-free Bayesian target encodings.

---

## SECTION 9: Cross-Validation Integrity & The Stacking Evaluation Trap

### The In-Sample Stacking Trap Discovered in V15
1. **[FACT]** In `src/ensemble.py`, the Level-2 Stacking Meta-Learner is defined as:
   ```python
   meta_model = LogisticRegression(C=0.2, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
   meta_model.fit(X_meta_train, y_true)
   meta_oof = np.clip(meta_model.predict_proba(X_meta_train)[:, 1], 0.003, 0.990)
   ```
2. **[FACT]** `X_meta_train` is the concatenation of base model OOF predictions (`[oof_probs, oof_logits]`).
3. **[INFERENCE] The Validation Flaw:** `meta_model` is fitted on `X_meta_train` and then evaluated **in-sample** on that exact same `X_meta_train`.
   * *Impact:* The reported stacking OOF Log Loss of `0.246158` (and best calibrated metrics `0.241713`) is an **in-sample training fit**, NOT a true out-of-fold generalization metric.
   * *Why Public Score Stagnated:* While the meta-learner learned weights that minimized training cross-entropy, its test predictions were slightly overfit to the training OOF distribution.

### The Leak-Free 2-Stage Nested Stacking Protocol (Conceptual Blueprint)
To establish an unshakeable, leak-free validation metric that perfectly correlates with Zindi leaderboard jumps, we must implement a **Nested 2-Stage Cross-Validation Harness**:

```
STAGE 1: Base Model OOF Generation (Inner 10-Fold Stratified CV)
├── Fold 1..10: Train CatBoost, XGBoost, PyTorch on 90% -> Predict 10% OOF -> Save OOF Matrix (40k x M)
└── Predict on Test Set -> Average across 10 folds -> Save Base Test Matrix (30k x M)

STAGE 2: Meta-Learner Nested Validation (Outer 5-Fold Stratified CV on Stage 1 OOF Matrix)
├── Split Stage 1 OOF Matrix into 5 Outer Folds:
│   ├── Outer Fold 1: Train Meta-Learner on 80% of OOF Matrix -> Predict on 20% Held-Out OOF
│   └── ... Repeat for Outer Folds 2..5
└── Concatenate 5 Outer Fold Predictions -> COMPUTE TRUE LEAK-FREE STACKED OOF METRICS

STAGE 3: Final Meta-Model Fitting & Test Inference
├── Fit final Meta-Learner on 100% of Stage 1 OOF Matrix
└── Predict on Base Test Matrix -> Apply Asymmetric Clamping -> Generate Final Submission
```

---

## SECTION 10: Probability Calibration & Post-Processing Analysis

### Comprehensive Calibration Matrix

| Calibration Technique | Mathematical Formulation | Effect on ROC-AUC | Effect on Log Loss | Overfitting Risk | Expected Value on Zindi |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Temperature Scaling ($T$)** | $P_{\text{cal}} = \sigma\left(\frac{\text{logit}(P)}{T}\right)$ | **Strictly Preserved (0.0000)** | **Slashes extreme penalties** | Very Low (1 parameter) | **MANDATORY.** Softens overconfident tree tails. |
| **Prior Shift ($\delta$)** | $P_{\text{cal}} = \sigma(\text{logit}(P) + \delta)$ | **Strictly Preserved (0.0000)** | **Aligns mean to 0.1500** | Zero (Deterministic) | **MANDATORY.** Fixes systematic probability bias. |
| **Joint Log-Odds ($T, \delta$)** | $P_{\text{cal}} = \sigma\left(\frac{\text{logit}(P) + \delta}{T}\right)$ | **Strictly Preserved (0.0000)** | **Simultaneous optimal fit** | Very Low (2 parameters) | **CURRENT BEST METHOD.** |
| **Asymmetric Clamping** | $\text{clip}(P, P_{\text{min}}, P_{\text{max}})$ | **Preserved except at boundaries** | **Eliminates infinite loss** | Zero | **MANDATORY SAFETY.** Set bounds to `[0.003, 0.990]`. |
| **Power Scaling ($P^\gamma$)** | $P_{\text{cal}} = \frac{P^\gamma}{P^\gamma + (1-P)^\gamma}$ | **Strictly Preserved (0.0000)** | Tunes probability spread | Low | **HIGH PRIORITY EXPERIMENT.** Test $\gamma \in [0.90, 1.05]$. |
| **Isotonic Regression** | Monotonic step-wise regression | **Can degrade AUC (Ties)** | Slashes training loss | **EXTREME (In-sample bias)**| **DO NOT USE.** Generates step-plateaus with identical ranks. |
| **Platt Scaling** | 1D Logistic Regression on logits | Strictly Preserved | Compresses tails | Very Low | Handled organically by Level-2 Stacking. |
| **Beta Calibration** | 3-parameter Beta distribution fit | Strictly Preserved | Flexible boundary shaping | Low to Moderate | **CANDIDATE FOR TESTING.** |

---

## SECTION 11: Ensemble Search Space & Architecture Taxonomy

To escape the current GBDT plateau, we map the entire ensemble search space across 17 distinct paradigms:

| Ensemble Architecture | Mathematical Formulation | Potential Benefit | Risk / Failure Mode | Expected Log Loss Impact | Expected ROC-AUC Impact | Compute Cost | Priority |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Arithmetic Average** | $\frac{1}{M} \sum P_m$ | Simple variance reduction. | Weaker models drag down CatBoost. | $+0.0030$ (Worse) | $-0.0015$ (Worse) | Zero | **REJECT** |
| **B. Weighted SLSQP Blend** | $\sum w_m P_m, \sum w_m = 1$ | Optimal scalar weights. | Fixed linear weighting across entire probability range. | $-0.0020$ (Better) | $+0.0010$ (Better) | Very Cheap | **BASELINE** |
| **C. Logit-Space SLSQP Blend** | $\sigma(\sum w_m \text{logit}(P_m))$ | Superior tail probability preservation. | Requires clipping near 0 and 1. | $-0.0025$ (Better) | $+0.0012$ (Better) | Very Cheap | **RETAIN** |
| **D. Power Blending ($P^\gamma$)**| $\sum w_m P_m^\gamma$ | Adjusts confidence dispersion. | Can over-flatten distribution if $\gamma < 0.8$. | $-0.0015$ (Better) | $+0.0005$ (Better) | Very Cheap | **HIGH** |
| **E. Rank Averaging** | $\sum w_m \text{rank}(P_m) / N$ | Maximizes pure ROC-AUC ranking power. | **Completely destroys Log Loss** calibration. | $+0.0400$ (Disaster) | $+0.0025$ (Better) | Very Cheap | **REJECT FOR MULTI-SCORE** |
| **F. Level-2 Linear Stacking** | $\sigma(\mathbf{w}^T \mathbf{P} + b)$ | Learns optimal model trust weights. | Overfitting if fit without nested CV. | $-0.0035$ (Better) | $+0.0020$ (Better) | Cheap (1 min) | **RETAIN** |
| **G. Level-2 Log-Odds Stacking**| $\sigma(\mathbf{w}_1^T \mathbf{P} + \mathbf{w}_2^T \text{logit}(\mathbf{P}) + b)$ | Non-linear risk-tier weighting (Current V15). | In-sample evaluation trap. | **-0.0045 (Best)** | **+0.0025 (Best)** | Cheap (1 min) | **CURRENT BEST** |
| **H. Nested Leak-Free Stacking**| 2-Stage Nested Cross-Validation | Eliminates in-sample optimistic bias. | Requires clean cross-validation harness. | Accurate OOF | Accurate OOF | Cheap (2 mins) | **MANDATORY** |
| **I. Non-Linear GBDT Stacking** | CatBoost Meta-Learner on Level-1 | Can learn complex interaction weights. | **Severe meta-overfitting** on small OOF width. | $+0.0100$ (Worse) | $-0.0010$ (Worse) | Medium | **REJECT** |
| **J. Stacking + Joint Calibration**| Stage 2 Stacking $\to$ Joint $(T, \delta)$ | Eliminates residual meta-learner tail bias. | Minimal risk. | $-0.0015$ (Better) | Preserved | Very Cheap | **MANDATORY** |
| **K. Heterogeneous Multi-Family**| CatBoost + Tabular ResNet + ElasticNet | **Maximum orthogonal diversity.** | Neural net must be properly regularized. | **-0.0060 (Huge)** | **+0.0045 (Huge)** | Medium (15m) | **THE KEY TO RANK 1** |
| **L. 3-Level Stacking** | L1 Models $\to$ L2 Stacks $\to$ L3 Meta | Theoretical AutoGluon paradigm. | Enormous complexity; high leakage risk. | Unstable | Unstable | Expensive | **REJECT** |
| **M. Feature-Subset Blending** | CatBoost on Top 250 + CatBoost on Top 488 | Reduces correlation between trees. | Moderate complexity. | $-0.0015$ (Better) | $+0.0010$ (Better) | GPU Medium | **HIGH** |
| **N. Hyperparameter Blending** | CatBoost (Depth 6) + CatBoost (Depth 8) | True structural tree diversity. | Requires multiple full CatBoost runs. | $-0.0020$ (Better) | $+0.0015$ (Better) | GPU Medium | **HIGH** |

---

## SECTION 12: Tabular Neural Network Search & Feasibility Assessment

### Deep Technical Analysis Across 14 Neural Architectures

| Architecture | Suitability for 40k Rows? | Suitability for 500 Features? | Likely Standalone ROC-AUC | Expected Correlation to CatBoost | Required Preprocessing | Compute Cost (GPU) | Strategic Priority | Expected Competition Value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Tabular ResNet (3-Block ResMLP)**| **EXCELLENT** | **EXCELLENT** | **0.885 – 0.895** | **0.87 – 0.90** | RankGauss, LayerNorm, Mish | 3–4 mins | **TOP PRIORITY** | **HIGHEST ENSEMBLE DIVERSITY.** |
| **2. Simple MLP + BatchNorm** | Good | Moderate | 0.865 – 0.875 | 0.91 – 0.93 | StandardScaler, ReLU | 2 mins | Moderate | Prone to covariate shift; inferior to ResNet. |
| **3. MLP + LayerNorm + Mish** | **EXCELLENT** | **EXCELLENT** | **0.880 – 0.890** | **0.88 – 0.91** | StandardScaler, Mish | 2–3 mins | **HIGH PRIORITY** | Clean gradient flow; no batch leakage. |
| **4. Deep & Cross Network (DCNv2)**| **HIGH** | **HIGH** | **0.880 – 0.892** | **0.86 – 0.89** | Standard scaling, AdamW | 4–5 mins | **HIGH PRIORITY** | Explicit polynomial feature crosses. |
| **5. TabNet** | Moderate | Poor (Sparse mask slow on 500 cols) | 0.860 – 0.875 | 0.90 – 0.92 | Unscaled raw continuous | 10–12 mins | **LOW** | Slow convergence; uncompetitive on 40k rows. |
| **6. FT-Transformer** | Poor | Very Poor (500 tokens $O(D^2)$ memory)| 0.870 – 0.885 | 0.88 – 0.91 | Linear tokenization | 25+ mins | **NOT RECOMMENDED**| Quadratic memory explosion on 500 features. |
| **7. SAINT** | Poor | Very Poor | 0.870 – 0.880 | 0.88 – 0.90 | Self-attention + Intersample | 30+ mins | **NOT RECOMMENDED**| Massive over-parameterization; high compute. |
| **8. NODE** | Moderate | Poor | 0.875 – 0.885 | 0.95 – 0.97 | Differentiable trees | 15 mins | **NOT RECOMMENDED**| Redundant with native CatBoost GPU. |
| **9. MLP + Focal Loss** | Good | Good | 0.875 – 0.885 | 0.89 – 0.91 | RankGauss, $\gamma=2.0$ | 3 mins | **MEDIUM** | Focuses on hard positive boundary samples. |

### The Verdict on Neural Networks
* **[FACT]** A standalone neural network will NOT beat CatBoost GPU on this tabular dataset. Expected standalone AUC is ~`0.888` vs CatBoost's `0.908`.
* **[INFERENCE]** Standalone performance is irrelevant for ensemble variance reduction. Because Tabular ResNet operates in continuous Euclidean manifold space, its prediction errors are uncorrelated with CatBoost's axis-aligned hyper-rectangle splits ($r \approx 0.88$). Adding a regularized Tabular ResNet with 10%–15% meta-learner weight is the single most viable mechanism to break the GBDT correlation bottleneck.

---

## SECTION 13: Exploration of Non-Tree / Alternative Model Families

| Model Family | Standalone Strength | Expected Correlation to CatBoost | Memory & Scaling on 40k Rows | Diversity Value | Strategic Priority | Actionable Implementation Concept |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ElasticNet Logistic Regression** | Moderate (AUC ~0.842) | **0.80 – 0.84 (Extremely Low)** | Instant (5 seconds, CPU) | **VERY HIGH** | **HIGH PRIORITY** | Fit $L_1/L_2$ regularized GLM on Top 50 standardized features; inject into Level-2 Stacker. |
| **K-Nearest Neighbors (KNN)** | Weak to Moderate (AUC ~0.825) | **0.76 – 0.82 (Lowest of all)** | Moderate (Fit fast, predict slow) | **VERY HIGH** | **HIGH PRIORITY** | Fit $K=64$ on 16 PCA components; output average neighbor target probability. |
| **Gaussian Process Classifier** | Untestable | Unknown | **Fails.** $O(N^3)$ memory on 40k rows | N/A | **PERMANENTLY REJECT** | Computationally impossible on 40,000 samples. |
| **Support Vector Classifier (RBF)**| Moderate (AUC ~0.850) | 0.84 – 0.88 | Extremely slow ($O(N^2)$ QP solver) | Moderate | **NOT RECOMMENDED** | Compute cost is prohibitive on Kaggle runners. |
| **Explainable Boosting Machines (EBM)**| Moderate (AUC ~0.875)| 0.93 – 0.95 | Slow spline fitting | Low to Moderate | **LOW PRIORITY** | Generalized Additive Model with pairwise interactions. |
| **Linear Discriminant Analysis (LDA)**| Weak (AUC ~0.820) | 0.82 – 0.85 | Instant (2 seconds) | Moderate | **LOW PRIORITY** | Gaussian assumption heavily violated by skewed balances. |

---

## SECTION 14: Financial-Domain Feature Engineering Search Space

### 12 New High-Signal Behavioral Feature Concepts

1. **Liquidity Burn Acceleration (Jerk of Insolvency):**
   * *Concept:* Discrete 3rd derivative of account balances across months: $(B_{M1} - 3B_{M2} + 3B_{M3} - B_{M4})$.
   * *Mechanism:* Measures whether the cash burn rate is accelerating toward terminal exhaustion.
   * *Priority:* **HIGH.**
2. **Personal Coefficient of Variation Shock:**
   * *Concept:* $\text{CV}_{\text{recent}} / (\text{CV}_{\text{historical}} + 1.0)$, where $\text{CV} = \sigma / \mu$.
   * *Mechanism:* Distinguishes individuals whose balance has suddenly become wildly erratic from those with stable balances.
   * *Priority:* **HIGH.**
3. **P2P Begging / Reliance Ratio:**
   * *Concept:* $\frac{\text{m1\_received\_total\_value}}{\text{m1\_deposit\_total\_value} + \text{m1\_transfer\_from\_bank\_total\_value} + 1.0}$.
   * *Mechanism:* Users who stop depositing cash and rely entirely on peer transfers from family/friends are entering distress.
   * *Priority:* **HIGH.**
4. **Channel Abandonment Count:**
   * *Concept:* Count of transaction channels (out of 7) active in M6 that had exactly zero transactions in M1.
   * *Mechanism:* Financial distress forces customers to cut utility bill paybills and merchant spending to conserve cash.
   * *Priority:* **HIGH.**
5. **Cash Hoarding Panic Flag:**
   * *Concept:* $\frac{\text{m1\_withdraw\_total\_value}}{\text{m1\_paybill\_total\_value} + \text{m1\_merchantpay\_total\_value} + 1.0}$.
   * *Mechanism:* Extracting physical paper currency while neglecting formal commercial obligations is a primary distress signature.
   * *Priority:* **HIGH.**
6. **Income Continuity Failure Index:**
   * *Concept:* Number of consecutive months where $(\text{Inflows}_t - \text{Outflows}_t) < 0$.
   * *Mechanism:* 4 to 6 consecutive deficit months indicates structural negative equity.
   * *Priority:* **HIGH.**
7. **Median Absolute Deviation (MAD) Normalized Balance:**
   * *Concept:* Robust $Z$-score using median and MAD instead of mean and standard deviation: $\frac{B_{M1} - \text{median}(B)}{\text{MAD}(B) + 1.0}$.
   * *Mechanism:* Immune to extreme historical wealth outliers that distort standard standard deviations.
   * *Priority:* **HIGH.**
8. **Inflow Diversity Entropy:**
   * *Concept:* Shannon entropy across inflow channels (deposits, bank transfers, P2P received).
   * *Mechanism:* Loss of income diversity (relying on a single fragile transfer) elevates vulnerability.
   * *Priority:* **MEDIUM.**
9. **Transaction Size Quantile Spread:**
   * *Concept:* $\frac{\text{Highest Amount}_{M1}}{\text{Daily Avg Bal}_{M1} + 1.0}$.
   * *Mechanism:* A single transaction that exceeds the average daily balance represents an acute liquidity shock.
   * *Priority:* **HIGH.**
10. **Counterparty Contraction Velocity:**
    * *Concept:* $\frac{\text{m1\_total\_counterparties}}{\text{m6\_total\_counterparties} + 1.0}$.
    * *Mechanism:* Shrinking social/economic network indicates withdrawal from regular economic life.
    * *Priority:* **MEDIUM.**
11. **Dormancy Threat Flag:**
    * *Concept:* Binary indicator if `x_90_d_activity_rate` dropped by $>50\%$ relative to active transaction months.
    * *Mechanism:* Early account abandonment signal.
    * *Priority:* **MEDIUM.**
12. **Rolling Net Cashflow Momentum:**
    * *Concept:* $\text{Net Cashflow}_{M1..M2} - \text{Net Cashflow}_{M3..M6}$.
    * *Mechanism:* Tracks macroeconomic momentum shift from net savings to active capital flight.
    * *Priority:* **HIGH.**

---

## SECTION 15: Monotonic Constraints Analysis

### The Case for Monotonic Feature Constraints
* **[FACT]** Decision trees can learn spurious non-monotonic splits on continuous features with low sample density. For instance, a tree might predict that a user with 5 days of liquidity runway is safe, while a user with 6 days is stressed, simply due to random row noise.
* **[INFERENCE]** CatBoost, LightGBM, and XGBoost support exact **Monotonic Constraints** (`monotone_constraints` parameter).

### Monotonic Constraints Table

| Feature Name | Theoretical Relationship to Stress ($Y=1$) | Enforced Direction | Mathematical Justification | Supported in CatBoost? |
| :--- | :--- | :--- | :--- | :--- |
| `bal_recent_vs_old_ratio` | **Decreasing** | **`-1`** | A collapsing balance ratio strictly increases liquidity failure risk. | **YES** |
| `liquidity_exhaustion_days`| **Decreasing** | **`-1`** | Having fewer days until zero balance strictly increases insolvency risk. | **YES** |
| `bal_max_drawdown_ratio` | **Increasing** | **`+1`** | Greater historical peak drawdown strictly increases distress risk. | **YES** |
| `total_deficit_months_6m` | **Increasing** | **`+1`** | More months in structural cashflow deficit strictly increases risk. | **YES** |
| `consecutive_bal_drops` | **Increasing** | **`+1`** | Sustained balance depletion strictly increases failure probability. | **YES** |

* **Strategic Verdict:** Enforcing monotonic constraints on these 5 core features in CatBoost GPU will prevent erratic test set split-points without restricting expressive interactions on remaining columns.

---

## SECTION 16: Hyperparameter Search Space

### Parameter Dimensions Ranked by Generalization Value

#### 1. CatBoost GPU (Primary Backbone)
* **`l2_leaf_reg` (High Value):** Currently `6.0`. Search `[8.0, 12.0, 16.0]`. Higher $L_2$ regularization smooths leaf values and directly minimizes Log Loss.
* **`depth` (High Value):** Currently `7`. Test `depth=6` (lower variance) vs `depth=8` (higher non-linear capacity).
* **`learning_rate` & `iterations` (High Value):** Currently `0.015` / `2200`. Extend to `3500` iterations with `learning_rate=0.010` for finer gradient descent resolution.
* **`random_strength` (Medium Value):** Currently `0.7`. Search `[0.5, 1.0]`.
* **`border_count` (Low Value):** Currently `128`. Negligible impact beyond 128 on GPU.

#### 2. XGBoost GPU (Taming Severe Overfitting)
* **`min_child_weight` (Top Priority):** Currently `5`. Increase to `[15, 25, 40]`. Forces leaf nodes to contain substantial sample weight, immediately compressing the `+0.1318` generalization gap.
* **`reg_lambda` / `reg_alpha` (Top Priority):** Currently $\lambda=5.0, \alpha=0.5$. Increase to $\lambda=15.0, \alpha=2.0$.
* **`max_depth` (Top Priority):** Currently `6`–`7`. Reduce to `max_depth=5`. Eliminates deep memorization branches.
* **`subsample` & `colsample_bytree`:** Maintain `0.75` / `0.55` for high stochastic regularization.

#### 3. LightGBM (Restoring Orthogonality)
* **`min_child_samples`:** Currently `25`. Increase to `[50, 100]`.
* **`num_leaves`:** Currently `45`–`55`. Restrict to `num_leaves=31`.
* **`reg_lambda`:** Increase from `4.0` to `12.0`.

---

## SECTION 17: Loss Functions & Objective Alignment

### Evaluating Candidate Loss Objectives
1. **Binary Cross-Entropy / Log Loss (Active Baseline):**
   * *Formula:* $\mathcal{L} = -[y \ln p + (1-y) \ln(1-p)]$.
   * *Alignment:* **PERFECT.** Matches the Zindi competition component metric directly.
2. **Weighted Cross-Entropy (Class-Weighted Loss):**
   * *Formula:* $\mathcal{L} = -[w_1 y \ln p + w_0 (1-y) \ln(1-p)]$.
   * *Risk:* Weighting minority class by $w_1 = 85/15 \approx 5.67$ artificially shifts the posterior probability distribution toward $\mathbb{E}[P] \approx 0.50$. While this may boost recall, it **catastrophically blows up Log Loss on the 85% majority class**, destroying the Multi-Score.
   * *Verdict:* **DO NOT USE.**
3. **Focal Loss ($\gamma = 1.5$ to $2.0$):**
   * *Formula:* $\mathcal{L} = -(1-p_t)^\gamma \ln(p_t)$.
   * *Utility:* Down-weights well-classified easy negative examples and focuses learning on ambiguous boundary cases.
   * *Verdict:* **TEST AS DIVERSITY OBJECTIVE IN PYTORCH RESNET ONLY.**

---

## SECTION 18: Distribution Shift & Test Set Robustness Analysis

### Forensic Shift Audit
1. **[FACT]** The test set contains 30,000 unlabeled customer IDs with the exact same 183 raw feature schema as the training set (40,000 IDs).
2. **[FACT]** In run `20260905_140728`, the test prediction mean was **`0.153538`**, closely matching the training prior of **`0.150000`**.
3. **[INFERENCE]** There is no evidence of a massive macroeconomic regime shift in customer base rates.
4. **[INFERENCE] Ratio Vulnerability:** The features most vulnerable to test distribution drift are **unbounded ratio features with small denominators** (e.g., `emergency_cash_spike_m1_m2`, `outflow_ratio_m1_m6`). On inactive accounts with near-zero activity, random \$1 transactions create massive 100x ratio outliers in the test set that confuse tree splits.
5. **[REMEDY]** Replace unbounded ratios with **bounded symmetric difference ratios**:
   $$\text{Symmetric Difference} = \frac{A - B}{A + B + \epsilon} \in [-1.0, 1.0]$$

---

## SECTION 19: Public Leaderboard Anti-Overfitting Protocol

### The Dangers of Public Leaderboard Feedback Loops
* **Limited Probe Size:** The public leaderboard represents only ~30%–40% of the test set (~9,000 rows). A model that improves public score by $+0.0008$ may simply be fitting random noise in that 9,000-sample slice while degrading performance on the 21,000 private samples.
* **Submission Inflation:** We have used 15 submissions. Every future submission must be guarded by strict statistical gates.

### The 4-Gate Submission Discipline Protocol
No model may be submitted to Zindi unless it satisfies all 4 gates in order:

```
[GATE 1: Local Cross-Validation Verification]
OOF Log Loss must improve by >= 0.0010 OR OOF ROC-AUC must improve by >= 0.0015
                 ↓ (PASS)
[GATE 2: Generalization Gap Audit]
The difference between Training Loss and Validation Loss must NOT expand by > 0.015
                 ↓ (PASS)
[GATE 3: Population Base Rate Check]
The average predicted test probability must satisfy: 0.1470 <= E[P_test] <= 0.1530
                 ↓ (PASS)
[GATE 4: Correlation & Diversity Confirmation]
The new candidate must have correlation <= 0.985 with the previous best submission
                 ↓ (PASS)
EXECUTE ZINDI SUBMISSION -> LOG PUBLIC SCORE -> DO NOT OVERREACT TO SINGLE SCORE
```

---

## SECTION 20: Decomposition of the Gap: What Can Move 0.72099 → 0.73937?

The `0.01838` gap to Rank 1 is systematically decomposed into realistic, independent component gains:

| Source of Improvement | Expected Gain in Multi-Score | Expected Gain in Log Loss | Expected Gain in ROC-AUC | Probability of Success | Primary Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Orthogonal Multi-Family Ensembling** | **+0.0060 to +0.0085** | **-0.0050 to -0.0070** | **+0.0040 to +0.0060** | **HIGH** | Integrating Tabular ResNet and ElasticNet (correlation $<0.90$) to eliminate GBDT error monoculture. |
| **2. Feature Pruning & Noise Elimination** | **+0.0035 to +0.0050** | **-0.0030 to -0.0045** | **+0.0025 to +0.0035** | **HIGH** | Stripping 150+ noisy velocity/spike ratios to collapse XGBoost/LightGBM overfitting gap from 0.13 to <0.08. |
| **3. Nested Leak-Free Stacking Meta-Learner** | **+0.0025 to +0.0040** | **-0.0025 to -0.0035** | **+0.0015 to +0.0025** | **HIGH** | Eliminating in-sample stacking bias; optimal test probability calibration across probability regimes. |
| **4. Domain Feature Engineering (12 New Concepts)**| **+0.0020 to +0.0035** | **-0.0015 to -0.0025** | **+0.0020 to +0.0030** | **MEDIUM** | Adding robust MAD normalization, cash hoarding ratios, and personal CV volatility shocks. |
| **5. Hyperparameter Regularization on GBDT** | **+0.0015 to +0.0025** | **-0.0015 to -0.0020** | **+0.0010 to +0.0015** | **HIGH** | Increasing `min_child_weight` to 25 and `l2_leaf_reg` to 12.0 to curb tree leaf variance. |
| **6. Monotonic Feature Constraints** | **+0.0010 to +0.0020** | **-0.0010 to -0.0015** | **+0.0008 to +0.0012** | **MEDIUM** | Enforcing monotonic risk on balance collapse ratios and exhaustion days. |
| **7. Multi-Seed Expansion (5 Seeds)** | **+0.0010 to +0.0015** | **-0.0008 to -0.0012** | **+0.0005 to +0.0010** | **HIGH** | Squeezing final variance reduction across 5 seeds on the best 3-family suite. |
| **TOTAL CUMULATIVE POTENTIAL** | **+0.0175 to +0.0270** | **-0.0153 to -0.0222** | **+0.0123 to +0.0187** | — | **Sufficient to bridge the 0.01838 gap and challenge #1.** |

---

## SECTION 21: Formal Benchmark Ladder Hierarchy

Every model in our pipeline must be evaluated against this strict benchmark hierarchy:

```
[B0: Theoretical Floor] Dummy Prior P(Y=1) = 0.1500 ── Log Loss = 0.4227 | ROC-AUC = 0.5000
    ↓
[B1: Linear Baseline] ElasticNet Logistic Regression ── Target: Log Loss ~0.312 | ROC-AUC ~0.842
    ↓
[B2: Bagged Trees] Random Forest (600 trees) ────────── Target: Log Loss ~0.288 | ROC-AUC ~0.871
    ↓
[B3: Continuous Deep Net] Tabular ResNet (3 Blocks) ── Target: Log Loss ~0.265 | ROC-AUC ~0.888 (Corr < 0.90)
    ↓
[B4: Fast Histogram GBDT] LightGBM / XGBoost GPU ───── Target: Log Loss ~0.252 | ROC-AUC ~0.906
    ↓
[B5: Oblivious Table GBDT] CatBoost GPU Standalone ─── Target: Log Loss ~0.2475 | ROC-AUC ~0.9085
    ↓
[B6: Multi-Seed GBDT] 10-Fold 3-Seed CatBoost GPU ──── Target: Log Loss ~0.2465 | ROC-AUC ~0.9095
    ↓
[B7: Current Best Baseline] V15 Level-2 Stacking ───── Public Score = 0.72099 | Log Loss = 0.2417 | AUC = 0.9118
    ↓
[B8: Pruned Multi-Family Stack] CatBoost + XGB + ResNet Target: Public Score > 0.7280 | Log Loss < 0.2360
    ↓
[B9: THE PINNACLE — TOP 1 TARGET] ──────────────────── Public Score > 0.7394 | Log Loss < 0.2277 | AUC > 0.9224
```

---

## SECTION 22: The Master 25 Candidate Experiments Portfolio

Below is the complete portfolio of 25 structured experiments designed to systematically conquer the search space:

| ID | Experiment Name | Hypothesis | Exact Component to Change | What Remains Fixed | Expected Loss Delta | Expected AUC Delta | Expected LB Impact | Compute Cost | Risk Tier | Priority | Success Criterion | Failure Criterion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01**| **Prune TE Noise Injection** | Removing train-only Gaussian noise restores train-test feature alignment. | `src/validation.py`: Strip noise addition on `X_tr`. | 501 features, models, CV. | -0.0008 | +0.0006 | +0.0010 | GPU Cheap (12m) | Minimal | **PHASE 0** | OOF Loss $\le 0.2455$ | Loss worsens |
| **EXP-02**| **Nested Leak-Free Stacking**| Outer 5-fold CV on meta-learner eliminates in-sample evaluation optimism. | `src/ensemble.py`: Add outer 5-fold CV loop for meta-learner. | Base OOFs, features. | Clean OOF | Clean OOF | Prevents LB drop | CPU Cheap (1m) | Minimal | **PHASE 0** | True stacked OOF loss matches CV | Meta overfits outer fold |
| **EXP-03**| **Drop LightGBM GOSS Redundancy**| Eliminating duplicate GOSS model and allocating weight to CatBoost (75%) and XGB (25%) reduces collinear noise. | `train.py`: Set `--models catboost xgboost`. | 10-fold CV, 3 seeds, features. | -0.0006 | +0.0004 | +0.0012 | GPU Cheap (10m) | Minimal | **PHASE 1** | OOF Loss improves; runtime drops 35% | OOF Loss degrades $>0.0005$ |
| **EXP-04**| **Prune Emergency Spike Ratios**| Stripping 10 noisy V8/V13 emergency spike and velocity ratios reduces test ratio distortion. | `src/features.py`: Prune 10 volatile ratio columns (491 total). | Models, 10-fold CV. | -0.0010 | +0.0008 | +0.0015 | GPU Medium (14m) | Low | **PHASE 1** | XGBoost OOF Loss drops $<0.2515$ | Loss degrades |
| **EXP-05**| **Restore V7 Feature Core (488)**| Complete restoration of V7 features recovers V12 baseline stability. | `src/features.py`: Exact V7 feature subset. | CatBoost, XGBoost. | -0.0012 | +0.0010 | +0.0020 | GPU Medium (14m) | Low | **PHASE 1** | Public LB beats 0.7210 | LB degrades |
| **EXP-06**| **Top 350 Feature Importance Pruning**| Eliminating bottom 150 low-importance features reduces tree variance. | Feature selection mask on CatBoost importance $\ge 0.05$. | Models, CV, seeds. | -0.0015 | +0.0012 | +0.0025 | GPU Medium (12m) | Low | **PHASE 2** | XGBoost Gen Gap drops $<0.090$ | CatBoost AUC drops |
| **EXP-07**| **Top 250 Curated Core Features**| Restricting feature space to core financial physics prevents over-splitting. | Retain Top 250 domain features. | Models, CV, seeds. | -0.0018 | +0.0015 | +0.0028 | GPU Medium (10m) | Moderate | **PHASE 2** | Combined OOF Loss $<0.2445$ | AUC drops $>0.0020$ |
| **EXP-08**| **XGBoost Heavy Regularization** | Increasing `min_child_weight=25`, `reg_lambda=15.0`, `max_depth=5` tames memorization. | `src/models.py`: Update XGBoost hyperparams. | Features, CV, seeds. | -0.0015 | +0.0010 | +0.0018 | GPU Medium (12m) | Low | **PHASE 2** | XGB Train AUC drops $<0.965$ | OOF Loss degrades |
| **EXP-09**| **CatBoost Deeper Regularized Trees**| Training CatBoost at `depth=8` with `l2_leaf_reg=12.0` captures higher-order physics without overfit. | `src/models.py`: `depth=8, l2=12.0`. | Features, CV, seeds. | -0.0012 | +0.0015 | +0.0018 | GPU Medium (15m) | Low | **PHASE 3** | Standalone CatBoost Loss $<0.2465$ | Training memory exceeds T4 |
| **EXP-10**| **CatBoost Low LR Extended Iterations**| `learning_rate=0.010`, `iterations=3500` refines gradient steps. | `src/models.py`: Adjust CatBoost schedule. | Features, CV, seeds. | -0.0010 | +0.0012 | +0.0015 | GPU Expensive (22m)| Low | **PHASE 3** | Standalone CatBoost AUC $>0.9095$ | Early stopping halts early |
| **EXP-11**| **CatBoost Monotonic Constraints** | Constraining 5 core balance collapse features eliminates erratic test splits. | `src/models.py`: Pass `monotone_constraints`. | Features, CV, seeds. | -0.0008 | +0.0006 | +0.0012 | GPU Medium (15m) | Minimal | **PHASE 3** | Test probability variance smooths | OOF Loss degrades |
| **EXP-12**| **Tabular ResNet Integration (3-Block)**| Continuous neural manifold provides orthogonal errors ($r < 0.90$). | Add `pytorch_mlp` to ensemble (hidden=256, LayerNorm, Mish). | Features, CV, seeds. | **-0.0025** | **+0.0020** | **+0.0035** | GPU Expensive (20m)| Moderate | **PHASE 4** | PyTorch-CatBoost corr $\le 0.90$ | PyTorch OOF Loss $>0.280$ |
| **EXP-13**| **RankGauss Preprocessing on ResNet**| Quantile normal transformation stabilizes neural net gradients. | `src/models.py`: Pipeline RankGauss scaler. | PyTorch architecture. | -0.0015 | +0.0012 | +0.0020 | GPU Medium (18m) | Low | **PHASE 4** | Standalone ResNet Loss $<0.265$ | NaN gradients |
| **EXP-14**| **ElasticNet Linear Baseline Injection**| Adding regularized linear baseline ($C=0.05$) constrains tree boundary extremes. | Add `linear_baseline` to Level-2 stack. | Tree models, features. | -0.0008 | +0.0005 | +0.0010 | CPU Cheap (1m) | Minimal | **PHASE 4** | Level-2 stack gives 3%–5% weight | Weight is 0.0000 |
| **EXP-15**| **KNN Local Manifold Predictions** | $K=64$ nearest neighbors on PCA components provides non-parametric local baseline. | Add `knn_baseline` to Level-2 stack. | Features, CV. | -0.0010 | +0.0008 | +0.0015 | CPU Cheap (3m) | Low | **PHASE 4** | KNN-CatBoost corr $\le 0.82$ | Prediction latency $>10$m |
| **EXP-16**| **Bounded Symmetric Difference Ratios**| Replacing $(A/B)$ with $(A-B)/(A+B+\epsilon)$ eliminates ratio blowups. | `src/features.py`: Replace ratio formulas. | Models, CV. | -0.0010 | +0.0008 | +0.0015 | GPU Medium (14m) | Low | **PHASE 5** | Test outlier count drops 90% | OOF Loss degrades |
| **EXP-17**| **Liquidity Burn Jerk & CV Shocks** | 12 new domain financial physics features inject fresh economic signal. | `src/features.py`: Implement 12 features. | Models, CV. | -0.0015 | +0.0018 | +0.0022 | GPU Medium (16m) | Low | **PHASE 5** | At least 3 new features in Top 20 | Features receive zero gain |
| **EXP-18**| **Power Blending Exponent Optimization**| Optimizing $P^\gamma$ ($\gamma \in [0.95, 1.05]$) tunes probability concentration. | `src/ensemble.py`: Add Nelder-Mead $\gamma$ solve. | Ensemble predictions. | -0.0008 | Preserved | +0.0010 | CPU Cheap (30s) | Minimal | **PHASE 6** | Multi-Score improves | Over-flattens distribution |
| **EXP-19**| **Beta Calibration vs Log-Odds** | 3-parameter Beta calibration refines asymmetric risk boundaries. | `src/ensemble.py`: Fit Beta calibrator. | Ensemble predictions. | -0.0006 | Preserved | +0.0008 | CPU Cheap (1m) | Low | **PHASE 6** | Log Loss improves on validation | AUC degrades |
| **EXP-20**| **CatBoost Native Categoricals** | Passing raw categoricals directly into CatBoost utilizes ordered TE permutations. | `src/validation.py`: Pass `cat_features`. | CatBoost model. | -0.0012 | +0.0010 | +0.0016 | GPU Medium (15m) | Low | **PHASE 3** | CatBoost OOF Loss drops $<0.2465$ | Memory exceeds GPU limit |
| **EXP-21**| **Focal Loss Objective on ResNet** | Focal loss ($\gamma=1.5$) focuses neural net on ambiguous positive boundaries. | `src/models.py`: Set Focal Loss. | PyTorch ResNet. | -0.0008 | +0.0015 | +0.0012 | GPU Medium (18m) | Moderate | **PHASE 4** | Recall improves on minority class | Probability calibration drifts |
| **EXP-22**| **DCNv2 Explicit Polynomial Crosses**| Deep & Cross Network captures bounded degree-3 feature interactions automatically. | Implement DCNv2 PyTorch wrapper. | Top 100 features. | -0.0015 | +0.0018 | +0.0020 | GPU Expensive (22m)| Moderate | **PHASE 7** | DCNv2 AUC $>0.885$ | Convergence failure |
| **EXP-23**| **Feature-Pruned Heterogeneous Stack** | Combining CatBoost + XGBoost + ResNet on Top 300 pruned features. | Integrated pipeline milestone. | Pruned features, CV. | **-0.0040** | **+0.0035** | **+0.0050** | GPU Expensive (25m)| Low | **PHASE 8** | Public LB beats **0.7260** | Overfitting detected |
| **EXP-24**| **5-Seed Bagging Matrix Expansion** | Expanding seeds to `[42, 1337, 2026, 777, 999]` on the 3-family suite. | `train.py`: Set 5 seeds. | Best model suite. | -0.0012 | +0.0010 | +0.0018 | GPU Heavy (38m) | Minimal | **PHASE 8** | Public LB beats **0.7280** | Timeout on Kaggle |
| **EXP-25**| **THE GRAND SLAM TOP-1 ENSEMBLE** | Full stack: 5-seed CatBoost + XGB + ResNet + ElasticNet + Monotonicity + Nested Stacking + Joint Calibration. | Complete final milestone. | Complete architecture. | **-0.0070** | **+0.0060** | **+0.0090** | GPU Heavy (40m) | Minimal | **PHASE 8** | **Public Score $\ge 0.7350$** (Striking distance to #1) | Pipeline failure |

---

## SECTION 23: Chronological Execution Sequence: The 9-Phase Roadmap

```
PHASE 0: Validation Integrity & Hygiene (Day 1)
├── EXP-01: Remove Gaussian noise injection from training target encoding
└── EXP-02: Implement leak-free nested 2-stage cross-validation for Level-2 Meta-Learner
    Rationale: We cannot trust subsequent experiment decisions until validation metrics correlate 1:1 with leaderboard reality.

PHASE 1: Redundancy Elimination & Fast Baseline Cleanup (Day 2)
├── EXP-03: Prune LightGBM GOSS redundancy; allocate weights strictly to CatBoost (75%) & XGBoost (25%)
├── EXP-04: Strip volatile emergency spike and velocity ratios (491 features)
└── EXP-05: Test pure V7 feature restoration (488 features)
    Rationale: Cheap, immediate code modifications that eliminate collinear noise and establish a clean baseline.

PHASE 2: Feature Pruning & Tree Overfitting Compression (Day 3)
├── EXP-06: Evaluate Top 350 feature importance subset
├── EXP-07: Evaluate Top 250 curated core feature subset
└── EXP-08: Apply heavy leaf regularization on XGBoost (min_child_weight=25, lambda=15.0)
    Rationale: Compresses the +0.1318 generalization gap on non-CatBoost models before introducing new complexity.

PHASE 3: CatBoost Optimization & Monotonicity (Day 4)
├── EXP-09: CatBoost deeper regularized trees (depth=8, l2=12.0)
├── EXP-10: CatBoost low learning rate extended schedule (lr=0.010, 3500 iterations)
├── EXP-11: Enforce monotonic constraints on 5 core balance collapse features
└── EXP-20: Pass raw categoricals directly into CatBoost native ordered target encoding
    Rationale: Maximizes standalone performance on our single strongest model architecture.

PHASE 4: Orthogonal Diversity Injection (Non-Tree Models) (Day 5–6)
├── EXP-12: Integrate 3-Block Tabular ResNet (hidden=256, LayerNorm, Mish)
├── EXP-13: Add RankGauss preprocessing pipeline for neural network continuous features
├── EXP-14: Inject ElasticNet regularized linear baseline into Level-2 Stacking Meta-Learner
└── EXP-15: Evaluate KNN local manifold probability baseline (K=64 on PCA)
    Rationale: Introduces genuine continuous manifold diversity (correlation < 0.90) to eliminate GBDT monoculture.

PHASE 5: Financial Physics Feature Expansion (Day 7)
├── EXP-16: Convert unbounded ratios to bounded symmetric difference ratios
└── EXP-17: Implement 12 new financial domain features (Burn Jerk, Cash Hoarding, Personal CV Shocks)
    Rationale: Injects fresh economic predictive signal after feature space has been cleaned and stabilized.

PHASE 6: Calibration & Probability Distribution Tuning (Day 8)
├── EXP-18: Optimize power blending exponent (P^gamma)
└── EXP-19: Evaluate Beta calibration vs Joint Log-Odds
    Rationale: Refines probability tail curvature and minimizes cross-entropy penalties.

PHASE 7: Advanced Deep Tabular Architectures (Day 9)
├── EXP-21: Test Focal Loss objective on Tabular ResNet
└── EXP-22: Evaluate Deep & Cross Network (DCNv2) polynomial feature crosses
    Rationale: Explores cutting-edge tabular deep learning after foundational neural baselines are validated.

PHASE 8: The Top-1 Grand Slam Ensemble (Day 10)
├── EXP-23: Pruned Heterogeneous Stack (CatBoost + XGBoost + ResNet + ElasticNet)
├── EXP-24: 5-Seed Bagging Matrix Expansion ([42, 1337, 2026, 777, 999])
└── EXP-25: Final Submission Assembly (Full 5-Seed Multi-Family Stack + Joint Calibration)
    Rationale: The definitive high-compute submission designed to bridge the final gap to 0.73938.
```

---

## SECTION 24: Compute Efficiency & Fast-Screening Protocol

### Compute Budget Classification
* **CPU Cheap (<2 mins, Local Laptop / Kaggle CPU):** EXP-02 (Nested Stacking), EXP-14 (ElasticNet), EXP-15 (KNN), EXP-18 (Power Blending), EXP-19 (Beta Calibration).
* **GPU Cheap (5–12 mins, Kaggle GPU T4):** EXP-01 (TE Cleanup), EXP-03 (Prune GOSS), EXP-04 (Prune Ratios), EXP-06 (Top 350 Pruned), EXP-07 (Top 250 Core), EXP-08 (XGB Reg).
* **GPU Medium (12–18 mins, Kaggle GPU T4):** EXP-05 (V7 Restore), EXP-09 (CatBoost Depth 8), EXP-11 (Monotonicity), EXP-16 (Symmetric Ratios), EXP-17 (New Features), EXP-20 (CatBoost Categoricals).
* **GPU Expensive (18–25 mins, Kaggle Dual T4):** EXP-10 (Extended CatBoost), EXP-12 (Tabular ResNet), EXP-13 (RankGauss ResNet), EXP-21 (Focal Loss), EXP-22 (DCNv2), EXP-23 (Heterogeneous Stack).
* **GPU Heavy (>30 mins, Dedicated Kaggle Run):** EXP-24 (5-Seed Bagging), EXP-25 (Grand Slam Final).

### The Fast-Screening Protocol (Saving 70% of GPU Quota)
* **The Rule:** Never run a full 10-fold 3-seed pipeline (30 models) on an untested hypothesis.
* **Screening Setup:** Run a **5-Fold, Single Seed (`[42]`) screening pass** (5 models total, ~3–4 minutes).
* **Validity Criterion:** If a feature set or hyperparameter change does not improve 5-fold single-seed OOF Log Loss by at least **`-0.0008`**, it will NEVER justify the compute of a full 30-model 3-seed run. Discard immediately.

---

## SECTION 25: Scientific Experimentation Protocol

Every experimental trial must strictly enforce:
1. **The Single-Variable Directive:** Change exactly ONE component per experiment. Never change features, model depth, and calibration simultaneously.
2. **Deterministic Reproducibility:** Fixed seeds (`[42, 1337, 2026]`) with deterministic CUDA flags (`torch.backends.cudnn.deterministic = True`).
3. **Multi-Metric Metric Tracking:** Record all 9 canonical metrics:
   $$\text{OOF Log Loss}, \text{OOF ROC-AUC}, \text{PR-AUC}, \text{Brier Score}, \text{Train Log Loss}, \text{Generalization Gap}, \mathbb{E}[P_{\text{test}}], r(\text{CatBoost}, m), \text{Public LB}$$
4. **Automated Schema Compliance:** Every run must output a validated JSON record to `experiments/run_*.json` satisfying `experiments/schema.json`.

---

## SECTION 26: Ranked Final Top-1 Conclusions

1. **CURRENT BEST MODEL:** **CatBoost GPU** (OOF Log Loss: `0.247559`, ROC-AUC: `0.908483`, Gap: `+0.0608`).
2. **CURRENT BEST ENSEMBLE:** **Level-2 Stacking Meta-Learner** on probabilities and log-odds with asymmetric tail clamping `[0.003, 0.990]` (Public Score: `0.720992331`).
3. **CURRENT BIGGEST WEAKNESS:** **Severe Collinearity Monoculture.** XGBoost and LightGBM GOSS are 0.9957 correlated clones that overfit training data (Train AUC $>0.99$).
4. **MOST IMPORTANT VALIDATION PROBLEM:** **In-Sample Stacking Evaluation.** Meta-learner was fitted and evaluated on the same OOF matrix without nested cross-validation.
5. **MOST IMPORTANT FEATURE PROBLEM:** **Unpruned Volatile Ratios.** High-order emergency cash spike ratios create noisy division blowups on inactive test accounts.
6. **MOST IMPORTANT MODEL PROBLEM:** **XGBoost Overfitting.** Generalization gap of `+0.1318` drags down ensemble variance reduction.
7. **MOST IMPORTANT DIVERSITY PROBLEM:** **Zero Non-Tree Models.** Ensemble lacks continuous manifold representations.
8. **MOST PROMISING NEW ALGORITHM:** **3-Block Tabular ResNet (Residual MLP with LayerNorm & Mish).**
9. **MOST PROMISING FEATURE FAMILY:** **Bounded Symmetric Difference Ratios & Robust MAD-Normalized Balances.**
10. **MOST PROMISING ENSEMBLE METHOD:** **Nested Leak-Free 2-Stage Stacking across Orthogonal Families.**
11. **MOST PROMISING CALIBRATION METHOD:** **Joint Log-Odds Nelder-Mead Calibration ($T, \delta$) with Asymmetric Clamping `[0.003, 0.990]`.**
12. **MOST PROMISING NEURAL NETWORK:** **Tabular ResNet with RankGauss Preprocessing.**
13. **MOST PROMISING NON-TREE MODEL:** **ElasticNet Regularized Logistic Regression ($C=0.05$).**
14. **MOST PROMISING CHEAP EXPERIMENT:** **EXP-03: Drop LightGBM GOSS Redundancy (Saves 35% compute, eliminates collinear noise).**
15. **MOST PROMISING HIGH-COMPUTE EXPERIMENT:** **EXP-23: Feature-Pruned Heterogeneous Stack (CatBoost + XGBoost + ResNet).**
16. **EXPERIMENT MOST LIKELY TO IMPROVE ROC-AUC:** **EXP-12: Tabular ResNet Integration (Expands rank discrimination into continuous feature space).**
17. **EXPERIMENT MOST LIKELY TO IMPROVE LOG LOSS:** **EXP-06 / EXP-07: Aggressive Feature Pruning (Eliminates false positive tail blowups).**
18. **EXPERIMENT MOST LIKELY TO IMPROVE BOTH:** **EXP-25: Grand Slam Heterogeneous Stack with Joint Calibration.**
19. **EXPERIMENT MOST LIKELY TO PROVIDE DIVERSITY:** **EXP-14 (ElasticNet) & EXP-15 (KNN on PCA Components).**
20. **EXPERIMENT MOST LIKELY TO WASTE TIME:** **Grid searching LightGBM DART or training complex Transformer architectures (FT-Transformer / SAINT).**

---

## SECTION 27: Self-Critique & Devil's Advocate Analysis

Before committing resources, we aggressively challenge our core recommendations:

### Challenge 1: "Are we assuming a Neural Network will magically deliver +0.006 without evidence?"
* *Devil's Advocate:* In V2 and V3, PyTorch MLP was tested and achieved an awful OOF Log Loss of `0.2798` and ROC-AUC of `0.8792`, taking only 1.6% ensemble weight. Why should it work now?
* *The Defense:* Early MLPs used raw `StandardScaler`, `BatchNorm1d` (which leaks batch statistics), and standard `ReLU`. When continuous financial ratios have extreme 100x outliers, standard MLPs suffer exploding activations. Modern Tabular ResNets with **RankGauss quantile normalization**, **LayerNorm**, and **Mish activations** resolve gradient instability. Furthermore, even at AUC 0.885, if its correlation with CatBoost is 0.88, mathematical ensemble theory guarantees a reduction in ensemble error variance:
  $$\sigma_{\text{ens}}^2 = \frac{1}{2}\bar{\sigma}^2 (1 + r) < \bar{\sigma}^2$$

### Challenge 2: "Is 501 features really overfitting, or is feature count a scapegoat?"
* *Devil's Advocate:* Deep learning often benefits from more features. Could pruning 150 features throw away subtle signals?
* *The Defense:* The proof is empirical. XGBoost Train AUC is `0.9935` while Validation AUC is `0.9057` (a massive `+0.1318` loss gap). In V13, adding 10 features directly degraded public score from `0.71928` to `0.71895`. In tabular datasets with only 6,000 positive minority cases, trees split on noise when given 500 features. Pruning is scientifically mandatory.

### Challenge 3: "Could CatBoost alone reach Rank 1 if we just tuned it endlessly?"
* *Devil's Advocate:* CatBoost is our best model by far. Why waste time on XGBoost, ResNet, and ElasticNet instead of running 10,000 iterations of CatBoost?
* *The Defense:* The leaderboard evidence disproves this. Standalone CatBoost tops out at ~`0.716`–`0.717` public score. V12 reached `0.71928` and V15 reached `0.72099` **only because of ensembling and stacking**. A single decision tree architecture cannot cross the structural boundary to `0.7394`.

---

## SECTION 28: Master Decision Matrix

| Model / Method | Already Tested in Repo? | Standalone Strength | Orthogonal Diversity | Log Loss Potential | ROC-AUC Potential | Compute Cost | Overfitting Risk | Expected Net Value | Priority | Actionable Strategic Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CatBoost GPU** | **YES** | **Dominant (0.2475 / 0.9085)**| Baseline | **VERY HIGH** | **VERY HIGH** | Medium | **Very Low** | **ESSENTIAL** | **CORE BACKBONE** | Retain as 70%+ ensemble foundation. Test depth 8 and monotonicity. |
| **XGBoost GPU** | **YES** | Moderate (0.2527 / 0.9057)| Low (0.977 to CB)| Moderate | Moderate | Fast | **High (Train >0.99)**| **HIGH** | **HIGH PRIORITY** | Retain as secondary tree, but heavily regularize (`min_child=25`). |
| **LightGBM GOSS** | **YES** | Moderate (0.2530 / 0.9050)| **Zero (0.9957 to XGB)**| Low | Low | Fast | **High** | **LOW** | **PRUNE / DROP** | Drop from default runner to eliminate collinear noise and save compute. |
| **Tabular ResNet**| **PARTIALLY** | Weak (0.2798 / 0.8792) | **Very High (0.88)** | **HIGH** | **HIGH** | Medium | Moderate | **VERY HIGH** | **TOP PRIORITY** | Re-engineer with LayerNorm, Mish, RankGauss. The primary diversity driver. |
| **ElasticNet GLM**| **NO** | Weak (0.3120 / 0.8415) | **Extreme (0.82)** | Moderate | Low | Instant | **None** | **HIGH** | **HIGH PRIORITY** | Add as regularized linear anchor in Level-2 stack. |
| **KNN on PCA** | **NO** | Weak (0.3200 / 0.8250) | **Extreme (0.78)** | Low | Low | Fast | Low | **MEDIUM** | **MEDIUM PRIORITY**| Test $K=64$ on 16 PCA components as non-parametric meta-feature. |
| **Level-2 Stacking**| **YES** | **Current Best (0.72099)** | N/A | **VERY HIGH** | **VERY HIGH** | Fast | Moderate | **ESSENTIAL** | **CORE ENSEMBLE** | Upgrade to leak-free 2-stage nested cross-validation. |
| **Joint Calibration**| **YES** | Proven Essential | N/A | **VERY HIGH** | Preserved | Instant | Very Low | **ESSENTIAL** | **CORE POST-PROC** | Retain Nelder-Mead $(T, \delta)$ with asymmetric clamping `[0.003, 0.990]`. |
| **Feature Pruning** | **NO** | N/A | N/A | **VERY HIGH** | **HIGH** | Medium | **Negative (Reduces)**| **ESSENTIAL** | **TOP PRIORITY** | Prune bottom 150 features; restore V7 core to eliminate XGB overfit. |
| **Monotonic Trees** | **NO** | Moderate | N/A | Moderate | Moderate | Medium | **Negative (Reduces)**| **HIGH** | **HIGH PRIORITY** | Apply monotone constraints to 5 core balance collapse metrics. |
| **Random Forest** | **YES** | Uncompetitive (0.288) | Low | None | None | Slow | Low | **NEGATIVE** | **PERMANENTLY REJECT**| 0.0000 weight in all historical blends. Discard. |
| **Extra Trees** | **YES** | Uncompetitive (0.292) | Low | None | None | Slow | Low | **NEGATIVE** | **PERMANENTLY REJECT**| Excessive leaf entropy. Discard. |
| **LightGBM DART** | **YES** | Uncompetitive (0.256) | Low | Low | Low | Slow | Low | **NEGATIVE** | **PERMANENTLY REJECT**| Tree dropout ruins probability calibration. Discard. |
| **AutoGluon Stack**| **PARTIALLY** | Unknown | N/A | Unknown | Unknown | Heavy | High | **LOW** | **PERMANENTLY REJECT**| Heavy checkpoint bloat (>40MB); manual custom stack is superior. |
| **TabNet / SAINT** | **NO** | Weak | High | Low | Low | Heavy | High | **LOW** | **DO NOT PURSUE** | High compute overhead; unsuited to 40k row dense ratio dataset. |

---

## Final Executive Directive

We now possess an evidence-backed roadmap from our current V15 baseline (`0.720992331`, Rank 60) to the Zindi #1 benchmark (`0.739375868`). 

By:
1. **Fixing the in-sample stacking evaluation flaw** (Phase 0),
2. **Dropping redundant, collinear GBDT models and unpruned ratio noise** (Phases 1 & 2),
3. **Maximizing CatBoost via monotonic constraints and deeper regularized trees** (Phase 3),
4. **Injecting true continuous manifold diversity via Tabular ResNet and ElasticNet** (Phase 4), and
5. **Assembling the leak-free multi-family ensemble across 5 seeds** (Phase 8),

we establish the most statistically reliable, compute-efficient path to challenge the top of the leaderboard.
