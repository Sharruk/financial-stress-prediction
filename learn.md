# 📘 Master Project Defense & Viva Guide: Zindi Financial Stress Prediction

**Project Title:** Mobile-Money Liquidity Stress Prediction Challenge  
**Team Name:** `sem_5` (Nathaniel Christian, Sharruk S, Shalini M)  
**Platform & Competition:** Zindi Africa Data Science Track  
**Objective:** Predict the calibrated probability that a customer will experience liquidity/financial stress within the next 30 days based on their demographic profile and 6-month mobile-money transaction history.

---

## 🎯 STEP 1: The 60-Second "Elevator Pitch" for Your Professor

> *"Sir, traditional credit scoring models fail in emerging economies because millions of people do not have formal bank accounts, tax returns, or credit histories. However, they perform dozens of mobile-money transactions every month—paying utility bills, buying groceries, transferring money to family, and withdrawing cash.*  
> 
> *Our project builds an end-to-end Machine Learning intelligence pipeline that monitors 6 months of historical mobile-money behavior to predict financial distress 30 days before it occurs.*  
> 
> *We engineered 502 domain-specific financial physics features (such as Cashflow Elasticity, Balance Drawdown, and Emergency Cash Drains) and combined 5 distinct model families—CatBoost GPU, XGBoost, LightGBM GOSS, a Deep PyTorch Tabular ResNet Neural Network, and a Level-2 Stacking Meta-Learner—to achieve a competitive ROC-AUC exceeding 0.915 and a Log Loss near 0.24."*

---

## 💡 STEP 2: The Problem Formulation (In Simple Terms)

### 1. What is "Liquidity Stress"?
* A person may own assets, but if their immediate cash inflows freeze and their balance cannot cover upcoming mandatory obligations (food, bills, repayments), they experience **liquidity stress**.
* In our dataset:
  * **0 = Financially Healthy**: Customer has sufficient liquidity over the next 30 days.
  * **1 = Financial Stress**: Customer suffers an acute cash shortage / liquidity failure.

### 2. The 85 / 15 Imbalance Problem (Why Accuracy Fails!)
* In the training dataset of 40,000 customers:
  * **34,000 customers (85.0%)** are class `0` (Healthy).
  * **6,000 customers (15.0%)** are class `1` (Stressed).
* **Viva Question Alert:** *"Why didn't you use Accuracy as your evaluation metric?"*
  * **Answer:** *"Sir, if a dummy model simply guesses '0' (healthy) for every single person, it would achieve 85% accuracy while finding zero stressed customers! Accuracy is deceptive on imbalanced datasets. Instead, the competition evaluates us on **Logarithmic Loss (Log Loss)** and **ROC-AUC**, which evaluate probability confidence and ranking discrimination."*

---

## 🔬 STEP 3: What We Learnt & Built: Feature Engineering Engine (502 Features)

Raw transaction counts and totals are not enough. Machine learning models need **behavioral dynamics** and **financial physics**. We engineered 502 features in `src/features.py`:

### 1. Cashflow Elasticity ($\rho$)
* **What it means:** We calculate the mathematical Pearson correlation between a customer's monthly income (deposits + bank transfers) and outflows (paybills + merchant spending) across all 6 months.
* **Why it matters:** Healthy individuals cut spending when income drops ($\rho \approx +1.0$). Individuals heading toward collapse continue spending out of desperation even when inflows dry up ($\rho \le 0$).

### 2. Personal Historical $Z$-Score
* **What it means:** We compare a customer's current balance in Month 1 against their own 6-month mean and standard deviation:
  $$Z_{\text{user}} = \frac{\text{Balance}_{M1} - \mu_{6m}}{\sigma_{6m} + 1.0}$$
* **Why it matters:** An absolute balance of \$100 might be normal for a student, but for a business owner whose normal balance is \$5,000, \$100 represents a catastrophic 5-sigma collapse.

### 3. Emergency Cash Drain Acceleration
* **What it means:** We track sudden spikes in physical ATM withdrawals and bank transfers relative to commercial spending:
  $$\text{Emergency Spike} = \frac{\text{Withdrawals}_{M1} + \text{Bank Transfers}_{M1}}{\text{Withdrawals}_{M2} + \text{Bank Transfers}_{M2} + 1.0}$$
* **Why it matters:** When people panic, they extract physical cash to hoard or pay emergency obligations.

### 4. Channel Inactivity & Abandonment
* **What it means:** We count how many transaction channels (utility bills, merchant purchases, P2P sends) that were active in Month 6 went completely dark (zero activity) in Month 1.
* **Why it matters:** When a user stops paying bills or transacting, it is an early warning indicator of account dormancy or insolvency.

### 5. Unsupervised Financial Personas (K-Means Clustering)
* Using 8-cluster K-Means fitted strictly on training data, we segment customers into distinct behavioral archetypes and calculate each customer's Euclidean distance to all 8 cluster centroids.

---

## 🛡️ STEP 4: How We Prevented Data Leakage (Cross-Validation Harness)

* **Stratified 10-Fold Cross-Validation:** The 40,000 training samples are split into 10 folds, ensuring each fold maintains the exact 15% positive stress ratio.
* **Zero-Leakage Target Encoding:** Categorical variables (e.g., `segment`, `region`, `earning_pattern`) are encoded into probabilities using Bayesian smoothing ($m=15.0$). 
* **Noise Injection ($\mathcal{N}(0, 0.005)$):** A subtle Gaussian perturbation is added to the training encodings. This prevents decision trees from memorizing high-frequency categories, forcing them to learn generalizable patterns.
* **Strict Fold Isolation:** Scalers, KMeans personas, and target encodings are fitted **exclusively on the 9 training folds** and only evaluated on the 10th validation fold and the test set.

---

## ⚡ STEP 5: Multi-Family Modeling & Stacking Architecture

Instead of relying on a single model, we built a diverse ensemble across orthogonal model families in `src/models.py`:

| Model Family | Role in Ensemble | Key Hyperparameters |
| :--- | :--- | :--- |
| **1. CatBoost GPU** | Primary tree driver (symmetric oblivious trees, GPU accelerated) | `iterations=3000`, `depth=7`, `learning_rate=0.022`, `l2_leaf_reg=6.0` |
| **2. XGBoost GPU** | Asymmetric depth-wise histogram gradient boosting | `n_estimators=2500`, `max_depth=7`, `learning_rate=0.018`, `subsample=0.85` |
| **3. LightGBM GOSS** | One-side gradient sampling for fast, deep tree splits | `n_estimators=2200`, `max_depth=8`, `num_leaves=55`, `learning_rate=0.020` |
| **4. PyTorch Tabular ResNet** | Deep Neural Network with residual skip connections, LayerNorm, and Mish activations | 3 Residual Blocks, `hidden_dim=256`, `dropout=0.2`, `lr=0.001` (AdamW) |
| **5. Regularized Linear Baseline** | Scaled logistic baseline capturing pure monotonic linear vectors | $L_2$ penalty, $C=0.1$ |

### Level-2 Stacking Meta-Learner (`src/ensemble.py`)
* Rather than simple voting, the Out-of-Fold (OOF) predictions and log-odds (logits) from all 5 models are fed into a **Level-2 Regularized Logistic Meta-Learner**.
* The meta-learner learns the exact mathematical trust weights for each model across different probability regimes.

### Joint Log-Odds Calibration & Asymmetric Clamping
* Log Loss heavily punishes extreme overconfidence. A single false positive predicted at $p=0.999$ adds severe penalty to Log Loss.
* We apply **Nelder-Mead optimization** to find the optimal Temperature ($T$) and shift ($\delta$), and clamp probabilities to $[0.003, 0.990]$ to eliminate catastrophic tail errors.

---

## 🎓 STEP 6: Master Viva / Professor Q&A Cheat Sheet

### Q1: "What is your project doing in one sentence?"
> *"Our project predicts the 30-day forward probability of liquidity stress for mobile-money users using a zero-leakage 10-fold ensemble of gradient boosted trees, deep tabular neural networks, and micro-economic velocity features."*

### Q2: "What is Data Leakage, and how did you guarantee your model has none?"
> *"Data leakage happens when information from outside the training slice is used to create features or fit models. We guaranteed zero leakage by wrapping target encoding, standard scaling, and K-Means persona clustering strictly inside the 10-fold cross-validation loop. Nothing was computed globally across train and test together."*

### Q3: "Why did you combine CatBoost, XGBoost, and a Neural Network instead of just picking the best one?"
> *"Sir, different model families make different types of errors. CatBoost builds oblivious symmetric trees, XGBoost builds asymmetric deep trees, and PyTorch builds smooth continuous manifolds. By combining them with a Level-2 Stacking Meta-Learner, their individual errors cancel out, resulting in lower variance and superior generalization on unseen test data."*

### Q4: "What is Multi-Seed Bagging and why is it useful?"
> *"Decision trees rely on stochastic feature subsampling and random row splits. By training across 3 distinct seeds (e.g., 42, 1337, 2026) and averaging the predictions, we compress prediction variance on the 30,000 test set, which stabilizes the Log Loss."*

### Q5: "What is Log Loss and why is probability calibration necessary?"
> *"Log Loss measures the negative log-likelihood of true labels given predicted probabilities: $-\frac{1}{N}\sum [y\ln(p) + (1-y)\ln(1-p)]$. If a model predicts 0.99 for someone who does not experience stress, Log Loss penalizes it heavily. Calibration aligns predicted probabilities with empirical true rates using temperature scaling and tail clamping."*

---

## 📁 Repository Structure Quick Reference

```text
├── data/
│   ├── raw/                 # Train.csv (40k), Test.csv (30k), SampleSubmission.csv
│   └── submissions/         # Final generated calibrated submissions
├── src/
│   ├── data.py              # Raw data loader and schema verification
│   ├── features.py          # 502 financial physics features (Elasticity, Z-score, Drawdown)
│   ├── models.py            # CatBoost, XGBoost, LightGBM, PyTorch ResNet, Linear
│   ├── validation.py        # 10-fold stratified CV & noise-injected target encoding
│   ├── ensemble.py          # Stacking meta-learner & joint log-odds calibration
│   └── utils.py             # Evaluation metrics (Log Loss, ROC-AUC, Brier score)
├── train.py                 # Main CLI training pipeline with automatic hardware detection
├── kaggle_run.py            # Automated Kaggle cloud GPU orchestrator
├── explain.txt              # Extended technical documentation
└── learn.md                 # This presentation and viva guide
```
