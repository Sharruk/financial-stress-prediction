# Zindi Financial Stress Prediction ML Engine (v2.0)

A high-performance, modular machine learning pipeline built specifically for the **Zindi Financial Stress Prediction Challenge**.

---

## 🏆 Model Highlights & Strategy

1. **Leak-Free 5-Fold Stratified Cross-Validation**:
   - Preserves exact 85/15 target class distribution across folds.
   - Fold-aware out-of-fold target encoding prevents any data leakage.
2. **Domain Financial & Temporal Feature Engineering (297 Total Features)**:
   - **Linear Slopes ($M6 \rightarrow M1$)**: Balance, inflow, outflow, and net cashflow trends.
   - **Net Cashflow & Liquidity**: Inflows vs. Outflows, coverage ratios, and outflow-to-balance pressure indicators.
   - **Recency & Volatility**: $M1$ vs. $M6$ ratios and differences, balance standard deviation ($\sigma_{bal}$), and coefficient of variation ($CV$).
   - **Transaction Silences & Drop-offs**: Sudden zero-activity flags in $M1$ following active history.
3. **Multi-Model Ensembling Engine**:
   - Combines **LightGBM**, **XGBoost**, **Random Forest**, and a **PyTorch Tabular MLP**.
   - Automatic SciPy SLSQP optimization for weighted probability blending.

---

## 📁 Repository Structure

```
model/
├── data/
│   ├── raw/                 # Original Train.csv, Test.csv, SampleSubmission.csv
│   └── submissions/         # Generated Zindi submission CSV and ZIP files
├── src/
│   ├── config.py            # Central paths, seed, and column definitions
│   ├── data.py              # Raw data loader and validator
│   ├── features.py          # Domain feature engineering pipeline
│   ├── models.py            # LightGBM, XGBoost, Random Forest, PyTorch MLP wrappers
│   ├── validation.py        # 5-Fold Stratified CV engine with fold-aware target encoding
│   ├── ensemble.py          # SciPy weight optimization & stacking meta-learner
│   └── utils.py             # Metrics, logging, submission & ZIP packager
├── train.py                 # Main CLI script to run training & build submissions
├── predict.py               # Standalone inference script
└── requirements.txt         # Pinned Python dependencies
```

---

## 🚀 How to Run and Train the Model

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Model Training & Submission Generation
To run full 5-Fold Stratified CV across all 40,000 training rows with all models:
```bash
python train.py
```
This will:
- Load the raw data and run the feature engineering pipeline.
- Train LightGBM, XGBoost, Random Forest, and PyTorch MLP models using 5-Fold CV.
- Calculate OOF Log Loss and ROC-AUC metrics.
- Find the optimal ensemble weights.
- Generate `submission.csv` and a timestamped `.zip` archive inside `data/submissions/`.

### 3. Run Quick Verification Mode
For quick debugging or testing pipeline changes:
```bash
python train.py --quick
```

### 4. Custom Model Training
To train specific models only (e.g. LightGBM and XGBoost):
```bash
python train.py --models lightgbm xgboost
```

---

## 📤 Submitting to Zindi

1. Locate the generated submission zip file inside `data/submissions/` (e.g., `zindi_stress_sub_20260813_144515.zip` or `submission.csv`).
2. Go to the Zindi competition submission tab and upload the `.zip` or `.csv` file.
3. Track your public leaderboard score and compare it with the Out-Of-Fold (OOF) Log Loss / ROC-AUC score reported in terminal logs.
