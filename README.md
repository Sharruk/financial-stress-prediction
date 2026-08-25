# Financial Stress Prediction Challenge — ML Engineering Project

This repository serves as the Machine Learning Algorithms Laboratory project for our team. We are actively participating in the Zindi Financial Stress Prediction Challenge, implementing an end-to-end Machine Learning Engineering pipeline to predict and analyze financial stress.

## Team

| Name | Register No. | Zindi Username | Role |
|---|---|---|---|
| Nathaniel Christian | 3122247001037 | NathanielC | Team Member |
| Shalini M | 3122247001060 | shalini_1506 | Team Member |
| Sharruk S | 3122247001061 | Sharruk_S | Team Leader |

## Mentor

**Ajay Kumar Reddy Poreddy**  
Assistant Professor  
Department of Computer Science and Engineering  
Sri Sivasubramaniya Nad(ar) College of Engineering  

Email: ajaykumarreddyp@ssn.edu.in  
Course: **ICS1512 – Machine Learning Algorithms Laboratory**

## Competition

**Financial Stress Prediction Challenge - July Starter Track**

Platform: Zindi  
Competition period: 01 July 2026 – 28 September 2026  
Competition link: [https://zindi.africa/competitions/financial-stress-prediction-challenge-2026-07-01](https://zindi.africa/competitions/financial-stress-prediction-challenge-2026-07-01)

## About the Competition / Dataset

The objective is to predict whether a customer is likely to experience financial stress within the next 30 days using six months of mobile money transaction history.

The dataset contains information related to:
- customer/mobile money transaction behaviour
- balances
- deposits
- withdrawals
- MM Send activity
- monthly transaction behaviour
- categorical customer information
- target financial-stress label

Our project focuses heavily on implementing a rigorous ML engineering lifecycle, specifically:
- Data preprocessing
- EDA
- Feature engineering
- Machine learning
- Cross-validation
- Model evaluation
- Ensemble methods
- Prediction
- Competition submission

## Current Progress

**Project completion: ~70%**

`███████░░░ 70%`

The engineering and ML pipeline has been substantially implemented, including data preparation, feature engineering, model development, validation infrastructure, model persistence, experiment tracking, API, Streamlit dashboard, testing, and Docker support. 

However, the competition is still ongoing and the final model/competition result has NOT been finalized. The 70% metric refers strictly to the current project implementation and infrastructure snapshot.

## Current Zindi Result

Current best displayed public score:  
**0.704044593**  
*(Approximately **0.7040**)*

Submitter: **NathanielC**

> This is the current best public Zindi score at this stage of the competition. It is NOT the final competition result. The README will be updated after the competition ends.

## Project Pipeline

```text
Raw Data
   ↓
Data Cleaning
   ↓
EDA
   ↓
Feature Engineering
   ↓
Feature Selection / Model Preparation
   ↓
Cross-Validation
   ↓
Model Training
   ↓
Evaluation
   ↓
Ensembling
   ↓
Model Persistence
   ↓
Prediction
   ↓
Submission
   ↓
Experiment Tracking
   ↓
FastAPI
   ↓
Streamlit Dashboard
```

The web platform (FastAPI + Streamlit) provides deep visibility into the project's status, tracking experiment results, comprehensive model metrics, bias/variance information, and ensemble configurations statelessly.

## Repository Structure

```text
financial-stress-prediction/
│
├── api/
│   ├── main.py
│   └── dependencies.py
│
├── app/
│   ├── streamlit_app.py
│   ├── components/
│   │   └── charts.py
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Project_Progress.py
│       ├── 3_Model_Leaderboard.py
│       ├── 4_Model_Metrics.py
│       ├── 5_Bias_Variance.py
│       └── 6_Ensemble_Analysis.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── submissions/
│
├── experiments/
│   └── schema.json
│
├── models/
│
├── notebooks/
│
├── progress/
│   ├── README.md
│   └── progress.yaml
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── ensemble.py
│   ├── persistence.py
│   ├── utils.py
│   └── validation.py
│
├── tests/
│
├── train.py
├── predict.py
├── requirements.txt
├── requirements-web.txt
├── Dockerfile
└── docker-compose.yml
```

## Implementation Workflow

### Step 1 — Data and ML Pipeline
The ML pipeline is heavily implemented through:
- `train.py`
- `predict.py`
- `src/`

`src/` contains reusable modules for:
- configuration
- data processing
- feature engineering
- model definitions
- validation
- ensembling
- persistence
- evaluation utilities

### Step 2 — Experiment Tracking
Training results and comprehensive evaluations are recorded into:
`experiments/`

The strict experiment schema governing these logs is defined in:
`experiments/schema.json`

### Step 3 — API
FastAPI exposes project metadata and serves experiment/model information as a stateless API layer.

### Step 4 — Streamlit
Streamlit consumes the API to provide the visual interactive diagnostic dashboard.

### Step 5 — Docker
Docker fully containerizes the web/API layer, running the dashboard safely separate from the expensive ML training environment.

## The Kaggle → Local → Docker Workflow

This repository is designed to separate heavy ML training (on Kaggle) from the local analytics dashboard (via Docker). 
The local Streamlit dashboard only requires lightweight metadata, meaning you do **not** need to sync heavy model weights locally.

### 1. Kaggle One-Command Workflow

You can orchestrate the entire end-to-end Kaggle training workflow using a single command wrapper:

1. Open a Kaggle Notebook and set **Accelerator → GPU T4 × 2**.
2. Clone the `nat` branch:
   ```bash
   git clone -b nat git@github.com:Sharruk/financial-stress-prediction.git
   ```
3. Enter the repository:
   ```bash
   cd financial-stress-prediction
   ```
4. Run the one-command orchestration script (trains the 10-Fold 4-Model Multi-Seed Ensemble by default):
   ```bash
   python kaggle_run.py
   ```

> [!NOTE]
> **Default Kaggle Run Configuration:**
> - **Models:** 4-Model Orthogonal Diversity Ensemble (`catboost`, `xgboost`, `lightgbm_goss`, `hist_gbm`)
> - **Validation:** 10-Fold Stratified Cross-Validation (90% training data per fold)
> - **Multi-Seed Bagging:** Seeds `[42, 1337, 2026]` for variance reduction across folds
> - **Hardware & Speed:** ~20–25 minutes total wall clock time on Kaggle T4 × 2 GPU
> - **Output Artifacts:**
>   - Primary submission: `data/submissions/submission.csv`
>   - Timestamped submission: `data/submissions/zindi_stress_sub_*.csv`
>   - Experiment metadata & metrics: `experiments/run_*.json`
>   - Out-Of-Fold predictions & correlations: `experiments/oof_*.csv`

> [!NOTE]
> Kaggle SSH authentication (adding your private SSH key in Kaggle Secrets / `.ssh/id_rsa`) must already be configured to clone via SSH and to use `--push-results`.

#### Optional Execution Flags

- **Smoke test only** (verify environment and GPU hardware in seconds):
  ```bash
  python kaggle_run.py --smoke-only
  ```
- **Reproduce Previous CatBoost-Only 5-Fold Experiment**:
  ```bash
  python kaggle_run.py --models catboost --folds 5 --single-seed
  ```
- **Single-seed execution of 4-model ensemble**:
  ```bash
  python kaggle_run.py --single-seed
  ```
- **Custom GPU configuration & folds**:
  ```bash
  python kaggle_run.py --gpu --devices 0:1 --folds 10
  ```
- **Force CPU execution**:
  ```bash
  python kaggle_run.py --cpu
  ```
- **Automatically commit & push experiment records and submission artifacts**:
  ```bash
  python kaggle_run.py --push-results
  ```

#### Underlying Pipeline Commands (Manual Workflow)

Under the hood, `kaggle_run.py` automatically executes and verifies:
1. Smoke test: `python train.py --smoke-test --gpu --devices 0:1`
2. Full 10-fold multi-seed training: `python train.py --models catboost xgboost lightgbm_goss hist_gbm --folds 10 --multi-seed --gpu --devices 0:1`

### 2. Download Artifacts to Local Machine

After Kaggle training finishes, you do **NOT** need to download heavy model binaries (`.pkl`, `.pt`) unless you intend to run inference (`predict.py`) locally.

For the local dashboard and experiment comparison, **download only the lightweight artifacts:**
1. The experiment record: `experiments/run_<timestamp>.json`
2. The submission file (optional): `data/submissions/zindi_stress_sub*.csv`

### 3. Local Dashboard Setup

Once downloaded, place the artifacts in their respective directories in your local repository:
- `run_<timestamp>.json` → `experiments/`
- `zindi_stress_sub*.csv` → `data/submissions/`

Then, launch the web layer using Docker:

```bash
docker compose up --build
```

- **Streamlit Dashboard:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

The Streamlit dashboard will automatically discover the new `experiments/run_<timestamp>.json` files and populate the Model Leaderboard and Metrics pages, allowing you to seamlessly track and compare multiple Kaggle experiments locally!
