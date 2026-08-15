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

## How to Run

### A. Run the ML Pipeline (Training Machine)
On a machine with sufficient computational resources intended for ML training:

1. Install the full ML dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the dataset files into `data/raw/`.
3. Run the training pipeline:
   ```bash
   python train.py
   ```
4. Generate predictions and submissions:
   ```bash
   python predict.py
   ```

### B. Start the Web Platform (Development/Local Machine)
To run the lightweight FastAPI server and Streamlit dashboard using Docker:

1. Start the containers using Docker Compose:
   ```bash
   docker compose up -d --build
   ```
2. The FastAPI documentation will be available at `http://localhost:8000/docs`
3. The Streamlit dashboard will be available at `http://localhost:8501`

*(Alternatively, run locally by installing `requirements-web.txt` and launching `uvicorn` and `streamlit` manually).*
