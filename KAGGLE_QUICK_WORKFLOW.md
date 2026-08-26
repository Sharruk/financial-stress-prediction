# Kaggle Quick Workflow Guide

This document outlines the standard collaborative training and submission workflow between local environments (WSL) and Kaggle GPU instances for the Financial Stress Prediction project.

---

## Workflow Steps

### 1. Friend Pushes New Changes to `nat`
- Your collaborator pushes code updates, feature engineering scripts, or model configurations to the `nat` branch on GitHub.

---

### 2. Local WSL Terminal
Synchronize your local environment with the latest changes from the `nat` branch:
- Switch to the `nat` branch.
- Pull updates using fast-forward only.
- Check the Git status and confirm the latest commit hash and message.

```bash
git switch nat
git pull --ff-only origin nat
git status
git log -1 --oneline
```

---

### 3. Open Kaggle
In your Kaggle notebook or terminal instance with GPU acceleration enabled:

- Pull the latest `nat` branch updates.
- Verify the active commit matches your local/remote commit.
- Execute the training orchestration pipeline via [kaggle_run.py](file:///d:/Visual_Studio_Code/ML_Hackathons/financial-stress-prediction/kaggle_run.py).

#### Option A: Kaggle Terminal / Bash
```bash
git pull --ff-only origin nat
git log -1 --oneline
python kaggle_run.py
```

#### Option B: Kaggle Notebook Cell Syntax
```python
!git pull --ff-only origin nat
!git log -1 --oneline
!python kaggle_run.py
```

---

### 4. After Training Finishes on Kaggle
- Verify that `data/submissions/submission.csv` has been generated and validated (shape: 30000 rows, no NaNs/Infs, valid probability range `[0, 1]`).
- Check the generated experiment record JSON and OOF evaluation metrics in `experiments/`.
- Confirm that the run finished with `[SUCCESS]` before taking further action.
- Do **not** re-run training unnecessarily.

---

### 5. Push Kaggle Results Back to `nat`
Check the git status and use the repository's built-in push mechanism to commit and push generated submission and experiment files:

#### Check Status & Commit
```bash
git status
git log -1 --oneline
```

#### Push Results via `kaggle_run.py`
```bash
python kaggle_run.py --push-results
```
*(In a Kaggle Notebook cell: `!python kaggle_run.py --push-results`)*

> **Note (Manual Git Fallback):** If pushing manually:
> ```bash
> git add data/submissions/submission.csv experiments/
> git commit -m "Kaggle Run Results [$(date '+%Y-%m-%d %H:%M:%S')]"
> git push origin nat
> ```

---

### 6. Return to Local WSL Terminal
Pull the Kaggle-generated submission and experiment records into your local WSL repository:

```bash
git switch nat
git pull --ff-only origin nat
git status
git log -1 --oneline
```

---

### 7. Final Submission
The final competition submission file is:

```
data/submissions/submission.csv
```

Verify that the file exists and is populated locally before uploading to the Zindi competition platform.

---

## 8. Important Rules

- **No Local Heavy Compute:** Never run expensive training locally; always leverage Kaggle GPUs.
- **Pull Before Training:** Always pull the latest `nat` branch before starting a Kaggle run.
- **Verify Commit:** Always verify the active commit hash before launching training.
- **Wait for Completion:** Always let the training run to full completion before attempting to export artifacts.
- **Inspect Artifacts:** Always inspect the generated validation metrics and submission file prior to pushing.
- **Sync Back Locally:** Always pull Kaggle results back to the local terminal to maintain full version control.
- **Target Submission:** Submit the final `data/submissions/submission.csv` to Zindi.
- **No Force Pushes:** Do not use `git push --force` or `--force-with-lease`.
- **No Destructive Commands:** Do not run destructive Git commands like `git reset --hard` unless explicitly instructed.
- **Security & Cleanliness:** Never commit SSH keys, tokens, credentials, virtualenv caches, or temporary files.

---

## QUICK COPY-PASTE FLOW

### LOCAL:
```bash
git switch nat
git pull --ff-only origin nat
git status
git log -1 --oneline
```

### KAGGLE:
```bash
git pull --ff-only origin nat
git log -1 --oneline
python kaggle_run.py
python kaggle_run.py --push-results
```

### LOCAL:
```bash
git pull --ff-only origin nat
git status
git log -1 --oneline
```

### Submit:
```
data/submissions/submission.csv
```
