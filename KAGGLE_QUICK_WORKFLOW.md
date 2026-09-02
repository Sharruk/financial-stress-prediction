# Kaggle Quick Workflow Guide

This document defines the standard end-to-end workflow for running machine learning training on Kaggle GPU instances, synchronizing results via Git on the `nat` branch, and preparing final submissions for Zindi.

---

## Important Context & Principles

- **Branch:** We use the `nat` branch for all active ML development and Kaggle training.
- **Compute:** We do **not** run heavy ML training locally on WSL (laptops are not suitable for full training). Heavy training is always executed on Kaggle GPU.
- **Authentication:** SSH and GitHub authentication are already configured on both WSL and Kaggle.
- **Workflow Cycle:** 
  1. Collaborator pushes changes to `nat`
  2. Pull changes locally on WSL
  3. Pull changes into Kaggle and train on GPU
  4. Validate metrics and submission artifact
  5. Push generated results back to `nat` from Kaggle
  6. Pull results back into local WSL
  7. Submit `data/submissions/submission.csv` to Zindi

---

## Workflow Steps

### 1. Friend Pushes Changes
Collaborator pushes the latest ML model changes, feature engineering, or pipeline updates to:
```
nat
```

---

### 2. Local WSL — Get Latest Code
Synchronize your local WSL repository with the latest remote changes:

```bash
git switch nat
git pull --ff-only origin nat
git status
git log -1 --oneline
```

> **Why `--ff-only`?**  
> `--ff-only` ensures Git will only fast-forward if the local branch can be updated cleanly without creating a merge commit. This prevents accidental merge commits while synchronizing branch state.

---

### 3. Kaggle — Get Latest `nat`

#### If the repository already exists on Kaggle:
```bash
cd /kaggle/working/financial-stress-prediction
git switch nat
git pull --ff-only origin nat
git log -1 --oneline
```

#### If the repository does not exist on Kaggle:
```bash
git clone -b nat git@github.com:Sharruk/financial-stress-prediction.git
cd /kaggle/working/financial-stress-prediction
git log -1 --oneline
```

> **Verify Commit:** Confirm that the latest commit displayed by `git log -1 --oneline` on Kaggle matches the commit hash pulled on your local WSL terminal.

---

### 4. Kaggle — Train

Run the repository's training runner script [kaggle_run.py](file:///d:/Visual_Studio_Code/ML_Hackathons/financial-stress-prediction/kaggle_run.py):

#### Terminal:
```bash
python kaggle_run.py
```

#### Kaggle Notebook Cell:
```python
!python kaggle_run.py
```

> **IMPORTANT: Avoiding Accidental Double Training**  
> In [kaggle_run.py](file:///d:/Visual_Studio_Code/ML_Hackathons/financial-stress-prediction/kaggle_run.py), the `--push-results` flag is an end-to-end execution argument that executes full model training before pushing. Running `python kaggle_run.py --push-results` *after* running `python kaggle_run.py` will execute the entire training pipeline a second time.  
> **Never run `python kaggle_run.py --push-results` immediately after training.** Follow the safe manual Git push procedure in Step 6 instead.

---

### 5. Kaggle — Validate Results

After training completes successfully, inspect the generated artifacts:

```bash
git status
```

Check the generated files in:
- `data/submissions/submission.csv`
- `experiments/`

#### Submission File Validation Requirements:
- Exactly 30,000 prediction rows + 1 header row (30,001 lines total)
- Columns: `ID,Target`
- No `NaN` values
- No infinite (`inf`) values
- Target probabilities strictly bounded between `0.0` and `1.0`

#### Model Performance Metrics:
Inspect the experiment JSON record and out-of-fold predictions in `experiments/`. The primary evaluation metrics to verify are:
- **Log Loss** (competition primary optimization metric)
- **ROC-AUC** (discrimination quality)

Do not focus solely on column names; ensure the model metrics reflect a valid, convergent training run.

---

### 6. Kaggle — Push Results

Use the safe Git procedure to stage only the generated submission and experiment files:

```bash
git status

git add data/submissions/submission.csv experiments/

git status

git commit -m "Kaggle Run Results"

git push origin nat
```

> **Warning Regarding `git add .`:**  
> Do **NOT** use `git add .` unless you have run `git status` and verified every untracked/modified file.  
> **Never commit or push:**
> - SSH private/public keys
> - GitHub personal access tokens
> - Kaggle API secrets or environment credentials
> - Python virtual environment / pip caches (`.cache/`, `__pycache__/`)
> - CatBoost temporary logging directories (`catboost_info/`)
> - Unrelated scratch scripts or large temporary data files

---

### 7. Local WSL — Pull Kaggle Results

Once Kaggle has successfully pushed to GitHub, return to your local WSL terminal to pull the new submission and experiment records:

```bash
git switch nat
git pull --ff-only origin nat
git status
git log -1 --oneline
```

---

### 8. Final Submission

The final competition submission file is:
```
data/submissions/submission.csv
```

- Upload this file directly to the Zindi competition submission page.
- **Do not manually edit or modify** the generated `submission.csv` file (e.g. changing column names, re-saving through Excel, or altering probability values), as this can corrupt formatting or probability calibration.

---

## 9. Git & Workflow Safety Rules

1. **Always work on `nat`:** All development, testing, and Kaggle training runs stay on the `nat` branch.
2. **Pull before training:** Always pull latest `nat` updates with `--ff-only` before starting a Kaggle training run.
3. **Verify commit hash:** Check `git log -1 --oneline` on Kaggle to ensure it matches local WSL.
4. **No heavy local training:** Keep your local machine responsive; run heavy compute exclusively on Kaggle GPU.
5. **Never force-push:** Avoid `git push --force` or `git push -f` under all circumstances.
6. **No casual hard resets:** Do not run `git reset --hard` unless intentionally discarding uncommitted scratch work.
7. **Never commit secrets or tokens:** Keep credentials, keys, and tokens out of the repository.
8. **No blind `git add .`:** Stage specific artifact paths (`data/submissions/submission.csv`, `experiments/`).
9. **Do not train twice accidentally:** Do not run `kaggle_run.py --push-results` after already running `kaggle_run.py`.
10. **Do not push partial or failed runs:** Only commit when training completes with `[SUCCESS]` and metrics pass validation.
11. **Always wait for training completion:** Allow all folds and seeds to finish before attempting export.
12. **Always validate submission & metrics:** Check row count, probability bounds, Log Loss, and ROC-AUC before pushing.
13. **Always pull back to WSL:** Keep the local WSL repository in sync with the latest Kaggle experiment results.

---

## 10. Quick Copy-Paste Section

### LOCAL (WSL) — Before Training:
```bash
git switch nat
git pull --ff-only origin nat
git status
git log -1 --oneline
```

### KAGGLE — Pull & Train:
```bash
cd /kaggle/working/financial-stress-prediction
git switch nat
git pull --ff-only origin nat
git log -1 --oneline
python kaggle_run.py
```

### KAGGLE — Validate & Push Results:
```bash
git status
ls -lh data/submissions/submission.csv
ls -lh experiments/

git add data/submissions/submission.csv experiments/
git status
git commit -m "Kaggle Run Results"
git push origin nat
```

### LOCAL (WSL) — After Kaggle Push:
```bash
git switch nat
git pull --ff-only origin nat
git status
git log -1 --oneline
```

### FINAL SUBMISSION FILE:
```
data/submissions/submission.csv
```
