# AI Agent Guidelines

1. **RESPECT THE MACHINE SEPARATION**:
   - The Development Machine has limited CPU/GPU. NEVER run expensive ML training (`python train.py`, `python predict.py`, large CV, Optuna) on this machine.
   - The Training Machine handles the heavy lifting.
   - Only lightweight tests (`pytest tests/`, `py_compile`) are permitted on the Development Machine.

2. **PRESERVE ML LOGIC**:
   - Do NOT rewrite or "improve" existing feature engineering, model hyperparameters, CV logic, or ensembling logic unless explicitly directed. Extend around it.

3. **WORK ON DEV BRANCH**:
   - Always operate on the `dev` branch.
   - Do NOT switch branches.
   - Do NOT commit.
   - Do NOT push.

4. **DO NOT FABRICATE DATA**:
   - Do not create fake experiments, metrics, or test data. If no experiment runs exist in `experiments/`, the API and Dashboard should gracefully handle the empty state.
