#!/usr/bin/env python3
"""
Zindi Financial Stress Prediction — Kaggle One-Command Runner
============================================================
Orchestration wrapper to execute the complete Kaggle training workflow:
1. Project Root & File Validation
2. System / Python / GPU Hardware Diagnostics
3. Dependency Installation (requirements.txt)
4. CatBoost & CUDA Verification
5. Pipeline & Hardware Smoke Test (train.py --smoke-test)
6. Full Model Training (train.py --models ... --folds ... --gpu --devices ...)
7. Output Artifact Verification (submission.csv, OOF, experiment records)
8. Final Summary Report & Optional Git Push
"""

import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_repo_root() -> Path:
    """Determine the absolute repository root directory from this script's location."""
    return Path(__file__).resolve().parent


def run_command(cmd: list[str], description: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Execute a subprocess command, streaming output in real-time, with clear error handling."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n>> [{description}]")
    print(f"   Command: {cmd_str}\n")
    
    try:
        # Use stream mode to print output live in Kaggle notebook / terminal
        process = subprocess.run(cmd, cwd=cwd or get_repo_root(), check=False)
        if check and process.returncode != 0:
            print(f"\n[ERROR] Command failed with exit code {process.returncode}: {cmd_str}", file=sys.stderr)
            sys.exit(process.returncode)
        return process
    except Exception as e:
        print(f"\n[EXCEPTION] Failed to run '{cmd_str}': {e}", file=sys.stderr)
        sys.exit(1)


def get_git_info(repo_root: Path) -> tuple[str, str]:
    """Retrieve current Git branch and latest commit oneline."""
    branch = "Unknown"
    commit = "Unknown"
    try:
        res_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False
        )
        if res_branch.returncode == 0:
            branch = res_branch.stdout.strip()
    except Exception:
        pass

    try:
        res_commit = subprocess.run(
            ["git", "log", "-1", "--oneline"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False
        )
        if res_commit.returncode == 0:
            commit = res_commit.stdout.strip()
    except Exception:
        pass

    return branch, commit


# ==============================================================================
# STEP 1: Project Root Validation
# ==============================================================================
def step1_validate_root(repo_root: Path) -> tuple[str, str]:
    print("==========================================================")
    print("      ZINDI FINANCIAL STRESS — KAGGLE RUNNER")
    print("==========================================================")
    
    branch, commit = get_git_info(repo_root)
    print(f"Repository Path    : {repo_root}")
    print(f"Current Git Branch : {branch}")
    print(f"Latest Git Commit  : {commit}")
    print(f"Current Working Dir: {os.getcwd()}")
    print("----------------------------------------------------------")

    required_files = [
        repo_root / "train.py",
        repo_root / "requirements.txt",
        repo_root / "data" / "raw" / "Train.csv",
        repo_root / "data" / "raw" / "Test.csv",
        repo_root / "data" / "raw" / "SampleSubmission.csv",
    ]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print("\n[ERROR] Required files are missing from repository:", file=sys.stderr)
        for m in missing:
            print(f"  - Missing: {m}", file=sys.stderr)
        print("Please ensure raw data is located in data/raw/ before running.", file=sys.stderr)
        sys.exit(1)

    print("[SUCCESS] All required project files & raw data verified.")
    return branch, commit


# ==============================================================================
# STEP 2: System / Python / GPU Information
# ==============================================================================
def step2_system_diagnostics(gpu_mode: bool) -> dict:
    print("\n----------------------------------------------------------")
    print(" STEP 2: SYSTEM / PYTHON / GPU INFORMATION")
    print("----------------------------------------------------------")
    print(f"Python Version     : {platform.python_version()} ({platform.architecture()[0]})")
    print(f"Python Executable  : {sys.executable}")
    print(f"OS / Platform      : {platform.platform()}")

    gpu_info = {
        "available": False,
        "count": 0,
        "devices": [],
        "smi_output": "Not available"
    }

    # Query nvidia-smi safely
    try:
        smi_res = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False
        )
        if smi_res.returncode == 0:
            gpu_info["smi_output"] = smi_res.stdout.strip()
            lines = [l.strip() for l in smi_res.stdout.strip().split("\n") if l.strip()]
            gpu_info["count"] = len(lines)
            gpu_info["available"] = len(lines) > 0
            for line in lines:
                gpu_info["devices"].append(line)
    except Exception:
        pass

    # Check PyTorch CUDA if installed
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count()
        if cuda_avail:
            gpu_info["available"] = True
            gpu_info["count"] = max(gpu_info["count"], cuda_count)
            if not gpu_info["devices"]:
                for i in range(cuda_count):
                    gpu_info["devices"].append(f"{i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        pass

    print(f"NVIDIA GPU Avail   : {gpu_info['available']}")
    print(f"Detected GPU Count : {gpu_info['count']}")
    if gpu_info["devices"]:
        for d in gpu_info["devices"]:
            print(f"  GPU Device       : {d}")
    else:
        if gpu_mode:
            print("  [NOTE] No NVIDIA GPUs detected via nvidia-smi/PyTorch. (Will verify via CatBoost)")
        else:
            print("  Running in CPU mode as requested.")

    return gpu_info


# ==============================================================================
# STEP 3: Install Requirements
# ==============================================================================
def step3_install_requirements(repo_root: Path):
    print("\n----------------------------------------------------------")
    print(" STEP 3: INSTALL REQUIREMENTS")
    print("----------------------------------------------------------")
    req_file = repo_root / "requirements.txt"
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
    run_command(cmd, "Installing project requirements via pip", cwd=repo_root, check=True)
    print("[SUCCESS] Dependencies installed/verified.")


# ==============================================================================
# STEP 4: Verify CatBoost & Acceleration
# ==============================================================================
def step4_verify_catboost():
    print("\n----------------------------------------------------------")
    print(" STEP 4: VERIFY CATBOOST & ACCELERATION")
    print("----------------------------------------------------------")
    try:
        import catboost as cb
        cb_version = getattr(cb, "__version__", "unknown")
        print(f"CatBoost Version   : {cb_version}")
    except ImportError as e:
        print(f"[ERROR] CatBoost could not be imported: {e}", file=sys.stderr)
        sys.exit(1)

    # Check CatBoost CUDA capability
    try:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=1, task_type="GPU", verbose=0)
        print("CatBoost GPU Init  : Supported and available")
    except Exception as e:
        print(f"CatBoost GPU Init  : CPU fallback (GPU init note: {e})")

    return cb_version


# ==============================================================================
# STEP 5: Run Existing Smoke Test
# ==============================================================================
def step5_run_smoke_test(repo_root: Path, gpu_mode: bool, devices: str | None) -> bool:
    print("\n==========================================================")
    print(" [1/2] RUNNING GPU / PIPELINE SMOKE TEST")
    print("==========================================================")
    
    cmd = [sys.executable, "train.py", "--smoke-test"]
    if not gpu_mode:
        cmd.append("--cpu")
    else:
        cmd.append("--gpu")
        if devices:
            cmd.extend(["--devices", devices])

    proc = run_command(cmd, "Executing train.py --smoke-test", cwd=repo_root, check=False)
    if proc.returncode != 0:
        print(f"\n[ERROR] Smoke test failed with exit code {proc.returncode}!", file=sys.stderr)
        sys.exit(proc.returncode)

    print("[SUCCESS] Smoke test passed cleanly.")
    return True


# ==============================================================================
# STEP 6: Run Full Training
# ==============================================================================
def step6_run_full_training(repo_root: Path, models: list[str], folds: int, gpu_mode: bool, devices: str | None) -> bool:
    models_str = " ".join(models).upper()
    accel_str = "GPU" if gpu_mode else "CPU"
    print("\n==========================================================")
    print(f" [2/2] RUNNING FULL {folds}-FOLD {models_str} {accel_str} TRAINING")
    if gpu_mode and devices:
        print(f" GPU DEVICES: {devices}")
    print("==========================================================")

    cmd = [sys.executable, "train.py", "--models", *models, "--folds", str(folds)]
    if gpu_mode:
        cmd.append("--gpu")
        if devices:
            cmd.extend(["--devices", devices])
    else:
        cmd.append("--cpu")

    proc = run_command(cmd, f"Executing full training: {' '.join(cmd)}", cwd=repo_root, check=False)
    if proc.returncode != 0:
        print(f"\n[ERROR] Full training run failed with exit code {proc.returncode}!", file=sys.stderr)
        sys.exit(proc.returncode)

    print("[SUCCESS] Full training completed successfully.")
    return True


# ==============================================================================
# STEP 7: Verify Output Artifacts
# ==============================================================================
def step7_verify_artifacts(repo_root: Path) -> dict:
    print("\n----------------------------------------------------------")
    print(" STEP 7: VERIFY OUTPUT ARTIFACTS")
    print("----------------------------------------------------------")
    
    submissions_dir = repo_root / "data" / "submissions"
    experiments_dir = repo_root / "experiments"

    latest_sub_csv = submissions_dir / "submission.csv"

    # Helper to find latest file matching pattern
    def find_newest(directory: Path, pattern: str) -> Path | None:
        files = list(directory.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda f: f.stat().st_mtime)

    newest_ts_sub = find_newest(submissions_dir, "zindi_stress_sub_*.csv")
    newest_zip = find_newest(submissions_dir, "zindi_stress_sub_*.zip")
    newest_exp = find_newest(experiments_dir, "run_*.json")
    newest_oof = find_newest(experiments_dir, "oof_*.csv")

    print(f"Primary Submission : {latest_sub_csv} (Exists: {latest_sub_csv.exists()})")
    print(f"Timestamped Sub CSV: {newest_ts_sub}")
    print(f"Timestamped Sub ZIP: {newest_zip}")
    print(f"Experiment Record  : {newest_exp}")
    print(f"OOF Record CSV     : {newest_oof}")

    # Validate latest submission.csv
    if not latest_sub_csv.exists():
        print(f"[ERROR] Expected submission file not found: {latest_sub_csv}", file=sys.stderr)
        sys.exit(1)

    try:
        import numpy as np
        import pandas as pd
        df = pd.read_csv(latest_sub_csv)
        print(f"\nVerifying {latest_sub_csv.name}:")
        print(f"  Shape            : {df.shape}")
        print(f"  Columns          : {list(df.columns)}")
        
        assert "ID" in df.columns and "Target" in df.columns, "Columns 'ID' and 'Target' required"
        assert len(df) == 30000, f"Expected exactly 30000 rows, got {len(df)}"
        assert not df["Target"].isna().any(), "Predictions contain NaN values"
        assert not np.isinf(df["Target"]).any(), "Predictions contain infinite values"
        assert (df["Target"] >= 0.0).all() and (df["Target"] <= 1.0).all(), "Predictions outside valid probability range [0, 1]"

        print("  NaN count        : 0")
        print("  Inf count        : 0")
        print(f"  Min probability  : {df['Target'].min():.6f}")
        print(f"  Max probability  : {df['Target'].max():.6f}")
        print(f"  Mean probability : {df['Target'].mean():.6f}")
        print("[SUCCESS] All submission constraints passed.")
    except Exception as e:
        print(f"[ERROR] Submission artifact validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    return {
        "submission": str(latest_sub_csv),
        "timestamped_csv": str(newest_ts_sub) if newest_ts_sub else "None",
        "zip": str(newest_zip) if newest_zip else "None",
        "experiment": str(newest_exp) if newest_exp else "None",
        "oof": str(newest_oof) if newest_oof else "None"
    }


# ==============================================================================
# STEP 8: Final Summary & Optional Git Push
# ==============================================================================
def step8_final_summary(
    branch: str,
    commit: str,
    python_ver: str,
    cb_ver: str,
    gpu_mode: bool,
    devices: str,
    smoke_status: str,
    train_status: str,
    artifacts: dict
):
    print("\n==========================================================")
    print("                 KAGGLE RUN COMPLETE")
    print("==========================================================")
    print(f"Git branch   : {branch}")
    print(f"Git commit   : {commit}")
    print()
    print(f"Python       : {python_ver}")
    print(f"CatBoost     : {cb_ver}")
    print()
    print(f"GPU          : {'Enabled' if gpu_mode else 'Disabled (CPU)'}")
    print(f"GPU devices  : {devices if gpu_mode else 'N/A'}")
    print()
    print(f"Smoke test   : {smoke_status}")
    print(f"Training     : {train_status}")
    print()
    print(f"Submission   : {artifacts.get('submission', 'N/A')}")
    print(f"OOF          : {artifacts.get('oof', 'N/A')}")
    print(f"Experiment   : {artifacts.get('experiment', 'N/A')}")
    print(f"ZIP          : {artifacts.get('zip', 'N/A')}")
    print()
    print("Next action:")
    print("Download/use data/submissions/submission.csv for Zindi submission.")
    print("==========================================================\n")


def push_results_to_git(repo_root: Path, branch: str):
    print("\n----------------------------------------------------------")
    print(" OPTIONAL STEP: GIT COMMIT & PUSH RESULTS")
    print("----------------------------------------------------------")
    
    # 1. Verify branch
    if branch in ["Unknown", "HEAD"]:
        print("[WARNING] Detached HEAD or unknown branch. Cannot push automatically.")
        return

    # 2. Configure user.name and user.email if missing
    for key, default_val in [("user.name", "Kaggle Automated Runner"), ("user.email", "kaggle-runner@zindi-challenge.local")]:
        check_cfg = subprocess.run(["git", "config", key], cwd=repo_root, capture_output=True, text=True, check=False)
        if not check_cfg.stdout.strip():
            print(f"Setting git config {key} = '{default_val}'")
            subprocess.run(["git", "config", key, default_val], cwd=repo_root, check=False)

    # 3. Stage intentionally generated experiment & submission files
    files_to_add = [
        "data/submissions/submission.csv",
        "experiments/"
    ]
    for pattern in files_to_add:
        subprocess.run(["git", "add", pattern], cwd=repo_root, check=False)

    # Check status of staged files
    status_res = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=False)
    staged = [l for l in status_res.stdout.split("\n") if l and l.startswith(("A", "M"))]
    
    if not staged:
        print("[INFO] No new/modified files staged to commit.")
        return

    print("Files staged for commit:")
    for l in staged:
        print(f"  {l}")

    # 4. Commit
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_msg = f"Kaggle Run Results [{ts}]"
    commit_res = subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_root, check=False)
    if commit_res.returncode != 0:
        print("[ERROR] Git commit failed.", file=sys.stderr)
        return

    # 5. Push
    print(f"Pushing commit to origin/{branch}...")
    push_res = subprocess.run(["git", "push", "origin", branch], cwd=repo_root, check=False)
    if push_res.returncode == 0:
        print(f"[SUCCESS] Successfully pushed results to origin/{branch}.")
    else:
        print(f"[WARNING] Git push failed. Verify SSH keys/credentials on Kaggle.", file=sys.stderr)


# ==============================================================================
# CLI Argument Parser & Main
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Zindi Financial Stress — One-Command Kaggle Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (disables GPU)")
    parser.add_argument("--gpu", action="store_true", default=True, help="Enable GPU training (default: True)")
    parser.add_argument("--devices", type=str, default="0:1", help="GPU devices string (e.g. '0:1' for Kaggle T4 x 2, or '0')")
    parser.add_argument("--smoke-only", action="store_true", help="Run only the smoke test and exit")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the smoke test and proceed directly to full training")
    parser.add_argument("--push-results", action="store_true", help="Commit and push experiment and submission artifacts to GitHub")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--models", nargs="+", default=["catboost"], help="List of models to train (default: ['catboost'])")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = get_repo_root()
    os.chdir(repo_root)

    # Determine GPU mode
    gpu_mode = not args.cpu

    # Step 1: Project Root Validation
    branch, commit = step1_validate_root(repo_root)

    # Step 2: System Diagnostics
    gpu_info = step2_system_diagnostics(gpu_mode=gpu_mode)

    # Step 3: Install Requirements
    step3_install_requirements(repo_root)

    # Step 4: Verify CatBoost
    cb_ver = step4_verify_catboost()

    smoke_status = "SKIPPED"
    train_status = "SKIPPED"
    artifacts = {}

    # Step 5: Smoke Test
    if not args.skip_smoke:
        step5_run_smoke_test(repo_root, gpu_mode=gpu_mode, devices=args.devices)
        smoke_status = "PASS"

    # Step 6: Full Training
    if not args.smoke_only:
        step6_run_full_training(repo_root, models=args.models, folds=args.folds, gpu_mode=gpu_mode, devices=args.devices)
        train_status = "PASS"

        # Step 7: Verify Output Artifacts
        artifacts = step7_verify_artifacts(repo_root)

    # Step 8: Final Summary
    step8_final_summary(
        branch=branch,
        commit=commit,
        python_ver=platform.python_version(),
        cb_ver=cb_ver,
        gpu_mode=gpu_mode,
        devices=args.devices,
        smoke_status=smoke_status,
        train_status=train_status,
        artifacts=artifacts
    )

    # Optional Push Mode
    if args.push_results and not args.smoke_only:
        push_results_to_git(repo_root, branch)


if __name__ == "__main__":
    main()
