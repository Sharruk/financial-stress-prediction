import json
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
SUBMISSIONS_DIR = BASE_DIR / "data" / "submissions"
PROGRESS_FILE = BASE_DIR / "progress" / "progress.yaml"

def get_project_status():
    if not PROGRESS_FILE.exists():
        return {"error": "Progress file not found"}
    with open(PROGRESS_FILE, "r") as f:
        return yaml.safe_load(f)

def get_all_experiments():
    if not EXPERIMENTS_DIR.exists():
        return []
    
    experiments = []
    for file in EXPERIMENTS_DIR.glob("run_*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                experiments.append(data)
        except Exception:
            pass
            
    # Sort descending by timestamp
    experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return experiments

def get_latest_experiment():
    experiments = get_all_experiments()
    return experiments[0] if experiments else None

def get_all_models():
    """Extract distinct models that have been run from the experiments."""
    experiments = get_all_experiments()
    models_dict = {}
    
    for exp in experiments:
        if "_base_models" in exp:
            for bm in exp["_base_models"]:
                m_name = bm["name"]
                if m_name not in models_dict:
                    models_dict[m_name] = []
                # Keep lightweight reference
                models_dict[m_name].append({
                    "experiment_id": exp.get("experiment_id"),
                    "timestamp": exp.get("timestamp"),
                    "oof_log_loss": bm.get("oof_metrics", {}).get("log_loss"),
                    "oof_roc_auc": bm.get("oof_metrics", {}).get("roc_auc")
                })
                
    return models_dict

def get_submissions():
    if not SUBMISSIONS_DIR.exists():
        return []
        
    subs = []
    for file in SUBMISSIONS_DIR.glob("*.csv"):
        # Basic metadata since we don't have a formal db
        stat = file.stat()
        subs.append({
            "filename": file.name,
            "size_bytes": stat.st_size,
            "last_modified": stat.st_mtime
        })
    return subs
