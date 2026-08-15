import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SUBMISSION_DIR = DATA_DIR / "submissions"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = RAW_DATA_DIR / "Train.csv"
TEST_PATH = RAW_DATA_DIR / "Test.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "SampleSubmission.csv"
DATA_DICTIONARY_PATH = RAW_DATA_DIR / "data_dictionary.csv"

# Modeling Constants
SEED = 42
N_SPLITS = 10  # 10-Fold Stratified Cross-Validation for Grand Master stability
ID_COL = "ID"
TARGET_COL = "liquidity_stress_next_30d"

# Column Categories
CATEGORICAL_COLS = [
    "gender",
    "region",
    "smartphone",
    "segment",
    "earning_pattern"
]

PROFILE_NUM_COLS = [
    "arpu",
    "age",
    "x_90_d_activity_rate"
]

BALANCE_COLS = [f"m{i}_daily_avg_bal" for i in range(1, 7)]

TRANSACTION_TYPES = [
    "paybill",
    "merchantpay",
    "transfer_from_bank",
    "mm_send",
    "received",
    "deposit",
    "withdraw"
]

INFLOW_TYPES = ["deposit", "received", "transfer_from_bank"]
OUTFLOW_TYPES = ["withdraw", "paybill", "merchantpay", "mm_send"]

COUNTERPARTY_SUFFIX_MAP = {
    "paybill": "companies",
    "merchantpay": "merchants",
    "transfer_from_bank": "banks",
    "mm_send": "recipients",
    "received": "senders",
    "deposit": "agents",
    "withdraw": "agents"
}

MONTHS = [f"m{i}" for i in range(1, 7)]
