"""
tests/test_submission_format.py
--------------------------------
Validates the submission CSV produced by the training/prediction pipeline.

Reads from data/submissions/submission.csv (the "latest" copy always written by
format_and_save_submission()). Requires that train.py --quick (or predict.py) has
been run first to generate this file.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Project root on sys.path so src.config is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SUBMISSION_DIR, TEST_PATH, ID_COL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def submission_df():
    """Load the latest submission CSV produced by the pipeline."""
    sub_path = SUBMISSION_DIR / "submission.csv"
    if not sub_path.exists():
        pytest.skip(
            f"Submission file not found at {sub_path}. "
            "Run 'python train.py --quick' or 'python predict.py' first."
        )
    return pd.read_csv(sub_path)


@pytest.fixture(scope="module")
def test_df():
    """Load the raw Test.csv for reference row count and IDs."""
    if not TEST_PATH.exists():
        pytest.skip(f"Test.csv not found at {TEST_PATH}.")
    return pd.read_csv(TEST_PATH)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_submission_has_correct_columns(submission_df):
    """Submission must have exactly ID and Target columns (in that order)."""
    assert list(submission_df.columns) == [ID_COL, "Target"], (
        f"Expected columns ['{ID_COL}', 'Target'], got {list(submission_df.columns)}"
    )


def test_target_probabilities_in_valid_range(submission_df):
    """All Target values must be within [0, 1]."""
    out_of_range = submission_df["Target"][(submission_df["Target"] < 0) | (submission_df["Target"] > 1)]
    assert len(out_of_range) == 0, (
        f"{len(out_of_range)} Target values are outside [0, 1]: {out_of_range.head().tolist()}"
    )


def test_submission_row_count_matches_test(submission_df, test_df):
    """Submission must have the same number of rows as Test.csv."""
    assert len(submission_df) == len(test_df), (
        f"Submission has {len(submission_df)} rows but Test.csv has {len(test_df)} rows."
    )


def test_submission_ids_match_test_ids(submission_df, test_df):
    """Submission ID set must match Test.csv ID set (order-independent)."""
    sub_ids = set(submission_df[ID_COL].astype(str))
    test_ids = set(test_df[ID_COL].astype(str))

    only_in_sub = sub_ids - test_ids
    only_in_test = test_ids - sub_ids

    assert sub_ids == test_ids, (
        f"ID mismatch.\n"
        f"  In submission but not Test.csv ({len(only_in_sub)}): {list(only_in_sub)[:5]}\n"
        f"  In Test.csv but not submission ({len(only_in_test)}): {list(only_in_test)[:5]}"
    )
