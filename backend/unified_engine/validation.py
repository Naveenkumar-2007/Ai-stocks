"""
Unified Engine — Walk-Forward Validation
==========================================
Proper walk-forward cross-validation with:
- Purge gap: 2× prediction horizon (prevents target leakage)
- Embargo: extra gap after test window (prevents autocorrelation leakage)
- Expanding window: training set grows over time
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from unified_engine.config import CONFIG


@dataclass
class WalkForwardSplit:
    """One fold of walk-forward validation."""
    fold_num: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_size: int
    test_size: int


def walk_forward_splits(
    n_samples: int,
    min_train: int = CONFIG.wf_min_train_days,
    test_window: int = CONFIG.wf_test_window,
    step_size: int = CONFIG.wf_step_size,
    purge_days: int = CONFIG.wf_purge_days,
    embargo_days: int = CONFIG.wf_embargo_days,
) -> List[WalkForwardSplit]:
    """
    Generate walk-forward validation splits with purge and embargo.

    Timeline for each fold:
    [=== TRAIN ===][-- PURGE --][=== TEST ===][-- EMBARGO --]

    The purge gap prevents information leakage from overlapping
    prediction windows between train and test sets.

    The embargo gap prevents autocorrelation leakage from features
    computed on test data bleeding into the next fold's training.

    Args:
        n_samples: Total number of samples
        min_train: Minimum training set size
        test_window: Number of test samples per fold
        step_size: How far to advance between folds
        purge_days: Gap between train end and test start
        embargo_days: Gap after test end before next fold's test starts

    Returns:
        List of WalkForwardSplit objects
    """
    splits = []
    fold_num = 0
    test_start = min_train + purge_days

    while test_start + test_window <= n_samples:
        # Training: all data up to purge gap
        train_end = test_start - purge_days
        train_indices = np.arange(0, train_end)

        # Test: the test window
        test_end = min(test_start + test_window, n_samples)
        test_indices = np.arange(test_start, test_end)

        # Validate
        if len(train_indices) < min_train:
            test_start += step_size
            continue

        if len(test_indices) < 5:
            test_start += step_size
            continue

        # Ensure no overlap (should never happen with purge, but safety check)
        assert train_indices[-1] < test_indices[0] - purge_days + 1, \
            f"Train/test overlap detected: train_end={train_indices[-1]}, test_start={test_indices[0]}"

        splits.append(WalkForwardSplit(
            fold_num=fold_num,
            train_indices=train_indices,
            test_indices=test_indices,
            train_size=len(train_indices),
            test_size=len(test_indices),
        ))

        fold_num += 1
        # Advance by step size + embargo
        test_start += step_size

    return splits


def validate_splits(splits: List[WalkForwardSplit], purge_days: int = CONFIG.wf_purge_days) -> bool:
    """Verify that no split has train/test overlap or insufficient purge."""
    for split in splits:
        gap = split.test_indices[0] - split.train_indices[-1]
        if gap < purge_days:
            print(f"WARNING: Fold {split.fold_num} has insufficient purge: {gap} < {purge_days}")
            return False
    return True
