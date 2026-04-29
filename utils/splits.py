# utils/splits.py
import random


def get_train_test_ids(split_name, all_sample_ids, test_split_ratio=0.1, seed=42):
    """
    Returns train and test sample IDs based on the chosen spatial split.
    Split A: Current paper setup
    Split B: Center training, Edge testing (D1-D4 & D21-D24)
    """
    random.seed(seed)

    damaged_ids = [sid for sid in all_sample_ids if sid.startswith("D")]
    baseline_ids = [sid for sid in all_sample_ids if sid.startswith("baseline")]

    if split_name.upper() == "A":
        # The split currently used in your notebooks
        test_labels = ["D4", "D21", "D22", "D23", "D24", "D25"]
    elif split_name.upper() == "B":
        # The new split you requested: outer rings unseen
        test_labels = ["D1", "D2", "D3", "D4", "D21", "D22", "D23", "D24"]
    else:
        raise ValueError(f"Unknown split: {split_name}")

    # Filter test IDs matching the 100kHz suffix
    damaged_test_ids = [
        f"{label}_100kHz" for label in test_labels if f"{label}_100kHz" in damaged_ids
    ]
    damaged_train_ids = [sid for sid in damaged_ids if sid not in damaged_test_ids]

    # Split Baseline Samples randomly based on test_split_ratio
    random.shuffle(baseline_ids)
    split_point = int(len(baseline_ids) * (1.0 - test_split_ratio))
    baseline_train_ids = baseline_ids[:split_point]
    baseline_test_ids = baseline_ids[split_point:]

    # Combine
    train_ids = damaged_train_ids + baseline_train_ids
    test_ids = damaged_test_ids + baseline_test_ids

    # Final shuffle
    random.shuffle(train_ids)
    random.shuffle(test_ids)

    return train_ids, test_ids
