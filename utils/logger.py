# utils/logger.py
import json
import os


def log_result(split, model_name, test_loss, filepath="results.json"):
    """Saves the test loss to a shared JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            results = {}
    else:
        results = {}

    if split not in results:
        results[split] = {}

    results[split][model_name] = test_loss

    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)
