import subprocess
import argparse
import sys
import json
import os


def run_script(script_name, split, quick_mode=False, extra_args=None):
    """Executes a training script with the specified split and extra arguments."""
    cmd = [sys.executable, script_name, "--split", split]

    if extra_args:
        cmd.extend(extra_args)

    if quick_mode:
        cmd.extend(["--epochs", "2"])

    print(f"\n{'=' * 65}")
    print(f"Executing: {' '.join(cmd)}")
    print(f"{'=' * 65}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the full evaluation pipeline.")
    parser.add_argument(
        "--split",
        type=str,
        default="A",
        choices=["A", "B"],
        help="Split A (paper setup) or Split B (inner train, outer test D1-D4 & D21-D24)",
    )
    parser.add_argument(
        "--skip_data", action="store_true", help="Skip the data preparation phase"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run a rapid smoke test (2 epochs)"
    )
    args = parser.parse_args()

    # Clear previous results for this split if running fresh
    if not args.quick and os.path.exists("results.json"):
        with open("results.json", "r") as f:
            try:
                results = json.load(f)
                if args.split in results:
                    del results[args.split]
                with open("results.json", "w") as fw:
                    json.dump(results, fw, indent=4)
            except json.JSONDecodeError:
                pass

    if not args.skip_data:
        print("Preparing Data...")
        try:
            subprocess.run([sys.executable, "data/prepare_data.py"], check=True)
        except subprocess.CalledProcessError:
            print("Data preparation failed. Exiting.")
            sys.exit(1)

    # Define the scripts and any extra arguments they need
    scripts_to_run = [
        ("main_cnn.py", []),
        ("main_lstm.py", []),
        ("main_gnn_baselines.py", []),
        ("main_wavegraphnet.py", ["--mode", "inverse_only"]),
        ("main_wavegraphnet.py", ["--mode", "coupled"]),
    ]

    for script, extra_args in scripts_to_run:
        try:
            run_script(script, args.split, quick_mode=args.quick, extra_args=extra_args)
        except subprocess.CalledProcessError:
            print(
                f"Error occurred while running {script} with args {extra_args}. Exiting pipeline early."
            )
            sys.exit(1)

    # --- PRINT LEADERBOARD ---
    print(f"\n\n{'=' * 45}")
    print(f" FINAL BASELINE RESULTS (SPLIT {args.split})")
    print(f"{'=' * 45}")

    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            try:
                results = json.load(f)
                if args.split in results:
                    print(f"{'Model Name':<30} | {'Test Loc MSE':<12}")
                    print("-" * 45)
                    # Sort results by lowest loss
                    sorted_res = sorted(
                        results[args.split].items(), key=lambda item: item[1]
                    )
                    for model, loss in sorted_res:
                        print(f"{model:<30} | {loss:.6f}")
                else:
                    print(f"No results logged for Split {args.split}.")
            except json.JSONDecodeError:
                print("Could not read results.json properly.")
    else:
        print("results.json not found.")


if __name__ == "__main__":
    main()
