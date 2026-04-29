# run_all.py
import subprocess
import argparse
import sys


def run_script(script_name, split, quick_mode=False):
    """
    Executes a training script with the specified split.
    If quick_mode is enabled, it overrides the epochs argument to 2.
    """
    cmd = [sys.executable, script_name, "--split", split]

    # If doing a quick smoke test, force the models to run for only 2 epochs
    if quick_mode:
        cmd.extend(["--epochs", "2"])

    print(f"\n{'=' * 60}")
    print(f"Executing: {' '.join(cmd)}")
    print(f"{'=' * 60}")
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
        "--skip_data",
        action="store_true",
        help="Skip the data preparation phase if already processed",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a rapid smoke test (2 epochs) to verify the pipeline logic",
    )
    args = parser.parse_args()

    # 1. Define the models to run
    scripts_to_run = [
        "main_cnn.py",
        "main_lstm.py",
        "main_gnn_baselines.py",
        "main_wavegraphnet.py",
    ]

    # 3. Execute each model script sequentially
    for script in scripts_to_run:
        try:
            run_script(script, args.split, quick_mode=args.quick)
        except subprocess.CalledProcessError:
            print(f"Error occurred while running {script}. Exiting pipeline early.")
            sys.exit(1)

    # Final summary message
    mode_str = "QUICK SMOKE TEST" if args.quick else "FULL TRAINING"
    print(
        f"\n[{mode_str}] Complete! All models evaluated successfully for Split {args.split}."
    )


if __name__ == "__main__":
    main()
