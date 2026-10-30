"""
run_pipeline.py
-----------------
Runs the whole pipeline in order with the current Python interpreter,
stopping at the first failing step, and prints how long each step took:

  import_archive_products -> festival_calendar -> generate_synthetic_data ->
  build_features -> festival_uplift -> train_model --trials 0 ->
  forecast_regional -> build_bigbasket_regional_risk

train_model runs with --trials 0, i.e. it reuses the tuned parameters in
models/params_<segment>.json; run src/train_model.py --trials N to re-tune.

--alerts adds a last step, alerts --city all: a dry run that writes each
city's vendor alert to outputs/alerts/ without sending any email.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
STEPS = [
    ["import_archive_products.py"],
    ["festival_calendar.py"],
    ["generate_synthetic_data.py"],
    ["build_features.py"],
    ["festival_uplift.py"],
    ["train_model.py", "--trials", "0"],
    ["forecast_regional.py"],
    ["build_bigbasket_regional_risk.py"],
]
ALERT_STEP = ["alerts.py", "--city", "all"]  # dry run; never --send


def main():
    parser = argparse.ArgumentParser(description="Run the whole pipeline.")
    parser.add_argument("--alerts", action="store_true",
                        help="finish with a dry run of the vendor email alerts (nothing is sent)")
    args = parser.parse_args()

    timings = []
    started = time.perf_counter()
    for step in STEPS + ([ALERT_STEP] if args.alerts else []):
        name = " ".join(step)
        print(f"\n=== {name} ===", flush=True)
        step_started = time.perf_counter()
        result = subprocess.run([sys.executable, *step], cwd=SRC_DIR)
        elapsed = time.perf_counter() - step_started
        timings.append((name, elapsed))
        if result.returncode != 0:
            print(f"\nFAILED: {name} exited with code {result.returncode} after {elapsed:.1f}s; stopping.")
            sys.exit(result.returncode)

    print("\n=== Timings ===")
    for name, elapsed in timings:
        print(f"  {name:<40}{elapsed:>8.1f}s")
    print(f"  {'total':<40}{time.perf_counter() - started:>8.1f}s")


if __name__ == "__main__":
    main()
