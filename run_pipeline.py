"""
run_pipeline.py
-----------------
Runs the whole pipeline in order with the current Python interpreter,
stopping at the first failing step, and prints how long each step took:

  import_archive_products -> festival_calendar -> generate_synthetic_data ->
  build_features -> festival_uplift -> train_model --trials 0 ->
  forecast_regional -> serving_risk_report -> explain_forecast ->
  build_bigbasket_regional_risk

train_model only runs when there are no saved models (models/lgbm_*.txt)
or with --retrain, and then with --trials 0 (the tuned parameters in
models/params_<segment>.json); run src/train_model.py --trials N to re-tune.
Training and every evaluation result use the history up to
config.EVALUATION_END only. The rest of the pipeline serves from
config.SERVING_DATE with the saved models: generate_synthetic_data appends
the history extension, then forecasts, the serving risk report and the
regional risk report run from that date.

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
    ["serving_risk_report.py"],
    ["explain_forecast.py"],
    ["build_bigbasket_regional_risk.py"],
]
ALERT_STEP = ["alerts.py", "--city", "all"]  # dry run; never --send
MODELS_DIR = SRC_DIR.parent / "models"
TRAIN_STEP = "train_model.py"


def main():
    parser = argparse.ArgumentParser(description="Run the whole pipeline.")
    parser.add_argument("--alerts", action="store_true",
                        help="finish with a dry run of the vendor email alerts (nothing is sent)")
    parser.add_argument("--retrain", action="store_true",
                        help="retrain the models (and rewrite every evaluation result) even if saved models exist")
    args = parser.parse_args()
    saved_models = all((MODELS_DIR / f"lgbm_{segment}.txt").exists() for segment in ("perishable", "nonperishable"))

    timings = []
    started = time.perf_counter()
    for step in STEPS + ([ALERT_STEP] if args.alerts else []):
        name = " ".join(step)
        if step[0] == TRAIN_STEP and saved_models and not args.retrain:
            print(f"\n=== {name}: skipped, using the saved models (--retrain to retrain) ===", flush=True)
            continue
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
