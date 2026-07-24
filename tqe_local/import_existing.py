"""
import_existing.py
==================
Imports CSV results from a previous Kaggle run into the incremental tracker,
so run_local_parallel.py will skip those cells.

Usage:
    1. Put your previous Kaggle CSVs in a folder (e.g., kaggle_results/)
    2. Adjust IMPORT_DIR below
    3. python import_existing.py
"""

import os
import pandas as pd


IMPORT_DIR = "kaggle_results"   # change to the path of your downloaded CSVs
OUT_DIR = "outputs_focused"
INCREMENTAL_FILE = os.path.join(OUT_DIR, "all_cells.csv")

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    if not os.path.isdir(IMPORT_DIR):
        print(f"Import dir {IMPORT_DIR} does not exist. Edit IMPORT_DIR in this script.")
        return

    found_csvs = []
    for fn in os.listdir(IMPORT_DIR):
        if fn.endswith(".csv") and not fn.startswith("all_"):
            found_csvs.append(os.path.join(IMPORT_DIR, fn))

    if not found_csvs:
        print(f"No CSV files found in {IMPORT_DIR}/")
        return

    print(f"Found {len(found_csvs)} CSV files to import:")
    for fn in found_csvs:
        print(f"  - {fn}")

    dfs = [pd.read_csv(fn) for fn in found_csvs]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total imported rows: {len(df)}")

    # Sanity check columns
    required = {"ansatz", "noise", "n_qubits", "n_layers", "metric", "mean"}
    missing = required - set(df.columns)
    if missing:
        print(f"  WARNING: missing columns {missing}; skipping.")
        return

    # Ensure required tracker columns exist
    for col, default in [("n_params", None), ("std", 0.0), ("lo", float("nan")),
                         ("hi", float("nan")), ("n_runs", 1)]:
        if col not in df.columns:
            df[col] = default

    # Append to incremental, dedupe
    if os.path.exists(INCREMENTAL_FILE):
        existing = pd.read_csv(INCREMENTAL_FILE)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df

    combined = combined.drop_duplicates(
        subset=["metric", "ansatz", "noise", "n_qubits", "n_layers"],
        keep="last"
    )
    combined.to_csv(INCREMENTAL_FILE, index=False)
    print(f"Wrote {len(combined)} unique cells to {INCREMENTAL_FILE}")


if __name__ == "__main__":
    main()
