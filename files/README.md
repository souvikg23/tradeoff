# Local Parallel Run (Ryzen 7, 8 cores)

Designed to use all your cores. Expected runtime: 20-40 minutes for the full focused matrix.

## Files

- `run_local_parallel.py` — main runner, multiprocessing.Pool with resume support.
- `import_existing.py` — import your existing Kaggle CSVs so they're not re-run.
- `ansatze.py`, `noise.py`, `metrics.py`, `analysis.py`, `sweeps.py`, `plotting.py` — same as before.

## Setup

1. Put all 8 files in a folder, e.g. `C:\vqc_local\`
2. Open a terminal in that folder and install deps:
   ```
   pip install pennylane numpy scipy pandas matplotlib
   ```

## Before you run — laptop power settings (important)

Ryzen 5000 series throttles aggressively on battery. To get full speed:
- Plug into the wall.
- Windows: Settings → System → Power & battery → Power mode → "Best performance".
- Close Chrome and other heavy apps; this run is CPU-bound on all cores.

## Step 1: import what Kaggle already gave you (optional, but recommended)

Take your Kaggle output (the partial `outputs_focused` from the version that ran 9 hours) and:

1. Extract the zip into a folder called `kaggle_results/` next to the code:
   ```
   vqc_local/
     ├── run_local_parallel.py
     ├── (other .py files)
     └── kaggle_results/
           ├── grad_amplitude_realistic.csv
           ├── grad_depolarizing_realistic.csv
           ├── grad_noiseless_new_topologies.csv
           └── ...
   ```
2. Run:
   ```
   python import_existing.py
   ```

This tells the runner "these cells are already done, skip them." Saves you ~3 hours of recomputation.

## Step 2: run the full thing

```
python run_local_parallel.py
```

What you'll see:
- A line like `Using 6 parallel processes (of 8 cores)` (it leaves 2 cores free for the OS).
- A line like `To do: 312 cells` if you imported existing data, or `~1500` if you didn't.
- Progress every 20 completed cells.

The script saves results to `outputs_focused/all_cells.csv` *after every single cell finishes*. If your laptop crashes, runs out of battery, or you have to restart, just re-run `python run_local_parallel.py` and it picks up exactly where it left off.

At the end, it also writes per-noise CSVs (e.g. `grad_depolarizing_strong.csv`) for compatibility with the original schema.

## If you want to use fewer cores

Open `run_local_parallel.py` and change this line near the top:

```python
N_PROCESSES = max(1, mp.cpu_count() - 2)
```

To, say, `N_PROCESSES = 4` if you want to keep using your laptop for other things during the run. Each process uses about 200-400MB RAM (density matrix simulator) so you might also want to limit if RAM is tight.

## When it finishes

`outputs_focused/all_focused.csv` is the consolidated file. Send me that one (or the whole folder) and we move on to the paper rewrite.

## What if something errors out?

If a single cell errors (e.g. a pennylane glitch), the rest of the cells keep going. The error gets printed but doesn't kill the run. After the run, look in `all_cells.csv` and check which cells are missing — usually it's just a couple, and re-running the script will retry only those.

If the whole script errors out before starting (e.g. import error), paste the error and I'll fix it.
