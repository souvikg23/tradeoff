# tqe_local: code and data for the topology trade-off paper

Code, data and number pipeline for "Entanglement Topology Controls the
Depth-Width Trade-off in Variational Quantum Circuits" (S. Ghosh).
Tested with Python 3.12 and PennyLane 0.44.1.

## Layout

| Path | Contents |
|---|---|
| `ansatze.py`, `metrics.py`, `noise.py`, `sweeps.py` | Circuit topologies (incl. `heavy_hex_true`, the IBM Falcon-27 subgraphs, and the legacy `heavy_hex`, called "chord-chain" in the paper), metrics and noise channels (applied after every layer) |
| `run_fixed_rerun.py`, `kaggle/` | Main noiseless and noisy sweeps (run on Kaggle) |
| `data/cells_noiseless.csv`, `data/cells_noisy.csv` | Outputs of the main sweeps used in the paper |
| `run_f4_pathk.py`, `run_gap23.py`, `run_gap4_vqe.py`, `run_hh_true_extras.py` | Path-k topologies, richer rotations / larger n, VQE on the chain Ising model, IBM heavy-hex extras |
| `run_reviewer_checks.py`, `test_reviewer_checks.py` | VQE with Ising couplings on the heavy-hex edges; heavy-hex measured on qubit 1 (degree 3) |
| `outputs_*/` | Results of the scripts above |
| `numbers/compute_numbers.py` | Recomputes every number quoted in the paper and writes `numbers/numbers.json` |
| `numbers/make_figs.py` | Redraws the paper figures (to `figures/`, or to `$FIG_DIR`) |

## Reproduce the numbers and figures

```bash
pip install "pennylane==0.44.*" numpy scipy pandas matplotlib
python numbers/compute_numbers.py
python numbers/make_figs.py
```

Neither script runs a simulation; both read the CSV files above.

## Reviewer checks

```bash
python test_reviewer_checks.py          # tests; must print ALL TESTS PASSED
python run_reviewer_checks.py --analyze --out outputs_reviewer_checks
```

The full runs took about 1.5 to 3 hours on 4 Kaggle CPUs. Finished rows are
skipped on rerun. Note: the resume key does not include the Hamiltonian name
(VQE) or the sample count (gradient sweep), so write runs with other settings
to a separate output folder.
