# EP Tradeoff Local Run

This repository contains local experiment code for the EP tradeoff project.

## Structure

- `files/` — local parallel run code and helper scripts for the focused matrix.
- `tqe_local/` — local code for the TQE upgrade experiments and related runs.

## Notes

- `files/README.md` explains how to run the local parallel experiment, install dependencies, and use existing Kaggle results to resume work.
- `tqe_local/` contains its own experiment scripts and data outputs. Inspect that folder for its README and run scripts.

## Getting started

1. Open a terminal in the repository root.
2. Review `files/README.md` for the main local run workflow.
3. If you want to work with the TQE experiments, inspect `tqe_local/` directly.

## Recommended dependencies

The scripts in `files/` expect Python packages such as:

- `pennylane`
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

Install them with:

```bash
pip install pennylane numpy scipy pandas matplotlib
```

## Contact

If you need help running the experiments or updating the results, ask for guidance with the specific script or output file.
