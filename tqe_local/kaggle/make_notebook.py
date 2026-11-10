"""Builds tradeoff_fixed_rerun.ipynb with the current code files embedded.
Run from tqe_local/:  python kaggle/make_notebook.py [--smoke]"""
import json
import sys

SMOKE = "--smoke" in sys.argv
FILES = ["ansatze.py", "noise.py", "metrics.py", "analysis.py", "run_fixed_rerun.py", "test_fixes.py"]


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}


cells = [
    md("# Topology depth–width trade-off: fixed rerun (Nov 2026)\n\n"
       "**Setup (once):** Settings → Accelerator **None** (CPU; the GPU is not used by this code) and Internet **On**.\n\n"
       "**Two stages.** Set `STAGE` in the next cell: run `\"noisy\"` first, then `\"noiseless\"`. "
       "Use *Save Version → Save & Run All (Commit)* so it runs in the background.\n\n"
       "**If a stage stops at the time budget:** open the notebook, *Add Input → Your Work →* this notebook's last version, "
       "then *Save & Run All* again. Finished cells are copied in and skipped.\n\n"
       "**When both stages are done:** download `outputs_fixed.zip` from the Output tab and send it back."),
    code(f'STAGE = "noisy"        # "noisy" first, then "noiseless"\n'
         f'MAX_HOURS = 11.0       # stop cleanly before Kaggle\'s 12 h limit; rerun to continue\n'
         f'SMOKE = {SMOKE}         # True = tiny test grid (a few minutes); keep False for the real run'),
    code("!pip install -q 'pennylane==0.44.1'\n"
         "import pennylane as qml, os\nprint('PennyLane', qml.__version__, '|', os.cpu_count(), 'CPUs')"),
]
for f in FILES:
    cells.append(code("%%writefile " + f + "\n" + open(f).read()))
cells += [
    md("## Tests (must all PASS)"),
    code("import subprocess, sys\n"
         "r = subprocess.run([sys.executable, 'test_fixes.py'], capture_output=True, text=True)\n"
         "print(r.stdout[-3000:], r.stderr[-3000:])\n"
         "assert r.returncode == 0, 'Tests failed: stop and send this output back.'"),
    md("## Resume: copy finished cells from a previous version (if added as input)"),
    code("import glob, os, shutil\nimport pandas as pd\n"
         "os.makedirs('outputs_fixed', exist_ok=True)\n"
         "for src in glob.glob('/kaggle/input/**/outputs_fixed/cells_*.csv', recursive=True):\n"
         "    dst = os.path.join('outputs_fixed', os.path.basename(src))\n"
         "    old = pd.read_csv(src)\n"
         "    if os.path.exists(dst):\n"
         "        old = pd.concat([pd.read_csv(dst), old])\n"
         "    old.drop_duplicates(['ansatz', 'noise', 'n_qubits', 'n_layers', 'metric']).to_csv(dst, index=False)\n"
         "    print('resumed from', src)\n"
         "print(sorted(os.listdir('outputs_fixed')))"),
    md("## Run the stage"),
    code("if SMOKE:\n"
         "    extra = ['--quick', '--runs', '2', '--samples', '5', '--nmax', '4', '--nmax-noisy', '3']\n"
         "else:\n"
         "    extra = ['--nmax', '12'] if STAGE == 'noiseless' else []\n"
         "cmd = [sys.executable, 'run_fixed_rerun.py', '--group', STAGE, '--out', 'outputs_fixed',\n"
         "       '--procs', str(os.cpu_count()), '--max-hours', str(MAX_HOURS)] + extra\n"
         "print(' '.join(cmd))\n"
         "p = subprocess.run(cmd, text=True)\n"
         "assert p.returncode == 0, 'Runner failed: send the log above back.'"),
    md("## Summary and package"),
    code("for f in sorted(glob.glob('outputs_fixed/cells_*.csv')):\n"
         "    d = pd.read_csv(f)\n"
         "    n = len(d.drop_duplicates(['ansatz', 'noise', 'n_qubits', 'n_layers']))\n"
         "    print(f'{f}: {n} cells, {len(d)} rows, metrics: {sorted(d.metric.unique())}')\n"
         "shutil.make_archive('outputs_fixed', 'zip', '.', 'outputs_fixed')\n"
         "print('wrote outputs_fixed.zip', os.path.getsize('outputs_fixed.zip'), 'bytes')"),
]
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = "kaggle/tradeoff_fixed_rerun_SMOKE.ipynb" if SMOKE else "kaggle/tradeoff_fixed_rerun.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out, len(cells), "cells")
