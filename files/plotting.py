"""
plotting.py
===========
Paper-ready figures. Each function takes a DataFrame from sweeps.py.

All figures use a consistent style: matplotlib defaults, no seaborn dependency,
one figure per call, tight layout, dpi=300 for save.
"""

import numpy as np
import matplotlib.pyplot as plt


def _pivot_heatmap(df_subset, value_col="mean"):
    """Pivot a (n_qubits, n_layers, value) sub-frame into a 2D array."""
    pv = df_subset.pivot_table(
        index="n_qubits", columns="n_layers", values=value_col, aggfunc="mean"
    )
    return pv


def heatmap_metric(df, metric, ansatz, noise, value_col="mean",
                   title=None, cmap="viridis", save_path=None, log_scale=False):
    sub = df[(df["metric"] == metric) & (df["ansatz"] == ansatz) & (df["noise"] == noise)]
    if sub.empty:
        print(f"No data for {metric}/{ansatz}/{noise}")
        return None
    pv = _pivot_heatmap(sub, value_col=value_col)

    fig, ax = plt.subplots(figsize=(6, 5))
    data = pv.values
    if log_scale:
        data = np.log10(np.maximum(data, 1e-12))
        cbar_label = f"log10({metric})"
    else:
        cbar_label = metric

    im = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns)
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index)
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("Width (qubits)")
    if title is None:
        title = f"{metric} | {ansatz} | {noise}"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def line_vs_depth(df, metric, ansatz, noise, save_path=None,
                  title=None, log_y=False):
    sub = df[(df["metric"] == metric) & (df["ansatz"] == ansatz) & (df["noise"] == noise)]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for n, grp in sub.groupby("n_qubits"):
        grp = grp.sort_values("n_layers")
        ax.errorbar(grp["n_layers"], grp["mean"],
                    yerr=[grp["mean"] - grp["lo"], grp["hi"] - grp["mean"]],
                    marker="o", capsize=2, label=f"n={n}")
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel(metric)
    if log_y:
        ax.set_yscale("log")
    if title is None:
        title = f"{metric} vs depth | {ansatz} | {noise}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def line_vs_width(df, metric, ansatz, noise, save_path=None,
                  title=None, log_y=False):
    sub = df[(df["metric"] == metric) & (df["ansatz"] == ansatz) & (df["noise"] == noise)]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for L, grp in sub.groupby("n_layers"):
        grp = grp.sort_values("n_qubits")
        ax.errorbar(grp["n_qubits"], grp["mean"],
                    yerr=[grp["mean"] - grp["lo"], grp["hi"] - grp["mean"]],
                    marker="s", capsize=2, label=f"L={L}")
    ax.set_xlabel("Width (qubits)")
    ax.set_ylabel(metric)
    if log_y:
        ax.set_yscale("log")
    if title is None:
        title = f"{metric} vs width | {ansatz} | {noise}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def topology_comparison(df, metric, noise, n_qubits_fixed, save_path=None):
    """Compare three topologies on one plot at fixed width, varying depth."""
    sub = df[(df["metric"] == metric) & (df["noise"] == noise) & (df["n_qubits"] == n_qubits_fixed)]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for a, grp in sub.groupby("ansatz"):
        grp = grp.sort_values("n_layers")
        ax.errorbar(grp["n_layers"], grp["mean"],
                    yerr=[grp["mean"] - grp["lo"], grp["hi"] - grp["mean"]],
                    marker="o", capsize=2, label=a)
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} | n={n_qubits_fixed} | {noise}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def noise_comparison(df, metric, ansatz, n_qubits_fixed, save_path=None):
    """Compare noiseless vs noisy at fixed (ansatz, width), varying depth."""
    sub = df[(df["metric"] == metric) & (df["ansatz"] == ansatz) & (df["n_qubits"] == n_qubits_fixed)]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    for noise, grp in sub.groupby("noise"):
        grp = grp.sort_values("n_layers")
        ax.errorbar(grp["n_layers"], grp["mean"],
                    yerr=[grp["mean"] - grp["lo"], grp["hi"] - grp["mean"]],
                    marker="o", capsize=2, label=noise)
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} | {ansatz} | n={n_qubits_fixed}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def pareto_expressivity_trainability(df_expr, df_grad, ansatz, noise, save_path=None):
    """
    Pareto plot: x = expressivity (e.g. kernel rank or eff_dim), y = grad_var.
    Each point is a (n_qubits, n_layers) configuration.
    """
    e = df_expr[(df_expr["ansatz"] == ansatz) & (df_expr["noise"] == noise)]
    g = df_grad[(df_grad["ansatz"] == ansatz) & (df_grad["noise"] == noise)]
    merged = e.merge(g, on=["ansatz", "noise", "n_qubits", "n_layers"],
                     suffixes=("_expr", "_grad"))
    if merged.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(merged["mean_expr"], merged["mean_grad"],
                    c=merged["n_qubits"], cmap="plasma", s=40, alpha=0.85)
    ax.set_xlabel(f"Expressivity ({merged['metric_expr'].iloc[0]})")
    ax.set_ylabel(f"Trainability ({merged['metric_grad'].iloc[0]})")
    ax.set_yscale("log")
    fig.colorbar(sc, ax=ax, label="n_qubits")
    ax.set_title(f"Expressivity-Trainability frontier | {ansatz} | {noise}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig
