from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi


REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = REPO_ROOT / "output" / "tables"
RESULTS_DIR = REPO_ROOT / "output" / "results"
FIGS_DIR = REPO_ROOT / "output" / "figures"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGS_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGS_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_flow_mimic() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def draw_box(x: float, y: float, w: float, h: float, text: str, fc: str = "#EBF3FA", ec: str = "#2B5B84", fontsize: float = 9.5) -> None:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", facecolor=fc, edgecolor=ec, linewidth=1.4)
        ax.add_patch(box)
        ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fontsize, color="#1A1A1A", fontweight="medium")

    draw_box(0.4, 5.1, 4.3, 1.0, "MIMIC-IV v2.1 Database\nSepsis-3 Criteria + KDIGO AKI Criteria", fc="#D6EAF8", fontsize=10)
    draw_box(0.4, 3.6, 1.95, 0.9, "Sepsis\n(n = 33,199)", fc="#F2F4F4", fontsize=9.5)
    draw_box(2.75, 3.6, 1.95, 0.9, "AKI\n(n = 34,854)", fc="#F2F4F4", fontsize=9.5)
    draw_box(0.55, 2.0, 4.0, 1.0, "SA-AKI Candidate Cohort\nAKI within 7d of Sepsis (n = 20,317)", fc="#D1F2EB", ec="#117864", fontsize=9.5)

    draw_box(
        5.5, 2.8, 4.7, 2.6,
        "Exclusion Criteria (Total = 3,956):\n"
        "• Admitted < 24 hours (n = 1,523)\n"
        "• Age < 18 or > 89 years (n = 1,015)\n"
        "• End-Stage Kidney Disease (ESKD) (n = 304)\n"
        "• Missing vital signs / early censor (n = 1,114)",
        fc="#FDEDEC", ec="#922B21", fontsize=8.8,
    )

    draw_box(
        0.55, 0.3, 4.0, 1.0,
        "Final SA-AKI Study Cohort\n(n = 16,361)",
        fc="#FCF3CF", ec="#B7950B", fontsize=10.5,
    )
    draw_box(
        5.5, 0.3, 4.7, 1.8,
        "Multi-Modal EHR Extraction (48h Window):\n"
        "• Demographics (Age groups, Sex)\n"
        "• Comorbidities (7 ICD-9/10 conditions)\n"
        "• Vital Signs (6 series) & Lab Results (15 series)\n"
        "• Clinical Outcomes (Mortality, Dialysis, ICU LOS)",
        fc="#E8F8F5", ec="#117864", fontsize=8.8,
    )

    ax.annotate("", xy=(1.35, 4.5), xytext=(2.0, 5.1), arrowprops=dict(arrowstyle="->", lw=1.4, color="#2C3E50"))
    ax.annotate("", xy=(3.65, 4.5), xytext=(3.2, 5.1), arrowprops=dict(arrowstyle="->", lw=1.4, color="#2C3E50"))
    ax.annotate("", xy=(2.55, 3.0), xytext=(1.35, 3.6), arrowprops=dict(arrowstyle="->", lw=1.4, color="#2C3E50"))
    ax.annotate("", xy=(2.55, 3.0), xytext=(3.65, 3.6), arrowprops=dict(arrowstyle="->", lw=1.4, color="#2C3E50"))
    ax.annotate("", xy=(5.5, 3.8), xytext=(4.3, 2.6), arrowprops=dict(arrowstyle="->", lw=1.4, color="#922B21"))
    ax.annotate("", xy=(2.55, 1.3), xytext=(2.55, 2.0), arrowprops=dict(arrowstyle="->", lw=1.6, color="#117864"))
    ax.annotate("", xy=(5.5, 0.9), xytext=(4.3, 0.8), arrowprops=dict(arrowstyle="->", lw=1.4, color="#117864"))

    save_both(fig, "Fig3_Flow_MIMIC")


def plot_fig4_number_of_clusters() -> None:
    df = pd.read_csv(TABLES_DIR / "rq1_cluster_numbers.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    styles = {
        "GBMN": {"color": "#F39C12", "marker": "o", "lw": 2.0},
        "K-Means": {"color": "#E74C3C", "marker": "s", "lw": 1.6},
        "LSTM": {"color": "#6C3483", "marker": "^", "lw": 1.6},
        "GCN": {"color": "#F5B7B1", "marker": "D", "lw": 1.8},
    }

    metrics = [
        ("DBI", "DBI Comparison", "DBI (Lower is Better)"),
        ("CHI", "CHI Comparison", "CHI (Higher is Better)"),
        ("Silhouette", "Silhouette Comparison", "Silhouette (Higher is Better)"),
    ]

    for ax, (col, title, ylabel) in zip(axes, metrics):
        for method, st in styles.items():
            sub = df[df["Method"] == method].sort_values("k")
            ax.plot(sub["k"], sub[col], label=method, color=st["color"], marker=st["marker"], linewidth=st["lw"], markersize=6)
        ax.set_title(title)
        ax.set_xlabel("Number of Subphenotypes (k)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([2, 3, 4, 5, 6])
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(frameon=True)

    fig.tight_layout()
    save_both(fig, "Fig4_NumberOfCluster")


def plot_fig5_outcomes() -> None:
    df = pd.read_csv(TABLES_DIR / "rq3_subphenotype_outcomes.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    x = np.arange(len(df))
    w = 0.25

    dial = df["Dialysis_fig_pct"].values
    mort = df["Mortality_pct"].values
    los = df["Median_LOS_days"].values

    b1 = ax.bar(x - w, dial, width=w, label="Dialysis (%)", color="#AED6F1", edgecolor="#1B4F72", hatch="//")
    b2 = ax.bar(x, mort, width=w, label="Mortality (%)", color="#2E86C1", edgecolor="#1B4F72")
    b3 = ax.bar(x + w, los, width=w, label="Median LOS (days)", color="#F9E79F", edgecolor="#7D6608", hatch="..")

    for bars, suffix in [(b1, "%"), (b2, "%"), (b3, " days")]:
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2.0, h + 1.0, f"{h:.1f}{suffix}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(df["Subphenotype"].values)
    ax.set_ylabel("Percent (%) / Days")
    ax.set_ylim(0, 65)
    ax.set_title("Clinical Outcomes Across Identified SA-AKI Subphenotypes (P < 0.001)")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    save_both(fig, "Fig5_Outcomes")


def plot_fig6_tsne_voronoi() -> None:
    data = np.load(RESULTS_DIR / "rq3_tsne_voronoi.npz")
    coords = data["coords"]
    labels = data["labels"]

    fig, ax = plt.subplots(figsize=(6.4, 5.8))

    # Plot background Voronoi decision regions from fine grid + cluster centroids
    xx, yy = np.meshgrid(np.linspace(-1.05, 1.05, 300), np.linspace(-1.05, 1.05, 300))
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    centroids = np.array([coords[labels == c].mean(axis=0) for c in range(3)])
    dists = np.linalg.norm(grid_pts[:, None, :] - centroids[None, :, :], axis=-1)
    grid_Cluster = np.argmin(dists, axis=-1).reshape(xx.shape)

    bg_colors = ["#FADBD8", "#D4EFDF", "#D6EAF8"]
    from matplotlib.colors import ListedColormap
    ax.contourf(xx, yy, grid_Cluster, levels=[-0.5, 0.5, 1.5, 2.5], colors=bg_colors, alpha=0.38)
    ax.contour(xx, yy, grid_Cluster, levels=[0.5, 1.5], colors="#34495E", linewidths=1.5, linestyles="--")

    # Add Voronoi cell boundaries over subsampled generator points
    vor = Voronoi(np.vstack([centroids, [[-3, -3], [-3, 3], [3, -3], [3, 3]]]))
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            ax.plot(vor.vertices[simplex, 0], vor.vertices[simplex, 1], color="#2C3E50", lw=1.2, alpha=0.5)

    pt_colors = ["#E74C3C", "#27AE60", "#2980B9"]
    for c in range(3):
        mask = labels == c
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=14,
            c=pt_colors[c],
            label=f"Subphenotype {c + 1}",
            alpha=0.78,
            edgecolors="none",
        )

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("tSNE 1")
    ax.set_ylabel("tSNE 2")
    ax.set_title("t-SNE Projection with Voronoi Tessellation of SA-AKI Subphenotypes")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.2, linestyle=":")

    fig.tight_layout()
    save_both(fig, "Fig6_tSNE_Voronoi")


def plot_fig7_shap() -> None:
    df = pd.read_csv(TABLES_DIR / "rq3_shap_importance.csv").sort_values("SHAP_Value", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    y = np.arange(len(df))
    vals = df["SHAP_Value"].values
    colors = plt.cm.Blues(np.linspace(0.45, 0.88, len(df)))

    bars = ax.barh(y, vals, color=colors, edgecolor="#1B4F72", height=0.68)
    for b, v in zip(bars, vals):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2.0, f"{v:.2f}", va="center", ha="left", fontsize=9.5)

    ax.set_yticks(y)
    ax.set_yticklabels(df["Feature"].values)
    ax.set_xlim(0.0, 0.96)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Subphenotype Assignment)")
    ax.set_title("Top 10 Clinical Features for SA-AKI Subphenotype Identification")
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    fig.tight_layout()
    save_both(fig, "Fig7_SHAP")


def plot_fig8_parameter_sensitivity() -> None:
    df = pd.read_csv(TABLES_DIR / "rq4_parameter_sensitivity.csv")

    panels = [
        ("gnn_type", "Different GNNs", "GNN Architecture", "DifferentGNNs"),
        ("gnn_layers", "Number of GNN Layers", "GNN Layers", "GNNLayers"),
        ("mamba_blocks", "Number of Mamba Blocks", "Mamba Blocks", "MambaBlocks"),
        ("walk_length", "Random Walk Lengths", "Random Walk Length (m)", "RandomWalkLengths"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    axes_flat = axes.ravel()

    for ax, (sweep_key, title, xlabel, single_stem) in zip(axes_flat, panels):
        sub = df[df["Sweep"] == sweep_key]
        x = np.arange(len(sub))
        w = 0.34

        ax2 = ax.twinx()
        b1 = ax.bar(x - w / 2.0, sub["DBI_mean"].values, width=w, color="#5DADE2", edgecolor="#1B4F72", label="DBI (↓)")
        b2 = ax2.bar(x + w / 2.0, sub["CHI_mean"].values, width=w, color="#F4D03F", edgecolor="#7D6608", label="CHI (↑)")

        for b in b1:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2.0, h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, color="#1B4F72")
        for b in b2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2.0, h + 6.0, f"{int(round(h))}", ha="center", va="bottom", fontsize=8.5, color="#7D6608")

        ax.set_xticks(x)
        ax.set_xticklabels(sub["Setting"].astype(str).values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("DBI (Lower is Better)", color="#1B4F72")
        ax2.set_ylabel("CHI (Higher is Better)", color="#7D6608")
        ax.set_ylim(0.0, max(sub["DBI_mean"].max() * 1.25, 2.8))
        ax2.set_ylim(0.0, max(sub["CHI_mean"].max() * 1.25, 560.0))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

        # Also save standalone subfigure PDF/PNG matching manuscript Figs/ names
        s_fig, s_ax = plt.subplots(figsize=(5.2, 3.8))
        s_ax2 = s_ax.twinx()
        sb1 = s_ax.bar(x - w / 2.0, sub["DBI_mean"].values, width=w, color="#5DADE2", edgecolor="#1B4F72", label="DBI")
        sb2 = s_ax2.bar(x + w / 2.0, sub["CHI_mean"].values, width=w, color="#F4D03F", edgecolor="#7D6608", label="CHI")
        for b in sb1:
            h = b.get_height()
            s_ax.text(b.get_x() + b.get_width() / 2.0, h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)
        for b in sb2:
            h = b.get_height()
            s_ax2.text(b.get_x() + b.get_width() / 2.0, h + 6.0, f"{int(round(h))}", ha="center", va="bottom", fontsize=8.5)
        s_ax.set_xticks(x)
        s_ax.set_xticklabels(sub["Setting"].astype(str).values)
        s_ax.set_xlabel(xlabel)
        s_ax.set_ylabel("DBI")
        s_ax2.set_ylabel("CHI")
        s_ax.set_ylim(0.0, max(sub["DBI_mean"].max() * 1.25, 2.8))
        s_ax2.set_ylim(0.0, max(sub["CHI_mean"].max() * 1.25, 560.0))
        s_ax.set_title(title)
        s_fig.tight_layout()
        save_both(s_fig, single_stem)

    fig.tight_layout()
    save_both(fig, "Fig8_ParameterSensitivity")


def main() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    print("Rendering all publication-grade figures (Fig. 3 to Fig. 8)...")
    plot_fig3_flow_mimic()
    plot_fig4_number_of_clusters()
    plot_fig5_outcomes()
    plot_fig6_tsne_voronoi()
    plot_fig7_shap()
    plot_fig8_parameter_sensitivity()
    plot_fig9_optimization_tuning()
    print(f"All figures saved to {FIGS_DIR}")




def plot_fig9_optimization_tuning() -> None:
    p = TABLES_DIR / "table5_gbmn_optimization_tuning.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    short_labels = [
        "Stage 0\n(Untuned)",
        "Stage 1\n(+Temp/L2Norm)",
        "Stage 2\n(+LGI/Gate)",
        "Stage 3\n(+Bottleneck)",
        "Stage 4\n(Tuned Optimal)",
        "Stage 5\n(Over-Sharp τ=0.06)",
    ][: len(df)]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3))
    x = np.arange(len(df))

    # Panel 1: DBI
    axes[0].plot(x, df["DBI_mean"], marker="o", color="#2E86C1", lw=2.2, markersize=7)
    axes[0].fill_between(x, df["DBI_mean"] - df["DBI_std"], df["DBI_mean"] + df["DBI_std"], color="#2E86C1", alpha=0.18)
    for xi, yi in zip(x, df["DBI_mean"]):
        axes[0].text(xi, yi + 0.04, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_labels, rotation=20, ha="right", fontsize=8.5)
    axes[0].set_ylabel("DBI (Lower is Better)")
    axes[0].set_title("Uncalibrated DBI Progression")
    axes[0].set_ylim(0.7, 2.4)
    axes[0].grid(alpha=0.25, linestyle="--")

    # Panel 2: CHI
    axes[1].plot(x, df["CHI_mean"], marker="s", color="#D35400", lw=2.2, markersize=7)
    axes[1].fill_between(x, df["CHI_mean"] - df["CHI_std"], df["CHI_mean"] + df["CHI_std"], color="#D35400", alpha=0.18)
    for xi, yi in zip(x, df["CHI_mean"]):
        axes[1].text(xi, yi + 35.0, f"{yi:.0f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_labels, rotation=20, ha="right", fontsize=8.5)
    axes[1].set_ylabel("CHI (Higher is Better)")
    axes[1].set_title("Uncalibrated CHI Progression")
    axes[1].set_ylim(100, 1850)
    axes[1].grid(alpha=0.25, linestyle="--")

    # Panel 3: Silhouette
    axes[2].plot(x, df["Silhouette_mean"], marker="^", color="#27AE60", lw=2.2, markersize=7)
    axes[2].fill_between(x, df["Silhouette_mean"] - df["Silhouette_std"], df["Silhouette_mean"] + df["Silhouette_std"], color="#27AE60", alpha=0.18)
    for xi, yi in zip(x, df["Silhouette_mean"]):
        axes[2].text(xi, yi + 0.012, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_labels, rotation=20, ha="right", fontsize=8.5)
    axes[2].set_ylabel("Silhouette Score (Higher is Better)")
    axes[2].set_title("Uncalibrated Silhouette Progression")
    axes[2].set_ylim(0.08, 0.42)
    axes[2].grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    save_both(fig, "Fig9_GBMN_Optimization_Tuning")


if __name__ == "__main__":
    main()
