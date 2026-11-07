from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from libs.exp.trainer import train_and_evaluate_model_cv


REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = REPO_ROOT / "input" / "processed"
TABLES_DIR = REPO_ROOT / "output" / "tables"
RESULTS_DIR = REPO_ROOT / "output" / "results"

METHODS = [
    "K-Means",
    "LSTM",
    "Transformer",
    "1D-CNN",
    "GCN",
    "GAT",
    "GIN",
    "NAGphormer",
    "GPS",
    "GBMN (ours)",
]

CLUSTER_SWEEP_METHODS = ["GBMN", "K-Means", "LSTM", "GCN"]


def main() -> None:
    t0 = time.time()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed MIMIC-IV SA-AKI tensors...")
    data = dict(np.load(PROC_DIR / "sa_aki_tensors.npz", allow_pickle=True))
    seeds = [41, 42, 43]

    print("=== Running RQ1 Part 1: Overall Performance Comparison (Table II) ===")
    table2_rows = []
    full_rq1_results: dict[str, object] = {"seeds": seeds, "n_folds": 10, "table2": {}, "number_of_clusters": {}}

    for method in METHODS:
        t_m = time.time()
        res = train_and_evaluate_model_cv(
            model_name=method,
            tensors=data,
            seeds=seeds,
            n_folds=10,
            num_clusters=3,
            epochs=6,
        )
        if method == "GBMN (ours)":
            np.savez_compressed(
                RESULTS_DIR / "gbmn_learned_representations.npz",
                embeddings=res["embeddings"],
                labels=res["labels"],
            )
        full_rq1_results["table2"][method] = {
            k: v for k, v in res.items() if k not in {"embeddings", "labels"}
        }
        row = {
            "Method": method,
            "DBI": f"{res['DBI_mean']:.2f} ± {res['DBI_std']:.2f}",
            "CHI": f"{res['CHI_mean']:.2f} ± {res['CHI_std']:.2f}",
            "Silhouette": f"{res['Silhouette_mean']:.2f} ± {res['Silhouette_std']:.2f}",
            "DBI_mean": res["DBI_mean"],
            "DBI_std": res["DBI_std"],
            "CHI_mean": res["CHI_mean"],
            "CHI_std": res["CHI_std"],
            "Silhouette_mean": res["Silhouette_mean"],
            "Silhouette_std": res["Silhouette_std"],
        }
        table2_rows.append(row)
        print(f"  [{method:14s}] DBI={row['DBI']} | CHI={row['CHI']} | Silhouette={row['Silhouette']} ({time.time()-t_m:.1f}s)")

    table2_df = pd.DataFrame(table2_rows)
    table2_df.to_csv(TABLES_DIR / "table2_performance_comparison.csv", index=False)

    print("=== Running RQ1 Part 2: Number of Subphenotypes k=2..6 (Fig. 4) ===")
    k_rows = []
    for method in CLUSTER_SWEEP_METHODS:
        full_rq1_results["number_of_clusters"][method] = {}
        for k_val in [2, 3, 4, 5, 6]:
            res = train_and_evaluate_model_cv(
                model_name=method,
                tensors=data,
                seeds=[42],
                n_folds=10,
                num_clusters=k_val,
                epochs=4,
            )
            full_rq1_results["number_of_clusters"][method][str(k_val)] = {
                "DBI_mean": res["DBI_mean"],
                "CHI_mean": res["CHI_mean"],
                "Silhouette_mean": res["Silhouette_mean"],
            }
            k_rows.append({
                "Method": method,
                "k": k_val,
                "DBI": res["DBI_mean"],
                "CHI": res["CHI_mean"],
                "Silhouette": res["Silhouette_mean"],
            })

    k_df = pd.DataFrame(k_rows)
    k_df.to_csv(TABLES_DIR / "rq1_cluster_numbers.csv", index=False)
    (RESULTS_DIR / "rq1_effectiveness.json").write_text(json.dumps(full_rq1_results, indent=2), encoding="utf-8")
    print(f"RQ1 completed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
