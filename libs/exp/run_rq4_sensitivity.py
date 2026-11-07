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

RQ4_SWEEPS: dict[str, list[tuple[str, dict[str, object]]]] = {
    "gnn_type": [
        ("GCN", {"gnn_type": "GCN", "gnn_layers": 3, "mamba_blocks": 2, "walk_length": 2}),
        ("GAT", {"gnn_type": "GAT", "gnn_layers": 3, "mamba_blocks": 2, "walk_length": 2}),
        ("GIN", {"gnn_type": "GIN", "gnn_layers": 3, "mamba_blocks": 2, "walk_length": 2}),
    ],
    "gnn_layers": [
        ("2", {"gnn_type": "GAT", "gnn_layers": 2, "mamba_blocks": 2, "walk_length": 2}),
        ("3", {"gnn_type": "GAT", "gnn_layers": 3, "mamba_blocks": 2, "walk_length": 2}),
        ("4", {"gnn_type": "GAT", "gnn_layers": 4, "mamba_blocks": 2, "walk_length": 2}),
    ],
    "mamba_blocks": [
        ("1", {"gnn_type": "GAT", "gnn_layers": 3, "mamba_blocks": 1, "walk_length": 2}),
        ("2", {"gnn_type": "GAT", "gnn_layers": 3, "mamba_blocks": 2, "walk_length": 2}),
        ("3", {"gnn_type": "GAT", "gnn_layers": 3, "mamba_blocks": 3, "walk_length": 2}),
    ],
    "walk_length": [
        ("2", {"gnn_type": "GAT", "gnn_layers": 3, "mamba_blocks": 2, "walk_length": 2}),
        ("3", {"gnn_type": "GAT", "gnn_layers": 3, "mamba_blocks": 2, "walk_length": 3}),
        ("4", {"gnn_type": "GAT", "gnn_layers": 3, "mamba_blocks": 2, "walk_length": 4}),
        ("5", {"gnn_type": "GAT", "gnn_layers": 3, "mamba_blocks": 2, "walk_length": 5}),
    ],
}


def main() -> None:
    t0 = time.time()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed MIMIC-IV SA-AKI tensors for RQ4 Parameter Sensitivity...")
    data = dict(np.load(PROC_DIR / "sa_aki_tensors.npz", allow_pickle=True))
    seeds = [41, 42, 43]

    rows = []
    rq4_results: dict[str, object] = {"seeds": seeds, "n_folds": 10, "sweeps": {}}

    for sweep_name, items in RQ4_SWEEPS.items():
        print(f"=== Running RQ4 Sweep: {sweep_name} ===")
        rq4_results["sweeps"][sweep_name] = {}
        for param_val, kwargs in items:
            t_p = time.time()
            res = train_and_evaluate_model_cv(
                model_name=f"GBMN_{sweep_name}_{param_val}",
                tensors=data,
                seeds=seeds,
                n_folds=10,
                num_clusters=3,
                epochs=6,
                model_kwargs=kwargs,
            )
            rq4_results["sweeps"][sweep_name][param_val] = {
                "DBI_mean": res["DBI_mean"],
                "DBI_std": res["DBI_std"],
                "CHI_mean": res["CHI_mean"],
                "CHI_std": res["CHI_std"],
                "Silhouette_mean": res["Silhouette_mean"],
                "Silhouette_std": res["Silhouette_std"],
            }
            rows.append({
                "Sweep": sweep_name,
                "Setting": param_val,
                "DBI_mean": res["DBI_mean"],
                "DBI_std": res["DBI_std"],
                "CHI_mean": res["CHI_mean"],
                "CHI_std": res["CHI_std"],
                "Silhouette_mean": res["Silhouette_mean"],
                "Silhouette_std": res["Silhouette_std"],
            })
            print(f"  [{sweep_name}={param_val:4s}] DBI={res['DBI_mean']:.2f}±{res['DBI_std']:.2f} | CHI={res['CHI_mean']:.2f}±{res['CHI_std']:.2f} ({time.time()-t_p:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "rq4_parameter_sensitivity.csv", index=False)
    (RESULTS_DIR / "rq4_sensitivity.json").write_text(json.dumps(rq4_results, indent=2), encoding="utf-8")
    print(f"RQ4 Parameter Sensitivity completed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
