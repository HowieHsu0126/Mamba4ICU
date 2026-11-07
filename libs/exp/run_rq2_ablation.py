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

ABLATION_CONFIGS: list[tuple[str, dict[str, bool]]] = [
    ("Vanilla", {"use_comorbidity": True, "use_lgi": True, "use_gnn": True, "use_bimamba": True}),
    ("w/o Comorbidity", {"use_comorbidity": False, "use_lgi": True, "use_gnn": True, "use_bimamba": True}),
    ("w/o LGI", {"use_comorbidity": True, "use_lgi": False, "use_gnn": True, "use_bimamba": True}),
    ("w/o GNNs", {"use_comorbidity": True, "use_lgi": True, "use_gnn": False, "use_bimamba": True}),
    ("w/o BiMamba", {"use_comorbidity": True, "use_lgi": True, "use_gnn": True, "use_bimamba": False}),
]


def main() -> None:
    t0 = time.time()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed MIMIC-IV SA-AKI tensors for RQ2 Ablation Study...")
    data = dict(np.load(PROC_DIR / "sa_aki_tensors.npz", allow_pickle=True))
    seeds = [41, 42, 43]

    rows = []
    rq2_results: dict[str, object] = {"seeds": seeds, "n_folds": 10, "ablations": {}}

    for variant_name, kwargs in ABLATION_CONFIGS:
        t_v = time.time()
        res = train_and_evaluate_model_cv(
            model_name=variant_name,
            tensors=data,
            seeds=seeds,
            n_folds=10,
            num_clusters=3,
            epochs=6,
            model_kwargs=kwargs,
        )
        rq2_results["ablations"][variant_name] = {
            k: v for k, v in res.items() if k not in {"embeddings", "labels"}
        }
        row = {
            "Variants": variant_name,
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
        rows.append(row)
        print(f"  [{variant_name:16s}] DBI={row['DBI']} | CHI={row['CHI']} | Silhouette={row['Silhouette']} ({time.time()-t_v:.1f}s)")

    ab_df = pd.DataFrame(rows)
    ab_df.to_csv(TABLES_DIR / "table3_ablation_study.csv", index=False)
    (RESULTS_DIR / "rq2_ablation.json").write_text(json.dumps(rq2_results, indent=2), encoding="utf-8")
    print(f"RQ2 Ablation Study completed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
