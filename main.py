from __future__ import annotations

import argparse
from libs.data import extract_mimic_cohort, preprocess
from libs.exp import (
    export_checkpoints,
    plot_figures,
    run_all,
    run_rq1_effectiveness,
    run_rq2_ablation,
    run_rq3_clinical,
    run_rq4_sensitivity,
    run_tuning_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mamba4ICU (GBMN): Subphenotype Identification for Sepsis-Associated Acute Kidney Injury"
    )
    parser.add_argument(
        "--stage",
        choices=[
            "extract",
            "preprocess",
            "rq1",
            "rq2",
            "rq3",
            "rq4",
            "tune",
            "export-ckpts",
            "plot",
            "all",
        ],
        default="all",
        help="Pipeline stage to execute (default: all).",
    )
    args = parser.parse_args()

    dispatch = {
        "extract": extract_mimic_cohort.main,
        "preprocess": preprocess.main,
        "rq1": run_rq1_effectiveness.main,
        "rq2": run_rq2_ablation.main,
        "rq3": run_rq3_clinical.main,
        "rq4": run_rq4_sensitivity.main,
        "tune": run_tuning_experiment.main,
        "export-ckpts": export_checkpoints.main,
        "plot": plot_figures.main,
        "all": run_all.main,
    }
    dispatch[args.stage]()


if __name__ == "__main__":
    main()
