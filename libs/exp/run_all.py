from __future__ import annotations

import time
from libs.exp import (
    export_checkpoints,
    plot_figures,
    run_rq1_effectiveness,
    run_rq2_ablation,
    run_rq3_clinical,
    run_rq4_sensitivity,
    run_tuning_experiment,
)


def main() -> None:
    t0 = time.time()
    print("=================================================================")
    print("Starting Full GBMN / Mamba4ICU Experimental Pipeline (RQ1-RQ4)")
    print("=================================================================")
    run_rq1_effectiveness.main()
    run_rq2_ablation.main()
    run_rq3_clinical.main()
    run_rq4_sensitivity.main()
    run_tuning_experiment.main()
    export_checkpoints.main()
    plot_figures.main()
    print("=================================================================")
    print(f"Full Mamba4ICU Pipeline completed successfully in {time.time() - t0:.1f}s.")
    print("=================================================================")


if __name__ == "__main__":
    main()
