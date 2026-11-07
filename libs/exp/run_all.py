from __future__ import annotations

import time
from libs.exp import (
    run_rq1_effectiveness,
    run_rq2_ablation,
    run_rq3_clinical,
    run_rq4_sensitivity,
    plot_figures,
)


def main() -> None:
    t0 = time.time()
    print("=================================================================")
    print("Starting Full GBMN Reproduction Pipeline (RQ1, RQ2, RQ3, RQ4)")
    print("=================================================================")
    run_rq1_effectiveness.main()
    run_rq2_ablation.main()
    run_rq3_clinical.main()
    run_rq4_sensitivity.main()
    plot_figures.main()
    print("=================================================================")
    print(f"Full GBMN Reproduction Pipeline completed successfully in {time.time() - t0:.1f}s.")
    print("=================================================================")


if __name__ == "__main__":
    main()
