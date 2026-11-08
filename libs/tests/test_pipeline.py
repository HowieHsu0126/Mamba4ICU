from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from libs.models.gbmn import GBMN
from libs.models.baselines import SequenceBaselineEncoder, GraphBaselineEncoder
from libs.exp.metrics import evaluate_clustering_metrics, compute_cramers_v_and_p, compute_eta_squared_and_p
from libs.exp.run_tuning_experiment import TunedGBMNWrapper


def test_gbmn_forward_backward_and_ablations() -> None:
    torch.manual_seed(42)
    bsz, m_vars, seq_len, static_dim = 32, 21, 8, 31
    x_temp = torch.rand(bsz, m_vars, seq_len)
    x_demo = torch.stack([torch.randint(0, 2, (bsz,)), torch.randint(0, 5, (bsz,))], dim=1)
    x_comorb = torch.randint(0, 2, (bsz, 7)).float()
    x_static = torch.rand(bsz, static_dim)

    for kwargs in [
        {},
        {"use_comorbidity": False},
        {"use_lgi": False},
        {"use_gnn": False},
        {"use_bimamba": False},
        {"gnn_type": "GCN", "gnn_layers": 2, "mamba_blocks": 1, "walk_length": 3},
        {"gnn_type": "GIN", "gnn_layers": 4, "mamba_blocks": 3, "walk_length": 4},
    ]:
        model = GBMN(
            num_temporal_vars=m_vars,
            static_dim=static_dim,
            latent_dim=64,
            lstm_dim=16,
            demo_embed_dim=16,
            comorb_embed_dim=16,
            pe_dim=8,
            num_clusters=3,
            **kwargs,
        )
        out = model(x_temp, x_demo, x_comorb, x_static)
        assert out["embeddings"].shape == (bsz, 64)
        assert out["subspace_embeddings"].shape == (bsz, 3)
        assert out["probs"].shape == (bsz, 3)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()


def test_all_baselines_and_metrics() -> None:
    torch.manual_seed(42)
    bsz, m_vars, seq_len, static_dim = 32, 21, 8, 31
    x_temp = torch.rand(bsz, m_vars, seq_len)
    x_static = torch.rand(bsz, static_dim)

    for arch in ["LSTM", "Transformer", "1D-CNN"]:
        m = SequenceBaselineEncoder(arch=arch, num_temporal_vars=m_vars, static_dim=static_dim, hidden_dim=32, latent_dim=16)
        out = m(x_temp, x_static)
        assert out["embeddings"].shape == (bsz, 16)
        out["loss"].backward()

    for arch in ["GCN", "GAT", "GIN", "NAGphormer", "GPS"]:
        m = GraphBaselineEncoder(arch=arch, num_temporal_vars=m_vars, static_dim=static_dim, latent_dim=12)
        out = m(x_temp, x_static)
        assert out["embeddings"].shape[0] == bsz
        out["loss"].backward()

    emb = np.vstack([
        np.random.randn(20, 8) + 3.0,
        np.random.randn(20, 8) - 3.0,
        np.random.randn(20, 8),
    ])
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    res = evaluate_clustering_metrics(emb, labels)
    assert res["DBI"] > 0.0 and res["CHI"] > 1.0 and -1.0 <= res["Silhouette"] <= 1.0

    cv, p1, q1 = compute_cramers_v_and_p((labels == 2).astype(float), labels)
    eta, p2, q2 = compute_eta_squared_and_p(emb[:, 0], labels)
    assert cv > 0.5 and q1 == "L" and p1 < 0.001
    assert eta > 0.14 and q2 == "L" and p2 < 0.001


def test_tuned_gbmn_stages_and_outputs() -> None:
    torch.manual_seed(42)
    bsz, m_vars, seq_len, static_dim = 24, 21, 8, 31
    x_temp = torch.rand(bsz, m_vars, seq_len)
    x_demo = torch.stack([torch.randint(0, 2, (bsz,)), torch.randint(0, 5, (bsz,))], dim=1)
    x_comorb = torch.randint(0, 2, (bsz, 7)).float()
    x_static = torch.rand(bsz, static_dim)

    for stage in [0, 1, 2, 3, 4]:
        model = TunedGBMNWrapper(num_temporal_vars=m_vars, static_dim=static_dim, stage=stage)
        out = model(x_temp, x_demo, x_comorb, x_static)
        assert out["probs"].shape == (bsz, 3)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()

    root = Path(__file__).resolve().parents[2]
    for rel in [
        "output/tables/table1_cohort_characteristics.csv",
        "output/tables/table2_performance_comparison.csv",
        "output/tables/table3_ablation_study.csv",
        "output/tables/table4_subphenotype_characteristics.csv",
        "output/figures/Fig3_Flow_MIMIC.pdf",
        "output/figures/Fig4_NumberOfCluster.pdf",
        "output/figures/Fig5_Outcomes.pdf",
        "output/figures/Fig6_tSNE_Voronoi.pdf",
        "output/figures/Fig7_SHAP.pdf",
        "output/figures/Fig8_ParameterSensitivity.pdf",
    ]:
        p = root / rel
        assert p.exists() and p.stat().st_size > 0, f"Missing output artifact: {rel}"


def test_pretrained_checkpoints_loading() -> None:
    root = Path(__file__).resolve().parents[2]
    ckpt_path = root / "output" / "ckpts" / "gbmn_best.pt"
    assert ckpt_path.exists() and ckpt_path.stat().st_size > 0
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GBMN(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_temp = torch.rand(8, 21, 8)
    x_demo = torch.zeros(8, 2, dtype=torch.long)
    x_comorb = torch.zeros(8, 7)
    x_static = torch.rand(8, 31)
    with torch.no_grad():
        out = model(x_temp, x_demo, x_comorb, x_static)
        assert out["probs"].shape == (8, 3)
        assert out["subspace_embeddings"].shape == (8, 3)


def test_postgresql_scripts_present() -> None:
    root = Path(__file__).resolve().parents[2]
    sql_dir = root / "libs" / "data" / "sql"
    expected_sql = [
        "01_stay_master.sql",
        "02_kdigo_aki_labels.sql",
        "03_sepsis3_and_sa_aki_cohort.sql",
        "04_comorbidities_and_static.sql",
        "05_vitals_hourly_48h.sql",
        "06_labs_hourly_48h.sql",
        "07_urine_output_hourly_48h.sql",
    ]
    for name in expected_sql:
        p = sql_dir / name
        assert p.exists() and p.stat().st_size > 100, f"Missing or empty SQL script: {name}"
        content = p.read_text(encoding="utf-8").upper()
        assert "SELECT" in content and "FROM" in content


def test_cohort_flow_and_no_sorting_truncation() -> None:
    import json
    import pandas as pd

    root = Path(__file__).resolve().parents[2]
    preprocess_src = (root / "libs" / "data" / "preprocess.py").read_text(encoding="utf-8")
    assert "iloc[:" not in preprocess_src, "Sorting truncation (iloc[:N]) must not be used in preprocess.py"

    raw_flow = json.loads((root / "input" / "raw" / "flow_counts.json").read_text(encoding="utf-8"))
    cohort_flow = json.loads((root / "output" / "results" / "cohort_flow.json").read_text(encoding="utf-8"))

    assert raw_flow["sepsis_stays"] == 33199
    assert raw_flow["aki_stays"] == 34854
    assert raw_flow["sa_aki_within_7d_stays"] == 20317
    assert raw_flow["excluded_los_lt_24h"] == 1523
    assert raw_flow["excluded_age_out_of_range"] == 1015
    assert raw_flow["excluded_eskd"] == 304
    assert raw_flow["excluded_missing_vitals"] == 1114
    assert raw_flow["final_sa_aki_cohort_n"] == 16361

    assert cohort_flow["manuscript_sepsis_n"] == 33199
    assert cohort_flow["manuscript_aki_n"] == 34854
    assert cohort_flow["manuscript_sa_aki_initial_n"] == 20317
    assert cohort_flow["manuscript_excluded_los_lt_24h"] == 1523
    assert cohort_flow["manuscript_excluded_age_out_of_range"] == 1015
    assert cohort_flow["manuscript_excluded_eskd"] == 304
    assert cohort_flow["manuscript_excluded_missing_vitals"] == 1114
    assert cohort_flow["final_sa_aki_cohort_n"] == 16361

    static_df = pd.read_parquet(root / "input" / "raw" / "sa_aki_candidates_static.parquet")
    cohort_df = pd.read_parquet(root / "input" / "processed" / "sa_aki_cohort.parquet")
    assert len(static_df) == 20317
    assert len(cohort_df) == 16361
    assert bool(cohort_df["within_7d"].all()) is True
    assert bool(cohort_df["pass_hard_filters"].all()) is True
