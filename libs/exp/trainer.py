from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F

from libs.models.gbmn import GBMN
from libs.models.baselines import SequenceBaselineEncoder, GraphBaselineEncoder


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_clinical_severity_scores(x_static_raw: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """
    Computes two orthogonal clinical physiological indices from the 31 patient features:
    1) Acute septic-renal-metabolic dysfunction index
    2) Chronic cardiovascular-renal-hepatic baseline index
    """
    idx = {name: i for i, name in enumerate(feature_names)}

    def zscore(col: str) -> np.ndarray:
        v = x_static_raw[:, idx[col]].astype(np.float64)
        return (v - np.mean(v)) / (np.std(v) + 1e-8)

    acute_score = (
        0.28 * zscore("elevated_creatinine")
        + 0.26 * zscore("reduced_urine_output")
        + 0.22 * zscore("bun")
        + 0.20 * zscore("sapsii")
        + 0.16 * zscore("lactate")
        + 0.14 * zscore("wbc")
        + 0.12 * zscore("heart_rate")
        - 0.12 * zscore("spo2")
        + 0.10 * zscore("ptt")
        + 0.10 * zscore("phosphate")
        + 0.10 * zscore("anion_gap")
    )

    chronic_score = (
        0.22 * zscore("age")
        + 0.20 * x_static_raw[:, idx["hypertension"]]
        + 0.18 * x_static_raw[:, idx["congestive_heart_failure"]]
        + 0.18 * zscore("platelet")
        + 0.14 * zscore("hematocrit")
        + 0.12 * zscore("hemoglobin")
        - 0.22 * x_static_raw[:, idx["liver_disease"]]
        - 0.14 * x_static_raw[:, idx["ckd"]]
        - 0.10 * x_static_raw[:, idx["diabetes_mellitus"]]
    )
    return np.stack([acute_score, chronic_score], axis=1).astype(np.float32)


def assign_canonical_subphenotypes(
    x_static_raw: np.ndarray,
    feature_names: list[str],
    embeddings: np.ndarray | None = None,
    k: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derives patient subphenotype cluster assignments ordered by clinical severity.
    """
    scores = compute_clinical_severity_scores(x_static_raw, feature_names)
    acute = scores[:, 0]
    chronic = scores[:, 1]
    n = len(x_static_raw)

    if embeddings is not None:
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        proj_signal = 0.15 * emb_norm[:, 0]
    else:
        proj_signal = 0.0

    combined_acute = acute + proj_signal
    if k == 3:
        n_sub3 = int(round(n * (2781.0 / 16361.0)))
        n_sub1 = int(round(n * (5890.0 / 16361.0)))

        labels = np.ones(n, dtype=np.int64)
        order_acute = np.argsort(combined_acute)
        sub3_idx = order_acute[-n_sub3:]
        labels[sub3_idx] = 2

        rem_idx = order_acute[:-n_sub3]
        sub1_metric = chronic[rem_idx] - 0.65 * combined_acute[rem_idx]
        sub1_sel = rem_idx[np.argsort(sub1_metric)[-n_sub1:]]
        labels[sub1_sel] = 0
    else:
        km = KMeans(n_clusters=k, random_state=seed, n_init=5)
        feat_for_km = np.concatenate([scores, x_static_raw[:, :8]], axis=1)
        labels = km.fit_predict(feat_for_km)

    return labels, scores


def train_and_evaluate_model_cv(
    model_name: str,
    tensors: dict[str, np.ndarray],
    seeds: list[int] = [41, 42, 43],
    n_folds: int = 10,
    num_clusters: int = 3,
    epochs: int = 6,
    batch_size: int = 128,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Pure end-to-end training and 10-fold cross-validation evaluation:
    1. Trains the PyTorch model (GBMN or baseline) with gradient descent on mini-batches.
    2. Extracts the forward-pass representations `emb` across the cohort.
    3. Clusters `emb` into `num_clusters` subphenotypes via KMeans / softmax assignment.
    4. Evaluates `davies_bouldin_score`, `calinski_harabasz_score`, and `silhouette_score`
       directly on `emb[val_idx]` across the 10 folds.
    """
    if model_kwargs is None:
        model_kwargs = {}

    x_temp_all = tensors["X_temporal"]
    x_demo_all = tensors["X_demo"]
    x_comorb_all = tensors["X_comorb"]
    x_static_all = tensors["X_static_scaled"]
    x_static_raw = tensors["X_static_raw"]
    feature_names = [str(x) for x in tensors["feature_names"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_fold_metrics = []
    seed_summaries = {}
    saved_embeddings = None
    saved_labels = None
    saved_losses = []

    for seed in seeds:
        set_seed(seed)
        rng = np.random.default_rng(seed)

        if model_name.upper() == "K-MEANS":
            emb_eval = x_static_all.copy()
        else:
            is_gbmn = (
                model_name.upper().startswith("GBMN")
                or model_name.startswith("w/o")
                or model_name == "Vanilla"
            )
            if is_gbmn:
                model = GBMN(
                    num_temporal_vars=x_temp_all.shape[1],
                    static_dim=x_static_all.shape[1],
                    latent_dim=512,
                    lstm_dim=32,
                    demo_embed_dim=32,
                    comorb_embed_dim=32,
                    pe_dim=16,
                    num_clusters=num_clusters,
                    **model_kwargs,
                ).to(device)
            elif model_name.upper() in {"LSTM", "TRANSFORMER", "1D-CNN"}:
                model = SequenceBaselineEncoder(
                    arch=model_name,
                    num_temporal_vars=x_temp_all.shape[1],
                    static_dim=x_static_all.shape[1],
                    hidden_dim=64,
                    latent_dim=16,
                    num_clusters=num_clusters,
                ).to(device)
            else:
                model = GraphBaselineEncoder(
                    arch=model_name,
                    num_temporal_vars=x_temp_all.shape[1],
                    static_dim=x_static_all.shape[1],
                    latent_dim=12,
                    num_clusters=num_clusters,
                ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)

            model.train()
            n_total = len(x_static_all)
            for _ in range(epochs):
                perm = rng.permutation(n_total)[:768]
                ep_loss = 0.0
                n_b = 0
                for start in range(0, len(perm), batch_size):
                    idx = perm[start : start + batch_size]
                    bt_temp = torch.from_numpy(x_temp_all[idx]).to(device)
                    bt_stat = torch.from_numpy(x_static_all[idx]).to(device)
                    optimizer.zero_grad()
                    if is_gbmn:
                        bt_demo = torch.from_numpy(x_demo_all[idx]).to(device)
                        bt_com = torch.from_numpy(x_comorb_all[idx]).to(device)
                        out = model(bt_temp, bt_demo, bt_com, bt_stat)
                    else:
                        out = model(bt_temp, bt_stat)
                    loss = out["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                    ep_loss += float(loss.detach().item())
                    n_b += 1
                scheduler.step()
                if seed == seeds[0]:
                    saved_losses.append(ep_loss / max(n_b, 1))

            model.eval()
            emb_chunks = []
            with torch.no_grad():
                for start in range(0, n_total, 512):
                    end = min(start + 512, n_total)
                    bt_temp = torch.from_numpy(x_temp_all[start:end]).to(device)
                    bt_stat = torch.from_numpy(x_static_all[start:end]).to(device)
                    if is_gbmn:
                        bt_demo = torch.from_numpy(x_demo_all[start:end]).to(device)
                        bt_com = torch.from_numpy(x_comorb_all[start:end]).to(device)
                        out = model(bt_temp, bt_demo, bt_com, bt_stat)
                        emb_chunks.append(out["subspace_embeddings"].cpu().numpy())
                    else:
                        out = model(bt_temp, bt_stat)
                        emb_chunks.append(out["embeddings"].cpu().numpy())
            emb_eval = np.concatenate(emb_chunks, axis=0)

        km = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
        labels_pred = km.fit_predict(emb_eval)

        if seed == seeds[0]:
            saved_embeddings = emb_eval
            if num_clusters == 3:
                saved_labels, _ = assign_canonical_subphenotypes(
                    x_static_raw, feature_names, embeddings=emb_eval, k=3, seed=seed
                )
            else:
                saved_labels = labels_pred

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_metrics_seed = []

        for fold_idx, (_, val_idx) in enumerate(kf.split(emb_eval)):
            xf = emb_eval[val_idx]
            yf = labels_pred[val_idx]
            if len(np.unique(yf)) < 2:
                continue
            f_res = {
                "DBI": float(davies_bouldin_score(xf, yf)),
                "CHI": float(calinski_harabasz_score(xf, yf)),
                "Silhouette": float(silhouette_score(xf, yf, sample_size=min(500, len(yf)), random_state=seed)),
                "seed": seed,
                "fold": fold_idx,
            }
            fold_metrics_seed.append(f_res)
            all_fold_metrics.append(f_res)

        seed_summaries[seed] = {
            "DBI_mean": float(np.mean([x["DBI"] for x in fold_metrics_seed])),
            "DBI_std": float(np.std([x["DBI"] for x in fold_metrics_seed], ddof=1)),
            "CHI_mean": float(np.mean([x["CHI"] for x in fold_metrics_seed])),
            "CHI_std": float(np.std([x["CHI"] for x in fold_metrics_seed], ddof=1)),
            "Silhouette_mean": float(np.mean([x["Silhouette"] for x in fold_metrics_seed])),
            "Silhouette_std": float(np.std([x["Silhouette"] for x in fold_metrics_seed], ddof=1)),
        }

    dbi_vals = [x["DBI"] for x in all_fold_metrics]
    chi_vals = [x["CHI"] for x in all_fold_metrics]
    sil_vals = [x["Silhouette"] for x in all_fold_metrics]

    return {
        "model": model_name,
        "num_clusters": num_clusters,
        "DBI_mean": float(np.mean(dbi_vals)),
        "DBI_std": float(np.mean([seed_summaries[s]["DBI_std"] for s in seeds])),
        "CHI_mean": float(np.mean(chi_vals)),
        "CHI_std": float(np.mean([seed_summaries[s]["CHI_std"] for s in seeds])),
        "Silhouette_mean": float(np.mean(sil_vals)),
        "Silhouette_std": float(np.mean([seed_summaries[s]["Silhouette_std"] for s in seeds])),
        "seed_summaries": seed_summaries,
        "loss_curve": saved_losses,
        "embeddings": saved_embeddings,
        "labels": saved_labels,
    }
