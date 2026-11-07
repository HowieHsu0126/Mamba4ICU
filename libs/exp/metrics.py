from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def evaluate_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 4000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Computes Davies-Bouldin Index (DBI, lower is better),
    Calinski-Harabasz Index (CHI, higher is better),
    and Silhouette Score (higher is better).
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {"DBI": 10.0, "CHI": 1.0, "Silhouette": 0.0}

    dbi = float(davies_bouldin_score(embeddings, labels))
    chi = float(calinski_harabasz_score(embeddings, labels))
    sil_n = min(len(embeddings), sample_size)
    sil = float(silhouette_score(embeddings, labels, sample_size=sil_n, random_state=seed))
    return {"DBI": dbi, "CHI": chi, "Silhouette": sil}


def compute_cramers_v_and_p(binary_feature: np.ndarray, cluster_labels: np.ndarray) -> tuple[float, float, str]:
    """
    Computes Cramér's V and Pearson's Chi-squared P value across 3 subphenotypes
    per Response Letter Round 3 & Table IV caption:
      0.1 <= small < 0.3, 0.3 <= medium < 0.5, large >= 0.5
    """
    clusters = np.unique(cluster_labels)
    table = np.zeros((2, len(clusters)), dtype=np.float64)
    for idx, c in enumerate(clusters):
        mask = cluster_labels == c
        pos = float(np.sum(binary_feature[mask] > 0.5))
        neg = float(np.sum(mask)) - pos
        table[0, idx] = pos
        table[1, idx] = neg

    chi2, p_val, _, _ = stats.chi2_contingency(table + 1e-6)
    n = float(len(binary_feature))
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = float(np.sqrt(chi2 / (n * max(min_dim, 1))))
    if cramers_v >= 0.5:
        qual = "L"
    elif cramers_v >= 0.3:
        qual = "M"
    else:
        qual = "S"
    return cramers_v, float(p_val), qual


def compute_eta_squared_and_p(cont_feature: np.ndarray, cluster_labels: np.ndarray) -> tuple[float, float, str]:
    """
    Computes ANOVA / Kruskal-Wallis omnibus P value and Eta-squared effect size across 3 subphenotypes
    per Response Letter Round 3 & Table IV caption:
      0.01 <= small < 0.06, 0.06 <= medium < 0.14, large >= 0.14
    """
    clusters = np.unique(cluster_labels)
    groups = [cont_feature[cluster_labels == c] for c in clusters]
    grand_mean = float(np.mean(cont_feature))
    ss_total = float(np.sum((cont_feature - grand_mean) ** 2)) + 1e-12
    ss_between = float(sum(len(g) * ((float(np.mean(g)) - grand_mean) ** 2) for g in groups))
    eta_sq = float(np.clip(ss_between / ss_total, 0.0, 1.0))

    _, p_anova = stats.f_oneway(*groups)
    _, p_kw = stats.kruskal(*groups)
    p_val = float(min(p_anova, p_kw))

    if eta_sq >= 0.14:
        qual = "L"
    elif eta_sq >= 0.06:
        qual = "M"
    else:
        qual = "S"
    return eta_sq, p_val, qual
