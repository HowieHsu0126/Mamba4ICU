from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableModularityClustering(nn.Module):
    """
    Implements Section III-D: Differentiable Patient Subphenotyping.
    - Eq. 21: Z = softmax(y_output)
    - Eq. 22: L_c(Z; A) = -1/(2U) Tr(Z^T M Z) + sqrt(k)/N ||sum_i Z_i^T||_F - 1
      where M = A - (d d^T)/(2U) is the spectral modularity matrix.
    """

    def __init__(self, latent_dim: int = 512, num_clusters: int = 3, temperature: float = 0.2) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        self.cluster_head = nn.Linear(latent_dim, num_clusters)

    def forward(self, y_output: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """
        Args:
            y_output: (N, D)
            adj: (N, N)
        Returns:
            z: (N, k) soft cluster assignments
            loss_c: scalar clustering loss (Eq. 22)
            metrics: dict of modularity and collapse terms
        """
        n_nodes = y_output.shape[0]
        k = self.num_clusters

        logits = self.cluster_head(F.normalize(y_output, p=2, dim=-1)) / self.temperature
        z = F.softmax(logits, dim=-1)  # Eq. 21

        deg = adj.sum(dim=-1, keepdim=True)  # (N, 1)
        two_u = deg.sum().clamp(min=1e-6)    # 2U

        # Tr(Z^T M Z) = Tr(Z^T A Z) - 1/(2U) ||d^T Z||_2^2
        az = torch.matmul(adj, z)
        tr_z_a_z = torch.sum(z * az)
        d_z = torch.matmul(deg.transpose(0, 1), z)  # (1, k)
        null_term = torch.sum(d_z ** 2) / two_u
        modularity = (tr_z_a_z - null_term) / two_u
        loss_modularity = -modularity

        # Collapse regularization: sqrt(k)/N ||sum_{i=1}^N Z_i^T||_F - 1
        cluster_sizes = z.sum(dim=0)  # (k,)
        loss_collapse = (math.sqrt(float(k)) / float(n_nodes)) * torch.norm(cluster_sizes, p=2) - 1.0

        # Feature-prototype compactness loss to sharpen cluster separation in representation space
        y_norm = F.normalize(y_output, p=2, dim=-1)
        weights = z / (cluster_sizes.unsqueeze(0) + 1e-6)
        centroids = F.normalize(torch.matmul(weights.transpose(0, 1), y_norm), p=2, dim=-1)  # (k, D)
        sim_to_centroids = torch.matmul(y_norm, centroids.transpose(0, 1))
        loss_compact = -torch.mean(torch.sum(z * sim_to_centroids, dim=-1))

        loss_c = loss_modularity + loss_collapse + 0.5 * loss_compact
        return z, loss_c, {
            "modularity": float(modularity.detach().item()),
            "collapse_reg": float(loss_collapse.detach().item()),
            "loss_c": float(loss_c.detach().item()),
        }
