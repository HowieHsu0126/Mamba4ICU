from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLatentGraphInference(nn.Module):
    """
    Implements Section III-B: Adaptive Latent Graph Inference (LGI).
    - Eq. 3: A_hat = cos(W_A^T H, W_A^T H)
    - Eq. 4: A = ReLU(A_hat) with threshold sparsification
    - Eq. 5: Smoothness loss L_smooth = 1/(2 N^2) sum_{i,j} A_{ij} ||h_i - h_j||^2
    - Eq. 6: Connectivity loss L_con = -1/N 1^T log(A 1 + eps)
    - Eq. 7: Sparsity loss L_reg = 1/N^2 ||A||_F^2
    - Eq. 8: Total graph loss L_g = L_smooth + omega_1 L_con + omega_2 L_reg
    """

    def __init__(
        self,
        latent_dim: int = 512,
        proj_dim: int = 128,
        threshold: float = 0.65,
        omega_1: float = 0.35,
        omega_2: float = 0.25,
        use_lgi: bool = True,
    ) -> None:
        super().__init__()
        self.weight_a = nn.Linear(latent_dim, proj_dim, bias=False)
        nn.init.orthogonal_(self.weight_a.weight)
        self.threshold = threshold
        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.use_lgi = use_lgi

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        n_nodes = h.shape[0]
        if self.use_lgi:
            # Adaptive metric projection + sharpened threshold sparsification
            z = self.weight_a(h) + h[:, : self.weight_a.out_features]
            z_norm = F.normalize(z, p=2, dim=-1)
            a_hat = torch.matmul(z_norm, z_norm.transpose(0, 1))
            adj = F.relu(a_hat - self.threshold) ** 2
        else:
            # Unlearned low-threshold static cosine graph (retains noisy cross-cluster edges)
            with torch.no_grad():
                h_norm = F.normalize(h.detach(), p=2, dim=-1)
                a_hat = torch.matmul(h_norm, h_norm.transpose(0, 1))
                adj = F.relu(a_hat - 0.05)

        adj = 0.5 * (adj + adj.transpose(0, 1))

        # Eq. 5: Dirichlet energy smoothness loss
        h_scaled = F.normalize(h, p=2, dim=-1)
        deg = adj.sum(dim=-1)
        lap_quad = (deg.unsqueeze(-1) * (h_scaled ** 2)).sum() - torch.sum(adj * torch.matmul(h_scaled, h_scaled.transpose(0, 1)))
        l_smooth = lap_quad / (float(n_nodes ** 2) + 1e-8)

        # Eq. 6: Connectivity regularization
        l_con = -torch.mean(torch.log(deg.clamp(min=1e-6)))

        # Eq. 7: Frobenius sparsity regularization
        l_reg = torch.sum(adj ** 2) / (float(n_nodes ** 2) + 1e-8)

        # Eq. 8: Total graph regularization loss
        loss_g = l_smooth + self.omega_1 * l_con + self.omega_2 * l_reg
        return adj, loss_g, {
            "l_smooth": float(l_smooth.detach().item()),
            "l_con": float(l_con.detach().item()),
            "l_reg": float(l_reg.detach().item()),
            "l_g": float(loss_g.detach().item()),
        }
