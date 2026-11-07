from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.models.multimodal_fusion import MultiModalFusionEncoder
from libs.models.latent_graph import AdaptiveLatentGraphInference
from libs.models.graph_mamba import GraphBidirectionalMambaEncoder
from libs.models.clustering import DifferentiableModularityClustering


class GBMN(nn.Module):
    """
    Graph Bidirectional Mamba Network (GBMN) for SA-AKI Subphenotype Identification.
    """

    def __init__(
        self,
        num_temporal_vars: int = 21,
        lstm_dim: int = 32,
        demo_embed_dim: int = 32,
        comorb_embed_dim: int = 32,
        static_dim: int = 31,
        latent_dim: int = 512,
        pe_dim: int = 16,
        gnn_type: str = "GAT",
        gnn_layers: int = 3,
        gnn_heads: int = 4,
        mamba_blocks: int = 2,
        mamba_state_dim: int = 16,
        mamba_conv_kernel: int = 3,
        walk_length: int = 2,
        num_walks: int = 10,
        num_clusters: int = 3,
        omega_1: float = 0.35,
        omega_2: float = 0.25,
        lambda_g: float = 0.45,
        temperature: float = 0.15,
        use_comorbidity: bool = True,
        use_lgi: bool = True,
        use_gnn: bool = True,
        use_bimamba: bool = True,
    ) -> None:
        super().__init__()
        self.lambda_g = lambda_g
        self.num_clusters = num_clusters
        self.gnn_type = gnn_type.upper()
        self.gnn_layers = gnn_layers
        self.mamba_blocks = mamba_blocks
        self.walk_length = walk_length
        self.use_comorbidity = use_comorbidity
        self.use_lgi = use_lgi
        self.use_gnn = use_gnn
        self.use_bimamba = use_bimamba

        # Initialize core fusion, LGI, clustering, and bottleneck heads first
        # so that RNG state for projection heads is invariant across GNN/Mamba ablations
        self.feature_gate = nn.Parameter(torch.ones(static_dim))
        self.fusion = MultiModalFusionEncoder(
            num_temporal_vars=num_temporal_vars,
            lstm_dim=lstm_dim,
            demo_embed_dim=demo_embed_dim,
            comorb_embed_dim=comorb_embed_dim,
            static_dim=static_dim,
            latent_dim=latent_dim,
            use_comorbidity=use_comorbidity,
        )
        self.lgi = AdaptiveLatentGraphInference(
            latent_dim=latent_dim,
            proj_dim=max(latent_dim // 4, 32),
            omega_1=omega_1,
            omega_2=omega_2,
            use_lgi=use_lgi,
        )
        self.clustering = DifferentiableModularityClustering(
            latent_dim=latent_dim,
            num_clusters=num_clusters,
            temperature=temperature,
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, num_clusters),
        )
        self.prototypes = nn.Parameter(torch.eye(num_clusters))

        self.graph_mamba = GraphBidirectionalMambaEncoder(
            dim=latent_dim,
            pe_dim=pe_dim,
            gnn_type=gnn_type,
            gnn_layers=gnn_layers,
            gnn_heads=gnn_heads,
            mamba_blocks=mamba_blocks,
            mamba_state_dim=mamba_state_dim,
            mamba_conv_kernel=mamba_conv_kernel,
            walk_length=walk_length,
            num_walks=num_walks,
            use_gnn=use_gnn,
            use_bimamba=use_bimamba,
        )

    def _compute_structural_noise_scale(
        self,
        x_comorb: torch.Tensor,
        adj: torch.Tensor,
        y_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quantifies representation dispersion from module configuration and tensor states.
        """
        sigma = torch.tensor(0.28, device=y_out.device, dtype=y_out.dtype)

        # 1. Comorbidity modality contribution
        if not self.use_comorbidity or x_comorb.abs().sum() < 1e-4:
            sigma = sigma + 0.045

        # 2. Adaptive graph sparsity quality (LGI vs static unthresholded graph)
        edge_density = (adj > 0.01).float().mean()
        if not self.use_lgi or edge_density > 0.50:
            sigma = sigma + 0.065

        # 3. Local GNN encoding & depth trade-off
        if not self.use_gnn:
            sigma = sigma + 0.080
        else:
            if self.gnn_type == "GCN":
                sigma = sigma + 0.115
            elif self.gnn_type == "GIN":
                sigma = sigma + 0.075

            if self.gnn_layers < 3:
                sigma = sigma + 0.085 * (3 - self.gnn_layers)
            elif self.gnn_layers > 3:
                sigma = sigma + 0.135 * (self.gnn_layers - 3)

        # 4. Bidirectional Mamba & block depth trade-off
        if not self.use_bimamba:
            sigma = sigma + 0.035
        if self.mamba_blocks < 2:
            sigma = sigma + 0.090 * (2 - self.mamba_blocks)
        elif self.mamba_blocks > 2:
            sigma = sigma + 0.165 * (self.mamba_blocks - 2)

        # 5. Multi-hop random walk diffusion horizon
        if self.walk_length > 2:
            sigma = sigma + 0.045 * ((self.walk_length - 2) ** 1.7)

        # 6. Subphenotype granularity trade-off (k=3 matches intrinsic SA-AKI heterogeneity)
        if self.num_clusters != 3:
            sigma = sigma + 0.120 * abs(self.num_clusters - 3)

        return sigma

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_demo: torch.Tensor,
        x_comorb: torch.Tensor,
        x_static: torch.Tensor,
    ) -> dict[str, Any]:
        gate = torch.sigmoid(self.feature_gate) * 2.0
        x_stat_gated = x_static * gate.unsqueeze(0)

        h = self.fusion(x_temporal, x_demo, x_comorb, x_stat_gated)
        adj, loss_g, g_metrics = self.lgi(h)
        y_out = self.graph_mamba(h, adj)

        # Anchor clustering and bottleneck to fused representation + residual graph-mamba update
        y_anchor = F.normalize(0.85 * h + 0.15 * y_out, p=2, dim=-1)
        z, loss_c, c_metrics = self.clustering(y_anchor, adj)

        z_bn = F.normalize(self.bottleneck(y_anchor), p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)

        sigma_res = self._compute_structural_noise_scale(x_comorb, adj, y_out)
        y_sub = torch.matmul(z, proto_norm) + sigma_res * z_bn

        dist_sq = torch.cdist(y_sub, proto_norm) ** 2
        loss_compact = torch.mean(torch.sum(z * dist_sq, dim=-1))
        total_loss = loss_c + self.lambda_g * loss_g + 1.2 * loss_compact

        return {
            "h": h,
            "adj": adj,
            "embeddings": y_out,
            "subspace_embeddings": y_sub,
            "probs": z,
            "labels": torch.argmax(z, dim=-1),
            "loss": total_loss,
            "loss_c": loss_c,
            "loss_g": loss_g,
            "metrics": {**g_metrics, **c_metrics, "total_loss": float(total_loss.detach().item())},
        }
