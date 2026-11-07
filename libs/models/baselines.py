from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.models.graph_mamba import DenseGNNLayer


def build_precomputed_graph(h: torch.Tensor, threshold: float = 0.55) -> torch.Tensor:
    """Builds a symmetric, non-negative thresholded cosine similarity graph for static-graph baselines."""
    with torch.no_grad():
        h_norm = F.normalize(h, p=2, dim=-1)
        sim = torch.matmul(h_norm, h_norm.transpose(0, 1))
        adj = F.relu(sim - threshold)
        eye = torch.eye(h.shape[0], device=h.device, dtype=h.dtype)
        adj = 0.5 * (adj + adj.transpose(0, 1)) + eye
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1e-6).pow(-0.5)
        return deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(0)


class SequenceBaselineEncoder(nn.Module):
    """
    Implements Sequence Learning Baselines (Section IV-B):
    - LSTM
    - Transformer
    - 1D-CNN
    """

    def __init__(
        self,
        arch: str = "LSTM",
        num_temporal_vars: int = 21,
        static_dim: int = 31,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_clusters: int = 3,
    ) -> None:
        super().__init__()
        self.arch = arch.upper()
        self.in_proj = nn.Linear(num_temporal_vars, hidden_dim)
        if self.arch == "LSTM":
            self.seq_net = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
            )
        elif self.arch == "TRANSFORMER":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.seq_net = nn.TransformerEncoder(enc_layer, num_layers=2)
        elif self.arch == "1D-CNN":
            self.seq_net = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unsupported sequence baseline: {arch}")

        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.cluster_head = nn.Linear(latent_dim, num_clusters)
        self.recon_head = nn.Linear(latent_dim, static_dim)

    def forward(self, x_temporal: torch.Tensor, x_static: torch.Tensor) -> dict[str, torch.Tensor]:
        x_seq = self.in_proj(x_temporal.transpose(1, 2))
        if self.arch == "LSTM":
            seq_out, _ = self.seq_net(x_seq)
            h_seq = seq_out[:, -1, :]
        elif self.arch == "TRANSFORMER":
            seq_out = self.seq_net(x_seq)
            h_seq = seq_out.mean(dim=1)
        else:
            seq_out = self.seq_net(x_seq.transpose(1, 2)).transpose(1, 2)
            h_seq = seq_out.mean(dim=1)

        h_stat = F.gelu(self.static_proj(x_static))
        emb = self.out_proj(torch.cat([h_seq, h_stat], dim=-1))
        probs = F.softmax(self.cluster_head(emb), dim=-1)
        recon = self.recon_head(emb)
        loss = F.mse_loss(recon, x_static)
        return {"embeddings": emb, "probs": probs, "loss": loss}


class GraphBaselineEncoder(nn.Module):
    """
    Implements Graph-Based Baselines (Section IV-B):
    - GCN (Kipf & Welling 2017)
    - GAT (Veličković et al. 2018)
    - GIN (Xu et al. 2019)
    - NAGphormer (Chen et al. ICLR 2023)
    - GPS (Rampášek et al. NeurIPS 2022)
    """

    def __init__(
        self,
        arch: str = "GCN",
        num_temporal_vars: int = 21,
        static_dim: int = 31,
        latent_dim: int = 12,
        num_layers: int = 3,
        num_hops: int = 2,
        num_clusters: int = 3,
    ) -> None:
        super().__init__()
        self.arch = arch.upper()
        self.num_hops = num_hops
        hidden_dim = 64
        self.input_proj = nn.Sequential(
            nn.Linear(num_temporal_vars + static_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.static_skip = nn.Linear(static_dim, hidden_dim)

        if self.arch in {"GCN", "GAT", "GIN"}:
            self.layers = nn.ModuleList([
                DenseGNNLayer(dim=hidden_dim, gnn_type=self.arch, heads=4)
                for _ in range(num_layers)
            ])
            out_dim = latent_dim
        elif self.arch == "NAGPHORMER":
            self.hop_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
            self.hop_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.norm = nn.LayerNorm(hidden_dim)
            out_dim = max(latent_dim // 2, 6)
        elif self.arch == "GPS":
            self.local_layers = nn.ModuleList([
                DenseGNNLayer(dim=hidden_dim, gnn_type="GAT", heads=4)
                for _ in range(2)
            ])
            self.global_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
            out_dim = max(latent_dim // 2, 6)
        else:
            raise ValueError(f"Unsupported graph baseline: {arch}")

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.cluster_head = nn.Linear(out_dim, num_clusters)
        self.recon_head = nn.Linear(out_dim, static_dim)

    def forward(self, x_temporal: torch.Tensor, x_static: torch.Tensor) -> dict[str, torch.Tensor]:
        ts_mean = x_temporal.mean(dim=-1)
        h0 = self.input_proj(torch.cat([ts_mean, x_static], dim=-1)) + 0.5 * self.static_skip(x_static)
        adj_norm = build_precomputed_graph(h0)

        if self.arch in {"GCN", "GAT", "GIN"}:
            h = h0
            for layer in self.layers:
                h = layer(h, adj_norm)
            h_final = h
        elif self.arch == "NAGPHORMER":
            hop_tokens = [h0]
            curr = h0
            for _ in range(self.num_hops):
                curr = torch.matmul(adj_norm, curr)
                hop_tokens.append(curr)
            seq = torch.stack(hop_tokens, dim=1)
            attn_out, _ = self.hop_attn(seq, seq, seq)
            seq = self.norm(seq + attn_out)
            h_final = self.norm(seq[:, 0, :] + self.hop_ffn(seq.mean(dim=1)))
        else:  # GPS
            h = h0
            for layer in self.local_layers:
                h_local = layer(h, adj_norm)
                h_glob, _ = self.global_attn(h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0))
                h = self.norm(h_local + 0.3 * h_glob.squeeze(0))
            h_final = h

        emb = F.normalize(self.readout(h_final), p=2, dim=-1)
        probs = F.softmax(self.cluster_head(emb) / 0.25, dim=-1)
        recon = self.recon_head(emb)
        # Feature reconstruction + graph smoothness loss
        deg = adj_norm.sum(dim=-1, keepdim=True)
        smooth_loss = torch.mean((emb - torch.matmul(adj_norm, emb)) ** 2)
        loss = F.mse_loss(recon, x_static) + 0.2 * smooth_loss
        return {"embeddings": emb, "probs": probs, "loss": loss}
