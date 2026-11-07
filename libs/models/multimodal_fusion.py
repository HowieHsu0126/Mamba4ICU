from __future__ import annotations

import torch
import torch.nn as nn


class MultiModalFusionEncoder(nn.Module):
    """
    Implements Section III-A: Input Representation.
    1. Time Series Embedding (Eq. 1): Variable-specific LSTM encoders + interleaved attention.
    2. Demographics & Comorbidity Embedding: Age groups (5 bins) + Sex + 7 Comorbidities.
    3. Multi-Modal Embedding Fusion (Eq. 2): H = E_T || E_D || E_C projected to latent_dim (512).
    """

    def __init__(
        self,
        num_temporal_vars: int = 21,
        lstm_dim: int = 32,
        num_age_groups: int = 5,
        demo_embed_dim: int = 32,
        num_comorbidities: int = 7,
        comorb_embed_dim: int = 32,
        static_dim: int = 31,
        latent_dim: int = 512,
        use_comorbidity: bool = True,
    ) -> None:
        super().__init__()
        self.num_temporal_vars = num_temporal_vars
        self.lstm_dim = lstm_dim
        self.use_comorbidity = use_comorbidity

        # Shared-architecture per-variable LSTM encoders (implemented efficiently via grouped/batched LSTM + variable embeddings)
        self.var_input_proj = nn.Linear(1, lstm_dim)
        self.var_id_embed = nn.Embedding(num_temporal_vars, lstm_dim)
        self.temporal_lstm = nn.LSTM(
            input_size=lstm_dim,
            hidden_size=lstm_dim,
            num_layers=1,
            batch_first=True,
        )
        # Interleaved attention across variables
        self.interleaved_attn = nn.MultiheadAttention(
            embed_dim=lstm_dim,
            num_heads=4,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(lstm_dim)

        # Demographics embeddings: sex (2 categories) + age_group (5 categories: 18-30, 31-40, 41-50, 51-60, 61-89)
        self.sex_embed = nn.Embedding(2, demo_embed_dim // 2)
        self.age_embed = nn.Embedding(num_age_groups, demo_embed_dim - demo_embed_dim // 2)
        self.demo_proj = nn.Sequential(
            nn.Linear(demo_embed_dim, demo_embed_dim),
            nn.GELU(),
            nn.LayerNorm(demo_embed_dim),
        )

        # Comorbidity embedding matrix E_C
        self.comorb_proj = nn.Sequential(
            nn.Linear(num_comorbidities, comorb_embed_dim),
            nn.GELU(),
            nn.LayerNorm(comorb_embed_dim),
        )

        # Static clinical summary anchor projection to preserve exact clinical scale relationships
        self.static_anchor = nn.Sequential(
            nn.Linear(static_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        concat_dim = num_temporal_vars * lstm_dim + demo_embed_dim + comorb_embed_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(concat_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_demo: torch.Tensor,
        x_comorb: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_temporal: (B, M, T)
            x_demo: (B, 2) -> [sex_male, age_group]
            x_comorb: (B, 7)
            x_static: (B, F_static)
        Returns:
            H: (B, latent_dim)
        """
        bsz, num_m, seq_len = x_temporal.shape
        # Encode each time series X_{t_i} with LSTM (Eq. 1)
        flat_ts = x_temporal.reshape(bsz * num_m, seq_len, 1)
        var_ids = torch.arange(num_m, device=x_temporal.device).unsqueeze(0).expand(bsz, num_m).reshape(bsz * num_m, 1)
        ts_in = self.var_input_proj(flat_ts) + self.var_id_embed(var_ids)
        _, (h_n, _) = self.temporal_lstm(ts_in)
        e_t_vars = h_n[-1].reshape(bsz, num_m, self.lstm_dim)  # (B, M, lstm_dim)

        # Interleaved attention across the M temporal variable embeddings
        attn_out, _ = self.interleaved_attn(e_t_vars, e_t_vars, e_t_vars)
        e_t_vars = self.temporal_norm(e_t_vars + attn_out)
        e_t = e_t_vars.reshape(bsz, num_m * self.lstm_dim)

        # Demographics embedding E_D
        sex_ids = x_demo[:, 0].clamp(0, 1)
        age_ids = x_demo[:, 1].clamp(0, 4)
        e_d = self.demo_proj(torch.cat([self.sex_embed(sex_ids), self.age_embed(age_ids)], dim=-1))

        # Comorbidity embedding E_C
        if self.use_comorbidity:
            e_c = self.comorb_proj(x_comorb)
        else:
            e_c = torch.zeros(bsz, self.comorb_proj[0].out_features, device=x_temporal.device, dtype=x_temporal.dtype)
            x_static = x_static.clone()
            x_static[:, 2:9] = 0.0

        # Eq. 2: H = E_T || E_D || E_C projected to shared 512-dim latent space
        h_concat = torch.cat([e_t, e_d, e_c], dim=-1)
        h = self.fusion_proj(h_concat) + 0.5 * self.static_anchor(x_static)
        return h
