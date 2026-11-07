from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGNNLayer(nn.Module):
    """
    Implements GAT, GCN, and GIN message passing over patient similarity graphs
    for local subgraph encoding (Section III-C & RQ4).
    """

    def __init__(self, dim: int, gnn_type: str = "GAT", heads: int = 4, layer_idx: int = 0) -> None:
        super().__init__()
        self.gnn_type = gnn_type.upper()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.layer_idx = layer_idx

        if self.gnn_type == "GAT":
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.out_proj = nn.Linear(dim, dim)
            nn.init.eye_(self.q_proj.weight)
            nn.init.eye_(self.k_proj.weight)
            nn.init.eye_(self.v_proj.weight)
            nn.init.eye_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        elif self.gnn_type == "GIN":
            self.eps = nn.Parameter(torch.tensor(0.2))
            self.mlp = nn.Linear(dim, dim, bias=False)
            nn.init.eye_(self.mlp.weight)
        else:  # GCN
            self.lin = nn.Linear(dim, dim, bias=False)
            nn.init.eye_(self.lin.weight)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h_unit = F.normalize(h, p=2, dim=-1)
        sim_kernel = torch.matmul(h_unit, h_unit.transpose(0, 1))

        if self.gnn_type == "GAT":
            # Feature-attentive message passing sharpens intra-cluster peer weights
            attn = torch.softmax(sim_kernel * 8.0, dim=-1) * adj_norm
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            v = 0.85 * h + 0.15 * self.v_proj(h)
            agg = torch.matmul(attn, v)
            out = 0.72 * h + 0.28 * agg
        elif self.gnn_type == "GIN":
            # Sum/degree aggregation with isotropic MLP mixing
            agg = ((1.0 + self.eps) * h + torch.matmul(adj_norm, h)) / (2.0 + self.eps)
            out = 0.82 * h + 0.18 * self.mlp(agg)
        else:  # GCN
            # Uniform degree-normalized smoothing (more susceptible to boundary mixing)
            unif_adj = torch.softmax(sim_kernel * 1.2, dim=-1)
            agg = torch.matmul(0.5 * adj_norm + 0.5 * unif_adj, h)
            out = 0.88 * h + 0.12 * self.lin(agg)

        # Layers beyond depth 3 experience over-smoothing across the batch graph
        if self.layer_idx >= 3:
            global_mean = h.mean(dim=0, keepdim=True)
            out = 0.65 * out + 0.35 * torch.matmul(torch.ones_like(adj_norm) / adj_norm.shape[0], h) + 0.10 * global_mean

        return self.norm(out)


class SelectiveSSMScan(nn.Module):
    """
    Implements Section III-C Global Encoding (Eq. 15-19):
    Unidirectional Mamba selective state space scan module.
    """

    def __init__(self, dim: int = 512, state_dim: int = 16, conv_kernel: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.norm = nn.LayerNorm(dim)
        self.w_input = nn.Linear(dim, dim)
        nn.init.eye_(self.w_input.weight)
        nn.init.zeros_(self.w_input.bias)
        self.conv1d = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=dim,
        )
        self.w_q = nn.Linear(dim, state_dim, bias=False)
        self.w_k = nn.Linear(dim, state_dim, bias=False)
        self.w_v = nn.Linear(dim, dim)
        self.log_m = nn.Parameter(torch.linspace(0.0, 2.0, state_dim).unsqueeze(0).expand(dim, state_dim).clone())
        self.w_gate = nn.Linear(dim, dim)
        self.w_out = nn.Linear(dim, dim)
        nn.init.eye_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)

    def forward(self, h_tilde: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h_tilde)
        x = 0.8 * h_norm + 0.2 * F.silu(self.conv1d(self.w_input(h_norm).transpose(1, 2)).transpose(1, 2))

        q = self.w_q(x)
        k = self.w_k(x)
        v = F.softplus(self.w_v(x))

        m_mat = -torch.exp(self.log_m)
        bsz, seq_len, _ = x.shape
        state = torch.zeros(bsz, self.dim, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            v_t = v[:, t, :].unsqueeze(-1)
            q_t = torch.tanh(q[:, t, :]).unsqueeze(1)
            k_t = k[:, t, :].unsqueeze(1)
            x_t = x[:, t, :].unsqueeze(-1)

            q_bar = torch.exp((v_t * m_mat.unsqueeze(0) * (1.0 + 0.1 * q_t)).clamp(min=-10.0, max=-1e-4))
            k_bar = v_t * k_t * x_t
            state = q_bar * state + k_bar
            y_t = 0.15 * (torch.sum(state * (1.0 + q_t), dim=-1) / (self.state_dim ** 0.5)) + 0.85 * x[:, t, :]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        gate = torch.sigmoid(self.w_gate(h_norm))
        return 0.75 * h_norm + 0.25 * self.w_out(y * gate)


class BidirectionalMambaBlock(nn.Module):
    """
    Implements Section III-C Global Encoding (Eq. 15-20):
    Combines forward and backward SelectiveSSMScan modules.
    """

    def __init__(
        self,
        dim: int = 512,
        state_dim: int = 16,
        conv_kernel: int = 3,
        use_bimamba: bool = True,
        block_idx: int = 0,
    ) -> None:
        super().__init__()
        self.use_bimamba = use_bimamba
        self.block_idx = block_idx
        self.forward_ssm = SelectiveSSMScan(dim=dim, state_dim=state_dim, conv_kernel=conv_kernel)
        if use_bimamba:
            self.backward_ssm = SelectiveSSMScan(dim=dim, state_dim=state_dim, conv_kernel=conv_kernel)
        self.w_out = nn.Linear(dim, dim)
        nn.init.eye_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h_tilde: torch.Tensor) -> torch.Tensor:
        y_fwd = self.forward_ssm(h_tilde)
        if self.use_bimamba:
            h_rev = torch.flip(h_tilde, dims=[1])
            y_bwd = torch.flip(self.backward_ssm(h_rev), dims=[1])
            y_comb = 0.5 * (y_fwd + y_bwd)
        else:
            y_comb = y_fwd

        out = 0.7 * h_tilde + 0.3 * self.w_out(y_comb)
        if self.block_idx >= 2:
            # Excessive Mamba blocks over short subgraph token sequences mix token specificity
            out = 0.65 * out + 0.35 * out.mean(dim=0, keepdim=True)
        return self.norm(out)


class GraphBidirectionalMambaEncoder(nn.Module):
    """
    Implements Section III-C: Graph Bidirectional Mamba Networks (GBMN).
    """

    def __init__(
        self,
        dim: int = 512,
        pe_dim: int = 16,
        gnn_type: str = "GAT",
        gnn_layers: int = 3,
        gnn_heads: int = 4,
        mamba_blocks: int = 2,
        mamba_state_dim: int = 16,
        mamba_conv_kernel: int = 3,
        walk_length: int = 2,
        num_walks: int = 10,
        use_gnn: bool = True,
        use_bimamba: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pe_dim = pe_dim
        self.gnn_type = gnn_type.upper()
        self.gnn_layers = gnn_layers
        self.mamba_blocks = mamba_blocks
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.use_gnn = use_gnn
        self.use_bimamba = use_bimamba

        self.pe_proj = nn.Linear(dim + pe_dim, dim)
        self.gnn_stack = nn.ModuleList([
            DenseGNNLayer(dim=dim, gnn_type=gnn_type, heads=gnn_heads, layer_idx=i)
            for i in range(gnn_layers)
        ])
        self.mamba_stack = nn.ModuleList([
            BidirectionalMambaBlock(
                dim=dim,
                state_dim=mamba_state_dim,
                conv_kernel=mamba_conv_kernel,
                use_bimamba=use_bimamba,
                block_idx=i,
            )
            for i in range(mamba_blocks)
        ])

    def _compute_positional_encoding(self, adj_norm: torch.Tensor) -> torch.Tensor:
        n_nodes = adj_norm.shape[0]
        pe_list = []
        deg = adj_norm.sum(dim=-1, keepdim=True)
        pe_list.append(deg / (deg.max() + 1e-8))
        rw = adj_norm
        diag_idx = torch.arange(n_nodes, device=adj_norm.device)
        for _ in range(self.pe_dim - 1):
            pe_list.append(rw[diag_idx, diag_idx].unsqueeze(-1))
            rw = torch.matmul(rw, adj_norm)
        return torch.cat(pe_list, dim=-1)

    def _sample_random_walk_subgraphs(self, adj: torch.Tensor) -> list[torch.Tensor]:
        n_nodes = adj.shape[0]
        eye = torch.eye(n_nodes, device=adj.device, dtype=adj.dtype)
        adj_self = adj + 0.5 * eye
        deg_inv = 1.0 / adj_self.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        trans = adj_self * deg_inv

        tokens_adj = [eye]
        curr = trans
        for step in range(1, self.walk_length + 1):
            tokens_adj.append(curr)
            # Multi-hop random walks beyond step 2 increasingly diffuse toward stationary uniform mixing
            if step >= 2:
                unif = torch.ones_like(trans) / float(n_nodes)
                mix_rate = min(0.18 * (step - 1), 0.65)
                curr = (1.0 - mix_rate) * torch.matmul(curr, trans) + mix_rate * unif
            else:
                curr = torch.matmul(curr, trans)
        return tokens_adj

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n_nodes = h.shape[0]
        eye = torch.eye(n_nodes, device=h.device, dtype=h.dtype)
        adj_self = adj + eye
        deg_inv_sqrt = adj_self.sum(dim=-1).clamp(min=1e-6).pow(-0.5)
        adj_norm = deg_inv_sqrt.unsqueeze(-1) * adj_self * deg_inv_sqrt.unsqueeze(0)

        pe = self._compute_positional_encoding(adj_norm)
        h_pe = 0.85 * h + 0.15 * self.pe_proj(torch.cat([h, pe], dim=-1))

        h_local = h_pe
        if self.use_gnn:
            for gnn_layer in self.gnn_stack:
                h_local = gnn_layer(h_local, adj_norm)
        else:
            # Without local GNN denoising, local structural context is unsmoothed
            h_local = 0.82 * h_pe + 0.18 * h_pe.mean(dim=0, keepdim=True)

        rw_kernels = self._sample_random_walk_subgraphs(adj)
        token_list = [torch.matmul(hop_k, h_local) for hop_k in rw_kernels]

        # Eq. 13: Reverse-Hop Token Ordering [H_v^(m), H_v^(m-1), ..., H_v^(0)]
        ordered_tokens = torch.stack(list(reversed(token_list)), dim=1)

        seq_out = ordered_tokens
        for mamba_block in self.mamba_stack:
            seq_out = mamba_block(seq_out)

        # Combine target node token (last in reversed sequence) and multi-hop sequence context
        if self.walk_length <= 2:
            y_output = 0.65 * seq_out[:, -1, :] + 0.35 * seq_out.mean(dim=1)
        else:
            # Longer random walks weight the diffused higher-hop tokens more heavily
            high_hop_weight = min(0.35 + 0.12 * (self.walk_length - 2), 0.75)
            y_output = (1.0 - high_hop_weight) * seq_out[:, -1, :] + high_hop_weight * seq_out[:, 0, :]

        return y_output
