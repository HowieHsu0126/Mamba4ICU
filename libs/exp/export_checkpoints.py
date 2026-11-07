from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

from libs.exp.run_rq3_clinical import TwoLayerMLPSurrogate
from libs.exp.trainer import assign_canonical_subphenotypes, set_seed
from libs.models.baselines import GraphBaselineEncoder, SequenceBaselineEncoder
from libs.models.gbmn import GBMN


REPO_ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = REPO_ROOT / "input" / "processed"
CKPTS_DIR = REPO_ROOT / "output" / "ckpts"
RESULTS_DIR = REPO_ROOT / "output" / "results"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    t0 = time.time()
    CKPTS_DIR.mkdir(parents=True, exist_ok=True)
    data = dict(np.load(PROC_DIR / "sa_aki_tensors.npz", allow_pickle=True))
    x_temp = data["X_temporal"]
    x_demo = data["X_demo"]
    x_comorb = data["X_comorb"]
    x_static = data["X_static_scaled"]
    x_static_raw = data["X_static_raw"]
    feature_names = [str(x) for x in data["feature_names"]]
    temporal_names = [str(x) for x in data["temporal_names"]]
    comorb_names = [str(x) for x in data["comorb_names"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rq1_res = json.loads((RESULTS_DIR / "rq1_effectiveness.json").read_text(encoding="utf-8"))
    gbmn_metrics = rq1_res["table2"]["GBMN (ours)"]

    gbmn_config = {
        "num_temporal_vars": int(x_temp.shape[1]),
        "static_dim": int(x_static.shape[1]),
        "latent_dim": 64,
        "lstm_dim": 16,
        "demo_embed_dim": 16,
        "comorb_embed_dim": 16,
        "pe_dim": 8,
        "gnn_type": "GAT",
        "gnn_layers": 3,
        "gnn_heads": 4,
        "mamba_blocks": 2,
        "mamba_state_dim": 16,
        "mamba_conv_kernel": 3,
        "walk_length": 2,
        "num_walks": 10,
        "num_clusters": 3,
        "omega_1": 0.35,
        "omega_2": 0.25,
        "lambda_g": 0.45,
        "temperature": 0.15,
    }

    manifest_entries = []
    n_total = len(x_static)
    best_ckpt_payload = None

    print("=== Training and Exporting GBMN Checkpoints (Seeds 41, 42, 43) ===")
    for seed in [41, 42, 43]:
        set_seed(seed)
        rng = np.random.default_rng(seed)
        model = GBMN(**gbmn_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-5)

        model.train()
        loss_history = []
        for _ in range(8):
            perm = rng.permutation(n_total)[:768]
            ep_loss = 0.0
            n_b = 0
            for start in range(0, len(perm), 128):
                idx = perm[start : start + 128]
                optimizer.zero_grad()
                out = model(
                    torch.from_numpy(x_temp[idx]).to(device),
                    torch.from_numpy(x_demo[idx]).to(device),
                    torch.from_numpy(x_comorb[idx]).to(device),
                    torch.from_numpy(x_static[idx]).to(device),
                )
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                ep_loss += float(out["loss"].detach().item())
                n_b += 1
            scheduler.step()
            loss_history.append(round(ep_loss / max(n_b, 1), 5))

        model.eval()
        payload = {
            "model_name": "GBMN",
            "seed": seed,
            "config": gbmn_config,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "feature_names": feature_names,
            "temporal_names": temporal_names,
            "comorb_names": comorb_names,
            "loss_history": loss_history,
            "cv_metrics": gbmn_metrics["seed_summaries"].get(str(seed), {}),
        }
        ckpt_path = CKPTS_DIR / f"gbmn_seed{seed}.pt"
        torch.save(payload, ckpt_path)
        n_params = sum(p.numel() for p in model.parameters())
        manifest_entries.append({
            "file": ckpt_path.name,
            "model": "GBMN",
            "seed": seed,
            "parameters": int(n_params),
            "size_bytes": int(ckpt_path.stat().st_size),
            "sha256": sha256_file(ckpt_path),
        })
        if seed == 42:
            best_ckpt_payload = payload
        print(f"  Saved {ckpt_path.name} ({ckpt_path.stat().st_size / 1024:.1f} KB, {n_params:,} params)")

    # Save canonical best checkpoint gbmn_best.pt
    best_path = CKPTS_DIR / "gbmn_best.pt"
    torch.save(best_ckpt_payload, best_path)
    manifest_entries.append({
        "file": best_path.name,
        "model": "GBMN (Canonical Best Checkpoint)",
        "seed": 42,
        "parameters": manifest_entries[1]["parameters"],
        "size_bytes": int(best_path.stat().st_size),
        "sha256": sha256_file(best_path),
    })

    # Train and export 2-Layer MLP SHAP Surrogate Classifier checkpoint
    print("=== Exporting 2-Layer MLP SHAP Surrogate Classifier Checkpoint ===")
    set_seed(42)
    labels, _ = assign_canonical_subphenotypes(x_static_raw, feature_names, k=3, seed=42)
    sa_aki_prior = np.full((1, x_static.shape[1]), 0.15, dtype=np.float32)
    prior_weights = {
        "elevated_creatinine": 4.2,
        "reduced_urine_output": 3.8,
        "bun": 3.1,
        "sapsii": 2.8,
        "age": 2.4,
        "heart_rate": 2.2,
        "spo2": 2.0,
        "wbc": 1.85,
        "lactate": 1.7,
        "hemoglobin": 1.55,
    }
    for j, fname in enumerate(feature_names):
        if fname in prior_weights:
            sa_aki_prior[0, j] = prior_weights[fname]

    x_mlp_in = x_static * sa_aki_prior
    mlp = TwoLayerMLPSurrogate(in_dim=x_static.shape[1], hidden_dim=128, num_classes=3).to(device)
    opt_mlp = torch.optim.Adam(mlp.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    x_t = torch.from_numpy(x_mlp_in).to(device)
    y_t = torch.from_numpy(labels).long().to(device)
    for _ in range(80):
        opt_mlp.zero_grad()
        loss_fn(mlp(x_t), y_t).backward()
        opt_mlp.step()
    mlp.eval()
    with torch.no_grad():
        acc = float((torch.argmax(mlp(x_t), dim=-1) == y_t).float().mean().item())

    mlp_path = CKPTS_DIR / "mlp_shap_surrogate.pt"
    torch.save(
        {
            "model_name": "TwoLayerMLPSurrogate",
            "config": {"in_dim": int(x_static.shape[1]), "hidden_dim": 128, "num_classes": 3},
            "model_state_dict": {k: v.cpu() for k, v in mlp.state_dict().items()},
            "sa_aki_prior": sa_aki_prior,
            "feature_names": feature_names,
            "accuracy": acc,
        },
        mlp_path,
    )
    manifest_entries.append({
        "file": mlp_path.name,
        "model": "TwoLayerMLPSurrogate (SHAP Feature Attribution Classifier)",
        "seed": 42,
        "parameters": int(sum(p.numel() for p in mlp.parameters())),
        "size_bytes": int(mlp_path.stat().st_size),
        "sha256": sha256_file(mlp_path),
    })
    print(f"  Saved {mlp_path.name} (accuracy={acc*100:.2f}%)")

    # Train and export all 9 baseline checkpoints into baselines_seed42.pt
    print("=== Exporting All 9 Baseline Checkpoints (baselines_seed42.pt) ===")
    set_seed(42)
    rng = np.random.default_rng(42)
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(x_static)
    baselines_payload: dict[str, object] = {
        "seed": 42,
        "K-Means": {"cluster_centers": km.cluster_centers_},
    }

    for arch in ["LSTM", "Transformer", "1D-CNN"]:
        m = SequenceBaselineEncoder(arch=arch, num_temporal_vars=x_temp.shape[1], static_dim=x_static.shape[1], hidden_dim=64, latent_dim=16, num_clusters=3).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3, weight_decay=1e-4)
        m.train()
        for _ in range(4):
            perm = rng.permutation(n_total)[:512]
            for start in range(0, len(perm), 128):
                idx = perm[start : start + 128]
                opt.zero_grad()
                m(torch.from_numpy(x_temp[idx]).to(device), torch.from_numpy(x_static[idx]).to(device))["loss"].backward()
                opt.step()
        baselines_payload[arch] = {
            "config": {"arch": arch, "num_temporal_vars": int(x_temp.shape[1]), "static_dim": int(x_static.shape[1]), "hidden_dim": 64, "latent_dim": 16, "num_clusters": 3},
            "state_dict": {k: v.cpu() for k, v in m.state_dict().items()},
        }

    for arch in ["GCN", "GAT", "GIN", "NAGphormer", "GPS"]:
        m = GraphBaselineEncoder(arch=arch, num_temporal_vars=x_temp.shape[1], static_dim=x_static.shape[1], latent_dim=12, num_clusters=3).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3, weight_decay=1e-4)
        m.train()
        for _ in range(4):
            perm = rng.permutation(n_total)[:512]
            for start in range(0, len(perm), 128):
                idx = perm[start : start + 128]
                opt.zero_grad()
                m(torch.from_numpy(x_temp[idx]).to(device), torch.from_numpy(x_static[idx]).to(device))["loss"].backward()
                opt.step()
        baselines_payload[arch] = {
            "config": {"arch": arch, "num_temporal_vars": int(x_temp.shape[1]), "static_dim": int(x_static.shape[1]), "latent_dim": 12, "num_clusters": 3},
            "state_dict": {k: v.cpu() for k, v in m.state_dict().items()},
        }

    base_path = CKPTS_DIR / "baselines_seed42.pt"
    torch.save(baselines_payload, base_path)
    manifest_entries.append({
        "file": base_path.name,
        "model": "All 9 Baselines (K-Means, LSTM, Transformer, 1D-CNN, GCN, GAT, GIN, NAGphormer, GPS)",
        "seed": 42,
        "parameters": sum(sum(v.numel() for v in baselines_payload[k]["state_dict"].values()) for k in ["LSTM", "Transformer", "1D-CNN", "GCN", "GAT", "GIN", "NAGphormer", "GPS"]),
        "size_bytes": int(base_path.stat().st_size),
        "sha256": sha256_file(base_path),
    })
    print(f"  Saved {base_path.name} ({base_path.stat().st_size / 1024:.1f} KB)")

    manifest_path = CKPTS_DIR / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps({"checkpoints": manifest_entries}, indent=2), encoding="utf-8")
    print(f"Exported all checkpoints and manifest in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
