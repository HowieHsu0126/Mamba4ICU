# GBMN: Subphenotype Identification for Sepsis-Associated Acute Kidney Injury using Graph Bidirectional Mamba Networks

[![IEEE JBHI](https://img.shields.io/badge/IEEE%20JBHI-11359427-00629B?style=flat-square&logo=ieee)](https://ieeexplore.ieee.org/abstract/document/11359427/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

Official PyTorch implementation for the **IEEE Journal of Biomedical and Health Informatics (JBHI)** paper:

> **Subphenotype Identification for Sepsis-Associated Acute Kidney Injury using Graph Bidirectional Mamba Networks**  
> *Haowei Xu, Wentie Liu, Tongyue Shi, and Guilan Kong\**  
> **Published in:** *IEEE Journal of Biomedical and Health Informatics (JBHI)*  
> **Paper Link:** [https://ieeexplore.ieee.org/abstract/document/11359427/](https://ieeexplore.ieee.org/abstract/document/11359427/)

---

## Overview

Sepsis-associated acute kidney injury (SA-AKI) is a highly heterogeneous clinical syndrome and a leading cause of mortality in intensive care units (ICUs). Identifying clinically distinct subphenotypes of SA-AKI from electronic health records (EHRs) enables targeted clinical interventions and precision medicine.

<p align="center">
  <img src="output/figures/WorkFlow.png" width="58%" alt="SA-AKI Subphenotyping Workflow"/>
</p>

To address the challenges of longitudinal irregularity, latent inter-patient correlations, and high-dimensional sparsity in multi-modal EHRs, **GBMN** introduces a unified four-stage architecture:

<p align="center">
  <img src="output/figures/Overall.png" width="95%" alt="GBMN Framework Overview"/>
</p>

1. **Multi-Modal Information Fusion (`libs/models/multimodal_fusion.py`)**: Integrates variable-specific LSTM temporal encoders, interleaved attention across 48-hour physiological trajectories (vital signs and laboratory tests), age-stratified demographic embeddings, and ICD-9/10 comorbidity representations into a shared latent space:
   $$\mathbf{H} = \mathbf{E}_T \;\|\; \mathbf{E}_D \;\|\; \mathbf{E}_C$$
2. **Adaptive Latent Graph Inference (`libs/models/latent_graph.py`)**: Learns a non-negative, sparse, symmetric patient similarity graph $\mathbf{A} = \operatorname{ReLU}(\cos(\mathbf{W}_{\hat{\mathbf{A}}}^\top \mathbf{H}, \mathbf{W}_{\hat{\mathbf{A}}}^\top \mathbf{H}))$ jointly optimized with Dirichlet smoothness ($\mathcal{L}_{\text{smooth}}$), log-degree connectivity ($\mathcal{L}_{\text{con}}$), and Frobenius sparsity ($\mathcal{L}_{\text{reg}}$) regularizations:
   $$\mathcal{L}_g = \mathcal{L}_{\text{smooth}} + \omega_1 \mathcal{L}_{\text{con}} + \omega_2 \mathcal{L}_{\text{reg}}$$
3. **Graph Bidirectional Mamba Networks (`libs/models/graph_mamba.py`)**: Combines node positional encodings ($\mathbf{H} \| \mathbf{P}$), $m$-hop random-walk subgraph tokenization ($G[R_0(v)], \dots, G[R_m(v)]$), local GNN neighborhood encoding (`GAT` / `GCN` / `GIN`), reverse-hop token ordering, and a bidirectional selective state-space model (BiMamba) encoder.
4. **Differentiable Patient Subphenotyping (`libs/models/clustering.py`)**: Optimizes soft cluster assignments $\mathbf{Z} = \operatorname{softmax}(\boldsymbol{y}_{\text{output}})$ via spectral modularity maximization with collapse regularization:
   $$\mathcal{L}_c(\mathbf{Z}; \mathbf{A}) = -\frac{1}{2U} \operatorname{Tr}\left(\mathbf{Z}^\top \mathbf{M} \mathbf{Z}\right) + \frac{\sqrt{k}}{N}\left\|\sum_{i=1}^k \mathbf{Z}_i^\top\right\|_F - 1, \quad \mathcal{L} = \mathcal{L}_c(\mathbf{Z}; \mathbf{A}) + \lambda \mathcal{L}_g(\mathbf{H}; \mathbf{A})$$

---

## Repository Structure

```text
GBMN/
├── libs/
│   ├── configs/
│   │   └── default.yaml                      # Default model, loss, and cohort hyperparameters
│   ├── data/
│   │   ├── extract_mimic_cohort.py           # MIMIC-IV v2.1 SQL extraction for SA-AKI cohort & 48h series
│   │   └── preprocess.py                     # Inclusion/exclusion cascade, SAPS II, KNN imputation & scaling
│   ├── models/
│   │   ├── multimodal_fusion.py              # LSTM temporal encoder + demographic & comorbidity fusion
│   │   ├── latent_graph.py                   # Adaptive Latent Graph Inference (LGI) & graph regularization
│   │   ├── graph_mamba.py                    # Subgraph tokenization, GNN local encoder & Bidirectional Mamba
│   │   ├── clustering.py                     # Differentiable spectral modularity maximization clustering
│   │   ├── gbmn.py                           # End-to-end GBMN model with ablation & sensitivity switches
│   │   └── baselines.py                      # Baselines (K-Means, LSTM, Transformer, 1D-CNN, GCN, GAT, GIN, NAGphormer, GPS)
│   ├── exp/
│   │   ├── metrics.py                        # DBI, CHI, Silhouette, Cramér's V, and Eta-squared statistical tests
│   │   ├── trainer.py                        # 10-fold cross-validation training & evaluation engine
│   │   ├── run_rq1_effectiveness.py          # RQ1: Number of subphenotypes (k=2..6) & 10-method comparison
│   │   ├── run_rq2_ablation.py               # RQ2: Module ablation study (Comorbidity, LGI, GNNs, BiMamba)
│   │   ├── run_rq3_clinical.py               # RQ3: Clinical outcomes, Table IV stats, t-SNE Voronoi & MLP+SHAP
│   │   ├── run_rq4_sensitivity.py            # RQ4: Hyperparameter sensitivity analysis
│   │   ├── run_tuning_experiment.py          # Multi-stage optimization, hyperparameter tuning & scalability study
│   │   ├── plot_figures.py                   # Publication-grade figure rendering (PDF & 300-DPI PNG)
│   │   └── run_all.py                        # Master pipeline entry point
│   └── tests/
│       └── test_pipeline.py                  # Unit and integration test suite
├── output/
│   ├── figures/                              # Generated figures (Fig. 3 to Fig. 9)
│   ├── results/                              # JSON evaluation summaries
│   └── tables/                               # Generated CSV tables (Tables I to VII)
├── requirements.txt
└── LICENSE
```

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/HowieHsu0126/GBMN.git
   cd GBMN
   ```

2. Create a Python 3.10+ environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Preparation (MIMIC-IV)

The study cohort is constructed from the [MIMIC-IV database](https://physionet.org/content/mimiciv/) following Sepsis-3 and KDIGO AKI criteria ($N = 16,361$ SA-AKI patients):

<p align="center">
  <img src="output/figures/Fig3_Flow_MIMIC.png" width="78%" alt="MIMIC-IV Cohort Inclusion and Exclusion Flowchart"/>
</p>

1. Configure your PostgreSQL connection to MIMIC-IV and extract the raw SA-AKI candidate cohort and 48-hour hourly trajectories:
   ```bash
   export MIMIC_PG_HOST="localhost"
   export MIMIC_PG_PORT="5432"
   export MIMIC_PG_DB="mimiciv"
   export MIMIC_PG_USER="postgres"
   python -m libs.data.extract_mimic_cohort
   ```

2. Run cohort filtering, SAPS II calculation, 5-NN missing value imputation, and MinMax normalization:
   ```bash
   python -m libs.data.preprocess
   ```
   This generates the processed tensors in `input/processed/` and the study population summary table in [`output/tables/table1_cohort_characteristics.csv`](output/tables/table1_cohort_characteristics.csv).

---

## Running Experiments

To run the complete experimental suite (RQ1–RQ4) and regenerate all tables and figures:

```bash
python -m libs.exp.run_all
```

Or run individual research question modules:

```bash
# RQ1: Subphenotype granularity (k=2..6) and 10-fold CV baseline comparison
python -m libs.exp.run_rq1_effectiveness

# RQ2: Ablation study (Vanilla vs. w/o Comorbidity, w/o LGI, w/o GNNs, w/o BiMamba)
python -m libs.exp.run_rq2_ablation

# RQ3: Clinical outcomes, 30-feature statistical profile, t-SNE Voronoi & SHAP attribution
python -m libs.exp.run_rq3_clinical

# RQ4: Hyperparameter sensitivity (GNN type, GNN layers, Mamba blocks, random walk length)
python -m libs.exp.run_rq4_sensitivity

# Multi-stage optimization progression & scalability benchmark
python -m libs.exp.run_tuning_experiment

# Run unit tests
pytest -q libs/tests/test_pipeline.py
```

---

## Experimental Results

### 1. Optimal Number of Subphenotypes & Overall Performance (RQ1)

Across $k \in \{2, 3, 4, 5, 6\}$, **GBMN** achieves optimal cluster compactness and separation at **$k = 3$** subphenotypes (lowest Davies–Bouldin Index, highest Calinski–Harabasz Index, and highest Silhouette score), outperforming static clustering (`K-Means`), sequence models (`LSTM`, `1D-CNN`, `Transformer`), static-graph GNNs (`GCN`, `GAT`, `GIN`), and Graph Transformers (`NAGphormer`, `GPS`). Full results are available in [`output/tables/table2_performance_comparison.csv`](output/tables/table2_performance_comparison.csv) and [`output/tables/table5_gbmn_optimization_tuning.csv`](output/tables/table5_gbmn_optimization_tuning.csv).

<p align="center">
  <img src="output/figures/Fig4_NumberOfCluster.png" width="95%" alt="Performance across Number of Subphenotypes k=2..6"/>
</p>

<p align="center">
  <img src="output/figures/Fig9_GBMN_Optimization_Tuning.png" width="95%" alt="GBMN Optimization and Tuning Progression"/>
</p>

### 2. Clinical Significance & Interpretability (RQ3)

The three SA-AKI subphenotypes identified by GBMN exhibit distinct clinical trajectories and statistically significant differences ($P < 0.001$) across ICU mortality, dialysis requirement, and ICU length of stay (see [`output/tables/table4_subphenotype_characteristics.csv`](output/tables/table4_subphenotype_characteristics.csv) and [`output/tables/rq3_subphenotype_outcomes.csv`](output/tables/rq3_subphenotype_outcomes.csv)):

- **Subphenotype 1 ($n = 5,890, 36.0\%$):** Characterized by older age and higher prevalence of chronic cardiovascular comorbidities (hypertension, congestive heart failure), lowest acute illness severity (SAPS II, BUN, and lactate), preserved urine output, and the most favorable clinical prognosis.
- **Subphenotype 2 ($n = 7,690, 47.0\%$):** Intermediate age and acute severity with higher prevalence of chronic kidney disease (CKD), diabetes mellitus, and liver disease, exhibiting intermediate dialysis and mortality rates.
- **Subphenotype 3 ($n = 2,781, 17.0\%$):** Severe hyper-acute septic-renal-metabolic dysfunction (markedly elevated lactate, BUN, creatinine, SAPS II, phosphate, anion gap, and PTT with profound oliguria), associated with the highest dialysis requirement, highest ICU mortality, and longest length of stay.

<p align="center">
  <img src="output/figures/Fig5_Outcomes.png" width="48%" alt="Clinical Outcomes Across Subphenotypes"/>
  <img src="output/figures/Fig6_tSNE_Voronoi.png" width="45%" alt="t-SNE Projection with Voronoi Tessellation"/>
</p>

Using a two-layer MLP surrogate classifier trained on the GBMN subphenotypes followed by SHAP attribution ([`output/tables/rq3_shap_importance.csv`](output/tables/rq3_shap_importance.csv)), the top 10 most influential clinical features are **Age, Blood Urea Nitrogen, Serum Creatinine, SAPS II Score, Hemoglobin, Urine Output, Lactate, Oxygen Saturation, Heart Rate, and White Blood Cell count**:

<p align="center">
  <img src="output/figures/Fig7_SHAP.png" width="68%" alt="Top 10 SHAP Feature Importance"/>
</p>

### 3. Hyperparameter Sensitivity Analysis (RQ4)

Sensitivity analysis across GNN architectures (`GCN`, `GAT`, `GIN`), GNN depths ($2, 3, 4$), Mamba block counts ($1, 2, 3$), and random walk lengths ($2, 3, 4, 5$) confirms that **3-layer GAT + 2 Bidirectional Mamba blocks + 2-hop random walk tokenization** achieves the best balance between local neighborhood representation and global context without over-smoothing ([`output/tables/rq4_parameter_sensitivity.csv`](output/tables/rq4_parameter_sensitivity.csv)):

<p align="center">
  <img src="output/figures/Fig8_ParameterSensitivity.png" width="88%" alt="Hyperparameter Sensitivity Analysis"/>
</p>

---

## Citation

If you find this repository or our paper useful in your research, please cite:

```bibtex
@article{xu2026subphenotype,
  author  = {Xu, Haowei and Liu, Wentie and Shi, Tongyue and Kong, Guilan},
  title   = {Subphenotype Identification for Sepsis-Associated Acute Kidney Injury using Graph Bidirectional Mamba Networks},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  url     = {https://ieeexplore.ieee.org/abstract/document/11359427/}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
