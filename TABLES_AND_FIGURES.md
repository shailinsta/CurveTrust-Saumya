# Tables and Figures extracted from the manuscript

Source: `docs/CurvTrust_Computer_Networks_revised.docx`

## Tables

### Table 1. Summary of representative related work and the gaps addressed by CurvTrust-PAI.

*Files:* `results/tables/table01_related_work.md` · `results/tables/table01_related_work.csv`

| Strand | Representative focus | Gap relative to CurvTrust-PAI (proposed) |
| --- | --- | --- |
| Identity / zero-trust [5–7] | Continuous authentication, least-privilege access, and assurance patterns for CPS | Post-admission behavioural reliability of authenticated agents is left to complementary evidence. |
| Flow-based IDS [8–21] | Ensemble and deep classifiers on tabular flow features, including trust-aware federated intrusion detection and resource-aware federated detection | Per-record neighbourhood overlap against a clean reference is not used as an added feature; residual suppression is not part of the reviewed evaluations. |
| Graph-based IDS [22–31] | GNN, attention and federated graph models on host or flow graphs | Require explicit graph construction and training; not designed as a single scalar auxiliary statistic. |
| Geometric / curvature signals [3–4] | Ollivier–Ricci curvature of Markov chains and networks | Applied to interaction graphs, not to per-sample feature-space residuals; Wasserstein cost is higher than a Jaccard proxy. |
| Neighbourhood anomaly scores [44–46] | Local density (LOF), shared-nearest-neighbour overlap, and conformal outlier tests | Not compared experimentally here; the residual differs by using a clean reference set and a deviation-from-median score. |
| Multi-agent trust [7, 35, 36] | Cyber-trust, kinematic consistency, reputation, isolation evidence, and hierarchical multi-agent cybersecurity assessment | Trust evidence is rarely mapped continuously onto a physical separation margin. |
| Control barrier functions [2, 37–41] | Fixed-margin HOCBF/QP filters, low-complexity, graph and neural barriers | Margin typically a design constant; no network-layer residual as input. |
| Detection–control coupling [36–41] | Trust-modulated barriers and insider-attack detection | Residual not shared across layers; benefit over a fixed margin remains unresolved here. |

### Table 2. Datasets, sampling caps and label construction.

*Files:* `results/tables/table02_datasets.md` · `results/tables/table02_datasets.csv`

| Dataset | Source files | Sample cap | Label and feature handling |
| --- | --- | --- | --- |
| UNSW-NB15 | Official training and testing partitions, concatenated | 40,000 | Binary label; id and attack category removed |
| NF-ToN-IoT-v2 | One NetFlow Parquet file | 30,000 | IP addresses and attack-type fields removed |
| NF-CSE-CIC-IDS2018-v2 | One NetFlow Parquet file | 30,000 | IP addresses and attack-type fields removed |
| CIDDS-001 | External-server and OpenStack captures, equal share from each | 30,000 | Class “normal” mapped to 0; “attacker” and “victim” mapped to 1; byte and packet counts with K/M/G suffixes converted to numbers |
| Custom-EdgeIoT | CSV captures of benign traffic and of DDoS, ransomware and SQL-injection attacks | 30,000 | Label column used where present, otherwise the capture file name; text, payload and address fields excluded; stratified subsample |

### Table 3. Detector configurations.

*Files:* `results/tables/table03_detector_configs.md` · `results/tables/table03_detector_configs.csv`

| Method | Input | Settings |
| --- | --- | --- |
| CurvTrust-PAI (full) | features + r(x) | Random Forest, 120 trees, depth 14, minimum leaf size 4, balanced subsample weights |
| RF features-only | features | Same forest settings |
| Logistic regression (LR) | features | Balanced class weights, 500 iterations |
| Gradient boosting (GB) | features | 80 trees, depth 4 |
| LR + residual | features + r(x) | As LR |
| GB + residual | features + r(x) | As GB |
| Residual conformal | r(x) only | Threshold τα, α = 0.05 |

### Table 4. Swarm parameters.

*Files:* `results/tables/table04_swarm_params.md` · `results/tables/table04_swarm_params.csv`

| Parameter | Value |
| --- | --- |
| Agents / steps / time step | 10 / 500 / 0.1 s |
| Compromised agents | 2 (fraction 0.20) |
| Attack window | Steps 150–350 |
| Minimum separation dmin | 1.5 m |
| Control limit | 2 m s−2 |
| Margins (fixed / fixed-large / adaptive) | 1.5 m / 3.0 m / 1.5–4.5 m |
| Trust weights (id, κ, r, q) | 0.25, 0.35, 0.25, 0.15 |
| Sigmoid slope β, healthy trust target | 10, 0.90 |
| Trust smoothing factor | 0.5 |

### Table 5. F1 on the four public benchmarks (mean ± std over five seeds).

*Files:* `results/tables/table05_f1_public.md` · `results/tables/table05_f1_public.csv`

| Method | UNSW-NB15 | NF-ToN-IoT-v2 | NF-CSE-CIC-IDS2018-v2 | CIDDS-001 |
| --- | --- | --- | --- | --- |
| CurvTrust-PAI (full) | 0.938 ± 0.001 | 0.998 ± 0.001 | 0.900 ± 0.016 | 0.998 ± 0.003 |
| RF features-only | ≈0.94 | ≈0.998 | ≈0.90 | ≈0.998 |
| Gradient boosting | ≈0.94 | ≈0.998 | ≈0.895 | ≈0.998 |
| GB + residual | ≈0.94 | ≈0.998 | ≈0.895 | ≈0.998 |
| Logistic regression | ≈0.86 | ≈0.96 | ≈0.58 | ≈0.998 |
| LR + residual | ≈0.86 | ≈0.96 | ≈0.58 | ≈0.998 |
| Residual conformal | ≈0.10 | ≈0.29 | ≈0.05 | ≈0.01 |

### Table 6. Clean detection metrics on Custom-EdgeIoT (mean ± std over five seeds).

*Files:* `results/tables/table06_edgeiot_metrics.md` · `results/tables/table06_edgeiot_metrics.csv`

| Method | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | --- | --- | --- | --- |
| CurvTrust-PAI (full) | 0.9967 ± 0.0004 | 0.9903 ± 0.0010 | 0.9996 ± 0.0009 | 0.9950 ± 0.0006 | 0.9998 ± 0.0001 |
| RF features-only | 0.9969 ± 0.0005 | 0.9907 ± 0.0013 | 0.9999 ± 0.0002 | 0.9953 ± 0.0007 | 0.9998 ± 0.0001 |
| Gradient boosting | 0.9980 ± 0.0004 | 0.9949 ± 0.0015 | 0.9990 ± 0.0008 | 0.9970 ± 0.0006 | 0.9999 ± 0.0001 |
| Logistic regression | 0.9582 ± 0.0020 | 0.9162 ± 0.0035 | 0.9584 ± 0.0051 | 0.9368 ± 0.0027 | 0.9880 ± 0.0009 |
| Residual conformal | 0.6626 ± 0.0058 | 0.3342 ± 0.0420 | 0.0454 ± 0.0130 | 0.0796 ± 0.0215 | 0.4372 ± 0.0195 |

### Table 7. Approximate F1 under Gaussian noise (read from Fig. 5).

*Files:* `results/tables/table07_noise_f1.md` · `results/tables/table07_noise_f1.csv`

| Setting | Method | ε = 0.05 | ε = 0.15 | ε = 0.25 | ε = 0.40 |
| --- | --- | --- | --- | --- | --- |
| Public (pooled) | CurvTrust-PAI | ≈0.88 | ≈0.83 | ≈0.81 | ≈0.72 |
| Public (pooled) | RF features-only | ≈0.87 | ≈0.82 | ≈0.77 | ≈0.69 |
| Public (pooled) | Logistic regression | ≈0.84 | ≈0.83 | ≈0.81 | ≈0.77 |
| Public (pooled) | Gradient boosting | ≈0.84 | ≈0.77 | ≈0.66 | ≈0.56 |
| Custom-EdgeIoT | CurvTrust-PAI (= RF) | ≈0.83 | ≈0.75 | ≈0.72 | ≈0.685 |
| Custom-EdgeIoT | Logistic regression | ≈0.91 | ≈0.79 | ≈0.71 | ≈0.64 |
| Custom-EdgeIoT | Gradient boosting | ≈0.70 | ≈0.64 | ≈0.61 | ≈0.59 |
| Custom-EdgeIoT | Residual conformal | ≈0.08 | ≈0.10 | ≈0.12 | ≈0.14 |

### Table 8. Swarm safety metrics under the overt and authenticated attacks (mean ± std over five seeds).

*Files:* `results/tables/table08_swarm_safety.md` · `results/tables/table08_swarm_safety.csv`

| Controller | Total violations | Collisions | Mean min. separation (m) | Global min. separation (m) | Goal error (m) |
| --- | --- | --- | --- | --- | --- |
| Nominal | 1860.0 ± 139.6 | 549.6 ± 30.4 | 1.572 ± 0.103 | 0.086 ± 0.040 | 1.633 ± 0.024 |
| Fixed | 377.8 ± 105.2 | 8.0 ± 7.1 | 1.555 ± 0.091 | 0.368 ± 0.108 | 4.269 ± 0.374 |
| Fixed-large | 44.6 ± 32.4 | 7.8 ± 4.6 | 2.634 ± 0.103 | 0.358 ± 0.122 | 5.007 ± 0.048 |
| Trust-adaptive, overt | 30.8 ± 27.0 | 4.0 ± 3.2 | 2.119 ± 0.144 | 0.555 ± 0.306 | 5.193 ± 0.138 |
| Trust-adaptive, authenticated | 38.0 ± 34.7 | 4.8 ± 3.7 | 2.124 ± 0.115 | 0.555 ± 0.306 | 4.903 ± 0.172 |

### Table 9. Effect of perturbations on the residual and attack recall on Custom-EdgeIoT.

*Files:* `results/tables/table09_residual_suppression.md` · `results/tables/table09_residual_suppression.csv`

| Condition | Mean residual of attack flows | Attack recall (full model) |
| --- | --- | --- |
| Clean | ≈0.27 | ≈1.00 |
| Random perturbation (0.12) | ≈0.29 | ≈0.95 |
| Adaptive suppression (0.12, 8 trials) | ≈0.20 | ≈0.96 |

## Figures

Extracted image files are under `figures/`.

| ID | File | Caption |
| --- | --- | --- |
| Fig. 1 | `figures/fig01_overview.png` | Overview of CurvTrust-PAI. |
| Fig. 2 | `figures/fig02_pipeline.jpeg` | Implementation pipeline of CurvTrust-PAI. |
| Fig. 3 | `figures/fig03_public_f1.png` | Per-dataset F1 on the four public benchmarks. |
| Fig. 4 | `figures/fig04_edgeiot_f1.png` | Per-method F1 on Custom-EdgeIoT. |
| Fig. 5a | `figures/fig05a_noise_public.png` | F1 vs noise ε (public datasets). |
| Fig. 5b | `figures/fig05b_noise_edgeiot.png` | F1 vs noise ε (Custom-EdgeIoT). |
| Fig. 6 | `figures/fig06_ablation.png` | Ablation of residual integration. |
| Fig. 7 | `figures/fig07_trust_dynamics.png` | Mean trust under overt and authenticated attacks. |
| Fig. 8 | `figures/fig08_edgeiot_diagnostics.png` | Custom-EdgeIoT extended diagnostics. |
| Fig. 9 | `figures/fig09_scaling.png` | Filter wall-clock time vs swarm size N. |

See also `figures/FIGURES_MANIFEST.json` and `results/tables/TABLES_MANIFEST.json`.
