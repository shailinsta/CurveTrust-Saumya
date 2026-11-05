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
