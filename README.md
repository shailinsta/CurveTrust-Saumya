# CurvTrust-PAI

**Reference-anchored neighbourhood-overlap residual for network intrusion analysis and trust-based safety filtering in multi-agent cyber-physical systems**

## Overview

CurvTrust-PAI evaluates a **reference-anchored neighbourhood-overlap residual**—the absolute deviation of a Jaccard neighbourhood-overlap score from its median over a clean reference set—in two independent settings:

1. **Network-flow intrusion analysis** — residual as an auxiliary classifier feature and as a stand-alone conformal detector  
2. **Trust-conditioned safety filtering** in a simulated multi-agent swarm — residual combined with identity, kinematic innovation and graph-isolation evidence to set the separation margin of a relative-degree-two barrier filter

The two modules share the residual implementation but exchange **no** data at run time. Authentication does not guarantee honest behaviour: a compromised agent with valid credentials can still send false state information to its neighbours. CurvTrust-PAI treats behavioural reliability as a separate quantity from identity.

## Key Features

* **Shared residual statistic**  
  * \(k\)-nearest-neighbour graph on a clean reference set (\(n_{\text{ref}} = 1000\))  
  * Mean Jaccard overlap of closed neighbourhoods, mapped to \([-1, 1]\)  
  * Residual \(r(x) = |\kappa(x) - \kappa_{\text{med}}|\)

* **Network-flow detection**  
  * Residual as extra feature to Random Forest / Gradient Boosting / Logistic Regression  
  * Residual alone as conformal score (\(\alpha = 0.05\))  
  * Noise robustness and black-box residual-suppression tests

* **Swarm trust & safety**  
  * Trust fuses residual, kinematic innovation, graph isolation and identity  
  * Trust sets the effective separation margin of a HOCBF-style quadratic program filter  
  * Overt vs authenticated attack modes (identity score drops or stays high)

* **Datasets**  
  * UNSW-NB15, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2, CIDDS-001, Custom-EdgeIoT

* **Baselines & diagnostics**  
  * Features-only Random Forest, gradient boosting, logistic regression  
  * Residual-only conformal detector  
  * Fixed-margin controllers (1.5 m, 3.0 m) vs trust-adaptive filter  
  * Scaling of filter latency with swarm size

## Installation

### Clone Repository

```bash
git clone https://github.com/<your-username>/CurvTrust-PAI.git
cd CurvTrust-PAI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas pyarrow scipy scikit-learn matplotlib
```

Optional GPU acceleration (RAPIDS / cuML on compatible systems, e.g. DGX Spark):

```bash
python -m cuml.accel code/curvtrust_pai_dgx_full.py --quick --workers 1
```

## Datasets

Five intrusion-detection datasets are used. They are **not** redistributed in this repository; place the local Parquet/CSV files under `~/Curve-Test/` (or edit `BASE_DIR` in the scripts).

| Dataset                 | Source / notes                                      |
| ----------------------- | --------------------------------------------------- |
| UNSW-NB15               | Official train/test Parquet files                   |
| NF-ToN-IoT-v2           | NF-ToN-IoT.parquet                                  |
| NF-CSE-CIC-IDS2018-v2   | NF-CSE-CIC-IDS2018.parquet                          |
| CIDDS-001               | cidds-001-externalserver + openstack Parquet        |
| Custom-EdgeIoT          | CSV folder under `Custom-Dataset-EdgeIOT/`          |

**Data protocol (as implemented):**

* Robust (median/IQR) scaler fitted on training rows only; values clipped to \([-10, 10]\)
* Host/session identifiers and non-numeric text dropped where present
* Stratified record-level random splits (or official train/test when `--protocol official` is used for UNSW)
* Per-dataset sample caps (default 30k–40k) for practical run times

## Setup Instructions

### Step 1: Place data

Copy the required Parquet/CSV files into `~/Curve-Test/` (exact filenames are listed in the loaders inside `code/curvtrust_pai_dgx_full.py`).

### Step 2: Run the pipeline

```bash
# Quick smoke test (2 seeds, smaller samples)
python code/curvtrust_pai_dgx_full.py --quick --workers 1 --out-dir ./results

# Strict four-dataset runner (six physical Parquet files only; no EdgeIoT)
python code/curvtrust_pai_dgx_full_6datasets.py --quick --workers 1

# Full run with Custom-EdgeIoT extended analysis
python code/curvtrust_pai_dgx_full.py --extended-analysis --workers 1 --out-dir ./results
```

### Useful flags

| Flag | Description |
| ---- | ----------- |
| `--quick` | Fewer seeds, smaller samples, shorter swarm horizon |
| `--datasets unsw,nf-ton,nf-cse,cidds,edgeiot` | Select logical datasets |
| `--skip-digital` / `--skip-swarm` | Run only one module |
| `--protocol random\|official` | Random stratified split or official train/test when available |
| `--extended-analysis` | Extra Custom-EdgeIoT figures (α-sensitivity, learning curves, …) |
| `--out-dir PATH` | Output directory for JSON, CSVs and figures |
| `--workers N` | Seed-level process parallelism (start with 1 under cuML) |

## Project Structure

```
CurvTrust-PAI/
│
├── code/
│   ├── curvtrust_pai_dgx_full.py            # Full pipeline (digital + swarm + EdgeIoT)
│   └── curvtrust_pai_dgx_full_6datasets.py  # Strict 4-dataset (6-file) runner
│
├── results/
│   └── results.json                         # Aggregated experiment summary
│
├── figures/                                 # Figures generated by the scripts
├── docs/
│   └── CurvTrust_Computer_Networks_revised.docx
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Pipeline Workflow

1. Load and sample datasets; drop identifiers and non-numeric columns  
2. Fit robust scaler on training split only  
3. Build reference set and compute neighbourhood-overlap residual  
4. **Digital path:** train RF / GB / LR with and without residual; residual-only conformal detector  
5. Noise robustness and eight-candidate residual-suppression search  
6. **Swarm path:** simulate 10 double-integrator agents; two compromised in a fixed window  
7. Estimate trust from residual + kinematic innovation + isolation + identity  
8. Apply trust-adaptive HOCBF margin filter; compare with nominal and fixed-margin controllers  
9. Measure separation violations, collisions, goal error and filter latency  
10. Optional Custom-EdgeIoT extended diagnostics (α-sensitivity, learning curves, update diagnostics)  
11. Export JSON, CSVs and figures  

## Residual Definition

Composite distance (in experiments \(\gamma = 1\), so ordinary Euclidean):

\[
d_\gamma(x, x')^2 = \|x_s - x'_s\|_2^2 + \gamma \|x_e - x'_e\|_2^2
\]

Closed \(k\)-neighbourhoods on a clean reference set; Jaccard overlap mapped to \([-1, 1]\); residual:

\[
r(x) = |\kappa(x) - \kappa_{\text{med}}|
\]

The framework name refers to curvature only by motivation: the statistic is a **Jaccard neighbourhood-overlap residual**, not Ollivier–Ricci curvature.

## Trust Model (swarm)

Trust combines four evidence terms (weights set by hand; no optimality claimed):

| Component | Role |
| --------- | ---- |
| Residual | Neighbourhood-overlap deviation of reported state |
| Kinematic innovation | Discrepancy between reported and predicted motion |
| Graph isolation | Structural isolation evidence |
| Identity | High under authenticated attack; low under overt attack |

Trust is EMA-smoothed and mapped to an effective pairwise separation margin for the barrier filter. Compromised agents bypass the filter.

## Evaluation Metrics

**Digital**

* Accuracy, precision, recall, F1, AUC (binary)  
* Residual-only conformal F1 at \(\alpha \in \{0.01, \ldots, 0.20\}\)  
* Attack recall under clean / random / adaptive residual suppression  

**Swarm**

* Separation violations (honest–honest, honest–compromised)  
* Collisions  
* Global and mean minimum separation  
* Goal-tracking error of honest agents  
* Mean trust of honest vs compromised agents  
* Filter wall-clock time vs swarm size \(N\)

## Experimental Results (summary)

### Network intrusion (clean F1)

Adding the residual did **not** improve clean F1 of a features-only Random Forest on any of the five datasets. Residual-only conformal F1 was weak (≈ 0.01–0.29 on public sets, ≈ 0.08 on Custom-EdgeIoT; AUC ≈ 0.437 on Custom-EdgeIoT).

### Residual suppression (Custom-EdgeIoT, approx.)

| Condition | Mean residual (attack) | Attack recall (full model) |
| --------- | ---------------------- | -------------------------- |
| Clean | ≈ 0.27 | ≈ 1.00 |
| Random perturbation (0.12) | ≈ 0.29 | ≈ 0.95 |
| Adaptive suppression (0.12, 8 trials) | ≈ 0.20 | ≈ 0.96 |

Recall loss under adaptive search is comparable to random perturbation of the same size; the residual is not a single point of failure.

### Swarm safety (10 agents, 2 compromised)

* Trust fell during both **overt** and **authenticated** attacks (identity stayed high in the latter).  
* Trust-adaptive filter reduced separation violations by ~98% and collisions by ~99% vs the unfiltered controller.  
* Five seeds showed **no clear advantage** over a fixed 3.0 m margin.  
* Goal-tracking error of honest agents rose from ~1.63 m (nominal) to ~5.19 m (trust-adaptive).  
* Mean global minimum separation was 0.555 m against a required 1.5 m (compromised agents bypass the filter).  

### Computational scaling

One complete filter step (sequential, test platform):

| \(N\) | Time (ms) |
| ----- | --------- |
| 5 | ≈ 1.3 |
| 10 | ≈ 4.5 |
| 15 | ≈ 9.7 |
| 20 | ≈ 17.3 |
| 30 | ≈ 51 |

Growth is approximately quadratic.

## Key Insights

* The residual is more useful as an **input to behavioural trust estimation** than as an addition to conventional tabular intrusion detection.  
* Authentication does not guarantee honest behaviour: trust fell for compromised agents even when the identity score stayed high.  
* Adaptive margin filtering improves safety over an unfiltered controller, but a fixed large margin achieved similar separation at similar tracking cost under the conditions tested.  
* Residual-suppression search did not preferentially break the detector beyond what random noise of the same size already does.  

## Research Contributions

1. **Residual definition** — Jaccard neighbourhood-overlap residual anchored to a clean reference population, applied to flow records and to reported multi-agent states.  
2. **Network evaluation** — residual as added classifier feature and as conformal score on five traffic datasets, under clean, noisy and residual-suppression conditions.  
3. **Swarm evaluation** — residual combined with kinematic, isolation and identity evidence into a trust value that sets the separation margin of a barrier filter; compared with unfiltered and fixed-margin controllers under overt and authenticated attacks.  
4. **Reported limitations** — no gain in clean F1, weak stand-alone detection, higher tracking error from filtering, no demonstrated advantage over a fixed large margin, and growth of computation time with swarm size.

## Important Notes / Limitations

* Record-level random splits may be optimistic; chronological, host- or capture-disjoint splits were not evaluated.  
* Adversarial evaluation is limited to eight Gaussian perturbations of fixed strength; not protocol-preserving.  
* Trust weights and margin gain were set by hand; no ablation of individual evidence terms.  
* Barrier construction treats the trust-based margin as constant; forward invariance with a time-varying margin is not guaranteed.  
* Five seeds are descriptive only; no significance or equivalence tests were run.  
* The swarm is a simulation stand-in; no real robot experiments are claimed.  
* Jaccard overlap is a **proxy**, not Wasserstein Ollivier–Ricci curvature.  

## Common Errors

| Error | Cause | Fix |
| ----- | ----- | --- |
| `FileNotFoundError` for Parquet/CSV | Data not under `~/Curve-Test/` | Place files as listed in the loaders, or edit `BASE_DIR` / `DATA_FILES` |
| `Only one class present` (EdgeIoT) | Missing benign or attack CSVs | Ensure both classes are present under `Custom-Dataset-EdgeIOT/` |
| Slow runs / OOM | Large samples + many seeds | Use `--quick` or lower `max_samples` in `Cfg` |
| `ModuleNotFoundError` | Dependencies missing | `pip install -r requirements.txt` |

## DOI

DOI: to be added after the Zenodo release is published.

## Authors

| Name | Affiliation |
| ---- | ----------- |
| Saumya Mishra | Department of Computer Science and Engineering, Motilal Nehru National Institute of Technology Allahabad (MNNIT Allahabad), Prayagraj 211004, Uttar Pradesh, India |
| Shailendra Mishra | Department of Computer Engineering, College of Computer and Information Sciences, Majmaah University, Al-Majmaah 11952, Saudi Arabia |

**Contact**

* Saumya Mishra — 20253577.saumya@mnnit.ac.in  
* Shailendra Mishra — s.mishra@mu.edu.sa (corresponding author)

## Citation

```bibtex
@article{mishra2026curvtrust,
  title   = {Reference-anchored neighbourhood-overlap residual for network intrusion analysis and trust-based safety filtering in multi-agent cyber-physical systems},
  author  = {Mishra, Saumya and Mishra, Shailendra},
  journal = {},
  year    = {},
  note    = {}
}
```

## License

This project is released under the MIT License (see `LICENSE`). Intended for academic and research use.
