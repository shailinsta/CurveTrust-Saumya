#!/usr/bin/env python3
"""
CurvTrust-PAI — DGX Spark runner
================================
Single-file version configured for the datasets already uploaded to:

    /Curve-Test/

Expected files:
    /Curve-Test/UNSW_NB15_training-set.parquet
    /Curve-Test/UNSW_NB15_testing-set.parquet
    /Curve-Test/NF-ToN-IoT.parquet
    /Curve-Test/NF-CSE-CIC-IDS2018.parquet
    /Curve-Test/cidds-001-externalserver.parquet
    /Curve-Test/cidds-001-openstack.parquet

Edge-IIoTset is not included because it was not visible in the uploaded folder.

DGX Spark:
    Recommended launch:
        python -m cuml.accel /Curve-Test/curvtrust_pai_dgx.py --quick

    Full run:
        python -m cuml.accel /Curve-Test/curvtrust_pai_dgx.py

Notes:
- Uses float32 for lower memory pressure / better accelerator compatibility.
- Dataset loaders use the exact /Curve-Test file paths above.
- The scientific pipeline remains the same: curvature residual, RF/LR/GB baselines,
  conformal thresholding, noise/adaptive tests, swarm trust, HOCBF-QP, statistics.
- SLSQP HOCBF optimization remains CPU-side.
- --workers controls seed-level parallelism. On a single GB10, start with 1 worker
  under cuml.accel; benchmark 2/4 only after confirming stability.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# =============================================================================
# DGX SPARK / DATA CONFIGURATION
# =============================================================================
BASE_DIR = Path.home() / "Curve-Test"
DATA_FILES = {
    "unsw_train": BASE_DIR / "UNSW_NB15_training-set.parquet",
    "unsw_test": BASE_DIR / "UNSW_NB15_testing-set.parquet",
    "nf_ton": BASE_DIR / "NF-ToN-IoT.parquet",
    "nf_cse": BASE_DIR / "NF-CSE-CIC-IDS2018.parquet",
    "cidds_external": BASE_DIR / "cidds-001-externalserver.parquet",
    "cidds_openstack": BASE_DIR / "cidds-001-openstack.parquet",
}

# Newly added custom EdgeIoT dataset folder.
EDGEIOT_DIR = BASE_DIR / "Custom-Dataset-EdgeIOT"


BETA = 10.0
TAU_HEALTHY = 0.90
CLIP = 10.0
NOISE_EPS = [0.05, 0.10, 0.15, 0.25, 0.35, 0.40]


@dataclass
class Cfg:
    seeds: list = field(default_factory=lambda: [42, 43, 44, 45, 46])
    data_dir: str = str(BASE_DIR)
    out_dir: str = str(BASE_DIR / "results_dgx")
    protocol: str = "random"
    workers: int = 1
    k: int = 5
    gamma: float = 1.0
    n_ref: int = 1000
    alpha: float = 0.05
    cal_frac: float = 0.20
    rf_trees: int = 120
    rf_depth: int = 14
    cv_trees: int = 80
    cv_depth: int = 12
    do_cv: bool = True
    adv_eps: float = 0.12
    adv_trials: int = 8
    max_attack_eval: int = 2000
    n_agents: int = 10
    n_steps: int = 500
    dt: float = 0.1
    d_min: float = 1.5
    umax: float = 2.0
    kp: float = 1.0
    kd: float = 1.5
    leg_steps: int = 100
    d_emb: int = 8
    attack_window: tuple = (0.30, 0.70)
    comp_frac: float = 0.25
    spoof_bias: float = 1.0
    spoof_sigma: float = 0.10
    emb_poison: float = 0.8
    maneuver_sigma: float = 0.5
    s_id_honest: float = 0.95
    s_id_comp: float = 0.25
    lambda_tau: float = 2.0
    zeta: float = 1.0
    omega: float = 1.5
    sense_range: float = 8.0
    ema: float = 0.5
    attack_modes: list = field(default_factory=lambda: ["overt", "authenticated"])
    scaling_N: list = field(default_factory=lambda: [5, 10, 15, 20, 30])
    scaling_reps: int = 10
    max_samples: dict = field(default_factory=lambda: {
        "UNSW-NB15": 40000,
        "NF-ToN-IoT-v2": 30000,
        "NF-CSE-CIC-IDS2018-v2": 30000,
        "CIDDS-001": 30000,
        "Custom-EdgeIoT": 30000,
    })


# =============================================================================
# Utilities
# =============================================================================
def fmt_ms(vals, nd=3):
    v = np.asarray(vals, dtype=float)
    return f"{np.nanmean(v):.{nd}f}±{np.nanstd(v):.{nd}f}"


def binary_metrics(y, pred, score):
    acc = accuracy_score(y, pred)
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y, score)
    except Exception:
        auc = float("nan")
    return dict(acc=acc, prec=p, rec=r, f1=f, auc=auc)


def paired_stats(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    out = dict(
        mean_diff=float(d.mean()), n=n,
        ci_lo=float("nan"), ci_hi=float("nan"), p=float("nan")
    )
    if n > 1:
        se = d.std(ddof=1) / np.sqrt(n)
        t = stats.t.ppf(0.975, n - 1)
        out["ci_lo"] = float(d.mean() - t * se)
        out["ci_hi"] = float(d.mean() + t * se)
        if se > 0:
            out["p"] = float(stats.ttest_rel(a, b).pvalue)
    return out


def print_table(title, headers, rows):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
        for i, h in enumerate(headers)
    ]
    fmt = "".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-" * 100)
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))
    print("=" * 100)


def read_parquet_exact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    return pd.read_parquet(path)


def sample_df(df, target, seed=0):
    if len(df) <= target:
        return df.reset_index(drop=True)
    return df.sample(n=target, random_state=seed).reset_index(drop=True)


def _clean_numeric(df):
    return (
        df.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )


LABEL_CANDIDATES = [
    "Label", "label", "LABEL", "Attack_label", "attack_label", "is_attack"
]


def _find_label(df, extra=()):
    for c in list(extra) + LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(
        f"No supported label column found. First columns: {list(df.columns)[:60]}"
    )


def binarize(series):
    if pd.api.types.is_numeric_dtype(series) or series.dtype == bool:
        return (
            pd.to_numeric(series, errors="coerce").fillna(0) > 0
        ).astype(int).values
    return (
        ~series.astype(str).str.strip().str.lower().isin(
            ["benign", "normal", "0", "false", "0.0", "none"]
        )
    ).astype(int).values


def _features(df, drop=(), onehot=()):
    d = df.drop(columns=[c for c in drop if c in df.columns])
    oh = [c for c in onehot if c in d.columns]
    parts = [_clean_numeric(d.drop(columns=oh))]
    for c in oh:
        parts.append(pd.get_dummies(d[c].astype(str), prefix=c).astype(np.float32))
    return pd.concat(parts, axis=1)


# =============================================================================
# Exact dataset loaders for /Curve-Test
# =============================================================================
def load_unsw(pool):
    a = read_parquet_exact(DATA_FILES["unsw_train"])
    b = read_parquet_exact(DATA_FILES["unsw_test"])

    # Preserve source indicator before optional sampling.
    a["_source_split_"] = 0
    b["_source_split_"] = 1
    df = pd.concat([a, b], ignore_index=True)
    df = sample_df(df, pool, seed=0)

    src = df.pop("_source_split_").astype(int).values
    lc = _find_label(df, extra=("label", "Label"))
    y = binarize(df[lc])
    X = _features(df, drop=(lc, "id", "attack_cat"))

    return X.to_numpy(dtype=np.float32), y, "UNSW-NB15", src


def load_nf_ton(pool):
    df = sample_df(read_parquet_exact(DATA_FILES["nf_ton"]), pool, seed=0)
    lc = _find_label(df)
    y = binarize(df[lc])
    X = _features(
        df,
        drop=(
            lc, "Attack", "attack", "Attack_type",
            "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
        ),
    )
    return X.to_numpy(dtype=np.float32), y, "NF-ToN-IoT-v2", None


def load_nf_cse(pool):
    df = sample_df(read_parquet_exact(DATA_FILES["nf_cse"]), pool, seed=0)
    lc = _find_label(df)
    y = binarize(df[lc])
    X = _features(
        df,
        drop=(
            lc, "Attack", "attack", "Attack_type",
            "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
        ),
    )
    return X.to_numpy(dtype=np.float32), y, "NF-CSE-CIC-IDS2018-v2", None


def _convert_cidds_numeric(df):
    for c in ("Bytes", "Packets"):
        if c in df.columns and df[c].dtype == object:
            s = df[c].astype(str).str.strip()
            mult = (
                s.str.extract(r"([KMG])$")[0]
                .map({"K": 1e3, "M": 1e6, "G": 1e9})
                .fillna(1.0)
            )
            df[c] = (
                pd.to_numeric(
                    s.str.replace(r"[KMG]$", "", regex=True),
                    errors="coerce",
                ) * mult
            )
    return df


def load_cidds(pool):
    per_file = max(1, pool // 2)
    dfs = [
        sample_df(read_parquet_exact(DATA_FILES["cidds_external"]), per_file, 0),
        sample_df(read_parquet_exact(DATA_FILES["cidds_openstack"]), per_file, 1),
    ]
    df = pd.concat(dfs, ignore_index=True)

    cc = next(
        (c for c in ("class", "Class", "label", "Label") if c in df.columns),
        None,
    )
    if cc is None:
        raise KeyError(
            f"CIDDS-001: no class/label column; columns={list(df.columns)[:60]}"
        )

    cls = df[cc].astype(str).str.strip().str.lower()
    keep = cls.isin(
        ["normal", "attacker", "victim", "benign", "0", "1"]
    )
    df = df.loc[keep].reset_index(drop=True)
    cls = cls.loc[keep].reset_index(drop=True)
    y = (~cls.isin(["normal", "benign", "0"])).astype(int).values

    df = _convert_cidds_numeric(df)
    drop = (
        cc, "attackType", "attackID", "attackDescription",
        "Date first seen", "Src IP Addr", "Dst IP Addr",
        "Date_first_seen", "Src_IP_Addr", "Dst_IP_Addr",
    )
    X = _features(df, drop=drop, onehot=("Proto", "Flags"))
    return X.to_numpy(dtype=np.float32), y, "CIDDS-001", None



def _edgeiot_label_column(df):
    for c in (
        "Label", "label", "Attack_label", "attack_label",
        "Attack", "attack", "Class", "class",
        "Category", "category", "Attack_type", "attack_type",
        "Type", "type",
    ):
        if c in df.columns:
            return c
    return None


def _edgeiot_binary_label(series):
    if pd.api.types.is_numeric_dtype(series) or series.dtype == bool:
        v = pd.to_numeric(series, errors="coerce").fillna(0)
        return (v > 0).astype(int).values

    s = series.astype(str).str.strip().str.lower()
    benign = {
        "0", "0.0", "false", "normal", "benign", "none",
        "no attack", "no_attack", "legitimate",
    }
    return (~s.isin(benign)).astype(int).values


def _edgeiot_filename_is_benign(path):
    """
    Filename fallback used only when a CSV has no supported label column.

    The uploaded Custom-Dataset-EdgeIoT folder contains normal IoT sensor
    captures (Distance, Flame Sensor, Heart Rate, IR Receiver, Modbus,
    pH Value, Soil Moisture, Sound Sensor, Temperature/Humidity, Water Level)
    plus explicitly named attack captures. Explicit attack names are class 1;
    the known sensor captures are class 0.

    Dataset-wide files such as ML-EdgeIIoT-dataset.csv and ciciot23.csv should
    preferably contain their own Label/Class/Attack column; if they do not,
    they are not assigned a class from the filename.
    """
    s = path.stem.lower().replace("-", "_").replace(" ", "_")
    attack_tokens = ("attack", "ddos", "ransomware", "sql_injection")
    if any(tok in s for tok in attack_tokens):
        return False

    benign_sensor_names = {
        "distance", "flame_sensor", "heart_rate", "ir_receiver", "modbus",
        "phvalue", "soil_moisture", "sound_sensor",
        "temperature_and_humidity", "water_level",
    }
    return s in benign_sensor_names


def _edgeiot_prepare_csv(df, path):
    """
    Convert one custom EdgeIoT CSV into numeric ML features.

    If a label column exists, it is used. Otherwise the filename is used:
    files containing 'benign' or 'normal' are class 0; other attack files
    are class 1.
    """
    df = df.copy()
    lc = _edgeiot_label_column(df)

    if lc is not None:
        y = _edgeiot_binary_label(df[lc])
        df = df.drop(columns=[lc])
    else:
        s = path.stem.lower().replace("-", "_").replace(" ", "_")
        explicit_attack = any(
            tok in s for tok in ("attack", "ddos", "ransomware", "sql_injection")
        )
        explicit_benign = _edgeiot_filename_is_benign(path)
        if not explicit_attack and not explicit_benign:
            raise ValueError(
                f"{path.name}: no supported label column and filename does not "
                "unambiguously identify benign or attack traffic."
            )
        y = np.full(len(df), 0 if explicit_benign else 1, dtype=int)

    # Drop common text labels/identifiers that should not become shortcuts.
    drop_names = {
        "attack_type", "attack type", "category", "sub_category",
        "sub category", "flow_id", "flow id",
    }
    df = df.drop(
        columns=[c for c in df.columns if str(c).strip().lower() in drop_names],
        errors="ignore",
    )

    # Convert numeric-looking strings. Non-numeric text (IPs, URLs, payloads,
    # timestamps, etc.) is excluded from this tabular model.
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(np.int8)
        elif df[c].dtype == object:
            v = pd.to_numeric(df[c], errors="coerce")
            denom = max(1, int(df[c].notna().sum()))
            if v.notna().sum() / denom >= 0.90:
                df[c] = v

    X = _clean_numeric(df)
    X.columns = [str(c) for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    return X, y


def load_edgeiot(pool):
    """
    Load every CSV under:
        ~/Curve-Test/Custom-Dataset-EdgeIOT/

    The current folder may contain matching PCAP files; they are intentionally
    ignored because this CurvTrust pipeline operates on tabular features.
    """
    if not EDGEIOT_DIR.exists():
        raise FileNotFoundError(
            f"Custom EdgeIoT folder not found: {EDGEIOT_DIR}"
        )

    files = sorted(EDGEIOT_DIR.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under: {EDGEIOT_DIR}"
        )

    print(f"[EDGEIOT] Found {len(files)} CSV file(s)")
    frames, labels = [], []
    per_file = max(1, int(np.ceil(pool / len(files))))

    for i, path in enumerate(files):
        df = pd.read_csv(path, low_memory=False)
        if len(df) > per_file:
            df = df.sample(
                n=per_file, random_state=100 + i
            ).reset_index(drop=True)

        try:
            Xf, yf = _edgeiot_prepare_csv(df, path)
        except Exception as e:
            print(f"[WARN] EdgeIoT: {path.name} skipped — {type(e).__name__}: {e}")
            continue
        if Xf.shape[1] == 0:
            print(
                f"[WARN] EdgeIoT: {path.name} skipped "
                "(no numeric features)."
            )
            continue

        frames.append(Xf)
        labels.append(yf)
        print(
            f"[EDGEIOT] {path.name}: rows={len(yf)}, "
            f"features={Xf.shape[1]}, "
            f"attack_rate={float(np.mean(yf)):.3f}"
        )

    if not frames:
        raise ValueError(
            "Custom EdgeIoT: no usable CSV feature files."
        )

    # Different CSVs can expose slightly different feature sets.
    all_cols = sorted(set().union(*(set(x.columns) for x in frames)))
    frames = [
        x.reindex(columns=all_cols, fill_value=0)
        for x in frames
    ]
    X = pd.concat(frames, ignore_index=True)
    y = np.concatenate(labels).astype(int)

    # Classification requires both benign and attack samples.
    if len(np.unique(y)) < 2:
        raise ValueError(
            "Custom EdgeIoT contains only one binary class. "
            "The visible DDoS, Ransomware and SQL-injection CSVs are "
            "attack datasets. Add at least one benign/normal CSV to "
            f"{EDGEIOT_DIR}; its filename can contain 'benign' or 'normal', "
            "or it can contain a supported Label/Class column."
        )

    if len(y) > pool:
        idx = np.arange(len(y))
        idx, _ = train_test_split(
            idx,
            train_size=pool,
            random_state=0,
            stratify=y,
        )
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]

    return (
        X.to_numpy(dtype=np.float32),
        y,
        "Custom-EdgeIoT",
        None,
    )


REGISTRY = {
    "unsw": load_unsw,
    "nf-ton": load_nf_ton,
    "nf-cse": load_nf_cse,
    "cidds": load_cidds,
    "edgeiot": load_edgeiot,
}
DEFAULT_DATASETS = ["unsw", "nf-ton", "nf-cse", "cidds", "edgeiot"]


# =============================================================================
# Per-sample Jaccard curvature residual
# =============================================================================
class CurvatureResidual:
    def __init__(self, k=5, gamma=1.0, n_ref=1000, n_struct=None, seed=0):
        self.k = k
        self.gamma = gamma
        self.n_ref = n_ref
        self.n_struct = n_struct
        self.seed = seed

    def _z(self, X):
        ns = self.n_struct if self.n_struct is not None else max(1, X.shape[1] // 2)
        return np.hstack(
            [X[:, :ns], np.sqrt(self.gamma) * X[:, ns:]]
        ).astype(np.float32, copy=False)

    def fit(self, X):
        rng = np.random.RandomState(self.seed)
        m = min(self.n_ref, X.shape[0])
        if m < 2:
            raise ValueError("CurvatureResidual requires at least 2 fitting samples.")
        self.ref_idx_ = rng.choice(X.shape[0], m, replace=False)
        self.R_ = self._z(X)[self.ref_idx_]
        self.k_eff_ = min(self.k, m - 1)
        self.nn_ = NearestNeighbors(
            n_neighbors=self.k_eff_ + 1
        ).fit(self.R_)
        nbr = self._neighbours(self.R_, np.arange(m))
        self.refnbr_ = nbr
        kap = self._kappa_from(nbr, np.arange(m))
        self.baseline_ = float(np.median(kap))
        self.ref_kappa_ = kap
        return self

    def _neighbours(self, Z, self_pos):
        idx = self.nn_.kneighbors(Z, return_distance=False)
        n, k = Z.shape[0], self.k_eff_
        mask = idx == self_pos[:, None]
        keep = ~mask
        nomatch = ~mask.any(axis=1)
        keep[nomatch, -1] = False
        return idx[keep].reshape(n, k)

    def _kappa_from(self, nbr, self_pos):
        k = self.k_eff_
        out = np.empty(nbr.shape[0], dtype=np.float32)
        for s in range(0, nbr.shape[0], 20000):
            e = min(s + 20000, nbr.shape[0])
            nb = nbr[s:e]
            sp = self_pos[s:e]
            nbj = self.refnbr_[nb]
            inter = (
                (nbj[:, :, :, None] == nb[:, None, None, :])
                .any(-1)
                .sum(-1)
                + 1
            )
            has_self = (sp >= 0)[:, None]
            inter = inter + (
                (nbj == sp[:, None, None]).any(-1) & has_self
            )
            union = 2 * (k + 1) - inter
            out[s:e] = (
                2.0 * inter / union - 1.0
            ).mean(axis=1)
        return out

    def kappa(self, X, self_pos=None):
        Z = self._z(X)
        sp = (
            np.full(X.shape[0], -1)
            if self_pos is None
            else np.asarray(self_pos)
        )
        return self._kappa_from(self._neighbours(Z, sp), sp)

    def transform(self, X, self_pos=None):
        return np.abs(
            self.kappa(X, self_pos) - self.baseline_
        ).reshape(-1, 1).astype(np.float32)

    def fit_transform(self, X):
        self.fit(X)
        sp = np.full(X.shape[0], -1)
        sp[self.ref_idx_] = np.arange(len(self.ref_idx_))
        return self.transform(X, sp)


def conformal_threshold(scores, alpha=0.05):
    n = len(scores)
    if n == 0:
        return float("inf")
    kk = int(np.ceil((n + 1) * (1 - alpha)))
    if kk > n:
        return float("inf")
    return float(np.sort(scores)[kk - 1])


def prep_scale(scaler, X):
    return np.clip(
        scaler.transform(X), -CLIP, CLIP
    ).astype(np.float32, copy=False)


def subsample(X, y, src, max_n, seed):
    if len(y) <= max_n:
        return X, y, src
    idx = np.random.RandomState(seed).choice(
        len(y), max_n, replace=False
    )
    return X[idx], y[idx], (src[idx] if src is not None else None)


# =============================================================================
# Digital intrusion detection
# =============================================================================
def run_digital_seed(X, y, src, name, seed, cfg: Cfg):
    X, y, src = subsample(
        X, y, src, cfg.max_samples.get(name, 20000), seed
    )
    X = X[:, X.std(axis=0) > 0]

    if cfg.protocol == "official" and src is not None and len(np.unique(src)) == 2:
        tr_i, te_i = np.where(src == 0)[0], np.where(src == 1)[0]
    else:
        tr_i, te_i = train_test_split(
            np.arange(len(y)),
            test_size=0.25,
            random_state=seed,
            stratify=y,
        )

    Xtr_raw, Xte_raw = X[tr_i], X[te_i]
    ytr, yte = y[tr_i], y[te_i]

    rng = np.random.RandomState(seed)
    normal = np.where(ytr == 0)[0]
    if len(normal) < 2:
        raise ValueError(
            f"{name}: insufficient normal training samples for calibration."
        )
    n_cal = min(len(normal) - 1, max(10, int(cfg.cal_frac * len(normal))))
    cal_idx = rng.choice(normal, n_cal, replace=False)
    fit_idx = np.setdiff1d(np.arange(len(ytr)), cal_idx)

    t0 = time.perf_counter()
    scaler = RobustScaler().fit(Xtr_raw[fit_idx])
    Xs_fit = prep_scale(scaler, Xtr_raw[fit_idx])
    curv = CurvatureResidual(
        cfg.k, cfg.gamma, cfg.n_ref, seed=seed
    )
    r_fit = curv.fit_transform(Xs_fit)
    t_curv_train = time.perf_counter() - t0

    Xs_cal = prep_scale(scaler, Xtr_raw[cal_idx])
    r_cal = curv.transform(Xs_cal).ravel()
    tau_conf = conformal_threshold(r_cal, cfg.alpha)

    def rf(trees=cfg.rf_trees, depth=cfg.rf_depth):
        # n_jobs=1 avoids CPU oversubscription if seed workers are enabled.
        return RandomForestClassifier(
            n_estimators=trees,
            max_depth=depth,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )

    y_fit = ytr[fit_idx]
    full = rf()
    t1 = time.perf_counter()
    full.fit(np.hstack([Xs_fit, r_fit]), y_fit)
    train_full_s = t_curv_train + (time.perf_counter() - t1)

    rf0 = rf()
    lr = LogisticRegression(
        max_iter=500, class_weight="balanced", random_state=seed
    )
    gb = GradientBoostingClassifier(
        n_estimators=80, max_depth=4, random_state=seed
    )
    rf0.fit(Xs_fit, y_fit)
    lr.fit(Xs_fit, y_fit)
    gb.fit(Xs_fit, y_fit)

    Xs_te = prep_scale(scaler, Xte_raw)
    t2 = time.perf_counter()
    r_te = curv.transform(Xs_te)
    _ = full.predict(np.hstack([Xs_te, r_te]))
    pred_full_s = time.perf_counter() - t2

    t3 = time.perf_counter()
    _ = rf0.predict(Xs_te)
    pred_rf0_s = time.perf_counter() - t3

    def evaluate(Xs, r):
        res = {}
        Xf = np.hstack([Xs, r])
        p = full.predict_proba(Xf)[:, 1]
        res["CurvTrust-PAI (full)"] = binary_metrics(
            yte, (p > 0.5).astype(int), p
        )
        p = rf0.predict_proba(Xs)[:, 1]
        res["RF features-only"] = binary_metrics(
            yte, (p > 0.5).astype(int), p
        )
        p = lr.predict_proba(Xs)[:, 1]
        res["Logistic Regression"] = binary_metrics(
            yte, (p > 0.5).astype(int), p
        )
        p = gb.predict_proba(Xs)[:, 1]
        res["Gradient Boosting"] = binary_metrics(
            yte, (p > 0.5).astype(int), p
        )
        rr = r.ravel()
        res["Residual conformal"] = binary_metrics(
            yte, (rr > tau_conf).astype(int), rr
        )
        return res

    clean = evaluate(Xs_te, r_te)

    noise = {}
    for eps in NOISE_EPS:
        nr = np.random.RandomState(seed * 1000 + int(eps * 100))
        Xn = (
            Xs_te + nr.normal(0, eps, Xs_te.shape).astype(np.float32)
        ).astype(np.float32)
        noise[eps] = {
            m: v["f1"]
            for m, v in evaluate(Xn, curv.transform(Xn)).items()
        }

    att = np.where(yte == 1)[0][:cfg.max_attack_eval]
    Xa = Xs_te[att]
    ra = curv.transform(Xa)
    Xa_adv, ra_adv = adaptive_residual_attack(
        Xa, curv, cfg.adv_eps, cfg.adv_trials, seed
    )
    Xa_rnd = (
        Xa + np.random.RandomState(seed + 7)
        .normal(0, cfg.adv_eps, Xa.shape)
        .astype(np.float32)
    ).astype(np.float32)
    ra_rnd = curv.transform(Xa_rnd)

    def recall(Xs, r, model_full):
        pr = (
            full.predict(np.hstack([Xs, r]))
            if model_full
            else rf0.predict(Xs)
        )
        return float(np.mean(pr == 1))

    adaptive = dict(
        resid_clean=float(ra.mean()),
        resid_random=float(ra_rnd.mean()),
        resid_adaptive=float(ra_adv.mean()),
        recall_full=dict(
            clean=recall(Xa, ra, True),
            random=recall(Xa_rnd, ra_rnd, True),
            adaptive=recall(Xa_adv, ra_adv, True),
        ),
        recall_rf=dict(
            clean=recall(Xa, ra, False),
            random=recall(Xa_rnd, ra_rnd, False),
            adaptive=recall(Xa_adv, ra_adv, False),
        ),
        recall_conf=dict(
            clean=float((ra.ravel() > tau_conf).mean()),
            random=float((ra_rnd.ravel() > tau_conf).mean()),
            adaptive=float((ra_adv.ravel() > tau_conf).mean()),
        ),
    )

    cv = None
    if cfg.do_cv:
        cv = dict(full=[], rf=[])
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for a, b in skf.split(Xtr_raw, ytr):
            sc = RobustScaler().fit(Xtr_raw[a])
            Xa_ = prep_scale(sc, Xtr_raw[a])
            Xb_ = prep_scale(sc, Xtr_raw[b])
            cc = CurvatureResidual(
                cfg.k, cfg.gamma, cfg.n_ref, seed=seed
            )
            ra_ = cc.fit_transform(Xa_)
            rb_ = cc.transform(Xb_)
            m1 = rf(cfg.cv_trees, cfg.cv_depth).fit(
                np.hstack([Xa_, ra_]), ytr[a]
            )
            m0 = rf(cfg.cv_trees, cfg.cv_depth).fit(Xa_, ytr[a])
            cv["full"].append(
                precision_recall_fscore_support(
                    ytr[b],
                    m1.predict(np.hstack([Xb_, rb_])),
                    average="binary",
                    zero_division=0,
                )[2]
            )
            cv["rf"].append(
                precision_recall_fscore_support(
                    ytr[b],
                    m0.predict(Xb_),
                    average="binary",
                    zero_division=0,
                )[2]
            )
        cv = dict(
            full=float(np.mean(cv["full"])),
            rf=float(np.mean(cv["rf"])),
        )

    return dict(
        seed=seed,
        n=int(len(y)),
        n_train=int(len(tr_i)),
        n_test=int(len(te_i)),
        attack_rate=float(y.mean()),
        clean=clean,
        noise=noise,
        adaptive=adaptive,
        cv=cv,
        tau_conf=tau_conf,
        train_s=train_full_s,
        pred_ms=pred_full_s / len(yte) * 1000,
        pred_ms_rf=pred_rf0_s / len(yte) * 1000,
        curv_baseline=curv.baseline_,
    )


def adaptive_residual_attack(X, curv, eps, trials, seed):
    rng = np.random.RandomState(seed + 99)
    best = X.copy()
    best_r = curv.transform(X).ravel()
    for _ in range(trials):
        cand = (
            X + rng.normal(0, eps, X.shape).astype(np.float32)
        ).astype(np.float32)
        rc = curv.transform(cand).ravel()
        better = rc < best_r
        best[better] = cand[better]
        best_r[better] = rc[better]
    return best, best_r.reshape(-1, 1)


# =============================================================================
# Swarm / Trust / HOCBF
# =============================================================================
def make_scenario(cfg, seed, attack, mode):
    rng = np.random.RandomState(seed)
    N, T, d = cfg.n_agents, cfg.n_steps, cfg.d_emb
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.uniform(0, 0.2)
    R = 4.0
    A = np.stack(
        [R * np.cos(ang), R * np.sin(ang), 1 + 0.2 * rng.randn(N)], 1
    )
    B = np.stack(
        [-A[:, 0], -A[:, 1], 1 + 0.2 * rng.randn(N)], 1
    ) + rng.randn(N, 3) * [0.4, 0.4, 0.1]
    n_comp = max(2, int(round(cfg.comp_frac * N)))
    comp = rng.choice(N, n_comp, replace=False) if attack else np.array([], int)
    dirs = rng.randn(N, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    t1, t2 = int(cfg.attack_window[0] * T), int(cfg.attack_window[1] * T)
    return dict(
        A=A, B=B, comp=comp, bias=cfg.spoof_bias * dirs,
        t1=t1, t2=t2, attack=attack, mode=mode,
        y0=rng.randn(N, d),
        emb_noise=rng.randn(T, N, d) * 0.05,
        proc=rng.randn(T, N, 3) * 0.01,
        white=rng.randn(T, N, 3) * cfg.spoof_sigma,
        poison=rng.randn(T, N, d) * cfg.emb_poison,
        maneuver=rng.randn(T, N, 3) * cfg.maneuver_sigma,
    )


def goal_at(sc, t, cfg):
    return sc["B"] if (t // cfg.leg_steps) % 2 == 0 else sc["A"]


def quarantine_scores(F, k=3, gamma=1.0, n_struct=6, iters=3):
    N = F.shape[0]
    Z = np.hstack([F[:, :n_struct], np.sqrt(gamma) * F[:, n_struct:]])
    D = np.linalg.norm(Z[:, None] - Z[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    kk = min(k, N - 1)
    Adj = np.zeros((N, N), bool)
    nn = np.argsort(D, axis=1)[:, :kk]
    for i in range(N):
        Adj[i, nn[i]] = True
    Adj |= Adj.T
    W = np.where(
        Adj,
        1.0 / (1.0 + np.where(np.isfinite(D), D, 0)),
        0.0,
    )
    S = (Adj | np.eye(N, dtype=bool)).astype(int)
    for _ in range(iters):
        inter = S @ S.T
        sz = S.sum(1)
        kap = (
            2.0 * inter / (sz[:, None] + sz[None] - inter) - 1.0
        )
        W = np.where(Adj & (kap < 0), W * (1 + 0.5 * kap), W)
    deg = W.sum(1)
    return 1.0 - deg / (deg.max() + 1e-12)


class TrustEstimator:
    W = dict(id=0.25, k=0.35, r=0.25, q=0.15)

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, rep_p, rep_v, rep_y, seed):
        cfg = self.cfg
        T, N, _ = rep_p.shape
        F = np.concatenate(
            [rep_p, rep_v, rep_y], axis=2
        ).reshape(T * N, -1)
        self.scaler = RobustScaler().fit(F)
        Fs = np.clip(self.scaler.transform(F), -CLIP, CLIP)
        self.curv = CurvatureResidual(
            cfg.k, cfg.gamma, cfg.n_ref,
            n_struct=6, seed=seed
        )
        rk = self.curv.fit_transform(Fs).reshape(T, N)
        rp = np.zeros((T, N))
        rp[1:] = np.linalg.norm(
            rep_p[1:] - (rep_p[:-1] + rep_v[:-1] * cfg.dt),
            axis=2,
        )
        Fs3 = Fs.reshape(T, N, -1)
        rq = np.stack(
            [quarantine_scores(Fs3[t]) for t in range(T)]
        )
        self.scale = dict(
            k=2 * np.quantile(rk, 0.95) + 1e-9,
            r=2 * np.quantile(rp[1:], 0.95) + 1e-9,
            q=2 * np.quantile(rq, 0.95) + 1e-9,
        )
        self.conf = dict(
            k=conformal_threshold(rk.ravel(), cfg.alpha),
            r=conformal_threshold(rp[1:].ravel(), cfg.alpha),
        )
        z = self._z(
            np.full((T, N), cfg.s_id_honest),
            self._n(rk, "k"),
            self._n(rp, "r"),
            self._n(rq, "q"),
        )
        self.theta = float(
            np.median(z)
            - np.log(TAU_HEALTHY / (1 - TAU_HEALTHY)) / BETA
        )
        return self

    def _n(self, e, key):
        return np.clip(e / self.scale[key], 0, 1)

    def _z(self, sid, ek, er, eq, w=None):
        w = w or self.W
        return (
            w["id"] * sid
            - w["k"] * ek
            - w["r"] * er
            - w["q"] * eq
        )

    def tau_from(self, sid, ek, er, eq, w=None):
        return 1.0 / (
            1.0 + np.exp(-BETA * (self._z(sid, ek, er, eq, w) - self.theta))
        )

    def evidence(self, p_rep, v_rep, y_rep, prev):
        cfg = self.cfg
        F = np.hstack([p_rep, v_rep, y_rep])
        Fs = np.clip(self.scaler.transform(F), -CLIP, CLIP)
        rk = self.curv.transform(Fs).ravel()
        rp = (
            np.zeros(len(p_rep))
            if prev is None
            else np.linalg.norm(
                p_rep - (prev[0] + prev[1] * cfg.dt), axis=1
            )
        )
        rq = quarantine_scores(Fs)
        return rk, rp, rq, Fs


def hocbf_qp(p_i, v_i, u_nom, p_o, v_o, d_eff, cfg, rho=1e3):
    info = dict(solved=0, active=0, relaxed=0, failed=0)
    rel_p, rel_v = p_i - p_o, v_i - v_o
    dist = np.linalg.norm(rel_p, axis=1)
    near = dist < cfg.sense_range
    if not near.any():
        return u_nom, info

    rel_p, rel_v, d_eff = rel_p[near], rel_v[near], d_eff[near]
    h = (rel_p ** 2).sum(1) - d_eff ** 2
    hd = 2 * (rel_p * rel_v).sum(1)
    A = 2 * rel_p
    b = (
        -2 * (rel_v ** 2).sum(1)
        - 2 * cfg.zeta * cfg.omega * hd
        - cfg.omega ** 2 * h
    )
    if np.all(A @ u_nom - b >= 0):
        return u_nom, info

    m = len(b)
    info["solved"] = 1
    x0 = np.r_[u_nom, np.maximum(0, b - A @ u_nom)]

    def f(x):
        return (
            np.sum((x[:3] - u_nom) ** 2)
            + rho * np.sum(x[3:] ** 2)
        )

    def g(x):
        gr = np.zeros_like(x)
        gr[:3] = 2 * (x[:3] - u_nom)
        gr[3:] = 2 * rho * x[3:]
        return gr

    cons = {
        "type": "ineq",
        "fun": lambda x: A @ x[:3] + x[3:] - b,
        "jac": lambda x: np.hstack([A, np.eye(m)]),
    }
    bnds = [(-cfg.umax, cfg.umax)] * 3 + [(0, None)] * m

    try:
        res = minimize(
            f, x0, jac=g, constraints=[cons], bounds=bnds,
            method="SLSQP",
            options={"maxiter": 60, "ftol": 1e-8},
        )
        if not res.success:
            info["failed"] = 1
            return u_nom, info
        info["active"] = int(
            np.linalg.norm(res.x[:3] - u_nom) > 1e-3
        )
        info["relaxed"] = int(res.x[3:].max() > 1e-3)
        return res.x[:3], info
    except Exception:
        info["failed"] = 1
        return u_nom, info


def simulate(cfg, sc, controller, est, log_evidence=False):
    N, T, dt, d_min = cfg.n_agents, cfg.n_steps, cfg.dt, cfg.d_min
    p, v = sc["A"].copy(), np.zeros((N, 3))
    comp = sc["comp"]
    is_comp = np.zeros(N, bool)
    is_comp[comp] = True
    prev = None
    tau_s = np.ones(N)
    iu = np.triu_indices(N, 1)
    pair_type = is_comp[iu[0]] | is_comp[iu[1]]
    viol = dict(hh=0, hc=0, hh_win=0, hc_win=0)
    coll = 0
    min_sep_t = np.zeros(T)
    prog = np.zeros(T)
    qp = dict(solves=0, active=0, relaxed=0, failed=0, agent_steps=0)
    filt_time = 0.0

    log = None
    if log_evidence:
        log = dict(
            sid=np.zeros((T, N)),
            rk=np.zeros((T, N)),
            rp=np.zeros((T, N)),
            rq=np.zeros((T, N)),
            tau=np.zeros((T, N)),
            F=np.zeros((T, N, 6 + cfg.d_emb)),
            lab=np.zeros((T, N), int),
            rep_p=np.zeros((T, N, 3)),
            rep_v=np.zeros((T, N, 3)),
            rep_y=np.zeros((T, N, cfg.d_emb)),
        )

    rep_hist = dict(p=[], v=[], y=[])
    large_d = d_min * (1 + cfg.lambda_tau * 0.5)

    for t in range(T):
        att = bool(sc["attack"] and sc["t1"] <= t < sc["t2"])
        p_rep, v_rep = p.copy(), v.copy()
        y_rep = sc["y0"] + sc["emb_noise"][t]
        sid = np.full(N, cfg.s_id_honest)

        if att and len(comp):
            p_rep[comp] += sc["bias"][comp] + sc["white"][t, comp]
            y_rep[comp] = sc["y0"][comp] + sc["poison"][t, comp]
            if sc["mode"] == "overt":
                sid[comp] = cfg.s_id_comp

        rep_hist["p"].append(p_rep.copy())
        rep_hist["v"].append(v_rep.copy())
        rep_hist["y"].append(y_rep.copy())

        if est is not None:
            rk, rp, rq, Fs = est.evidence(
                p_rep, v_rep, y_rep, prev
            )
            ek, er, eq = (
                est._n(rk, "k"),
                est._n(rp, "r"),
                est._n(rq, "q"),
            )
            tau = est.tau_from(sid, ek, er, eq)
            tau_s = (
                tau.copy()
                if t == 0
                else cfg.ema * tau_s + (1 - cfg.ema) * tau
            )
            if log is not None:
                log["sid"][t] = sid
                log["rk"][t] = rk
                log["rp"][t] = rp
                log["rq"][t] = rq
                log["tau"][t] = tau
                log["F"][t] = Fs
                log["lab"][t] = (is_comp & att).astype(int)

        prev = (p_rep.copy(), v_rep.copy())
        g = goal_at(sc, t, cfg)
        u_nom = np.clip(
            cfg.kp * (g - p) - cfg.kd * v,
            -cfg.umax, cfg.umax,
        )
        u = u_nom.copy()

        if controller != "nominal":
            ts = time.perf_counter()
            for i in range(N):
                if is_comp[i] and att:
                    continue
                o = np.arange(N) != i
                if controller == "fixed":
                    d_eff = np.full(N - 1, d_min)
                elif controller == "fixed_large":
                    d_eff = np.full(N - 1, large_d)
                else:
                    d_eff = d_min * (
                        1 + cfg.lambda_tau * (1 - tau_s[o])
                    )
                u[i], info = hocbf_qp(
                    p[i], v[i], u_nom[i],
                    p_rep[o], v_rep[o], d_eff, cfg
                )
                qp["solves"] += info["solved"]
                qp["active"] += info["active"]
                qp["relaxed"] += info["relaxed"]
                qp["failed"] += info["failed"]
                qp["agent_steps"] += 1
            filt_time += time.perf_counter() - ts

        if att and len(comp):
            u[comp] = np.clip(
                u_nom[comp] + sc["maneuver"][t, comp],
                -cfg.umax, cfg.umax,
            )

        D = np.linalg.norm(
            p[iu[0]] - p[iu[1]], axis=1
        )
        bad = D < d_min
        viol["hh"] += int((bad & ~pair_type).sum())
        viol["hc"] += int((bad & pair_type).sum())
        if att:
            viol["hh_win"] += int((bad & ~pair_type).sum())
            viol["hc_win"] += int((bad & pair_type).sum())
        coll += int((D < 0.5 * d_min).sum())
        min_sep_t[t] = D.min()
        hon = ~is_comp
        prog[t] = np.linalg.norm(
            p[hon] - g[hon], axis=1
        ).mean()

        v = v + (u + sc["proc"][t]) * dt
        p = p + v * dt

    if log is not None:
        log["rep_p"] = np.array(rep_hist["p"])
        log["rep_v"] = np.array(rep_hist["v"])
        log["rep_y"] = np.array(rep_hist["y"])

    return dict(
        viol=viol,
        collisions=coll,
        min_sep_mean=float(min_sep_t.mean()),
        min_sep_global=float(min_sep_t.min()),
        goal_err=float(prog.mean()),
        qp=qp,
        filter_ms_per_step=filt_time / T * 1000,
        log=log,
    )


def _clean_reports(cfg, sc):
    r = simulate(
        cfg, dict(sc, attack=False),
        "nominal", None, log_evidence=True
    )
    log = r["log"]
    return log["rep_p"], log["rep_v"], log["rep_y"]


def calibrate_estimator(cfg, seed):
    sc = make_scenario(
        cfg, seed + 1000, attack=False, mode="overt"
    )
    rep = _clean_reports(cfg, sc)
    return TrustEstimator(cfg).fit(*rep, seed=seed)


def auc_safe(y, s):
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def swarm_detection(cfg, est, log, seed):
    T, N = log["lab"].shape
    sp = int(0.6 * T)
    lab = log["lab"]
    ytr, yte = lab[:sp].ravel(), lab[sp:].ravel()
    out = dict(
        n_pos_train=int(ytr.sum()),
        n_pos_test=int(yte.sum()),
        n_test=int(len(yte)),
    )

    ek = est._n(log["rk"], "k")
    er = est._n(log["rp"], "r")
    eq = est._n(log["rq"], "q")
    sid = log["sid"]

    full = est.tau_from(sid, ek, er, eq)
    noid = est.tau_from(
        np.full_like(sid, cfg.s_id_honest), ek, er, eq
    )
    only_id = est.tau_from(
        sid, 0 * ek, 0 * er, 0 * eq
    )
    y_all = lab.ravel()

    out["auc_evidence"] = {
        "1-s_id": auc_safe(y_all, (1 - sid).ravel()),
        "curvature": auc_safe(y_all, ek.ravel()),
        "physics": auc_safe(y_all, er.ravel()),
        "quarantine": auc_safe(y_all, eq.ravel()),
        "1-tau (full)": auc_safe(y_all, (1 - full).ravel()),
        "1-tau (no identity)": auc_safe(
            y_all, (1 - noid).ravel()
        ),
        "1-tau (identity only)": auc_safe(
            y_all, (1 - only_id).ravel()
        ),
    }

    out["trust"] = dict(
        attack=float(full[lab == 1].mean()) if (lab == 1).any() else float("nan"),
        honest=float(full[lab == 0].mean()),
    )
    out["trust_noid"] = dict(
        attack=float(noid[lab == 1].mean()) if (lab == 1).any() else float("nan"),
        honest=float(noid[lab == 0].mean()),
    )
    out["quarantine"] = dict(
        mean=float(log["rq"].mean()),
        max=float(log["rq"].max()),
    )
    out["phys_resid"] = dict(
        all=float(log["rp"][1:].mean()),
        attack=float(log["rp"][lab == 1].mean()) if (lab == 1).any() else float("nan"),
        normal=float(log["rp"][1:][lab[1:] == 0].mean()),
    )

    res = {}
    Ftr = log["F"][:sp].reshape(sp * N, -1)
    Fte = log["F"][sp:].reshape((T - sp) * N, -1)
    rk_tr = log["rk"][:sp].reshape(-1, 1)
    rk_te = log["rk"][sp:].reshape(-1, 1)
    ev_tr = np.hstack([
        rk_tr,
        log["rp"][:sp].reshape(-1, 1),
        log["rq"][:sp].reshape(-1, 1),
    ])
    ev_te = np.hstack([
        rk_te,
        log["rp"][sp:].reshape(-1, 1),
        log["rq"][sp:].reshape(-1, 1),
    ])

    def rf():
        return RandomForestClassifier(
            n_estimators=80,
            max_depth=10,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )

    can_train = (
        len(np.unique(ytr)) == 2
        and len(np.unique(yte)) == 2
    )
    models = {}
    if can_train:
        specs = [
            ("RF features-only", Ftr, Fte),
            (
                "RF + curvature",
                np.hstack([Ftr, rk_tr]),
                np.hstack([Fte, rk_te]),
            ),
            (
                "RF + all evidence",
                np.hstack([Ftr, ev_tr]),
                np.hstack([Fte, ev_te]),
            ),
        ]
        for nm, a, b in specs:
            m = rf().fit(a, ytr)
            pr = m.predict_proba(b)[:, 1]
            res[nm] = binary_metrics(
                yte, (pr > 0.5).astype(int), pr
            )
            models[nm] = (m, b)

    tau_te = full[sp:].ravel()
    res["Trust threshold (tau<0.5)"] = binary_metrics(
        yte, (tau_te < 0.5).astype(int), 1 - tau_te
    )
    res["Physics conformal"] = binary_metrics(
        yte,
        (log["rp"][sp:].ravel() > est.conf["r"]).astype(int),
        log["rp"][sp:].ravel(),
    )
    res["Curvature conformal"] = binary_metrics(
        yte,
        (log["rk"][sp:].ravel() > est.conf["k"]).astype(int),
        log["rk"][sp:].ravel(),
    )
    out["detect"] = res

    if can_train:
        pos = np.where(yte == 1)[0]
        Xa = Fte[pos]
        ra = rk_te[pos]
        Xadv, radv = adaptive_residual_attack(
            Xa, est.curv, cfg.adv_eps, cfg.adv_trials, seed
        )
        Xrnd = (
            Xa + np.random.RandomState(seed + 5)
            .normal(0, cfg.adv_eps, Xa.shape)
        )
        rrnd = est.curv.transform(Xrnd)
        m_full, _ = models["RF + curvature"]
        m_rf, _ = models["RF features-only"]

        def rc(X, r):
            return float(
                m_full.predict(np.hstack([X, r])).mean()
            )

        out["adaptive"] = dict(
            resid_clean=float(ra.mean()),
            resid_random=float(rrnd.mean()),
            resid_adaptive=float(radv.mean()),
            recall_full=dict(
                clean=rc(Xa, ra),
                random=rc(Xrnd, rrnd),
                adaptive=rc(Xadv, radv),
            ),
            recall_rf=dict(
                clean=float(m_rf.predict(Xa).mean()),
                random=float(m_rf.predict(Xrnd).mean()),
                adaptive=float(m_rf.predict(Xadv).mean()),
            ),
        )
    return out


def run_swarm_seed(cfg, seed, mode):
    est = calibrate_estimator(cfg, seed)
    sc = make_scenario(cfg, seed, attack=True, mode=mode)
    ctrl = {}
    log = None

    for c in ["nominal", "fixed", "fixed_large", "trust"]:
        r = simulate(
            cfg, sc, c,
            est if c in ("nominal", "trust") else None,
            log_evidence=(c == "nominal"),
        )
        if c == "nominal":
            log = r["log"]
        r["log"] = None
        ctrl[c] = r

    det = swarm_detection(cfg, est, log, seed)
    comp = sc["comp"]
    hon = np.setdiff1d(np.arange(cfg.n_agents), comp)
    traj = dict(
        comp=log["tau"][:, comp].mean(1).tolist(),
        honest=log["tau"][:, hon].mean(1).tolist(),
    )
    return dict(
        seed=seed,
        mode=mode,
        controllers=ctrl,
        detection=det,
        traj=traj,
        window=(sc["t1"], sc["t2"]),
        n_comp=int(len(comp)),
    )


def run_scaling(cfg):
    rows = []
    rng = np.random.RandomState(0)
    for N in cfg.scaling_N:
        cdict = asdict(cfg)
        cdict["n_agents"] = N
        c = Cfg(**cdict)
        radius = 1.2 * N ** (1 / 3) * cfg.d_min
        times = []

        for _ in range(cfg.scaling_reps):
            p = rng.randn(N, 3)
            p = (
                p / np.linalg.norm(p, axis=1, keepdims=True)
                * radius * rng.uniform(0.3, 1, (N, 1))
            )
            v = rng.randn(N, 3) * 0.5
            u_nom = np.clip(
                -0.5 * p, -cfg.umax, cfg.umax
            )
            t0 = time.perf_counter()
            for i in range(N):
                o = np.arange(N) != i
                hocbf_qp(
                    p[i], v[i], u_nom[i],
                    p[o], v[o],
                    np.full(N - 1, cfg.d_min * 2.0),
                    c,
                )
            times.append(
                (time.perf_counter() - t0) * 1000
            )
        rows.append(
            (N, float(np.mean(times)), float(np.std(times)))
        )
    return rows


# =============================================================================
# Reporting
# =============================================================================
METHODS = [
    "RF features-only",
    "Logistic Regression",
    "Gradient Boosting",
    "Residual conformal",
    "CurvTrust-PAI (full)",
]


def report_digital(all_res):
    for name, runs in all_res.items():
        n = len(runs)
        print(
            f"\n>>> {name}: {n} seeds, n={runs[0]['n']}, "
            f"train={runs[0]['n_train']}, test={runs[0]['n_test']}, "
            f"attack rate={runs[0]['attack_rate']:.3f}"
        )
        rows = []
        for m in METHODS:
            rows.append(
                [m]
                + [
                    fmt_ms([r["clean"][m][k] for r in runs])
                    for k in ("acc", "f1", "auc")
                ]
            )
        print_table(
            f"Detection performance — {name} (mean±std over {n} seeds)",
            ["Method", "Accuracy", "F1", "AUC"],
            rows,
        )

        ps = paired_stats(
            [r["clean"]["CurvTrust-PAI (full)"]["f1"] for r in runs],
            [r["clean"]["RF features-only"]["f1"] for r in runs],
        )
        print(
            f"Paired F1 difference (full − RF): {ps['mean_diff']:+.4f} "
            f"95% CI [{ps['ci_lo']:+.4f}, {ps['ci_hi']:+.4f}] "
            f"p={ps['p']:.4g}"
        )
        print(
            f"Train time: {fmt_ms([r['train_s'] for r in runs], 2)} s | "
            f"Inference: {fmt_ms([r['pred_ms'] for r in runs], 4)} ms/sample | "
            f"RF: {fmt_ms([r['pred_ms_rf'] for r in runs], 4)} ms/sample"
        )

        if runs[0]["cv"]:
            print(
                f"5-fold CV F1: full "
                f"{fmt_ms([r['cv']['full'] for r in runs])} | RF "
                f"{fmt_ms([r['cv']['rf'] for r in runs])}"
            )

        rows = [
            [m]
            + [
                fmt_ms([r["noise"][e][m] for r in runs], 3)
                for e in NOISE_EPS
            ]
            for m in METHODS
        ]
        print_table(
            f"F1 under Gaussian noise — {name}",
            ["Method"] + [f"eps={e}" for e in NOISE_EPS],
            rows,
        )

        a = [r["adaptive"] for r in runs]
        print(
            "Adaptive residual-suppression attack:"
            f"\n  residual clean={fmt_ms([x['resid_clean'] for x in a],4)}"
            f" random={fmt_ms([x['resid_random'] for x in a],4)}"
            f" adaptive={fmt_ms([x['resid_adaptive'] for x in a],4)}"
        )


def report_swarm(sw):
    for mode, runs in sw.items():
        print("\n" + "#" * 100)
        print(f"# SWARM — attack mode: {mode} — {len(runs)} seeds")
        print("#" * 100)

        labels = dict(
            nominal="Nominal (no CBF)",
            fixed="Fixed-margin HOCBF",
            fixed_large="Fixed-LARGE-margin HOCBF",
            trust="Trust-adaptive HOCBF (C1)",
        )
        rows = []
        for c, nm in labels.items():
            g = lambda f: [f(r["controllers"][c]) for r in runs]
            rows.append([
                nm,
                fmt_ms(g(lambda x: x["viol"]["hh"] + x["viol"]["hc"]), 1),
                fmt_ms(g(lambda x: x["viol"]["hc"]), 1),
                fmt_ms(g(lambda x: x["viol"]["hh"]), 1),
                fmt_ms(g(lambda x: x["collisions"]), 1),
                fmt_ms(g(lambda x: x["min_sep_mean"]), 3),
                fmt_ms(g(lambda x: x["min_sep_global"]), 3),
                fmt_ms(g(lambda x: x["goal_err"]), 3),
            ])

        print_table(
            "Safety results",
            [
                "Controller", "Violations", "hc", "hh",
                "Collisions", "Mean min-sep",
                "Global min-sep", "Goal error",
            ],
            rows,
        )

        d = [r["detection"] for r in runs]
        keys = list(d[0]["auc_evidence"].keys())
        print_table(
            "AUC of evidence streams",
            ["Evidence", "AUC"],
            [
                [k, fmt_ms([x["auc_evidence"][k] for x in d])]
                for k in keys
            ],
        )

        dm = list(d[0]["detect"].keys())
        print_table(
            "Swarm detection",
            ["Method", "Accuracy", "F1", "AUC"],
            [
                [m]
                + [
                    fmt_ms([x["detect"][m][k] for x in d])
                    for k in ("acc", "f1", "auc")
                ]
                for m in dm
            ],
        )


def make_figures(dig, sw, scaling, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)

    if dig:
        names = list(dig.keys())
        fig, ax = plt.subplots(
            figsize=(max(7, 2.4 * len(names)), 4.5)
        )
        w = 0.8 / len(METHODS)
        for j, m in enumerate(METHODS):
            mu = [
                np.mean([r["clean"][m]["f1"] for r in dig[n]])
                for n in names
            ]
            sd = [
                np.std([r["clean"][m]["f1"] for r in dig[n]])
                for n in names
            ]
            ax.bar(
                np.arange(len(names)) + j * w - 0.4 + w / 2,
                mu, w, yerr=sd, label=m, capsize=2
            )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 (mean ± std)")
        ax.set_title("Per-dataset F1")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        fig.savefig(
            fig_dir / "fig_per_dataset_performance.png",
            dpi=150,
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m in METHODS:
            # aggregate over datasets and seeds
            mu = []
            sd = []
            for e in NOISE_EPS:
                vals = [
                    r["noise"][e][m]
                    for n in names for r in dig[n]
                ]
                mu.append(np.mean(vals))
                sd.append(np.std(vals))
            ax.errorbar(
                NOISE_EPS, mu, yerr=sd,
                marker="o", label=m, capsize=2
            )
        ax.set_xlabel("Noise epsilon")
        ax.set_ylabel("F1")
        ax.set_title("Noise robustness")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(
            fig_dir / "fig_noise_robustness.png",
            dpi=150,
        )
        plt.close(fig)

    if sw:
        ctl = ["nominal", "fixed", "fixed_large", "trust"]
        lab = ["Nominal", "Fixed", "Fixed-large", "Trust-adaptive"]
        for fname, getter, ylabel, title in [
            (
                "fig_min_separation.png",
                lambda x: x["min_sep_mean"],
                "Mean minimum true separation",
                "Mean minimum separation",
            ),
            (
                "fig_separation_violations.png",
                lambda x: x["viol"]["hh"] + x["viol"]["hc"],
                "Violations",
                "Separation violations",
            ),
        ]:
            modes = list(sw.keys())
            fig, axs = plt.subplots(
                1, len(modes),
                figsize=(5.2 * len(modes), 4),
                squeeze=False,
            )
            for ax, mode in zip(axs[0], modes):
                mu = [
                    np.mean([
                        getter(r["controllers"][c])
                        for r in sw[mode]
                    ])
                    for c in ctl
                ]
                sd = [
                    np.std([
                        getter(r["controllers"][c])
                        for r in sw[mode]
                    ])
                    for c in ctl
                ]
                ax.bar(lab, mu, yerr=sd, capsize=3)
                ax.set_title(f"{mode} attack")
                ax.tick_params(axis="x", rotation=15)
                ax.grid(axis="y", alpha=.3)
            axs[0][0].set_ylabel(ylabel)
            fig.suptitle(title)
            fig.tight_layout()
            fig.savefig(fig_dir / fname, dpi=150)
            plt.close(fig)

        modes = list(sw.keys())
        fig, axs = plt.subplots(
            1, len(modes),
            figsize=(5.2 * len(modes), 3.8),
            squeeze=False,
        )
        for ax, mode in zip(axs[0], modes):
            r0 = sw[mode][0]
            ax.plot(
                np.mean(
                    [r["traj"]["honest"] for r in sw[mode]], 0
                ),
                label="honest",
            )
            ax.plot(
                np.mean(
                    [r["traj"]["comp"] for r in sw[mode]], 0
                ),
                label="compromised",
            )
            ax.axvspan(
                *r0["window"], alpha=.2, label="attack window"
            )
            ax.set_title(f"{mode} attack")
            ax.set_xlabel("step")
            ax.set_ylim(0, 1)
            ax.grid(alpha=.3)
        axs[0][0].set_ylabel("mean trust tau")
        axs[0][0].legend(fontsize=7)
        fig.suptitle("Per-agent trust over time")
        fig.tight_layout()
        fig.savefig(
            fig_dir / "fig_trust_timeline.png",
            dpi=150,
        )
        plt.close(fig)

    if scaling:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        Ns, mu, sd = zip(*scaling)
        ax.errorbar(Ns, mu, yerr=sd, marker="o", capsize=3)
        ax.set_xlabel("number of agents N")
        ax.set_ylabel("full filter step (ms)")
        ax.set_title("HOCBF-QP wall-clock scaling")
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(
            fig_dir / "fig_hocbf_scaling.png",
            dpi=150,
        )
        plt.close(fig)


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def save_csv_outputs(dig, sw, scaling, out_dir):
    # Clean digital metrics
    rows = []
    for dataset, runs in dig.items():
        for r in runs:
            for method, met in r["clean"].items():
                rows.append({
                    "dataset": dataset,
                    "seed": r["seed"],
                    "method": method,
                    **met,
                    "train_s": r["train_s"],
                    "pred_ms": r["pred_ms"],
                })
    if rows:
        pd.DataFrame(rows).to_csv(
            out_dir / "digital_results.csv", index=False
        )

    rows = []
    for dataset, runs in dig.items():
        for r in runs:
            for eps, methods in r["noise"].items():
                for method, f1 in methods.items():
                    rows.append({
                        "dataset": dataset,
                        "seed": r["seed"],
                        "epsilon": eps,
                        "method": method,
                        "f1": f1,
                    })
    if rows:
        pd.DataFrame(rows).to_csv(
            out_dir / "noise_robustness.csv", index=False
        )

    rows = []
    for mode, runs in sw.items():
        for r in runs:
            for controller, x in r["controllers"].items():
                rows.append({
                    "mode": mode,
                    "seed": r["seed"],
                    "controller": controller,
                    "viol_hh": x["viol"]["hh"],
                    "viol_hc": x["viol"]["hc"],
                    "collisions": x["collisions"],
                    "min_sep_mean": x["min_sep_mean"],
                    "min_sep_global": x["min_sep_global"],
                    "goal_err": x["goal_err"],
                    "filter_ms_per_step": x["filter_ms_per_step"],
                })
    if rows:
        pd.DataFrame(rows).to_csv(
            out_dir / "swarm_safety.csv", index=False
        )

    if scaling:
        pd.DataFrame(
            scaling, columns=["N", "ms_mean", "ms_std"]
        ).to_csv(out_dir / "scaling.csv", index=False)


def run_seed_jobs(X, y, src, name, cfg):
    if cfg.workers <= 1:
        return [
            run_digital_seed(X, y, src, name, s, cfg)
            for s in cfg.seeds
        ]

    print(
        f"[INFO] Running {len(cfg.seeds)} seeds using "
        f"{cfg.workers} process workers."
    )
    results = []
    with ProcessPoolExecutor(
        max_workers=cfg.workers
    ) as ex:
        futs = {
            ex.submit(
                run_digital_seed,
                X, y, src, name, s, cfg
            ): s
            for s in cfg.seeds
        }
        for f in as_completed(futs):
            seed = futs[f]
            result = f.result()
            results.append(result)
            print(f"[WORKER] seed={seed} completed")
    return sorted(results, key=lambda r: r["seed"])


def validate_files(keys):
    required = []
    if "unsw" in keys:
        required += ["unsw_train", "unsw_test"]
    if "nf-ton" in keys:
        required += ["nf_ton"]
    if "nf-cse" in keys:
        required += ["nf_cse"]
    if "cidds" in keys:
        required += ["cidds_external", "cidds_openstack"]

    if "edgeiot" in keys:
        if not EDGEIOT_DIR.exists():
            print(f"\n[ERROR] Missing Custom EdgeIoT folder: {EDGEIOT_DIR}")
            raise SystemExit(2)
        edge_csvs = sorted(EDGEIOT_DIR.rglob("*.csv"))
        if not edge_csvs:
            print(f"\n[ERROR] No CSV files found in: {EDGEIOT_DIR}")
            raise SystemExit(2)
        print(f"[CHECK] Custom EdgeIoT: {len(edge_csvs)} CSV file(s)")
        for p in edge_csvs:
            print(f"        {p.name}")

    missing = [
        str(DATA_FILES[k])
        for k in required
        if not DATA_FILES[k].exists()
    ]
    if missing:
        print("\n[ERROR] Missing required dataset file(s):")
        for p in missing:
            print("  -", p)
        print(
            "\nIf Jupyter's actual server root differs from /Curve-Test, "
            "edit BASE_DIR at the top of this file."
        )
        raise SystemExit(2)



# =============================================================================
# Extended Custom-EdgeIoT publication analyses
# =============================================================================
def _edgeiot_train_eval_simple(X, y, seed, cfg, n_total=None):
    """Targeted CurvTrust/RF evaluation used by learning/saturation figures."""
    rng = np.random.RandomState(seed)
    if n_total is not None and len(y) > n_total:
        idx = rng.choice(len(y), int(n_total), replace=False)
        X0, y0 = X[idx], y[idx]
    else:
        X0, y0 = X, y

    X0 = X0[:, X0.std(axis=0) > 0]
    tr, te = train_test_split(
        np.arange(len(y0)), test_size=0.25, random_state=seed, stratify=y0
    )
    Xtr, Xte, ytr, yte = X0[tr], X0[te], y0[tr], y0[te]

    scaler = RobustScaler().fit(Xtr)
    A = prep_scale(scaler, Xtr)
    B = prep_scale(scaler, Xte)

    curv = CurvatureResidual(cfg.k, cfg.gamma, cfg.n_ref, seed=seed)
    rA = curv.fit_transform(A)
    rB = curv.transform(B)

    def rf():
        return RandomForestClassifier(
            n_estimators=cfg.rf_trees, max_depth=cfg.rf_depth,
            min_samples_leaf=4, class_weight="balanced_subsample",
            random_state=seed, n_jobs=1,
        )

    m_full = rf().fit(np.hstack([A, rA]), ytr)
    m_rf = rf().fit(A, ytr)

    pf = m_full.predict(np.hstack([B, rB]))
    pr = m_rf.predict(B)
    f_full = precision_recall_fscore_support(
        yte, pf, average="binary", zero_division=0
    )[2]
    f_rf = precision_recall_fscore_support(
        yte, pr, average="binary", zero_division=0
    )[2]
    return float(f_full), float(f_rf)


def run_edgeiot_extended_analysis(X, y, cfg, out_dir, existing_runs=None):
    """
    Produce measured publication-style diagnostics for Custom-EdgeIoT.

    Creates:
      Custom-EdgeIoT_alpha_sensitivity.png
      Custom-EdgeIoT_data_saturation.png
      Custom-EdgeIoT_learning_curves.png
      Custom-EdgeIoT_update_diagnostics.png
      worker_dataset_comparison.png
      figures/Custom-EdgeIoT/*.png
      extended_analysis.csv
    """
    print("\n" + "=" * 100)
    print("CUSTOM-EDGEIOT EXTENDED FIGURE ANALYSIS")
    print("=" * 100)

    fig_root = Path(out_dir) / "figures"
    detail = fig_root / "Custom-EdgeIoT"
    detail.mkdir(parents=True, exist_ok=True)

    # Keep the extended analysis practical while still using multiple seeds.
    ext_seeds = cfg.seeds
    rows = []

    # -------------------------------------------------------------------------
    # A. Alpha sensitivity: alpha is the conformal significance level in this
    # implementation. It changes the residual-conformal decision threshold.
    # CurvTrust full F1 is included as a reference, not claimed to depend on alpha.
    # -------------------------------------------------------------------------
    alpha_values = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20]
    alpha_res = {a: {"Residual conformal": [], "CurvTrust-PAI (full)": []}
                 for a in alpha_values}

    # Use one fixed split per seed and recompute conformal thresholds from normal
    # calibration residuals. This avoids fabricating alpha-dependent full-model results.
    for seed in ext_seeds:
        Xs, ys, _ = subsample(
            X, y, None, cfg.max_samples.get("Custom-EdgeIoT", 30000), seed
        )
        Xs = Xs[:, Xs.std(axis=0) > 0]
        tr, te = train_test_split(
            np.arange(len(ys)), test_size=0.25,
            random_state=seed, stratify=ys
        )
        Xtr_raw, Xte_raw = Xs[tr], Xs[te]
        ytr, yte = ys[tr], ys[te]
        rng = np.random.RandomState(seed)
        normal = np.where(ytr == 0)[0]
        n_cal = min(len(normal)-1, max(10, int(cfg.cal_frac * len(normal))))
        cal_idx = rng.choice(normal, n_cal, replace=False)
        fit_idx = np.setdiff1d(np.arange(len(ytr)), cal_idx)

        sc = RobustScaler().fit(Xtr_raw[fit_idx])
        Af = prep_scale(sc, Xtr_raw[fit_idx])
        Ac = prep_scale(sc, Xtr_raw[cal_idx])
        Bt = prep_scale(sc, Xte_raw)
        cc = CurvatureResidual(cfg.k, cfg.gamma, cfg.n_ref, seed=seed)
        rf_fit = cc.fit_transform(Af)
        rcal = cc.transform(Ac).ravel()
        rte = cc.transform(Bt)

        model = RandomForestClassifier(
            n_estimators=cfg.rf_trees, max_depth=cfg.rf_depth,
            min_samples_leaf=4, class_weight="balanced_subsample",
            random_state=seed, n_jobs=1,
        ).fit(np.hstack([Af, rf_fit]), ytr[fit_idx])
        full_pred = model.predict(np.hstack([Bt, rte]))
        full_f1 = precision_recall_fscore_support(
            yte, full_pred, average="binary", zero_division=0
        )[2]

        for a in alpha_values:
            tau = conformal_threshold(rcal, a)
            cp = (rte.ravel() > tau).astype(int)
            cf1 = precision_recall_fscore_support(
                yte, cp, average="binary", zero_division=0
            )[2]
            alpha_res[a]["Residual conformal"].append(float(cf1))
            alpha_res[a]["CurvTrust-PAI (full)"].append(float(full_f1))
            rows.append(dict(
                analysis="alpha_sensitivity", seed=seed, x=a,
                metric="f1", method="Residual conformal", value=float(cf1)
            ))
            rows.append(dict(
                analysis="alpha_sensitivity", seed=seed, x=a,
                metric="f1", method="CurvTrust-PAI (full)", value=float(full_f1)
            ))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for method in ["Residual conformal", "CurvTrust-PAI (full)"]:
        mu = [np.mean(alpha_res[a][method]) for a in alpha_values]
        sd = [np.std(alpha_res[a][method]) for a in alpha_values]
        ax.errorbar(alpha_values, mu, yerr=sd, marker="o", capsize=3, label=method)
    ax.set_xlabel("Conformal significance level α")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Custom-EdgeIoT: α sensitivity")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    for p in [fig_root / "Custom-EdgeIoT_alpha_sensitivity.png",
              detail / "alpha_sensitivity.png"]:
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # B/C. Data saturation and learning curves.
    # These are genuine retraining experiments at increasing dataset sizes.
    # -------------------------------------------------------------------------
    sizes = [2500, 5000, 10000, 20000, 30000]
    sizes = [s for s in sizes if s <= len(y)]
    sat = {s: {"CurvTrust-PAI (full)": [], "RF features-only": []} for s in sizes}

    for s in sizes:
        print(f"[EXT] learning/data saturation n={s}")
        for seed in ext_seeds:
            ff, fr = _edgeiot_train_eval_simple(X, y, seed, cfg, n_total=s)
            sat[s]["CurvTrust-PAI (full)"].append(ff)
            sat[s]["RF features-only"].append(fr)
            rows.append(dict(
                analysis="data_saturation", seed=seed, x=s,
                metric="f1", method="CurvTrust-PAI (full)", value=ff
            ))
            rows.append(dict(
                analysis="data_saturation", seed=seed, x=s,
                metric="f1", method="RF features-only", value=fr
            ))

    def plot_learning(stem, title):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for method in ["CurvTrust-PAI (full)", "RF features-only"]:
            mu = [np.mean(sat[s][method]) for s in sizes]
            sd = [np.std(sat[s][method]) for s in sizes]
            ax.errorbar(sizes, mu, yerr=sd, marker="o", capsize=3, label=method)
        ax.set_xlabel("Total samples used")
        ax.set_ylabel("Test F1")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        for p in [fig_root / f"Custom-EdgeIoT_{stem}.png",
                  detail / f"{stem}.png"]:
            fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)

    plot_learning("data_saturation", "Custom-EdgeIoT: data saturation")
    plot_learning("learning_curves", "Custom-EdgeIoT: learning curves")

    # -------------------------------------------------------------------------
    # D. Update/adaptive-attack diagnostics from the actual final seed runs.
    # -------------------------------------------------------------------------
    if existing_runs:
        labels = ["Clean", "Random perturbation", "Adaptive suppression"]
        resid = [
            [r["adaptive"]["resid_clean"] for r in existing_runs],
            [r["adaptive"]["resid_random"] for r in existing_runs],
            [r["adaptive"]["resid_adaptive"] for r in existing_runs],
        ]
        recall_full = [
            [r["adaptive"]["recall_full"]["clean"] for r in existing_runs],
            [r["adaptive"]["recall_full"]["random"] for r in existing_runs],
            [r["adaptive"]["recall_full"]["adaptive"] for r in existing_runs],
        ]

        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        x = np.arange(3)
        ax.bar(x-.18, [np.mean(v) for v in resid], width=.36,
               yerr=[np.std(v) for v in resid], capsize=3, label="Mean residual")
        ax2 = ax.twinx()
        ax2.bar(x+.18, [np.mean(v) for v in recall_full], width=.36,
                yerr=[np.std(v) for v in recall_full], capsize=3,
                label="CurvTrust attack recall")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Residual magnitude")
        ax2.set_ylabel("Attack recall")
        ax2.set_ylim(0, 1.05)
        ax.set_title("Custom-EdgeIoT: update / adaptive-attack diagnostics")
        ax.grid(axis="y", alpha=.25)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, loc="best", fontsize=8)
        fig.tight_layout()
        for p in [fig_root / "Custom-EdgeIoT_update_diagnostics.png",
                  detail / "update_diagnostics.png"]:
            fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)

        for r in existing_runs:
            for state, key in [("clean", "clean"), ("random", "random"),
                               ("adaptive", "adaptive")]:
                rows.append(dict(
                    analysis="update_diagnostics", seed=r["seed"], x=state,
                    metric="residual", method="Curvature residual",
                    value=float(r["adaptive"][f"resid_{key}"])
                ))
                rows.append(dict(
                    analysis="update_diagnostics", seed=r["seed"], x=state,
                    metric="recall", method="CurvTrust-PAI (full)",
                    value=float(r["adaptive"]["recall_full"][key])
                ))

    # -------------------------------------------------------------------------
    # E. Worker/dataset-style comparison: clean metrics for this worker dataset.
    # -------------------------------------------------------------------------
    if existing_runs:
        methods = list(existing_runs[0]["clean"].keys())
        f1mu = [np.mean([r["clean"][m]["f1"] for r in existing_runs]) for m in methods]
        f1sd = [np.std([r["clean"][m]["f1"] for r in existing_runs]) for m in methods]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(methods, f1mu, yerr=f1sd, capsize=3)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 (mean ± std)")
        ax.set_title("Custom-EdgeIoT worker dataset comparison")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        fig.savefig(fig_root / "worker_dataset_comparison.png",
                    dpi=300, bbox_inches="tight")
        fig.savefig(detail / "worker_dataset_comparison.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    if rows:
        pd.DataFrame(rows).to_csv(
            Path(out_dir) / "extended_analysis.csv", index=False
        )

    print(f"[EXT] Extended figures saved to: {fig_root}")
    print(f"[EXT] Detailed folder: {detail}")

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default=str(BASE_DIR / "results_dgx")
    )
    ap.add_argument(
        "--seeds", default="42,43,44,45,46"
    )
    ap.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Seed-level process workers. For one DGX Spark GPU, "
            "start with 1 under cuml.accel; use --workers 4 only "
            "after benchmarking."
        ),
    )
    ap.add_argument(
        "--protocol",
        default="random",
        choices=["random", "official"],
    )
    ap.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma list: unsw,nf-ton,nf-cse,cidds,edgeiot",
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-digital", action="store_true")
    ap.add_argument("--skip-swarm", action="store_true")
    ap.add_argument(
        "--extended-analysis", action="store_true",
        help="Run measured Custom-EdgeIoT alpha sensitivity, data saturation, learning curves, and diagnostics."
    )
    args = ap.parse_args(argv)

    keys = [
        k.strip()
        for k in args.datasets.split(",")
        if k.strip()
    ]
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        raise SystemExit(
            f"Unknown dataset key(s): {unknown}. "
            f"Valid: {list(REGISTRY)}"
        )

    validate_files(keys)

    cfg = Cfg(
        seeds=[int(s) for s in args.seeds.split(",")],
        out_dir=args.out_dir,
        protocol=args.protocol,
        workers=max(1, args.workers),
    )

    if args.quick:
        cfg.seeds = cfg.seeds[:2]
        cfg.do_cv = False
        cfg.n_steps = 300
        cfg.max_samples = {
            k: 4000 for k in cfg.max_samples
        }
        cfg.scaling_N = [5, 10, 15]
        cfg.scaling_reps = 3

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CurvTrust-PAI — NVIDIA DGX Spark")
    print(f"Data root : {BASE_DIR}")
    print(f"Output    : {out_dir}")
    print(f"Datasets  : {keys}")
    print(f"Seeds     : {cfg.seeds}")
    print(f"Workers   : {cfg.workers}")
    print(f"Protocol  : {cfg.protocol}")
    print("=" * 100)

    print("\nDataset paths:")
    for k, p in DATA_FILES.items():
        print(f"  {k:<16} {p}")
    print(f"  {'edgeiot_dir':<16} {EDGEIOT_DIR}")

    dig = {}
    coverage = []
    edgeiot_loaded = None

    if not args.skip_digital:
        # Loader pool: enough for later per-seed max_samples while
        # preventing unnecessarily large in-memory frames.
        pool = max(cfg.max_samples.values()) * 3

        for key in keys:
            loader = REGISTRY[key]
            try:
                print(f"\n[LOAD] {key}")
                X, y, name, src = loader(pool)
                if len(np.unique(y)) < 2:
                    raise ValueError(
                        "Only one class present after label conversion."
                    )
                print(
                    f"[RUN] {name}: {X.shape[0]} rows, "
                    f"{X.shape[1]} features, "
                    f"attack rate={y.mean():.3f}"
                )
                dig[name] = run_seed_jobs(
                    X, y, src, name, cfg
                )
                if key == "edgeiot":
                    edgeiot_loaded = (X, y, src, name)
                coverage.append([name, "evaluated"])
            except Exception as e:
                coverage.append([
                    key,
                    f"NOT evaluated — {type(e).__name__}: {e}",
                ])
                print(
                    f"[WARN] {key}: {type(e).__name__}: {e}"
                )

        print_table(
            "DATASET COVERAGE",
            ["Dataset", "Status"],
            coverage,
        )
        if dig:
            report_digital(dig)

    sw = {}
    scaling = None
    if not args.skip_swarm:
        for mode in cfg.attack_modes:
            print(f"\n[RUN] swarm mode={mode}")
            # Swarm is CPU/SLSQP-heavy; run sequentially for reproducibility.
            sw[mode] = [
                run_swarm_seed(cfg, s, mode)
                for s in cfg.seeds
            ]
        report_swarm(sw)
        scaling = run_scaling(cfg)
        print_table(
            "Scaling: one full filter step",
            ["N", "ms mean", "ms std"],
            [
                [n, f"{m:.2f}", f"{s:.2f}"]
                for n, m, s in scaling
            ],
        )

    full_out = {
        "config": to_py(asdict(cfg)),
        "data_files": {
            k: str(v) for k, v in DATA_FILES.items()
        },
        "digital": to_py(dig),
        "swarm": to_py(sw),
        "scaling": to_py(scaling),
        "coverage": coverage,
    }

    (out_dir / "results.json").write_text(
        json.dumps(full_out, indent=2)
    )
    save_csv_outputs(dig, sw, scaling, out_dir)
    make_figures(
        dig, sw, scaling, out_dir / "figures"
    )

    if args.extended_analysis and edgeiot_loaded is not None:
        X_e, y_e, src_e, name_e = edgeiot_loaded
        run_edgeiot_extended_analysis(
            X_e, y_e, cfg, out_dir,
            existing_runs=dig.get(name_e)
        )

    print("\n" + "=" * 100)
    print("RUN COMPLETE")
    print("=" * 100)
    print(f"Full JSON : {out_dir / 'results.json'}")
    print(f"CSV files : {out_dir}")
    print(f"Figures   : {out_dir / 'figures'}")
    print(
        "\nScientific reporting reminders:\n"
        "1. Jaccard overlap is a curvature proxy, not Wasserstein Ollivier-Ricci curvature.\n"
        "2. The swarm is a simulation stand-in.\n"
        "3. In overt mode, s_id contains simulated ground-truth identity degradation.\n"
        "4. Trust uses the calibrated logistic described in the code.\n"
        "5. Report only datasets shown as evaluated."
    )


if __name__ == "__main__":
    main()
