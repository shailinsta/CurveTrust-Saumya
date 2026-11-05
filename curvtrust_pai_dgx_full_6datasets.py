#!/usr/bin/env python3
"""
CurvTrust-PAI — DGX Spark strict six-file runner
=================================================

This runner uses ONLY these six physical Parquet files from ~/Curve-Test:

1. cidds-001-externalserver.parquet
2. cidds-001-openstack.parquet
3. NF-CSE-CIC-IDS2018.parquet
4. NF-ToN-IoT.parquet
5. UNSW_NB15_testing-set.parquet
6. UNSW_NB15_training-set.parquet

They are evaluated as four logical benchmark datasets:
- CIDDS-001 (two physical files combined)
- NF-CSE-CIC-IDS2018-v2
- NF-ToN-IoT-v2
- UNSW-NB15 (official train/test files combined for random protocol,
  or source split retained for --protocol official)

Place this file in the SAME ~/Curve-Test directory as:
    curvtrust_pai_dgx_full.py

The scientific pipeline remains in curvtrust_pai_dgx_full.py. This strict
runner overrides its dataset paths before main() executes, validates all six
files, and prevents accidental use of Edge-IIoT or any other dataset.

Recommended test:
    python -m cuml.accel curvtrust_pai_dgx_full_6datasets.py --quick --workers 1

Full run:
    python -m cuml.accel curvtrust_pai_dgx_full_6datasets.py --workers 1
"""

from pathlib import Path
import sys

BASE_DIR = Path.home() / "Curve-Test"

# Import the full scientific pipeline.
try:
    import curvtrust_pai_dgx_full as pipeline
except ImportError as exc:
    raise SystemExit(
        "\n[ERROR] Could not import curvtrust_pai_dgx_full.py.\n"
        "Put this new runner in the same ~/Curve-Test directory as "
        "curvtrust_pai_dgx_full.py, then run it from there.\n"
        f"Original error: {exc}"
    )

# ---------------------------------------------------------------------------
# STRICT DATASET CONFIGURATION
# ---------------------------------------------------------------------------
# Exact filenames visible in the supplied ~/Curve-Test screenshot.
STRICT_DATA_FILES = {
    "unsw_train": BASE_DIR / "UNSW_NB15_training-set.parquet",
    "unsw_test": BASE_DIR / "UNSW_NB15_testing-set.parquet",
    "nf_ton": BASE_DIR / "NF-ToN-IoT.parquet",
    "nf_cse": BASE_DIR / "NF-CSE-CIC-IDS2018.parquet",
    # NOTE: screenshot filename is "externalsever" (without the second 'r').
    "cidds_external": BASE_DIR / "cidds-001-externalserver.parquet",
    "cidds_openstack": BASE_DIR / "cidds-001-openstack.parquet",
}

# Override all dataset paths used by the imported pipeline.
pipeline.BASE_DIR = BASE_DIR
pipeline.DATA_FILES = STRICT_DATA_FILES

# Keep exactly these four logical loaders (six physical files).
pipeline.REGISTRY = {
    "unsw": pipeline.load_unsw,
    "nf-ton": pipeline.load_nf_ton,
    "nf-cse": pipeline.load_nf_cse,
    "cidds": pipeline.load_cidds,
}
pipeline.DEFAULT_DATASETS = ["unsw", "nf-ton", "nf-cse", "cidds"]


def strict_validate():
    """Require exactly the six intended physical source files."""
    print("=" * 100)
    print("STRICT SIX-FILE DATASET CHECK")
    print("=" * 100)

    missing = []
    for key, path in STRICT_DATA_FILES.items():
        if path.is_file():
            try:
                size_gb = path.stat().st_size / (1024 ** 3)
                print(f"[OK] {key:<16} {path.name:<42} {size_gb:8.3f} GB")
            except OSError:
                print(f"[OK] {key:<16} {path.name}")
        else:
            print(f"[MISSING] {key:<11} {path}")
            missing.append(path)

    if missing:
        print("\n[ERROR] The strict runner will not start because required files are missing.")
        print("Check the exact filenames with:")
        print("    cd ~/Curve-Test && ls -lh *.parquet")
        raise SystemExit(2)

    print("\n[OK] All six required physical dataset files were found.")
    print("[OK] No Edge-IIoT loader is registered.")
    print("[OK] No other Parquet file will be loaded by this runner.")
    print("=" * 100)


def enforce_cli_dataset_scope():
    """
    If --datasets is supplied, reject anything outside the approved four
    logical dataset keys. If omitted, pipeline.main() uses the four approved
    defaults patched above.
    """
    allowed = {"unsw", "nf-ton", "nf-cse", "cidds"}

    if "--datasets" in sys.argv:
        i = sys.argv.index("--datasets")
        if i + 1 >= len(sys.argv):
            raise SystemExit("[ERROR] --datasets requires a comma-separated value.")
        requested = {x.strip() for x in sys.argv[i + 1].split(",") if x.strip()}
        bad = requested - allowed
        if bad:
            raise SystemExit(
                f"[ERROR] Dataset(s) not allowed by strict runner: {sorted(bad)}\n"
                f"Allowed logical datasets: {sorted(allowed)}"
            )


if __name__ == "__main__":
    strict_validate()
    enforce_cli_dataset_scope()
    pipeline.main()
