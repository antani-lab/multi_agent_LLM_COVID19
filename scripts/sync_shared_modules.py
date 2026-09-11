#!/usr/bin/env python3
"""Synchronize notebook-local helper modules with the importable package copies."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    ROOT / "notebooks/stage_B/cxr_metrics.py": ROOT / "src/multi_agent_cxr/metrics.py",
    ROOT / "notebooks/stage_C/stage_c_reasoner.py": ROOT / "src/multi_agent_cxr/reasoner.py",
    ROOT / "notebooks/stage_D/stage_d_stats.py": ROOT / "src/multi_agent_cxr/statistics.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check for drift without writing.")
    args = parser.parse_args()

    drift = []
    for source, target in MODULES.items():
        if source.read_bytes() == target.read_bytes():
            print(f"OK: {target.relative_to(ROOT)}")
            continue
        drift.append((source, target))
        if not args.check:
            shutil.copy2(source, target)
            print(f"UPDATED: {target.relative_to(ROOT)}")

    if args.check and drift:
        for source, target in drift:
            print(f"DRIFT: {target.relative_to(ROOT)} differs from {source.relative_to(ROOT)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
