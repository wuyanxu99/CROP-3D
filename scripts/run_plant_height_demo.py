#!/usr/bin/env python3
"""Regenerate the plant-height demo figures from packaged reference outputs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = REPO_ROOT / "examples" / "plant_height_demo" / "reference_output"
OUTPUT = REPO_ROOT / "outputs" / "plant_height"


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ("heights.csv", "errors.csv", "ordered_heights.txt", "canopy.ply"):
        src = REFERENCE / name
        if src.exists():
            shutil.copy2(src, OUTPUT / name)

    run(
        [
            sys.executable,
            "-m",
            "crop3d.phenotyping.plant_height.plot_test_iteration",
            "--current-csv",
            str(OUTPUT / "heights.csv"),
            "--output-dir",
            str(OUTPUT / "figures" / "metrics"),
            "--label",
            "Adaptive soft growth fixed046 demo",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "crop3d.phenotyping.plant_height.plot_canopy_xz_by_group",
            "--results-csv",
            str(OUTPUT / "heights.csv"),
            "--canopy-ply",
            str(OUTPUT / "canopy.ply"),
            "--qc-dir",
            str(REFERENCE / "qc_all"),
            "--output-dir",
            str(OUTPUT / "figures" / "xz_side"),
            "--max-points-per-group",
            "80000",
        ]
    )
    print(f"Plant-height demo outputs: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

