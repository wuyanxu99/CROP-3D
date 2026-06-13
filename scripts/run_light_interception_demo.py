#!/usr/bin/env python3
"""Regenerate the packaged light-interception demo from one cropped canopy ROI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "examples" / "light_interception_demo"
INPUT = DEMO / "input"
REFERENCE = DEMO / "reference_output"
OUTPUT = REPO_ROOT / "outputs" / "light_interception"


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "crop3d.phenotyping.light_interception.calculate_light_interception",
            "--plant-ply",
            str(INPUT / "plant_local.ply"),
            "--roi-summary",
            str(INPUT / "roi_summary.json"),
            "--model",
            str(INPUT / "ppfd_light_field_model.pkl"),
            "--output-dir",
            str(OUTPUT),
            "--settings",
            "170,205,240",
        ]
    )

    regions_dir = OUTPUT / "regions"
    regions_dir.mkdir(parents=True, exist_ok=True)
    for name in ("region_comparison_setting_240.csv", "region_comparison_setting_240.json"):
        src = DEMO / "five_regions" / name
        if src.exists():
            shutil.copy2(src, regions_dir / name)
    for png in REFERENCE.glob("*.png"):
        shutil.copy2(png, OUTPUT / png.name)

    print(f"Light-interception demo outputs: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

