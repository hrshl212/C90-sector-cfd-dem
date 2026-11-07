#!/usr/bin/env python3
"""Run public-tree structure, link, CSV, and leakage checks."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "NOTICE.md",
    "docs/formulation.md",
    "docs/implementation-overview.md",
    "docs/validation-methods.md",
    "docs/validation-results.md",
    "docs/limitations.md",
    "data/README.md",
    "data/sanitized/particle_retention_timeseries.csv",
    "data/sanitized/particle_count_timeseries.csv",
    "scripts/plot_particle_population.py",
    "scripts/render_snapshot_comparison.py",
    "figures/validation/matched_liquid_flow.png",
    "figures/validation/matched_liquid_velocity_cross_section_z5mm.png",
    "figures/validation/matched_particle_geometry.png",
    "figures/validation/piston_fluid_force_c90_vs_full360_interim.png",
    "figures/validation/particle_count_c90_vs_full360_interim.png",
)
TEXT_SUFFIXES = {".md", ".py", ".cu", ".cpp", ".h", ".sh", ".yml", ".yaml", ".cff", ".txt", ".csv", ".svg"}
PROHIBITED = (
    re.compile("/" + "scratch" + "/", re.IGNORECASE),
    re.compile("private" + "_validation", re.IGNORECASE),
    re.compile(
        r"(?:^|\W)(?:sba" + r"tch|sr" + r"un|sl" + r"urm)(?:\W|$)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:^|\W)(?:gau" + r"tschi|hr" + r"aut)(?:\W|$)", re.IGNORECASE),
    re.compile("hand" + "offs/", re.IGNORECASE),
    re.compile(r"\bjob\s+[0-9]{6,}\b", re.IGNORECASE),
)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_required() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        fail(f"missing required files: {', '.join(missing)}")


def public_text_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and (path.suffix.lower() in TEXT_SUFFIXES or path.name == "LICENSE")
        and ".git" not in path.parts
    )


def check_leakage(files: list[Path]) -> None:
    for path in files:
        text = path.read_text(encoding="utf-8")
        for pattern in PROHIBITED:
            if pattern.search(text):
                fail(f"prohibited identifier in {path.relative_to(ROOT)}")


def check_links(files: list[Path]) -> None:
    for path in files:
        if path.suffix.lower() != ".md":
            continue
        for target in LINK.findall(path.read_text(encoding="utf-8")):
            target = target.strip()
            if not target or target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                fail(f"link escapes repository: {path.relative_to(ROOT)} -> {target}")
            if not resolved.exists():
                fail(f"broken link: {path.relative_to(ROOT)} -> {target}")


def check_data_tests() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        check=False,
    )
    if result.returncode:
        fail("public-data unit tests failed")


def main() -> None:
    check_required()
    files = public_text_files()
    check_leakage(files)
    check_links(files)
    check_data_tests()
    print(f"PASS: checked {len(files)} text files and all required artifacts")


if __name__ == "__main__":
    main()
