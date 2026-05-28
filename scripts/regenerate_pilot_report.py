"""Regenerate ``results/v01_full_pilot/report.html`` with all 4 dimensions.

The bench writes the report at the end of each invocation, but each
invocation only declares the metrics that *that* run requested. The
cross-dimension v0.1 pilot needs a single regeneration after Day 4
with the full ``{calibration, consistency, robustness, safety}``
requested_metrics set so the header reflects the cross-dimension
artifact rather than the last invocation's metrics.

Per ``results/v01_full_pilot/manifest.json`` §`outputs_to_commit`.

Safety lives under a separate output subdir (``…/safety/<model>/safety.json``)
because ADR-0007 §G splits the safety-bank dispatch from the per-task
benchmark dispatch. The pilot reuses the 2026-05-14 safety run at
``results/safety_pilot_001/`` (manifest §`day_4_safety`); this script
symlinks/copies those into the v01_full_pilot/safety/ tree so the
single report renders all four dimensions from one output_dir.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from steadfast.reporting.html import write_html_report

REPO_ROOT = Path(__file__).resolve().parent.parent
PILOT_DIR = REPO_ROOT / "results" / "v01_full_pilot"
SAFETY_PILOT_DIR = REPO_ROOT / "results" / "safety_pilot_001"
REPORT_PATH = PILOT_DIR / "report.html"

MODELS = ["claude-opus-4-7", "gpt-5.2", "gemini-2.5-pro"]


def _stage_safety() -> None:
    """Copy the 2026-05-14 safety pilot files into the v01 pilot tree.

    write_html_report's safety-section discovery walks
    ``output_dir/<model>/safety.json`` per :func:`_render_safety_section`
    in reporting/html.py — same layout as the per-task dimensions. The
    safety pilot was run separately into safety_pilot_001/; copying the
    JSONs lets the cross-dimension report find them without forking the
    renderer.
    """
    if not SAFETY_PILOT_DIR.exists():
        print(
            f"warning: safety pilot dir not found at {SAFETY_PILOT_DIR}; "
            "safety section will be empty"
        )
        return
    for model in MODELS:
        src = SAFETY_PILOT_DIR / model / "safety.json"
        if not src.exists():
            print(f"warning: no safety.json for {model} at {src}")
            continue
        dst = PILOT_DIR / model / "safety.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"staged {dst.relative_to(REPO_ROOT)}")


def main() -> None:
    _stage_safety()
    write_html_report(
        output_dir=PILOT_DIR,
        benchmark_name="all",
        target_models=MODELS,
        requested_metrics=frozenset({"calibration", "consistency", "robustness", "safety"}),
        report_path=REPORT_PATH,
    )
    print(f"wrote {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
