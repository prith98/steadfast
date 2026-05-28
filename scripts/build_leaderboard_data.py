"""Stage v0.1 pilot results into leaderboard/src/data/pilot.json.

Reads the per-model JSON files under ``results/v01_full_pilot/`` (and
``results/v01_full_pilot/safety/``), flattens them into a single
leaderboard-shaped document, and writes ``leaderboard/src/data/pilot.json``
for the Next.js app to import at build time.

The flattening keeps the CIs and known-asterisk surfaces; per-task /
per-case drill-downs are intentionally NOT included (the rendered
``report.html`` is the authoritative drill-down view, the leaderboard is
the cross-model headline).

Run:

    python3 scripts/build_leaderboard_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
PILOT_DIR = REPO_ROOT / "results" / "v01_full_pilot"
SAFETY_PILOT_DIR = REPO_ROOT / "results" / "safety_pilot_001"  # 2026-05-14 run
OUT_PATH = REPO_ROOT / "leaderboard" / "src" / "data" / "pilot.json"

MODELS = ["claude-opus-4-7", "gpt-5.2", "gemini-2.5-pro"]


def _load(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _calibration(model: str) -> dict[str, Any] | None:
    data = _load(PILOT_DIR / model / "calibration.json")
    if data is None:
        return None
    brier = data["brier"]
    ece = data["ece"]
    refusal = data["refusal"]
    over = data["overconfidence"]
    return {
        "n_total": brier["n_total"],
        "n_used": brier["n"],
        "n_refused": brier["n_refused"],
        "brier_verbalized": brier["verbalized"],
        "brier_logprob": brier.get("logprob"),
        "ece_verbalized": ece["verbalized"],
        "ece_logprob": ece.get("logprob"),
        "refusal_sensitivity": refusal["sensitivity"],
        "refusal_sensitivity_ci": refusal.get("sensitivity_ci"),
        "refusal_specificity": refusal["specificity"],
        "refusal_specificity_ci": refusal.get("specificity_ci"),
        "overconfidence_rate": over["rate"],
        "overconfidence_ci": over["ci"],
    }


def _consistency(model: str) -> dict[str, Any] | None:
    """Aggregate per-task consistency files into a model-level cell."""
    model_dir = PILOT_DIR / model
    if not model_dir.exists():
        return None
    files = sorted(model_dir.glob("consistency_*.json"))
    if not files:
        return None
    scores: list[float] = []
    n_empty_total = 0
    n_total = 0
    for fp in files:
        d = json.loads(fp.read_text())
        # Each per-task consistency JSON exposes a mean rubric score
        # (rubric_mean.point_estimate) and n_empty_answers count.
        rm = d.get("rubric_mean") or {}
        pt = rm.get("point_estimate")
        if pt is not None:
            scores.append(pt)
        n_empty_total += int(d.get("n_empty_answers", 0))
        n_total += int(d.get("n_paraphrases_total", 0) or d.get("n_total", 0))
    if not scores:
        return None
    return {
        "n_tasks": len(scores),
        "mean_rubric_score": sum(scores) / len(scores),
        "min_rubric_score": min(scores),
        "max_rubric_score": max(scores),
        "n_empty_answers": n_empty_total,
        "n_paraphrases_total": n_total,
    }


def _robustness(model: str) -> dict[str, Any] | None:
    data = _load(PILOT_DIR / model / "robustness.json")
    if data is None:
        return None
    out: dict[str, Any] = {"sub_metrics": {}}
    sub = data.get("sub_metrics", {})
    for kind, payload in sub.items():
        # Best-effort flatten — actual schema varies per kind.
        out["sub_metrics"][kind] = payload
    return out


def _safety(model: str) -> dict[str, Any] | None:
    # v0.1 pilot reuses the 2026-05-14 safety run per manifest.
    data = _load(SAFETY_PILOT_DIR / model / "safety.json")
    if data is None:
        return None
    return {
        "n_traps": data["n_traps"],
        "n_benigns": data["n_benigns"],
        "catastrophic_rate": data["catastrophic"]["rate"],
        "catastrophic_ci": data["catastrophic"]["ci"],
        "n_harmful": data["catastrophic"]["n_harmful"],
        "refusal_f1": data["refusal"]["f1"],
        "refusal_f1_ci": data["refusal"]["f1_ci"],
        "refusal_precision": data["refusal"]["precision"],
        "refusal_recall": data["refusal"]["recall"],
    }


def _per_model(model: str) -> dict[str, Any]:
    return {
        "model_id": model,
        "calibration": _calibration(model),
        "consistency": _consistency(model),
        "robustness": _robustness(model),
        "safety": _safety(model),
    }


def main() -> None:
    manifest = _load(PILOT_DIR / "manifest.json") or {}
    doc = {
        "pilot_id": manifest.get("pilot_id", "v01_full_pilot"),
        "generated_from_commit": manifest.get("code", {}).get("git_commit"),
        "package_version": manifest.get("code", {}).get("package_version"),
        "run_started_at": manifest.get("meta", {}).get("run_started_at"),
        "run_completed_at": manifest.get("meta", {}).get("run_completed_at"),
        "scope_decisions": manifest.get("scope_decisions"),
        "known_asterisks": manifest.get("known_asterisks"),
        "benchmarks": {
            "customer_support_n_tasks": manifest.get("benchmarks", {})
            .get("customer_support", {})
            .get("task_count"),
            "code_repair_n_tasks": manifest.get("benchmarks", {})
            .get("code_repair", {})
            .get("task_count"),
            "multi_hop_research_n_tasks": manifest.get("benchmarks", {})
            .get("multi_hop_research", {})
            .get("task_count"),
            "safety_n_traps": manifest.get("benchmarks", {})
            .get("safety", {})
            .get("trap_count"),
            "safety_n_benigns": manifest.get("benchmarks", {})
            .get("safety", {})
            .get("benign_count"),
            "long_context_task_subset": manifest.get("benchmarks", {})
            .get("long_context_task_subset", {})
            .get("task_ids"),
        },
        "models": [_per_model(m) for m in MODELS],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, indent=2, sort_keys=False))
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
