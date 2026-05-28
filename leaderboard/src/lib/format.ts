// Display helpers — keep formatting decisions in one place so the
// leaderboard table and the per-cell drill-downs render identically.

import type { Ci, WilsonCi } from "./types";

export function pct(x: number | null | undefined, digits = 1): string {
  if (x == null || Number.isNaN(x)) return "N/A";
  return `${(x * 100).toFixed(digits)}%`;
}

export function num(x: number | null | undefined, digits = 3): string {
  if (x == null || Number.isNaN(x)) return "N/A";
  return x.toFixed(digits);
}

export function ciOnly(ci: Ci | WilsonCi | null | undefined, digits = 3): string {
  if (!ci) return "";
  return `[${ci.ci_lower.toFixed(digits)}, ${ci.ci_upper.toFixed(digits)}]`;
}

export function pctCi(
  ci: WilsonCi | Ci | null | undefined,
  digits = 1,
): string {
  if (!ci) return "";
  return `[${(ci.ci_lower * 100).toFixed(digits)}%, ${(ci.ci_upper * 100).toFixed(digits)}%]`;
}

// Format a Ci-with-point-estimate as "0.029 [0.012, 0.063]" for tables.
export function ciCell(ci: Ci | null | undefined, digits = 3): string {
  if (!ci || ci.point_estimate == null) return "N/A";
  return `${ci.point_estimate.toFixed(digits)} ${ciOnly(ci, digits)}`;
}

export function pctCiCell(
  rate: number | null | undefined,
  ci: WilsonCi | Ci | null | undefined,
  digits = 1,
): string {
  if (rate == null) return "N/A";
  if (!ci) return pct(rate, digits);
  return `${pct(rate, digits)} ${pctCi(ci, digits)}`;
}
