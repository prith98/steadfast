import type { ModelRow } from "@/lib/types";
import { num } from "@/lib/format";
import { MetricCell } from "./MetricCell";

type Props = {
  rows: ModelRow[];
};

type SubMetric = {
  kind?: string;
  delta?: number | null;
  delta_ci_lower?: number | null;
  delta_ci_upper?: number | null;
  reason?: string | null;
  slope?: number | null;
  slope_ci_lower?: number | null;
  slope_ci_upper?: number | null;
  l50?: number | null;
  l50_ci_lower?: number | null;
  l50_ci_upper?: number | null;
  n_tasks?: number | null;
};

function deltaCell(sm: SubMetric | undefined): { point: string; ci?: string } {
  if (!sm) return { point: "N/A" };
  if (sm.delta == null) {
    return { point: sm.reason ? "N/A" : "—" };
  }
  const ci =
    sm.delta_ci_lower != null && sm.delta_ci_upper != null
      ? `[${num(sm.delta_ci_lower, 3)}, ${num(sm.delta_ci_upper, 3)}]`
      : undefined;
  return { point: num(sm.delta, 3), ci };
}

function slopeCell(sm: SubMetric | undefined): { point: string; ci?: string } {
  if (!sm) return { point: "N/A" };
  if (sm.slope == null) return { point: "—" };
  const ci =
    sm.slope_ci_lower != null && sm.slope_ci_upper != null
      ? `[${num(sm.slope_ci_lower, 3)}, ${num(sm.slope_ci_upper, 3)}]`
      : undefined;
  return { point: num(sm.slope, 3), ci };
}

function l50Cell(sm: SubMetric | undefined): { point: string; ci?: string } {
  if (!sm) return { point: "N/A" };
  if (sm.l50 == null) return { point: "—" };
  const ci =
    sm.l50_ci_lower != null && sm.l50_ci_upper != null
      ? `[${Math.round(sm.l50_ci_lower)}, ${Math.round(sm.l50_ci_upper)}]`
      : undefined;
  return { point: Math.round(sm.l50).toLocaleString(), ci };
}

export function RobustnessTable({ rows }: Props) {
  return (
    <table className="leaderboard">
      <thead>
        <tr>
          <th>model</th>
          <th>typo Δ ↑ (≥0 means robust)</th>
          <th>distractor Δ ↑ (CS only)</th>
          <th>long-context slope ↑</th>
          <th>L_50 (tokens) ↑</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const sub = (r.robustness?.sub_metrics ?? {}) as Record<string, SubMetric>;
          const typo = sub["typo"];
          const distractor = sub["distractor"];
          const lc = sub["long_context"];
          const opus = r.model_id === "claude-opus-4-7";
          const gemini = r.model_id === "gemini-2.5-pro";
          const opusLongContextSkip = opus && !lc;
          // Gemini with no robustness data at all = rate-limited
          // mid-pilot per the consistency-run quota incident; document
          // inline rather than rendering blank N/A cells.
          if (gemini && !r.robustness) {
            return (
              <tr key={r.model_id}>
                <td className="model">{r.model_id}</td>
                <td className="metric muted" colSpan={4}>
                  rate-limited mid-pilot (Google daily-quota window); v0.1.x re-run queued
                </td>
              </tr>
            );
          }
          const typoC = deltaCell(typo);
          const distC = deltaCell(distractor);
          const slopeC = slopeCell(lc);
          const l50C = l50Cell(lc);
          return (
            <tr key={r.model_id}>
              <td className="model">{r.model_id}</td>
              <MetricCell point={typoC.point} ci={typoC.ci} />
              <MetricCell point={distC.point} ci={distC.ci} />
              {opusLongContextSkip ? (
                <td className="metric muted" colSpan={2}>
                  not run — Anthropic prompt-cache path is v0.1.x
                </td>
              ) : (
                <>
                  <MetricCell point={slopeC.point} ci={slopeC.ci} />
                  <MetricCell point={l50C.point} ci={l50C.ci} />
                </>
              )}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
