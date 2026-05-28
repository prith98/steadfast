import type { ModelRow } from "@/lib/types";
import { ciCell, pct, pctCi } from "@/lib/format";
import { MetricCell } from "./MetricCell";

type Props = {
  rows: ModelRow[];
  geminiAsteriskReason: string;
  wilsonOverlapReason: string;
};

export function SafetyTable({
  rows,
  geminiAsteriskReason,
  wilsonOverlapReason,
}: Props) {
  return (
    <table className="leaderboard">
      <thead>
        <tr>
          <th>model</th>
          <th>catastrophic rate ↓</th>
          <th>refusal F1 ↑</th>
          <th>refusal precision / recall</th>
          <th>n harmful / n traps</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const s = r.safety;
          const gemini = r.model_id === "gemini-2.5-pro";
          if (!s) {
            return (
              <tr key={r.model_id}>
                <td className="model">{r.model_id}</td>
                <td className="metric muted" colSpan={4}>
                  not run
                </td>
              </tr>
            );
          }
          return (
            <tr key={r.model_id}>
              <td className="model">{r.model_id}</td>
              <MetricCell
                point={pct(s.catastrophic_rate)}
                ci={pctCi(s.catastrophic_ci)}
                asteriskReason={wilsonOverlapReason}
              />
              <MetricCell
                point={ciCell({ ...s.refusal_f1_ci, point_estimate: s.refusal_f1 })}
                asteriskReason={gemini ? geminiAsteriskReason : undefined}
              />
              <td className="metric">
                {pct(s.refusal_precision)} / {pct(s.refusal_recall)}
              </td>
              <td className="metric">
                {s.n_harmful} / {s.n_traps}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
