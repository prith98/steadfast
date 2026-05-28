import type { ModelRow } from "@/lib/types";
import { ciCell, num, pct, pctCi } from "@/lib/format";
import { MetricCell } from "./MetricCell";

type Props = {
  rows: ModelRow[];
  geminiAsteriskReason: string;
};

export function CalibrationTable({ rows, geminiAsteriskReason }: Props) {
  return (
    <table className="leaderboard">
      <thead>
        <tr>
          <th>model</th>
          <th>Brier (verbalized) ↓</th>
          <th>ECE (verbalized) ↓</th>
          <th>Refusal sensitivity (hard) ↑</th>
          <th>Refusal specificity (normal) ↑</th>
          <th>Overconfidence rate ↓</th>
          <th>n used / n total</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const c = r.calibration;
          const gemini = r.model_id === "gemini-2.5-pro";
          if (!c) {
            return (
              <tr key={r.model_id}>
                <td className="model">{r.model_id}</td>
                <td className="metric muted" colSpan={6}>
                  not run
                </td>
              </tr>
            );
          }
          return (
            <tr key={r.model_id}>
              <td className="model">{r.model_id}</td>
              <MetricCell
                point={ciCell(c.brier_verbalized, 3)}
                asteriskReason={gemini ? geminiAsteriskReason : undefined}
              />
              <MetricCell point={num(c.ece_verbalized, 3)} />
              <MetricCell
                point={pct(c.refusal_sensitivity)}
                ci={c.refusal_sensitivity_ci ? pctCi(c.refusal_sensitivity_ci) : undefined}
                asteriskReason={gemini ? geminiAsteriskReason : undefined}
              />
              <MetricCell
                point={pct(c.refusal_specificity)}
                ci={c.refusal_specificity_ci ? pctCi(c.refusal_specificity_ci) : undefined}
              />
              <MetricCell
                point={pct(c.overconfidence_rate)}
                ci={pctCi(c.overconfidence_ci)}
              />
              <td className="metric">
                {c.n_used} / {c.n_total}
                {c.n_refused ? <span className="ci">{c.n_refused} refused</span> : null}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
