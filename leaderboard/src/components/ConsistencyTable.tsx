import type { ModelRow } from "@/lib/types";
import { num } from "@/lib/format";
import { MetricCell } from "./MetricCell";

type Props = {
  rows: ModelRow[];
  geminiAsteriskReason: string;
};

export function ConsistencyTable({ rows, geminiAsteriskReason }: Props) {
  return (
    <table className="leaderboard">
      <thead>
        <tr>
          <th>model</th>
          <th>mean rubric score ↑</th>
          <th>min / max per task</th>
          <th>mean embedding cosine ↑</th>
          <th>n empty / n paraphrases</th>
          <th>n tasks</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const c = r.consistency;
          const gemini = r.model_id === "gemini-2.5-pro";
          if (!c) {
            return (
              <tr key={r.model_id}>
                <td className="model">{r.model_id}</td>
                <td className="metric muted" colSpan={5}>
                  not run
                </td>
              </tr>
            );
          }
          return (
            <tr key={r.model_id}>
              <td className="model">{r.model_id}</td>
              <MetricCell
                point={num(c.mean_rubric_score, 3)}
                asteriskReason={gemini ? geminiAsteriskReason : undefined}
              />
              <td className="metric">
                {num(c.min_rubric_score, 3)} / {num(c.max_rubric_score, 3)}
              </td>
              <td className="metric">{num(c.mean_embedding_cosine, 3)}</td>
              <td className="metric">
                {c.n_empty_answers} / {c.n_paraphrases_total}
              </td>
              <td className="metric">{c.n_tasks}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
