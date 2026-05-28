import { pilot } from "@/lib/data";
import { CalibrationTable } from "@/components/CalibrationTable";
import { ConsistencyTable } from "@/components/ConsistencyTable";
import { RobustnessTable } from "@/components/RobustnessTable";
import { SafetyTable } from "@/components/SafetyTable";

const GEMINI_ASTERISK =
  pilot.known_asterisks?.gemini_empty_responses ??
  "Gemini content-filter empty responses; v0.2 refusal_v2 reclassification pending.";

const WILSON_OVERLAP =
  pilot.known_asterisks?.safety_wilson_ci_overlap ??
  "All pairwise Wilson CIs overlap at safety bank-size 10; pairwise ranking is not defensible.";

function shortCommit(c: string | null): string {
  if (!c) return "—";
  return c.length > 8 ? c.slice(0, 8) : c;
}

export default function HomePage() {
  return (
    <main>
      <header className="site">
        <h1>Steadfast — v0.1 reliability leaderboard</h1>
        <p className="tagline">
          Rigorous, reproducible reliability benchmark for AI agents.
          51 cross-domain tasks × N=10 reps + 20 safety cases × 3 frontier
          models. Methodology pinned; CIs on every metric.
        </p>
        <div className="meta">
          <span>pilot: <code>{pilot.pilot_id}</code></span>
          <span>commit: <code>{shortCommit(pilot.generated_from_commit)}</code></span>
          <span>package: <code>{pilot.package_version ?? "—"}</code></span>
          <span>started: <code>{pilot.run_started_at ?? "—"}</code></span>
        </div>
      </header>

      <section>
        <h2>Calibration</h2>
        <p className="lede">
          Brier and ECE on verbalized confidence; refusal sensitivity on the
          hard-task subset (uncertain-by-construction; higher refusal is
          better) and specificity on normal tasks (confident answers; higher
          is better). N={pilot.benchmarks.customer_support_n_tasks ?? 17} CS + {pilot.benchmarks.code_repair_n_tasks ?? 17} code-repair + {pilot.benchmarks.multi_hop_research_n_tasks ?? 17} multi-hop-research tasks × 10 reps.
        </p>
        <CalibrationTable rows={pilot.models} geminiAsteriskReason={GEMINI_ASTERISK} />
      </section>

      <section>
        <h2>Consistency</h2>
        <p className="lede">
          K=5 semantically-equivalent paraphrases per task; pairwise rubric
          score across the {`{`}clean, paraphrased{`}`} response set.
          n_empty_answers high on Gemini is the known content-filter
          artifact, not a methodology bug.
        </p>
        <ConsistencyTable rows={pilot.models} geminiAsteriskReason={GEMINI_ASTERISK} />
      </section>

      <section>
        <h2>Robustness</h2>
        <p className="lede">
          Typo delta: paired clean-vs-perturbed rubric score, signed so ≥0
          means the model is robust to typos. Distractor runs on
          customer_support only (v0.1.x adds the code-repair / multi-hop
          banks). Long-context measures success-rate ladder at {`{`}4 k, 16 k,
          64 k, 128 k{`}`} tokens with sigmoid fit; slope and L<sub>50</sub>{" "}
          bootstrap CIs. Long-context cohort:{" "}
          <code>
            {(pilot.benchmarks.long_context_task_subset ?? []).join(", ") || "—"}
          </code>
          {" "}(5-task stratified subset per METHODOLOGY §Known limitations;
          full-suite version is v0.1.x).
        </p>
        <RobustnessTable rows={pilot.models} />
      </section>

      <section>
        <h2>Safety</h2>
        <p className="lede">
          Catastrophic rate on a 10-trap × 4-category bank (Wilson 95% CI);
          refusal F1 over the 10 traps + 10 benign cases (percentile-bootstrap
          95% CI, n=10k resamples). All pairwise CIs on catastrophic rates
          overlap at bank-size 10 — pairwise ranking is not defensible at
          v0.1; v0.2 bank growth past 50 cases unlocks it.
        </p>
        <SafetyTable
          rows={pilot.models}
          geminiAsteriskReason={GEMINI_ASTERISK}
          wilsonOverlapReason={WILSON_OVERLAP}
        />
      </section>

      <section>
        <h2>Scope & known limitations</h2>
        <div className="scope-grid">
          <div className="scope-card">
            <h3>3 models, not 5</h3>
            <p>
              {pilot.scope_decisions?.models_3_not_5?.rationale ??
                "Llama-4 and Mistral Large are v0.1.x via a Together/Groq adapter ADR."}
            </p>
          </div>
          <div className="scope-card">
            <h3>Opus skipped on long-context</h3>
            <p>
              {pilot.scope_decisions?.opus_long_context_skip?.rationale ??
                "Per-rep distinct filler windows bypass Anthropic prompt-caching; full ladder would be ~$160. v0.1.x adds prompt-cache headers."}
            </p>
          </div>
          <div className="scope-card">
            <h3>Distractor on customer-support only</h3>
            <p>
              {pilot.scope_decisions?.distractor_cs_only?.rationale ??
                "Other two domains' distractor cells surface as N/A; banks are v0.1.x."}
            </p>
          </div>
          <div className="scope-card">
            <h3>Long-context on a 5-task subset</h3>
            <p>
              Per-task input is dominated by the fixed filler ladder, so
              cost scales linearly in task count. Full 51-task long-context
              run would price at ~$291 vs the envelope's $30–80; the 5-task
              stratified subset (2 CS + 2 code-repair + 1 multi-hop-research)
              restores the envelope at ~$29. Per-tier Wilson CIs still pool
              50 trials per tier; task-level bootstrap CIs on slope and{" "}
              L<sub>50</sub> are wider at n=5. v0.2 candidates: prompt-cache
              support or a 10-task subset.
            </p>
          </div>
        </div>
      </section>

      <section>
        <h2>How to read this</h2>
        <ul className="notes">
          <li>
            Every numeric cell is a point estimate + 95% confidence interval
            (Brier / refusal F1 / consistency: bootstrap, 10k resamples;
            refusal sensitivity / specificity / catastrophic rate / overconfidence:
            Wilson). ECE is reported without a CI because the binning
            estimator is biased; see METHODOLOGY §"Calibration".
          </li>
          <li>
            Cells marked with{" "}
            <span className="asterisk">*</span> carry a pre-registered known
            limitation. Hover for the citation. Per METHODOLOGY's "Known
            limitations" framing, asterisks are documented at v0.1 rather
            than papered over.
          </li>
          <li>
            We do not aggregate to a single scalar reliability score —
            METHODOLOGY §"Aggregation" rejects the implied tradeoff. The
            leaderboard reports per-dimension headlines + per-domain
            breakdowns in the report.html drill-down.
          </li>
          <li>
            Pairwise ranking claims (e.g., "Model A is more reliable than
            Model B") require the CI of the paired difference to exclude
            zero. v0.1 surface lacks per-pair-difference CIs; the
            methodology writeup quotes the safety Wilson-overlap example as
            the canonical reason this is gated.
          </li>
        </ul>
      </section>

      <footer className="site">
        <p>
          Steadfast v0.1 · open-source · see{" "}
          <a href="https://github.com/anthropics/steadfast">repo</a>{" "}
          (placeholder) for methodology, ADRs, and reproducibility manifest.
          Generated from <code>{shortCommit(pilot.generated_from_commit)}</code>.
        </p>
      </footer>
    </main>
  );
}
