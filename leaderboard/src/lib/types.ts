// Shape mirrors scripts/build_leaderboard_data.py output. Kept narrow
// enough that the leaderboard renders with partial data — every cell is
// nullable so a missing dimension surfaces as "N/A" rather than crashing.

export type Ci = {
  ci_lower: number;
  ci_upper: number;
  confidence_level: number;
  point_estimate?: number;
  method?: string;
};

export type WilsonCi = {
  ci_lower: number;
  ci_upper: number;
  proportion: number;
  confidence_level: number;
};

export type Calibration = {
  n_total: number;
  n_used: number;
  n_refused: number;
  brier_verbalized: Ci;
  brier_logprob: Ci | null;
  ece_verbalized: number;
  ece_logprob: number | null;
  refusal_sensitivity: number;
  refusal_sensitivity_ci: WilsonCi | null;
  refusal_specificity: number;
  refusal_specificity_ci: WilsonCi | null;
  overconfidence_rate: number;
  overconfidence_ci: WilsonCi;
};

export type Consistency = {
  n_tasks: number;
  mean_rubric_score: number;
  min_rubric_score: number;
  max_rubric_score: number;
  mean_embedding_cosine: number;
  n_empty_answers: number;
  n_paraphrases_total: number;
};

export type RobustnessTypo = {
  delta?: Ci | number | null;
  n?: number;
  reason?: string | null;
};

export type Robustness = {
  sub_metrics: Record<string, unknown>;
};

export type Safety = {
  n_traps: number;
  n_benigns: number;
  catastrophic_rate: number;
  catastrophic_ci: WilsonCi;
  n_harmful: number;
  refusal_f1: number;
  refusal_f1_ci: Ci;
  refusal_precision: number;
  refusal_recall: number;
};

export type ModelRow = {
  model_id: string;
  calibration: Calibration | null;
  consistency: Consistency | null;
  robustness: Robustness | null;
  safety: Safety | null;
};

export type KnownAsterisks = {
  gemini_empty_responses?: string;
  safety_wilson_ci_overlap?: string;
  code_repair_rubric_only?: string;
  multi_hop_research_no_retrieval?: string;
};

export type ScopeDecisions = {
  models_3_not_5?: { decision: string; rationale: string };
  opus_long_context_skip?: { decision: string; rationale: string };
  distractor_cs_only?: { decision: string; rationale: string };
};

export type BenchmarksMeta = {
  customer_support_n_tasks: number | null;
  code_repair_n_tasks: number | null;
  multi_hop_research_n_tasks: number | null;
  safety_n_traps: number | null;
  safety_n_benigns: number | null;
  long_context_task_subset: string[] | null;
};

export type Pilot = {
  pilot_id: string;
  generated_from_commit: string | null;
  package_version: string | null;
  run_started_at: string | null;
  run_completed_at: string | null;
  scope_decisions: ScopeDecisions | null;
  known_asterisks: KnownAsterisks | null;
  benchmarks: BenchmarksMeta;
  models: ModelRow[];
};
