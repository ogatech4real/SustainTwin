import numpy as np
import pandas as pd
from .config import WP1_DIR, WP2_DIR, WP3_DIR, EVIDENCE_DIR
from .utils import ensure_dirs, save_json


def _best(metrics: pd.DataFrame, task: str) -> dict:
    g = metrics[metrics["task"] == task].copy()
    if g.empty:
        return {}
    col = "oof_macro_f1_expected_labels" if "oof_macro_f1_expected_labels" in g else "oof_accuracy"
    return g.sort_values(col, ascending=False).iloc[0].to_dict()


def run() -> dict:
    ensure_dirs(EVIDENCE_DIR)
    q = pd.read_csv(WP1_DIR / "run_quality_audit.csv")
    em = pd.read_csv(WP2_DIR / "electromechanical_validation_summary.csv")
    metrics = pd.read_csv(WP3_DIR / "model_comparison_metrics.csv")
    loso = pd.read_csv(WP3_DIR / "leave_one_severity_out_fault_active.csv")
    rep = pd.read_csv(WP3_DIR / "experimental_replication_audit.csv")
    fold = pd.read_csv(WP3_DIR / "grouped_cv_fold_class_audit.csv")

    best_full = _best(metrics, "full_temporal_multiclass")
    best_active = _best(metrics, "fault_active_multiclass")
    best_binary = _best(metrics, "binary_healthy_fault")
    erow = em.iloc[0] if len(em) else pd.Series(dtype=float)
    total = int(erow.get("pairs_total", 0))
    comparable = int(erow.get("pearson_pairs_empirically_comparable", 0))

    # FINAL FREEZE: no further experiment is required for the scoped manuscript.
    actions = pd.DataFrame([
        {
            "action_type": "VALIDATION_EXPERIMENT", "fault_code": "F0", "severity": 0.0,
            "replicate": np.nan, "priority": "OPTIONAL", "decision": "DEFERRED_NOT_REQUIRED_FOR_FREEZE",
            "reason": "A controlled F0 speed-envelope test could strengthen direct RPM-dependent cross-domain comparison, but it is outside the frozen evidence scope and is not required for the current manuscript claims."
        },
        {
            "action_type": "EXPERIMENT", "fault_code": "F1/F2/F3/F4", "severity": 0.3,
            "replicate": 2, "priority": "LOW", "decision": "NOT_REQUIRED_FOR_FREEZE",
            "reason": "Additional condition replication may be future work; the current paper reports run-grouped predictive evidence and does not claim replicated-condition population variance."
        }
    ])
    actions.to_csv(EVIDENCE_DIR / "recommended_targeted_actions.csv", index=False)
    actions.to_csv(EVIDENCE_DIR / "recommended_targeted_additional_runs.csv", index=False)

    main = fold[fold["task"].isin(["full_temporal_multiclass", "fault_active_multiclass"])]
    class_complete = bool((main["missing_expected_labels"].fillna("").astype(str).str.len() == 0).all())
    conditions_3plus = int((rep["independent_runs"] >= 3).sum())
    summary = {
        "pipeline_version": "2.2-final",
        "evidence_status": "FROZEN_FOR_MANUSCRIPT",
        "experimental_campaign": "14 accepted controlled Build 3 runs; clean F2 S0.6 R2 replaces the earlier sensitivity-only F2 S0.6 run",
        "wp1_primary_runs": int((q["run_inclusion_status"] == "PRIMARY").sum()),
        "wp1_sensitivity_runs": int((q["run_inclusion_status"] == "SENSITIVITY_ONLY").sum()),
        "wp1_excluded_runs": int((q["run_inclusion_status"] == "EXCLUDE").sum()),
        "wp2_electromechanical_pairs_total": total,
        "wp2_electromechanical_pairs_empirically_comparable": comparable,
        "wp2_comparable_only_pearson_sign_agreement_pct": None if pd.isna(erow.get("pearson_sign_agreement_pct_comparable_only", np.nan)) else float(erow["pearson_sign_agreement_pct_comparable_only"]),
        "best_full_temporal": best_full,
        "best_fault_active": best_active,
        "best_binary": best_binary,
        "main_multiclass_folds_class_complete": class_complete,
        "leave_one_severity_out_rows": int(len(loso)),
        "conditions_with_3plus_independent_runs": conditions_3plus,
        "additional_experiments_required": False,
        "decision_rule": "Evidence is frozen. Do not add experiments unless peer review or a changed research question requires them. Optional F0 speed-envelope validation is explicitly deferred.",
    }
    save_json(summary, EVIDENCE_DIR / "research_readiness_summary.json")

    index = pd.DataFrame([
        ["WP1 run quality", "outputs/wp1_experiments/run_quality_audit.csv", "protocol integrity and final inclusion"],
        ["WP1 severity response", "outputs/wp1_experiments/severity_response_summary.csv", "three-level controlled severity consistency"],
        ["WP1 fault deltas", "outputs/wp1_experiments/fault_vs_baseline_deltas.csv", "within-run baseline-to-fault effect sizes"],
        ["WP2 electromechanical", "outputs/wp2_real_validation/electromechanical_validation.csv", "bounded cross-domain behavioural evidence"],
        ["WP2 current secondary", "outputs/wp2_real_validation/current_secondary_validation.csv", "current-power/vibration external correspondence"],
        ["WP3 grouped models", "outputs/wp3_ai/model_comparison_metrics.csv", "pooled OOF run-grouped evaluation"],
        ["WP3 fold audit", "outputs/wp3_ai/grouped_cv_fold_class_audit.csv", "class-complete fold verification"],
        ["WP3 ablation", "outputs/wp3_ai/physics_feature_ablation.csv", "incremental value of physics-derived features"],
        ["WP3 severity transfer", "outputs/wp3_ai/leave_one_severity_out_fault_active.csv", "within-twin severity transfer"],
        ["Final figures", "outputs/final_analysis/figures", "publication-ready PNG/PDF figures"],
        ["Final tables", "outputs/final_analysis/tables", "manuscript-ready result tables"],
    ], columns=["evidence_block", "file", "purpose"])
    index.to_csv(EVIDENCE_DIR / "key_results_index.csv", index=False)

    claims = "# SustainTwin manuscript claim boundaries — V2.2 FINAL\n\n"
    claims += "- The controlled Build 3 campaign is frozen at 14 accepted PRIMARY runs. The clean F2 S0.6 R2 is the retained middle-severity F2 experiment.\n"
    claims += "- Severity R² is descriptive consistency evidence across the programmed three-level fault continuum; it is not a population-level physical law.\n"
    claims += "- Industrial historian data provide external behavioural/operating plausibility, not F1–F4 ground-truth fault validation.\n"
    claims += "- Undefined SustainTwin F0 RPM correlations are reported as undefined and never counted as cross-domain agreement.\n"
    claims += "- Current–power and current–vibration comparisons are secondary external evidence where both domains contain sufficient variation.\n"
    claims += "- Rows from one experiment never cross train/test boundaries. Main grouped folds are audited for class completeness.\n"
    claims += "- Near-perfect fault-active classification describes separability inside the controlled digital-twin experiment; it is not evidence of industrial-domain classifier generalisation.\n"
    claims += "- Leave-one-severity-out results measure transfer inside the programmed severity continuum, not unseen-machine generalisation.\n"
    claims += "- Most fault×severity conditions have one independent run. Do not claim replicated-condition variance, confidence intervals across machines, or population-level reliability.\n"
    claims += "- The optional F0 speed-envelope experiment is deferred. The manuscript must therefore avoid claiming direct empirical validation of RPM-dependent F0 correlations.\n"
    claims += "- No further experiments are required for the frozen scoped manuscript unless the research question changes or reviewers request additional validation.\n"
    (EVIDENCE_DIR / "manuscript_claim_boundaries.md").write_text(claims, encoding="utf-8")
    return summary
