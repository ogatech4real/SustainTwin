from pipeline import (
    wp1_preprocess, wp1_analysis, wp2_preprocess, wp2_validation,
    wp3_prepare_ml, wp3_train_evaluate, build_evidence_pack, final_analysis
)


def main():
    stages = [
        ("WP1.1 Experiment preprocessing + inclusion policy", wp1_preprocess.run),
        ("WP1.2 Corrected severity + sensitivity scientific analysis", wp1_analysis.run),
        ("WP2.1 Industrial historian preprocessing", wp2_preprocess.run),
        ("WP2.2 Corrected external behavioural validation", wp2_validation.run),
        ("WP3.1 Leakage-safe ML + run-level replication audit", wp3_prepare_ml.run),
        ("WP3.2 Corrected run-grouped multi-task AI evaluation", wp3_train_evaluate.run),
        ("V2.2 FINAL evidence freeze", build_evidence_pack.run),
        ("V2.2 publication figures + manuscript tables", final_analysis.run),
    ]
    print("SustainTwin Research Work Package Pipeline — Version 2.2 — FINAL ANALYSIS FREEZE")
    print("=" * 66)
    for label, func in stages:
        print(f"\n>>> {label}")
        result = func()
        print(result)
    print("\nVersion 2.2 — FINAL ANALYSIS FREEZE pipeline complete.")
    print("Evidence frozen. Review outputs/final_analysis and outputs/manuscript_evidence for manuscript writing.")


if __name__ == "__main__":
    main()
