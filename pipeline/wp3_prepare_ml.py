import numpy as np
import pandas as pd

from .config import WP1_DIR, WP3_DIR, ML_OBSERVABLE_FEATURES, ML_PHYSICS_FEATURES
from .utils import ensure_dirs, save_json

FORBIDDEN_FEATURE_PREFIXES = (
    "selected_fault", "embedded_fault", "fault_", "experiment_", "run_id", "source_file",
    "current_multiplier", "vibration_offset", "effective_mechanical_load", "effective_process_heat_load",
)
META = ["group_run", "target_fault", "target_binary", "experiment_phase", "selected_fault_severity", "run_inclusion_status"]


def run() -> dict:
    ensure_dirs(WP3_DIR, WP3_DIR / "figures", WP3_DIR / "models")
    df = pd.read_csv(WP1_DIR / "experiments_master_clean.csv")
    df["target_fault"] = df["embedded_fault_code"].astype(str)
    df["target_binary"] = (df["target_fault"] != "F0").astype(int)
    df["group_run"] = df["run_id"].astype(str)
    if "run_inclusion_status" not in df.columns:
        df["run_inclusion_status"] = "PRIMARY"

    df = df[df["target_fault"].isin(["F0", "F1", "F2", "F3", "F4"])].copy()
    df = df[df["run_inclusion_status"] != "EXCLUDE"].copy()

    def build(features, filename, primary_only=False):
        src = df[df["run_inclusion_status"] == "PRIMARY"].copy() if primary_only else df.copy()
        use = [f for f in features if f in src.columns]
        out = src[META + use].copy()
        for f in use:
            out[f] = pd.to_numeric(out[f], errors="coerce")
        out["selected_fault_severity"] = pd.to_numeric(out["selected_fault_severity"], errors="coerce")
        out = out.dropna(subset=use + ["target_fault", "group_run"])
        out.to_csv(WP3_DIR / filename, index=False)
        return out, use

    obs, obs_features = build(ML_OBSERVABLE_FEATURES, "ml_dataset_observable.csv")
    phy, phy_features = build(ML_PHYSICS_FEATURES, "ml_dataset_physics_augmented.csv")
    obs_p, _ = build(ML_OBSERVABLE_FEATURES, "ml_dataset_observable_primary_only.csv", primary_only=True)
    phy_p, _ = build(ML_PHYSICS_FEATURES, "ml_dataset_physics_augmented_primary_only.csv", primary_only=True)

    audit = []
    for c in df.columns:
        if c in {"energy_wh", "co2_g", "ts_ms", "ts_iso", "experiment_elapsed_s"}:
            reason = "time/cumulative leakage risk"
        elif c.startswith(FORBIDDEN_FEATURE_PREFIXES) or c in {"anomaly", "health_overall", "health_status", "ai_fault_code", "ai_confidence", "ai_fault_label", "ai_severity"}:
            reason = "ground-truth/derived label leakage or post-diagnostic metadata"
        elif c in obs_features:
            reason = "observable feature"
        elif c in phy_features:
            reason = "physics-derived candidate feature"
        else:
            reason = "not used in primary diagnostic feature set"
        audit.append({"column": c, "used_observable": c in obs_features,
                      "used_physics_augmented": c in phy_features, "reason": reason})
    pd.DataFrame(audit).to_csv(WP3_DIR / "feature_leakage_audit.csv", index=False)

    class_counts = (phy.groupby(["target_fault", "run_inclusion_status"]).size()
                    .rename("rows").reset_index().sort_values(["target_fault", "run_inclusion_status"]))
    class_counts.to_csv(WP3_DIR / "ml_class_distribution.csv", index=False)

    # V2.1 correction: replication is a run-level property of the SELECTED experimental condition,
    # not a row-level property of embedded F0 phases inside fault experiments.
    run_conditions = (df[["group_run", "selected_fault_severity", "run_inclusion_status"]]
                      .drop_duplicates(subset=["group_run"]).copy())
    selected = (pd.read_csv(WP1_DIR / "experiments_master_clean.csv")
                .groupby("run_id", as_index=False)
                .agg(selected_fault_code=("selected_fault_code", "first"),
                     selected_fault_severity=("selected_fault_severity", "first"),
                     run_inclusion_status=("run_inclusion_status", "first")))
    selected["selected_fault_severity"] = pd.to_numeric(selected["selected_fault_severity"], errors="coerce")
    replication = (selected.groupby(["selected_fault_code", "selected_fault_severity", "run_inclusion_status"], dropna=False)
                   .agg(independent_runs=("run_id", "nunique"))
                   .reset_index()
                   .rename(columns={"selected_fault_code": "fault_code", "selected_fault_severity": "severity"})
                   .sort_values(["fault_code", "severity", "run_inclusion_status"]))
    replication["replication_interpretation"] = np.where(
        replication["independent_runs"] >= 3,
        "3+ independent runs available",
        "limited independent replication; rows within runs are not independent replicates"
    )
    replication.to_csv(WP3_DIR / "experimental_replication_audit.csv", index=False)

    run_manifest = selected.rename(columns={"run_id": "group_run", "selected_fault_code": "selected_condition_fault",
                                            "selected_fault_severity": "selected_condition_severity"})
    run_manifest.to_csv(WP3_DIR / "run_condition_manifest.csv", index=False)

    manifest = {
        "observable_rows": int(len(obs)), "physics_rows": int(len(phy)),
        "runs_non_excluded": int(phy["group_run"].nunique()),
        "runs_primary_only": int(phy_p["group_run"].nunique()),
        "observable_features": obs_features,
        "physics_augmented_features": phy_features,
        "target_multiclass": "embedded_fault_code -> target_fault",
        "target_binary": "F0=0; F1-F4=1",
        "grouping": "run_id -> group_run",
        "replication_audit_basis": "selected experimental condition at run level",
        "replication_warning": "Most fault×severity conditions contain one independent run; baseline/recovery F0 rows inside fault experiments do not count as independent F0 severity runs.",
        "sensitivity_policy": "Non-stationary-baseline runs are retained in non-excluded ML data and separately evaluated against PRIMARY-only data."
    }
    save_json(manifest, WP3_DIR / "wp3_prepare_manifest.json")
    return manifest


if __name__ == "__main__":
    print(run())
