from pathlib import Path
import numpy as np
import pandas as pd

from .config import EXPERIMENT_DIR, WP1_DIR
from .utils import ensure_dirs, read_csv_robust, save_json

CRITICAL_WARNINGS = {
    "BASELINE_NOT_F0", "FAULT_CODE_MISMATCH", "SEVERITY_MISMATCH",
    "RECOVERY_NOT_F0", "SEQUENCE_GAP", "LOW_ROW_COUNT"
}
SENSITIVITY_WARNINGS = {
    "NONSTATIONARY_BASELINE_RPM", "NONSTATIONARY_BASELINE_HEAT"
}


def _inclusion_status(warnings: list[str], quality_flag: str) -> tuple[str, str]:
    ws = set(warnings)
    if quality_flag == "FAIL" or ws.intersection(CRITICAL_WARNINGS):
        return "EXCLUDE", "critical protocol/data-integrity warning"
    if ws.intersection(SENSITIVITY_WARNINGS):
        return "SENSITIVITY_ONLY", "non-stationary nominal baseline; retain for sensitivity analysis"
    if warnings:
        return "PRIMARY", "minor warning retained with transparent reporting"
    return "PRIMARY", "meets primary run-quality criteria"


def _quality_for_run(df: pd.DataFrame, source_file: str) -> dict:
    elapsed = pd.to_numeric(df.get("experiment_elapsed_s"), errors="coerce")
    seq = pd.to_numeric(df.get("sequence_id"), errors="coerce")
    ts = pd.to_datetime(df.get("ts_iso"), errors="coerce", utc=True)
    dseq = seq.diff().dropna()
    dt = elapsed.diff().dropna()

    baseline = df[df["experiment_phase"].astype(str).eq("BASELINE")]
    fault = df[df["experiment_phase"].astype(str).eq("FAULT_ACTIVE")]
    recovery = df[df["experiment_phase"].astype(str).eq("RECOVERY")]

    selected_code = str(df["selected_fault_code"].dropna().iloc[0]) if df["selected_fault_code"].notna().any() else ""
    sev_series = pd.to_numeric(df["selected_fault_severity"], errors="coerce")
    selected_sev = float(sev_series.dropna().iloc[0]) if sev_series.notna().any() else np.nan

    fault_code_match = True
    severity_match = True
    if selected_code != "F0" and not fault.empty:
        fault_code_match = bool((fault["embedded_fault_code"].astype(str) == selected_code).mean() >= 0.98)
        sev = pd.to_numeric(fault["embedded_fault_severity"], errors="coerce")
        severity_match = bool((np.isclose(sev.dropna(), selected_sev, atol=1e-6)).mean() >= 0.98) if sev.notna().any() else False
    elif selected_code == "F0" and not fault.empty:
        fault_code_match = bool((fault["embedded_fault_code"].astype(str) == "F0").mean() >= 0.98)

    baseline_f0 = bool((baseline["embedded_fault_code"].astype(str) == "F0").mean() >= 0.98) if not baseline.empty else False
    recovery_f0 = bool((recovery["embedded_fault_code"].astype(str) == "F0").mean() >= 0.95) if not recovery.empty else False

    warnings = []
    if len(df) < 560:
        warnings.append("LOW_ROW_COUNT")
    if not baseline_f0:
        warnings.append("BASELINE_NOT_F0")
    if not fault_code_match:
        warnings.append("FAULT_CODE_MISMATCH")
    if not severity_match:
        warnings.append("SEVERITY_MISMATCH")
    if not recovery_f0:
        warnings.append("RECOVERY_NOT_F0")
    if (dseq > 1).sum() > 0:
        warnings.append("SEQUENCE_GAP")
    if (dt > 2.0).sum() > 1:
        warnings.append("SAMPLING_DELAY")

    if not baseline.empty:
        rpm = pd.to_numeric(baseline["actual_rpm"], errors="coerce")
        heat = pd.to_numeric(baseline["process_heat_load_w"], errors="coerce")
        if rpm.notna().any() and rpm.max() - rpm.min() > 400:
            warnings.append("NONSTATIONARY_BASELINE_RPM")
        if heat.notna().any() and heat.max() - heat.min() > 50:
            warnings.append("NONSTATIONARY_BASELINE_HEAT")

    quality_flag = "PASS" if not warnings else ("REVIEW" if len(warnings) <= 2 else "FAIL")
    inclusion_status, inclusion_reason = _inclusion_status(warnings, quality_flag)

    return {
        "source_file": source_file,
        "run_id": str(df["run_id"].dropna().iloc[0]) if df["run_id"].notna().any() else Path(source_file).stem,
        "fault_code": selected_code,
        "severity": selected_sev,
        "replicate": int(pd.to_numeric(df["replicate"], errors="coerce").dropna().iloc[0]),
        "rows": int(len(df)),
        "duration_s": float(elapsed.max() - elapsed.min()) if elapsed.notna().any() else np.nan,
        "median_sampling_s": float(dt.median()) if not dt.empty else np.nan,
        "max_sampling_s": float(dt.max()) if not dt.empty else np.nan,
        "sequence_gaps": int((dseq > 1).sum()),
        "duplicate_sequence_ids": int(seq.duplicated().sum()),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "baseline_rows": int(len(baseline)),
        "fault_rows": int(len(fault)),
        "recovery_rows": int(len(recovery)),
        "baseline_f0_ok": baseline_f0,
        "fault_code_ok": fault_code_match,
        "severity_ok": severity_match,
        "recovery_f0_ok": recovery_f0,
        "quality_flag": quality_flag,
        "run_inclusion_status": inclusion_status,
        "inclusion_reason": inclusion_reason,
        "warnings": ";".join(warnings),
    }


def run() -> dict:
    ensure_dirs(WP1_DIR, WP1_DIR / "figures")
    files = sorted(EXPERIMENT_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No experiment CSV files found in {EXPERIMENT_DIR}")

    frames, quality = [], []
    text_cols = {
        "source_file", "run_id", "experiment_phase", "selected_fault_code", "asset_id", "schema_version",
        "model_version", "stream_status", "synchronisation_status", "control_mode", "operating_mode",
        "health_status", "embedded_fault_code", "embedded_fault_label", "ai_fault_label", "ai_severity"
    }
    for path in files:
        df = read_csv_robust(path)
        df["source_file"] = path.name
        df["ts_iso"] = pd.to_datetime(df["ts_iso"], errors="coerce", utc=True)
        for col in df.columns:
            if col in text_cols:
                continue
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= df[col].notna().sum() * 0.95:
                df[col] = converted
        quality.append(_quality_for_run(df, path.name))
        frames.append(df)

    master = pd.concat(frames, ignore_index=True, sort=False)
    quality_df = pd.DataFrame(quality).sort_values(["fault_code", "severity", "replicate"])
    inclusion = quality_df[["run_id", "run_inclusion_status", "inclusion_reason", "quality_flag", "warnings"]].copy()
    master = master.merge(inclusion[["run_id", "run_inclusion_status"]], on="run_id", how="left")

    master.to_csv(WP1_DIR / "experiments_master_clean.csv", index=False)
    quality_df.to_csv(WP1_DIR / "run_quality_audit.csv", index=False)
    inclusion.to_csv(WP1_DIR / "run_inclusion_manifest.csv", index=False)

    manifest = {
        "input_directory": str(EXPERIMENT_DIR),
        "files": len(files),
        "rows": int(len(master)),
        "columns": int(master.shape[1]),
        "quality_counts": quality_df["quality_flag"].value_counts().to_dict(),
        "inclusion_counts": quality_df["run_inclusion_status"].value_counts().to_dict(),
        "policy": "PRIMARY for manuscript main analysis; SENSITIVITY_ONLY retained separately; EXCLUDE removed from primary analyses"
    }
    save_json(manifest, WP1_DIR / "wp1_preprocess_manifest.json")
    return manifest


if __name__ == "__main__":
    print(run())
