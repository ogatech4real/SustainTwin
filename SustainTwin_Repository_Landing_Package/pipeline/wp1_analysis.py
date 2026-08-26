import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import WP1_DIR, STEADY_FAULT_START_S, STEADY_FAULT_END_S
from .utils import ensure_dirs, robust_iqr_summary, save_heatmap

CORE_SIGNALS = [
    "temperature_c", "actual_rpm", "current_a", "vibration_g", "power_w",
    "cooling_rate_c_per_s", "cooling_efficiency", "thermal_resistance_c_per_w",
]


def _severity_fit(grouped: pd.Series) -> dict:
    grouped = grouped.dropna().sort_index()
    x = grouped.index.to_numpy(dtype=float)
    y = grouped.to_numpy(dtype=float)
    n_levels = len(grouped)
    out = {
        "n_severity_levels": n_levels,
        "severity_levels": ";".join(f"{v:.1f}" for v in x),
        "values": ";".join(f"{a:.1f}:{b:.6g}" for a, b in zip(x, y)),
        "slope_per_severity": np.nan,
        "r2": np.nan,
        "fit_status": "INSUFFICIENT_LEVELS",
        "fit_interpretation": "At least 3 distinct severity levels are required before R² is treated as evidence.",
    }
    if n_levels >= 2:
        coef = np.polyfit(x, y, 1)
        out["slope_per_severity"] = float(coef[0])
    if n_levels >= 3:
        pred = np.polyval(np.polyfit(x, y, 1), x)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        out["r2"] = 1.0 - ss_res / ss_tot if ss_tot else 1.0
        out["fit_status"] = "VALID_3PLUS_LEVEL_FIT"
        out["fit_interpretation"] = "R² is descriptive model-consistency evidence across 3+ controlled severity levels."
    elif n_levels == 2:
        out["fit_status"] = "TWO_LEVEL_DIRECTION_ONLY"
        out["fit_interpretation"] = "Slope/direction may be reported; R² is intentionally withheld because two points trivially define a line."
    return out


def _analyse_subset(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    steady = master[(master["experiment_phase"] == "FAULT_ACTIVE") &
                    (master["experiment_elapsed_s"] >= STEADY_FAULT_START_S) &
                    (master["experiment_elapsed_s"] <= STEADY_FAULT_END_S)].copy()
    rows = []
    for keys, g in steady.groupby(["run_id", "selected_fault_code", "selected_fault_severity"]):
        rec = {"run_id": keys[0], "fault_code": keys[1], "severity": keys[2], "n": len(g)}
        for signal in CORE_SIGNALS + ["effective_process_heat_load_w"]:
            if signal not in g.columns:
                continue
            stats = robust_iqr_summary(g[signal])
            for k, v in stats.items():
                rec[f"{signal}_{k}"] = v
        rows.append(rec)
    steady_summary = pd.DataFrame(rows)
    if not steady_summary.empty:
        steady_summary = steady_summary.sort_values(["fault_code", "severity"])

    baseline = master[(master["experiment_phase"] == "BASELINE") & (master["experiment_elapsed_s"] >= 20)].copy()
    delta_rows = []
    for run_id, fault_g in steady.groupby("run_id"):
        base_g = baseline[baseline["run_id"] == run_id]
        if base_g.empty:
            continue
        code = fault_g["selected_fault_code"].iloc[0]
        severity = float(fault_g["selected_fault_severity"].iloc[0])
        rec = {"run_id": run_id, "fault_code": code, "severity": severity}
        for s in CORE_SIGNALS + ["effective_process_heat_load_w"]:
            if s not in fault_g.columns or s not in base_g.columns:
                continue
            b = pd.to_numeric(base_g[s], errors="coerce").mean()
            f = pd.to_numeric(fault_g[s], errors="coerce").mean()
            rec[f"{s}_baseline_mean"] = b
            rec[f"{s}_fault_mean"] = f
            rec[f"{s}_delta"] = f - b
            rec[f"{s}_delta_pct"] = ((f - b) / abs(b) * 100.0) if pd.notna(b) and b != 0 else np.nan
        delta_rows.append(rec)
    deltas = pd.DataFrame(delta_rows)
    if not deltas.empty:
        deltas = deltas.sort_values(["fault_code", "severity"])

    primary = {
        "F1": ["vibration_g", "current_a", "power_w"],
        "F2": ["current_a", "power_w", "temperature_c"],
        "F3": ["temperature_c", "cooling_efficiency", "thermal_resistance_c_per_w"],
        "F4": ["temperature_c", "effective_process_heat_load_w", "power_w"],
    }
    sev_rows = []
    for fault, signals in primary.items():
        fg = steady[steady["selected_fault_code"] == fault]
        for signal in signals:
            if signal not in fg.columns:
                continue
            grouped = fg.groupby("selected_fault_severity")[signal].mean().dropna()
            if len(grouped) >= 2:
                rec = {"fault_code": fault, "signal": signal}
                rec.update(_severity_fit(grouped))
                sev_rows.append(rec)
    return steady_summary, deltas, pd.DataFrame(sev_rows)


def run() -> dict:
    ensure_dirs(WP1_DIR / "figures")
    master = pd.read_csv(WP1_DIR / "experiments_master_clean.csv")
    master["experiment_elapsed_s"] = pd.to_numeric(master["experiment_elapsed_s"], errors="coerce")
    if "run_inclusion_status" not in master.columns:
        master["run_inclusion_status"] = "PRIMARY"

    phase_summary = (master.groupby(["run_id", "selected_fault_code", "selected_fault_severity", "experiment_phase"], dropna=False)
                     .size().rename("rows").reset_index())
    phase_summary.to_csv(WP1_DIR / "phase_summary.csv", index=False)

    primary_master = master[master["run_inclusion_status"] == "PRIMARY"].copy()
    sensitivity_master = master[master["run_inclusion_status"].isin(["PRIMARY", "SENSITIVITY_ONLY"])].copy()

    steady_summary, deltas, severity = _analyse_subset(primary_master)
    steady_summary.to_csv(WP1_DIR / "fault_steady_state_summary.csv", index=False)
    deltas.to_csv(WP1_DIR / "fault_vs_baseline_deltas.csv", index=False)
    severity.to_csv(WP1_DIR / "severity_response_summary.csv", index=False)

    s_steady, s_deltas, s_severity = _analyse_subset(sensitivity_master)
    s_steady.to_csv(WP1_DIR / "sensitivity_fault_steady_state_summary.csv", index=False)
    s_deltas.to_csv(WP1_DIR / "sensitivity_fault_vs_baseline_deltas.csv", index=False)
    s_severity.to_csv(WP1_DIR / "sensitivity_severity_response_summary.csv", index=False)

    healthy = primary_master[primary_master["selected_fault_code"] == "F0"].copy()
    corr_cols = ["actual_rpm", "power_w", "vibration_g", "temperature_c", "current_a"]
    corr = healthy[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
    corr.to_csv(WP1_DIR / "synthetic_healthy_correlation.csv")
    save_heatmap(corr, "SustainTwin F0 correlation structure", WP1_DIR / "figures" / "wp1_synthetic_healthy_correlation.png")

    steady_primary = primary_master[(primary_master["experiment_phase"] == "FAULT_ACTIVE") &
                                    (primary_master["experiment_elapsed_s"] >= STEADY_FAULT_START_S) &
                                    (primary_master["experiment_elapsed_s"] <= STEADY_FAULT_END_S)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_map = {"F1": "vibration_g", "F2": "current_a", "F3": "temperature_c", "F4": "temperature_c"}
    for ax, (fault, signal) in zip(axes.ravel(), plot_map.items()):
        fg = steady_primary[steady_primary["selected_fault_code"] == fault].groupby("selected_fault_severity")[signal].mean().sort_index()
        if len(fg):
            ax.plot(fg.index, fg.values, marker="o")
        ax.set_title(f"{fault}: {signal}")
        ax.set_xlabel("Severity"); ax.set_ylabel(signal); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(WP1_DIR / "figures" / "wp1_severity_response.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    for fault in ["F1", "F2", "F3", "F4"]:
        g = primary_master[(primary_master["selected_fault_code"] == fault) &
                           np.isclose(pd.to_numeric(primary_master["selected_fault_severity"], errors="coerce"), 0.6)].copy()
        sensitivity_suffix = ""
        if g.empty:
            g = sensitivity_master[(sensitivity_master["selected_fault_code"] == fault) &
                                   np.isclose(pd.to_numeric(sensitivity_master["selected_fault_severity"], errors="coerce"), 0.6)].copy()
            sensitivity_suffix = " (sensitivity run)"
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        for signal in ["temperature_c", "current_a", "vibration_g"]:
            s = pd.to_numeric(g[signal], errors="coerce")
            std = s.std(ddof=0)
            z = (s - s.mean()) / std if std else s * 0
            ax.plot(g["experiment_elapsed_s"], z, label=signal)
        ax.axvline(60, linestyle="--", linewidth=1); ax.axvline(540, linestyle="--", linewidth=1)
        ax.set_title(f"{fault} severity 0.6: normalized temporal response{sensitivity_suffix}")
        ax.set_xlabel("Experiment elapsed time (s)"); ax.set_ylabel("Within-run z-score")
        ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
        fig.savefig(WP1_DIR / "figures" / f"wp1_{fault.lower()}_s06_temporal.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    invalid_r2 = int((severity.get("fit_status", pd.Series(dtype=str)) != "VALID_3PLUS_LEVEL_FIT").sum()) if len(severity) else 0
    return {
        "master_rows": int(len(master)),
        "primary_rows": int(len(primary_master)),
        "primary_runs": int(primary_master["run_id"].nunique()),
        "sensitivity_or_primary_runs": int(sensitivity_master["run_id"].nunique()),
        "severity_rows_with_r2_withheld": invalid_r2,
        "outputs": str(WP1_DIR),
    }


if __name__ == "__main__":
    print(run())
