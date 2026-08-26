import numpy as np
import pandas as pd

from .config import REAL_DATA_FILE, WP2_DIR, REAL_NORMAL_MIN_RPM, REAL_NORMAL_MIN_POWER_KW
from .utils import ensure_dirs, read_csv_robust, save_json


def run() -> dict:
    ensure_dirs(WP2_DIR, WP2_DIR / "figures")
    df = read_csv_robust(REAL_DATA_FILE)
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()

    # Normalize the two headers affected by non-breaking spaces/spacing.
    rename = {
        "Bag Filter  Inlet Temp.": "Bag Filter Inlet Temp.",
        "Motor Power ( kW)": "Motor Power ( kW)",
    }
    df = df.rename(columns=rename)
    df["timestamp"] = pd.to_datetime(df["Timestamp"], format="%d.%m.%Y %H:%M:%S", errors="coerce")

    for col in df.columns:
        if col in {"Timestamp", "timestamp"}:
            continue
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")

    df = df[df["timestamp"].notna()].sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["minute_gap"] = df["timestamp"].diff().dt.total_seconds().div(60)

    # Derived thermal context only; not treated as equivalent to SustainTwin cooling rate.
    if "Bag Filter Inlet Temp." in df.columns and "Bag Filter Outlet Temp." in df.columns:
        df["bagfilter_delta_t_c"] = df["Bag Filter Inlet Temp."] - df["Bag Filter Outlet Temp."]

    core = ["Speed (RPM)", "Motor Power ( kW)", "Fan Vibration (mm/s)", "Bag Filter Outlet Temp."]
    df["core_complete"] = df[core].notna().all(axis=1)
    df["normal_operation"] = (
        df["core_complete"] &
        (df["Speed (RPM)"] >= REAL_NORMAL_MIN_RPM) &
        (df["Motor Power ( kW)"] >= REAL_NORMAL_MIN_POWER_KW)
    )
    df["current_available"] = df.get("Current", pd.Series(np.nan, index=df.index)).notna()

    df.to_csv(WP2_DIR / "real_fan_clean_all.csv", index=False)
    df[df["core_complete"]].to_csv(WP2_DIR / "real_fan_core_complete.csv", index=False)
    df[df["normal_operation"]].to_csv(WP2_DIR / "real_fan_normal_operating.csv", index=False)
    df[df["normal_operation"] & df["current_available"]].to_csv(WP2_DIR / "real_fan_current_overlap.csv", index=False)

    # Missingness and cadence summaries.
    summary = []
    for col in [c for c in df.columns if c not in {"Timestamp", "timestamp", "date"}]:
        summary.append({"column": col, "valid": int(df[col].notna().sum()), "missing": int(df[col].isna().sum()),
                        "valid_pct": float(df[col].notna().mean() * 100)})
    pd.DataFrame(summary).to_csv(WP2_DIR / "real_data_completeness.csv", index=False)

    cadence = (df.groupby("date")["minute_gap"]
               .agg(["count", "median", "mean", "min", "max"]).reset_index())
    cadence.to_csv(WP2_DIR / "real_sampling_cadence_by_day.csv", index=False)

    manifest = {
        "source": str(REAL_DATA_FILE),
        "rows_valid_timestamp": int(len(df)),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "core_complete_rows": int(df["core_complete"].sum()),
        "normal_operating_rows": int(df["normal_operation"].sum()),
        "normal_current_overlap_rows": int((df["normal_operation"] & df["current_available"]).sum()),
        "normal_thresholds": {"min_rpm": REAL_NORMAL_MIN_RPM, "min_power_kw": REAL_NORMAL_MIN_POWER_KW},
    }
    save_json(manifest, WP2_DIR / "wp2_preprocess_manifest.json")
    return manifest


if __name__ == "__main__":
    print(run())
