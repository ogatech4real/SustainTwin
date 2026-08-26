import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import WP1_DIR, WP2_DIR
from .utils import ensure_dirs, robust_iqr_summary, save_heatmap, save_json


def _rename_real(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "Speed (RPM)": "rpm",
        "Motor Power ( kW)": "power",
        "Fan Vibration (mm/s)": "vibration",
        "Bag Filter Outlet Temp.": "temperature",
        "Bag Filter Inlet Temp.": "temperature_inlet",
        "Current": "current",
        "Bag Filter Diff. Pressure 2": "diff_pressure",
    })


def _rename_synth(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={
        "actual_rpm": "rpm",
        "power_w": "power",
        "vibration_g": "vibration",
        "temperature_c": "temperature",
        "current_a": "current",
    })


def _safe_pair_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    g = pd.concat([pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")], axis=1).dropna()
    if len(g) < 3 or g.iloc[:, 0].nunique() < 2 or g.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(g.iloc[:, 0].corr(g.iloc[:, 1], method=method))


def _corr(df: pd.DataFrame, cols: list[str], method: str) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for a in cols:
        out.loc[a, a] = 1.0 if pd.to_numeric(df[a], errors="coerce").nunique() >= 2 else np.nan
        for b in cols:
            if a != b:
                out.loc[a, b] = _safe_pair_corr(df[a], df[b], method)
    return out


def _percentile_normalized_slope(df: pd.DataFrame, x: str, y: str) -> tuple[float, int]:
    g = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(g) < 5:
        return np.nan, len(g)
    xlo, xhi = g[x].quantile([0.05, 0.95]); ylo, yhi = g[y].quantile([0.05, 0.95])
    if xhi == xlo or yhi == ylo:
        return np.nan, len(g)
    xn = (g[x] - xlo) / (xhi - xlo); yn = (g[y] - ylo) / (yhi - ylo)
    return float(np.polyfit(xn, yn, 1)[0]), len(g)


def _relationship_table(synth: pd.DataFrame, real: pd.DataFrame, pairs: list[tuple[str, str]], domain: str) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        sg = synth[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        rg = real[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        sp = _safe_pair_corr(sg[a], sg[b], "pearson")
        rp = _safe_pair_corr(rg[a], rg[b], "pearson")
        ss = _safe_pair_corr(sg[a], sg[b], "spearman")
        rs = _safe_pair_corr(rg[a], rg[b], "spearman")
        s_slope, sn = _percentile_normalized_slope(sg, a, b)
        r_slope, rn = _percentile_normalized_slope(rg, a, b)
        empirical = pd.notna(sp) and pd.notna(rp)
        rows.append({
            "validation_domain": domain,
            "variable_1": a,
            "variable_2": b,
            "twin_pearson_r": sp,
            "real_pearson_r": rp,
            "pearson_empirically_comparable": empirical,
            "pearson_sign_agreement": bool(np.sign(sp) == np.sign(rp)) if empirical else np.nan,
            "pearson_abs_difference": abs(sp-rp) if empirical else np.nan,
            "twin_spearman_rho": ss,
            "real_spearman_rho": rs,
            "spearman_empirically_comparable": pd.notna(ss) and pd.notna(rs),
            "spearman_sign_agreement": bool(np.sign(ss) == np.sign(rs)) if pd.notna(ss) and pd.notna(rs) else np.nan,
            "twin_normalized_slope": s_slope,
            "real_normalized_slope": r_slope,
            "normalized_slope_empirically_comparable": pd.notna(s_slope) and pd.notna(r_slope),
            "normalized_slope_sign_agreement": bool(np.sign(s_slope) == np.sign(r_slope)) if pd.notna(s_slope) and pd.notna(r_slope) else np.nan,
            "twin_n": sn,
            "real_n": rn,
            "twin_variable_1_unique": int(pd.to_numeric(sg[a], errors="coerce").nunique()),
            "twin_variable_2_unique": int(pd.to_numeric(sg[b], errors="coerce").nunique()),
            "real_variable_1_unique": int(pd.to_numeric(rg[a], errors="coerce").nunique()),
            "real_variable_2_unique": int(pd.to_numeric(rg[b], errors="coerce").nunique()),
            "evidence_class": "EMPIRICAL_CROSS_DOMAIN" if empirical else "INDUSTRIAL_RELATIONSHIP_PLUS_TWIN_MODEL_STRUCTURE",
            "interpretation_rule": "Only finite relationships from both datasets count as empirical cross-domain agreement. Undefined twin relationships are never scored as agreement.",
        })
    return pd.DataFrame(rows)


def _summary_table(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    finite_p = df["pearson_empirically_comparable"].fillna(False)
    finite_s = df["spearman_empirically_comparable"].fillna(False)
    finite_sl = df["normalized_slope_empirically_comparable"].fillna(False)
    p_agree = df.loc[finite_p, "pearson_sign_agreement"].dropna()
    s_agree = df.loc[finite_s, "spearman_sign_agreement"].dropna()
    sl_agree = df.loc[finite_sl, "normalized_slope_sign_agreement"].dropna()
    return pd.DataFrame([{
        "domain": domain,
        "pairs_total": int(len(df)),
        "pearson_pairs_empirically_comparable": int(finite_p.sum()),
        "pearson_pairs_undefined_for_cross_domain": int((~finite_p).sum()),
        "pearson_sign_agreement_pct_comparable_only": float(p_agree.mean()*100) if len(p_agree) else np.nan,
        "spearman_pairs_empirically_comparable": int(finite_s.sum()),
        "spearman_sign_agreement_pct_comparable_only": float(s_agree.mean()*100) if len(s_agree) else np.nan,
        "normalized_slope_pairs_empirically_comparable": int(finite_sl.sum()),
        "normalized_slope_sign_agreement_pct_comparable_only": float(sl_agree.mean()*100) if len(sl_agree) else np.nan,
        "mean_abs_pearson_difference_comparable_only": float(df.loc[finite_p, "pearson_abs_difference"].mean()) if finite_p.any() else np.nan,
        "claim_boundary": "Agreement percentages use only relationships empirically estimable in both domains; undefined twin correlations are reported, not imputed or counted as agreement.",
    }])


def _plot_relationship(real: pd.DataFrame, synth: pd.DataFrame, x: str, y: str, outpath, title: str):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plotted = 0
    for label, d in [("Industrial", real), ("SustainTwin F0", synth)]:
        g = d[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(g) < 3:
            continue
        xlo, xhi = g[x].quantile([0.05, 0.95]); ylo, yhi = g[y].quantile([0.05, 0.95])
        if xhi == xlo or yhi == ylo:
            continue
        xn = (g[x]-xlo)/(xhi-xlo); yn = (g[y]-ylo)/(yhi-ylo)
        ax.scatter(xn, yn, s=9, alpha=0.28, label=label)
        coef = np.polyfit(xn, yn, 1)
        xx = np.linspace(max(-0.1, xn.quantile(0.01)), min(1.1, xn.quantile(0.99)), 50)
        ax.plot(xx, np.polyval(coef, xx), linewidth=2)
        plotted += 1
    ax.set_xlabel(f"{x} (within-domain normalized)"); ax.set_ylabel(f"{y} (within-domain normalized)")
    ax.set_title(title)
    if plotted:
        ax.legend()
    ax.grid(True, alpha=0.2); fig.tight_layout(); fig.savefig(outpath, dpi=220, bbox_inches="tight"); plt.close(fig)


def run() -> dict:
    ensure_dirs(WP2_DIR / "figures")
    synth_all = pd.read_csv(WP1_DIR / "experiments_master_clean.csv")
    real = pd.read_csv(WP2_DIR / "real_fan_normal_operating.csv")
    real_current = pd.read_csv(WP2_DIR / "real_fan_current_overlap.csv")

    if "run_inclusion_status" in synth_all.columns:
        synth_all = synth_all[synth_all["run_inclusion_status"] == "PRIMARY"].copy()
    s0 = _rename_synth(synth_all[synth_all["embedded_fault_code"].astype(str) == "F0"].copy())
    r0 = _rename_real(real.copy()); rc = _rename_real(real_current.copy())

    common4 = ["rpm", "power", "vibration", "temperature"]
    mech_pairs = [("rpm", "power"), ("rpm", "vibration"), ("power", "vibration")]
    thermal_pairs = [("rpm", "temperature"), ("power", "temperature"), ("vibration", "temperature")]
    current_pairs = [("current", "power"), ("current", "rpm"), ("current", "vibration")]

    for method in ["pearson", "spearman"]:
        sc = _corr(s0, common4, method); rcorr = _corr(r0, common4, method)
        sc.to_csv(WP2_DIR / f"sustaintwin_f0_common_{method}_correlation.csv")
        rcorr.to_csv(WP2_DIR / f"industrial_fan_common_{method}_correlation.csv")
        (rcorr-sc).to_csv(WP2_DIR / f"correlation_difference_real_minus_twin_{method}.csv")
        save_heatmap(sc, f"SustainTwin F0 {method.title()} correlation", WP2_DIR / "figures" / f"wp2_twin_{method}_correlation.png", cbar_label=("Pearson r" if method == "pearson" else "Spearman ρ"))
        save_heatmap(rcorr, f"Industrial fan {method.title()} correlation", WP2_DIR / "figures" / f"wp2_real_{method}_correlation.png", cbar_label=("Pearson r" if method == "pearson" else "Spearman ρ"))

    electromech = _relationship_table(s0, r0, mech_pairs, "electromechanical")
    electromech.to_csv(WP2_DIR / "electromechanical_validation.csv", index=False)
    em_summary = _summary_table(electromech, "electromechanical")
    em_summary.to_csv(WP2_DIR / "electromechanical_validation_summary.csv", index=False)

    thermal = _relationship_table(s0, r0, thermal_pairs, "thermal_context")
    thermal.to_csv(WP2_DIR / "thermal_relationship_context.csv", index=False)
    _summary_table(thermal, "thermal_context").to_csv(WP2_DIR / "thermal_relationship_summary.csv", index=False)

    current = _relationship_table(s0, rc, current_pairs, "current_secondary")
    current.to_csv(WP2_DIR / "current_secondary_validation.csv", index=False)
    _summary_table(current, "current_secondary").to_csv(WP2_DIR / "current_secondary_validation_summary.csv", index=False)

    env_rows = []
    for domain, d in [("SustainTwin_F0", s0), ("Industrial_normal", r0)]:
        for c in common4:
            env_rows.append({"domain": domain, "variable": c, **robust_iqr_summary(d[c])})
    pd.DataFrame(env_rows).to_csv(WP2_DIR / "cross_domain_operating_envelopes.csv", index=False)

    thermal_context = []
    for domain, d in [("SustainTwin_F0", s0), ("Industrial_normal", r0)]:
        t = pd.to_numeric(d["temperature"], errors="coerce")
        thermal_context.append({"domain": domain, "temperature_n": int(t.notna().sum()),
                                "temperature_mean": t.mean(), "temperature_std": t.std(),
                                "temperature_cv": (t.std()/abs(t.mean())) if t.notna().sum() > 1 and t.mean() != 0 else np.nan,
                                "temperature_min": t.min(), "temperature_max": t.max()})
    if "temperature_inlet" in r0.columns:
        dt = pd.to_numeric(r0["temperature_inlet"], errors="coerce") - pd.to_numeric(r0["temperature"], errors="coerce")
        thermal_context.append({"domain": "Industrial_deltaT", "temperature_n": int(dt.notna().sum()),
                                "temperature_mean": dt.mean(), "temperature_std": dt.std(),
                                "temperature_cv": (dt.std()/abs(dt.mean())) if dt.notna().sum() > 1 and dt.mean() != 0 else np.nan,
                                "temperature_min": dt.min(), "temperature_max": dt.max()})
    pd.DataFrame(thermal_context).to_csv(WP2_DIR / "thermal_context_summary.csv", index=False)

    ep = electromech.set_index(["variable_1", "variable_2"]); cp = current.set_index(["variable_1", "variable_2"])
    mech = pd.DataFrame([
        {"fault": "F1 mechanical imbalance", "twin_mechanism": "vibration/current/power increase",
         "industrial_evidence": "power-vibration and current-vibration coupling",
         "real_pearson_primary": ep.loc[("power", "vibration"), "real_pearson_r"] if ("power", "vibration") in ep.index else np.nan,
         "real_pearson_secondary": cp.loc[("current", "vibration"), "real_pearson_r"] if ("current", "vibration") in cp.index else np.nan,
         "claim": "supportive mechanistic plausibility; no real imbalance label"},
        {"fault": "F2 electrical overcurrent", "twin_mechanism": "current/power increase",
         "industrial_evidence": "current-power coupling",
         "real_pearson_primary": cp.loc[("current", "power"), "real_pearson_r"] if ("current", "power") in cp.index else np.nan,
         "real_pearson_secondary": np.nan,
         "claim": "supportive electrical coupling; no real overcurrent event label"},
        {"fault": "F3 cooling degradation", "twin_mechanism": "degraded cooling changes thermal state",
         "industrial_evidence": "broad real process-temperature variability under fan operation",
         "real_pearson_primary": np.nan, "real_pearson_secondary": np.nan,
         "claim": "thermal context only; not direct cooling-degradation validation"},
        {"fault": "F4 thermal overload", "twin_mechanism": "additional heat load raises temperature",
         "industrial_evidence": "real thermal operating envelope and inlet-outlet temperature difference",
         "real_pearson_primary": np.nan, "real_pearson_secondary": np.nan,
         "claim": "thermal context only; not direct overload validation"},
    ])
    mech.to_csv(WP2_DIR / "mechanistic_plausibility_evidence.csv", index=False)

    for x, y in mech_pairs:
        _plot_relationship(r0, s0, x, y, WP2_DIR / "figures" / f"wp2_normalized_{x}_{y}.png", f"External behavioural comparison: {x} vs {y}")

    rplot = r0.copy()
    if "timestamp" in rplot.columns:
        rplot["timestamp"] = pd.to_datetime(rplot["timestamp"], errors="coerce"); rplot = rplot.sort_values("timestamp")
        fig, ax = plt.subplots(figsize=(10, 4))
        for c in common4:
            s = pd.to_numeric(rplot[c], errors="coerce"); std = s.std(ddof=0); z = (s-s.mean())/std if std else s*0
            ax.plot(rplot["timestamp"], z, label=c, linewidth=1)
        ax.set_title("Industrial fan: normalized natural operating variability"); ax.set_ylabel("Within-domain z-score")
        ax.legend(ncol=4); ax.grid(True, alpha=0.25); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(WP2_DIR / "figures" / "wp2_real_normalized_variability.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    erow = em_summary.iloc[0]
    manifest = {
        "synthetic_healthy_rows": int(len(s0)),
        "industrial_normal_rows": int(len(r0)),
        "real_current_overlap_rows": int(len(rc)),
        "electromechanical_pairs_total": int(erow["pairs_total"]),
        "electromechanical_pairs_empirically_comparable": int(erow["pearson_pairs_empirically_comparable"]),
        "electromechanical_pairs_undefined_for_cross_domain": int(erow["pearson_pairs_undefined_for_cross_domain"]),
        "electromechanical_pearson_sign_agreement_pct_comparable_only": None if pd.isna(erow["pearson_sign_agreement_pct_comparable_only"]) else float(erow["pearson_sign_agreement_pct_comparable_only"]),
        "validation_boundary": "Industrial data validate behavioural/operating plausibility. Undefined twin correlations are not counted as agreement. No industrial F1-F4 ground truth is inferred.",
        "outputs": str(WP2_DIR),
    }
    save_json(manifest, WP2_DIR / "wp2_validation_manifest.json")
    return manifest


if __name__ == "__main__":
    print(run())
