import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .config import WP3_DIR, RANDOM_STATE, N_GROUP_FOLDS, FAULT_CLASSES, FAULT_ONLY_CLASSES, SEVERITIES
from .utils import ensure_dirs, save_json

META_COLS = {
    "group_run", "target_fault", "target_binary", "experiment_phase",
    "selected_fault_severity", "run_inclusion_status"
}


def _models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=160, class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1, min_samples_leaf=2, max_depth=14
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, max_iter=90, max_depth=7
        ),
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1800, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def _macro(y_true, y_pred, labels):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return float(p), float(r), float(f1)


def _metric_dict(y_true, y_pred, expected_labels):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    present_labels = [lab for lab in expected_labels if lab in set(y_true.tolist())]
    ep, er, ef = _macro(y_true, y_pred, expected_labels)
    pp, pr, pf = _macro(y_true, y_pred, present_labels) if present_labels else (np.nan, np.nan, np.nan)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision_expected_labels": ep,
        "macro_recall_expected_labels": er,
        "macro_f1_expected_labels": ef,
        "macro_precision_present_labels": pp,
        "macro_recall_present_labels": pr,
        "macro_f1_present_labels": pf,
        "present_labels": ";".join(str(x) for x in present_labels),
        "missing_expected_labels": ";".join(str(x) for x in expected_labels if x not in present_labels),
        "n_present_classes": len(present_labels),
        "n_expected_classes": len(expected_labels),
    }


def _save_confusion(y_true, y_pred, labels, stem, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(WP3_DIR / f"confusion_{stem}.csv")
    fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(cm, aspect="auto")
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(WP3_DIR / "figures" / f"confusion_{stem}.png", dpi=220, bbox_inches="tight"); plt.close(fig)


def _per_class_rows(y_true, y_pred, labels, dataset, model, task):
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return [{"dataset": dataset, "model": model, "task": task, "class": label,
             "precision": p[i], "recall": r[i], "f1": f1[i], "support": int(support[i])}
            for i, label in enumerate(labels)]


def _run_condition_table(work: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, g in work.groupby("group_run"):
        faults = [x for x in g["target_fault"].astype(str).unique().tolist() if x != "F0"]
        intended = faults[0] if faults else "F0"
        sev = pd.to_numeric(g["selected_fault_severity"], errors="coerce")
        rows.append({"group_run": str(run), "intended_fault": intended,
                     "severity": float(sev.dropna().iloc[0]) if sev.notna().any() else 0.0,
                     "run_inclusion_status": str(g["run_inclusion_status"].iloc[0])})
    return pd.DataFrame(rows)


def _class_complete_fold_map(work: pd.DataFrame, expected_labels, task_name: str, target_col: str) -> tuple[dict, str]:
    """Construct deterministic run-level folds that are class-complete where the campaign supports it.

    For F1-F4 the three severity runs are distributed across three folds with rotating offsets,
    preventing one fold from being equivalent to one severity level. F0-only runs are distributed
    across folds but F0 also appears in baseline/recovery of fault runs for full-temporal/binary tasks.
    """
    runs = _run_condition_table(work)
    mapping = {}
    offsets = {"F1": 0, "F2": 1, "F3": 2, "F4": 0, "F0": 0}
    for fault, g in runs.groupby("intended_fault"):
        g = g.sort_values(["severity", "group_run"]).reset_index(drop=True)
        off = offsets.get(fault, 0)
        for i, row in g.iterrows():
            mapping[row["group_run"]] = (i + off) % N_GROUP_FOLDS

    # Verify expected class completeness in test data for each fold.
    complete = True
    details = []
    for fold in range(N_GROUP_FOLDS):
        test_runs = {r for r, f in mapping.items() if f == fold}
        tg = work[work["group_run"].astype(str).isin(test_runs)]
        raw_present = tg[target_col].dropna().unique().tolist() if target_col in tg else []
        present = set(str(x) for x in raw_present)
        needed = set(str(x) for x in expected_labels)
        missing = needed - present
        if missing:
            complete = False
        details.append(f"fold{fold+1}:present={','.join(sorted(present))};missing={','.join(sorted(missing))}")
    status = "CLASS_COMPLETE" if complete else "BEST_EFFORT_NOT_CLASS_COMPLETE"
    return mapping, status + " | " + " | ".join(details)


def _cv_splits(work, X, y, groups, expected_labels, task_name, target_col):
    # Use class-complete deterministic grouped folds for main multiclass/binary campaign where possible.
    fold_map, status = _class_complete_fold_map(work, expected_labels, task_name, target_col)
    splits = []
    for fold in range(N_GROUP_FOLDS):
        test_mask = groups.map(fold_map).to_numpy() == fold
        te = np.flatnonzero(test_mask); tr = np.flatnonzero(~test_mask)
        if len(te) and len(tr):
            splits.append((tr, te))
    if len(splits) == N_GROUP_FOLDS:
        return splits, "deterministic_run_condition_balanced", status, fold_map

    # Defensive fallback, primarily for PRIMARY-only sensitivity analyses with missing conditions.
    cv = StratifiedGroupKFold(n_splits=N_GROUP_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(X, y, groups)), "StratifiedGroupKFold_fallback", "FALLBACK", {}


def _group_cv(df, dataset_name, task_name, target_col, labels, subset_rule=None, save_confusions=True, model_names=None):
    work = df.copy()
    if subset_rule is not None:
        work = subset_rule(work).copy()
    features = [c for c in work.columns if c not in META_COLS]
    X = work[features].apply(pd.to_numeric, errors="coerce")
    y = work[target_col]
    groups = work["group_run"].astype(str)
    valid = X.notna().all(axis=1) & y.notna() & groups.notna()
    X, y, groups, work = X.loc[valid], y.loc[valid], groups.loc[valid], work.loc[valid]
    if work["group_run"].nunique() < N_GROUP_FOLDS:
        raise ValueError(f"Not enough independent runs for {task_name}: {work['group_run'].nunique()}")

    splits, cv_method, fold_design_status, fold_map = _cv_splits(work, X, y, groups, labels, task_name, target_col)
    metric_rows, fold_rows, pred_rows, per_class_rows = [], [], [], []
    models = _models(); selected = model_names if model_names is not None else list(models.keys())

    for model_name in selected:
        all_true, all_pred, fold_metrics = [], [], []
        for fold, (tr, te) in enumerate(splits, start=1):
            mdl = clone(models[model_name]); mdl.fit(X.iloc[tr], y.iloc[tr]); pred = mdl.predict(X.iloc[te]); truth = y.iloc[te].to_numpy()
            fm = _metric_dict(truth, pred, labels)
            fm.update({"dataset": dataset_name, "model": model_name, "task": task_name, "fold": fold,
                       "cv_method": cv_method, "fold_design_status": fold_design_status})
            train_runs = sorted(groups.iloc[tr].unique()); test_runs = sorted(groups.iloc[te].unique())
            fm.update({"train_runs": ";".join(train_runs), "test_runs": ";".join(test_runs),
                       "train_rows": len(tr), "test_rows": len(te), "test_run_count": len(test_runs)})
            fold_metrics.append(fm); fold_rows.append(fm.copy())
            all_true.extend(truth.tolist()); all_pred.extend(pred.tolist())

            meta = work.iloc[te]
            for pos, idx in enumerate(meta.index):
                pred_rows.append({"dataset": dataset_name, "model": model_name, "task": task_name,
                                  "row_index": int(idx), "group_run": str(meta.loc[idx, "group_run"]),
                                  "experiment_phase": str(meta.loc[idx, "experiment_phase"]),
                                  "selected_fault_severity": meta.loc[idx, "selected_fault_severity"],
                                  "run_inclusion_status": str(meta.loc[idx, "run_inclusion_status"]),
                                  "true": truth[pos], "pred": pred[pos], "fold": fold})

        oof = _metric_dict(all_true, all_pred, labels)
        row = {"dataset": dataset_name, "model": model_name, "task": task_name,
               **{f"oof_{k}": v for k, v in oof.items()}, "features": ";".join(features),
               "cv_method": cv_method, "fold_design_status": fold_design_status}
        for key in ["accuracy", "balanced_accuracy", "macro_f1_expected_labels", "macro_f1_present_labels"]:
            vals = [f[key] for f in fold_metrics]
            row[f"{key}_mean"] = float(np.nanmean(vals)); row[f"{key}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        metric_rows.append(row)
        per_class_rows.extend(_per_class_rows(all_true, all_pred, labels, dataset_name, model_name, task_name))
        if save_confusions:
            stem = f"{task_name}_{dataset_name}_{model_name}".replace(" ", "_")
            _save_confusion(all_true, all_pred, labels, stem, f"{task_name} — {dataset_name} — {model_name}")

    return pd.DataFrame(metric_rows), pd.DataFrame(fold_rows), pd.DataFrame(pred_rows), pd.DataFrame(per_class_rows), features


def _per_run_performance(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    full = preds[preds["task"] == "full_temporal_multiclass"].copy()
    for keys, g in full.groupby(["dataset", "model", "group_run"]):
        truth = g["true"].astype(str); pred = g["pred"].astype(str)
        fault_mask = truth != "F0"; healthy_mask = truth == "F0"
        sev = pd.to_numeric(g["selected_fault_severity"], errors="coerce")
        rows.append({"dataset": keys[0], "model": keys[1], "group_run": keys[2],
                     "selected_fault_severity": sev.dropna().iloc[0] if sev.notna().any() else np.nan,
                     "run_inclusion_status": g["run_inclusion_status"].iloc[0], "rows": len(g),
                     "accuracy": accuracy_score(truth, pred),
                     "healthy_recall": float((pred[healthy_mask] == "F0").mean()) if healthy_mask.any() else np.nan,
                     "fault_multiclass_recall": float((pred[fault_mask] == truth[fault_mask]).mean()) if fault_mask.any() else np.nan,
                     "fault_detection_recall_any_fault": float((pred[fault_mask] != "F0").mean()) if fault_mask.any() else np.nan,
                     "intended_fault": next((v for v in truth.unique() if v != "F0"), "F0")})
    return pd.DataFrame(rows)


def _severity_performance(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []; fa = preds[preds["task"] == "fault_active_multiclass"].copy()
    fa["selected_fault_severity"] = pd.to_numeric(fa["selected_fault_severity"], errors="coerce")
    for keys, g in fa.groupby(["dataset", "model", "selected_fault_severity"]):
        if pd.isna(keys[2]): continue
        m = _metric_dict(g["true"], g["pred"], FAULT_ONLY_CLASSES)
        rows.append({"dataset": keys[0], "model": keys[1], "severity": keys[2], "rows": len(g), **m})
    return pd.DataFrame(rows)


def _leave_one_severity_out(df: pd.DataFrame, dataset_name: str, model_names=None) -> pd.DataFrame:
    work = df[(df["experiment_phase"].astype(str) == "FAULT_ACTIVE") & (df["target_fault"].astype(str) != "F0")].copy()
    features = [c for c in work.columns if c not in META_COLS]
    X = work[features].apply(pd.to_numeric, errors="coerce"); valid = X.notna().all(axis=1)
    work, X = work.loc[valid], X.loc[valid]; y = work["target_fault"].astype(str)
    sev = pd.to_numeric(work["selected_fault_severity"], errors="coerce"); rows = []
    models = _models(); selected = model_names if model_names is not None else list(models.keys())
    for held in SEVERITIES:
        train = ~np.isclose(sev, held); test = np.isclose(sev, held)
        if train.sum() == 0 or test.sum() == 0: continue
        for model_name in selected:
            mdl = clone(models[model_name]); mdl.fit(X.loc[train], y.loc[train]); pred = mdl.predict(X.loc[test])
            m = _metric_dict(y.loc[test], pred, FAULT_ONLY_CLASSES)
            rows.append({"dataset": dataset_name, "model": model_name, "held_out_severity": held,
                         "train_rows": int(train.sum()), "test_rows": int(test.sum()),
                         "train_runs": int(work.loc[train, "group_run"].nunique()), "test_runs": int(work.loc[test, "group_run"].nunique()),
                         **m,
                         "interpretation": "severity-transfer within the controlled digital-twin fault model; not external industrial generalisation"})
    return pd.DataFrame(rows)


def _fit_deployment_artifact(df, dataset_name, model_name, features):
    X = df[features].apply(pd.to_numeric, errors="coerce"); y = df["target_fault"].astype(str)
    valid = X.notna().all(axis=1) & y.notna(); mdl = _models()[model_name]; mdl.fit(X.loc[valid], y.loc[valid])
    artifact = {"pipeline": mdl, "feature_cols": features, "labels": FAULT_CLASSES, "training_scope": dataset_name,
                "evaluation_basis": "run-grouped class-balanced cross-validation",
                "deployment_caveat": "Artifact is fitted on all non-excluded synthetic experimental rows after evaluation; it is not independently externally validated."}
    joblib.dump(artifact, WP3_DIR / "models" / f"fan_fault_classifier_{dataset_name}.joblib")


def run() -> dict:
    ensure_dirs(WP3_DIR / "figures", WP3_DIR / "models")
    metrics_all, folds_all, preds_all, class_all, best = [], [], [], [], {}
    dataset_specs = [("ml_dataset_observable.csv", "observable"), ("ml_dataset_physics_augmented.csv", "physics_augmented")]
    loaded = {name: pd.read_csv(WP3_DIR / fname) for fname, name in dataset_specs}

    for _, name in dataset_specs:
        df = loaded[name]
        m, f, p, c, features = _group_cv(df, name, "full_temporal_multiclass", "target_fault", FAULT_CLASSES)
        metrics_all.append(m); folds_all.append(f); preds_all.append(p); class_all.append(c)
        best_row = m.sort_values("oof_macro_f1_expected_labels", ascending=False).iloc[0]
        best[name] = str(best_row["model"]); _fit_deployment_artifact(df, name, best[name], features)

    for _, name in dataset_specs:
        df = loaded[name]; chosen = [best[name]]
        m2, f2, p2, c2, _ = _group_cv(
            df, name, "fault_active_multiclass", "target_fault", FAULT_ONLY_CLASSES,
            subset_rule=lambda d: d[(d["experiment_phase"].astype(str) == "FAULT_ACTIVE") & (d["target_fault"].astype(str) != "F0")],
            model_names=chosen)
        metrics_all.append(m2); folds_all.append(f2); preds_all.append(p2); class_all.append(c2)
        m3, f3, p3, c3, _ = _group_cv(df, name, "binary_healthy_fault", "target_binary", [0, 1], model_names=chosen)
        metrics_all.append(m3); folds_all.append(f3); preds_all.append(p3); class_all.append(c3)

    metrics = pd.concat(metrics_all, ignore_index=True); folds = pd.concat(folds_all, ignore_index=True)
    preds = pd.concat(preds_all, ignore_index=True); per_class = pd.concat(class_all, ignore_index=True)
    metrics.to_csv(WP3_DIR / "model_comparison_metrics.csv", index=False); folds.to_csv(WP3_DIR / "grouped_cv_fold_metrics.csv", index=False)
    preds.to_csv(WP3_DIR / "grouped_cv_predictions.csv", index=False); per_class.to_csv(WP3_DIR / "per_class_performance.csv", index=False)

    # Dedicated fold audit makes class completeness transparent.
    fold_audit = folds[["dataset", "model", "task", "fold", "cv_method", "present_labels", "missing_expected_labels",
                        "n_present_classes", "n_expected_classes", "test_runs", "test_rows", "fold_design_status"]].copy()
    fold_audit.to_csv(WP3_DIR / "grouped_cv_fold_class_audit.csv", index=False)

    _per_run_performance(preds).to_csv(WP3_DIR / "per_run_performance.csv", index=False)
    _severity_performance(preds).to_csv(WP3_DIR / "per_severity_fault_active_performance.csv", index=False)

    loso = pd.concat([_leave_one_severity_out(loaded[name], name, model_names=[best[name]]) for _, name in dataset_specs], ignore_index=True)
    loso.to_csv(WP3_DIR / "leave_one_severity_out_fault_active.csv", index=False)

    full_metrics = metrics[metrics["task"] == "full_temporal_multiclass"].copy()
    ablation = full_metrics.pivot_table(index=["model", "task"], columns="dataset", values="oof_macro_f1_expected_labels").reset_index()
    if "observable" in ablation.columns and "physics_augmented" in ablation.columns:
        ablation["oof_macro_f1_gain_physics"] = ablation["physics_augmented"] - ablation["observable"]
    ablation.to_csv(WP3_DIR / "physics_feature_ablation.csv", index=False)

    sensitivity_rows = []
    for fname, name in [("ml_dataset_observable_primary_only.csv", "observable"), ("ml_dataset_physics_augmented_primary_only.csv", "physics_augmented")]:
        dfp = pd.read_csv(WP3_DIR / fname)
        try:
            sm, _, _, _, _ = _group_cv(dfp, name, "primary_only_full_temporal", "target_fault", FAULT_CLASSES,
                                        save_confusions=False, model_names=[best[name]])
            sensitivity_rows.append(sm)
        except ValueError as exc:
            sensitivity_rows.append(pd.DataFrame([{"dataset": name, "task": "primary_only_full_temporal", "error": str(exc)}]))
    pd.concat(sensitivity_rows, ignore_index=True, sort=False).to_csv(WP3_DIR / "primary_only_sensitivity_metrics.csv", index=False)

    rep = pd.read_csv(WP3_DIR / "experimental_replication_audit.csv")
    rep["replication_sufficient_for_variance_estimation"] = rep["independent_runs"] >= 3
    rep.to_csv(WP3_DIR / "experimental_replication_audit.csv", index=False)

    manifest = {
        "best_deployment_models_by_full_temporal_oof_macro_f1": best,
        "evaluation": "3-fold run-grouped deterministic condition-balanced CV; fallback to StratifiedGroupKFold only when class-complete grouping is impossible",
        "fold_metric_correction": "Both expected-label and present-label macro metrics are reported. Absent test-fold classes no longer silently depress the present-label macro-F1.",
        "fold_class_audit": "grouped_cv_fold_class_audit.csv",
        "model_benchmark_scope": "All three candidate models benchmarked on full-temporal F0-F4; secondary tasks use the selected model per feature set.",
        "tasks": ["full temporal F0-F4", "fault-active F1-F4", "binary healthy/fault", "leave-one-severity-out F1-F4"],
        "replication_warning": "Most fault×severity conditions have one independent run. Run grouping prevents row leakage but does not create replicated-condition variance estimates.",
        "severity_transfer_boundary": "Leave-one-severity-out measures transfer within the programmed controlled severity continuum, not unseen-machine or industrial-domain generalisation.",
        "deployment_warning": "Saved joblib artifacts are fitted to all non-excluded synthetic data after grouped evaluation and are not external industrial validation models.",
        "rows_within_run_never_split_across_train_test": True,
    }
    save_json(manifest, WP3_DIR / "wp3_training_manifest.json"); return manifest


if __name__ == "__main__":
    print(run())
