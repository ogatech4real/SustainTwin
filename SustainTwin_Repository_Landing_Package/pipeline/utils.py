from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
    """Read CSV exports that may come from Windows/industrial historian encodings."""
    errors = []
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeDecodeError("unknown", b"", 0, 1, "Unable to decode CSV: " + " | ".join(errors))


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.strip().replace({"": np.nan}), errors="coerce")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def safe_corr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    return df[cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")


def zscore_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in columns:
        s = pd.to_numeric(df[c], errors="coerce")
        std = s.std(ddof=0)
        out[c] = (s - s.mean()) / std if std and not math.isnan(std) else 0.0
    return out


def save_heatmap(matrix: pd.DataFrame, title: str, path: Path, vmin=-1, vmax=1, cbar_label="Correlation") -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix.values, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def robust_iqr_summary(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan,
                "q1": np.nan, "q3": np.nan, "min": np.nan, "max": np.nan, "cv": np.nan}
    mean = s.mean()
    return {
        "n": int(s.size),
        "mean": float(mean),
        "std": float(s.std(ddof=1)) if s.size > 1 else 0.0,
        "median": float(s.median()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
        "cv": float(s.std(ddof=1) / abs(mean)) if s.size > 1 and mean != 0 else np.nan,
    }
