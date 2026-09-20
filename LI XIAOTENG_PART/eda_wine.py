"""Reproducible audit, cleaning demonstration and EDA for CA6000.

Run from this directory: python eda_wine.py
Requires pandas and numpy. Writes eda_results.json and SVG charts under figs/.
The source CSV is read only. The cleaning demonstration operates on a copy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "archive" / "winequality-red.csv"
FIGS = ROOT / "figs"
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
COLUMNS = FEATURES + ["quality"]
COLORS = {"low": "#b45554", "medium": "#bf8d37", "high": "#3e8978"}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(SOURCE, sep=",")
    if list(df.columns) != COLUMNS or df.shape != (1599, 12):
        raise ValueError("Unexpected CSV schema or row count")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
        raise ValueError("Non-numeric column detected")
    return df


def invalid_counts(df: pd.DataFrame) -> dict[str, int]:
    nonnegative = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "sulphates",
    ]
    return {
        "negative_chemical_values": int(df[nonnegative].lt(0).sum().sum()),
        "density_nonpositive": int(df["density"].le(0).sum()),
        "alcohol_nonpositive": int(df["alcohol"].le(0).sum()),
        "pH_outside_0_14": int((~df["pH"].between(0, 14)).sum()),
        "quality_outside_0_10_or_noninteger": int(
            ((~df["quality"].between(0, 10)) |
             (df["quality"].notna() & df["quality"].mod(1).ne(0))).sum()
        ),
        "free_sulfur_above_total": int(
            df["free sulfur dioxide"].gt(df["total sulfur dioxide"]).sum()
        ),
    }


def audit(df: pd.DataFrame) -> dict:
    quartiles = df[FEATURES].quantile([0.25, 0.75])
    lower = quartiles.loc[0.25] - 1.5 * (quartiles.loc[0.75] - quartiles.loc[0.25])
    upper = quartiles.loc[0.75] + 1.5 * (quartiles.loc[0.75] - quartiles.loc[0.25])
    predictors = df[FEATURES]
    conflict_groups = int(df.groupby(FEATURES, dropna=False)["quality"].nunique().gt(1).sum())
    return {
        "rows": len(df), "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "nonfinite_numeric_cells": int((~np.isfinite(df.to_numpy(dtype=float))).sum()),
        "duplicate_full_rows": int(df.duplicated().sum()),
        "duplicate_predictor_rows": int(predictors.duplicated().sum()),
        "predictor_groups_with_conflicting_quality": conflict_groups,
        "invalid": invalid_counts(df),
        "iqr_feature_flags": {
            col: int((df[col].lt(lower[col]) | df[col].gt(upper[col])).sum())
            for col in FEATURES
        },
        "iqr_any_feature_rows": int(((predictors.lt(lower)) | (predictors.gt(upper))).any(axis=1).sum()),
        "iqr_residual_sugar_upper": float(upper["residual sugar"]),
    }


def cleaning_demonstration(df: pd.DataFrame) -> dict:
    demo = df.copy(deep=True)
    demo.loc[0, "alcohol"] = np.nan
    demo.loc[1, "quality"] = 99
    demo.loc[2, "chlorides"] = -0.2
    demo.loc[3, "residual sugar"] = 100.0
    before = {
        "missing_cells": int(demo.isna().sum().sum()),
        "invalid_quality": int((~demo["quality"].between(0, 10)).sum()),
        "negative_chlorides": int(demo["chlorides"].lt(0).sum()),
        "residual_sugar_over_20": int(demo["residual sugar"].gt(20).sum()),
    }
    # Medians and the IQR cap are learned from the unmodified reference
    # only for this synthetic exercise. Real model preprocessing must be fitted
    # on training observations after splitting.
    demo["alcohol"] = demo["alcohol"].fillna(df["alcohol"].median())
    demo["quality"] = demo["quality"].mask(~demo["quality"].between(0, 10))
    # A target label should be recovered from a trusted record, not guessed.
    demo["quality"] = demo["quality"].fillna(df["quality"]).astype(int)
    demo["chlorides"] = demo["chlorides"].mask(demo["chlorides"].lt(0))
    demo["chlorides"] = demo["chlorides"].fillna(df["chlorides"].median())
    q1, q3 = df["residual sugar"].quantile([0.25, 0.75])
    cap = float(q3 + 1.5 * (q3 - q1))
    demo.loc[3, "residual sugar"] = demo.loc[[3], "residual sugar"].clip(upper=cap).iloc[0]
    after = {
        "missing_cells": int(demo.isna().sum().sum()),
        "invalid_quality": int((~demo["quality"].between(0, 10)).sum()),
        "negative_chlorides": int(demo["chlorides"].lt(0).sum()),
        "residual_sugar_over_20": int(demo["residual sugar"].gt(20).sum()),
    }
    assert all(v == 0 for v in after.values())
    return {
        "injected_cells": ["row 0: alcohol = NaN", "row 1: quality = 99",
                           "row 2: chlorides = -0.2", "row 3: residual sugar = 100"],
        "before": before, "after": after, "iqr_cap": cap,
        "repaired_values": {
            "alcohol_row_0": float(demo.loc[0, "alcohol"]),
            "quality_row_1": int(demo.loc[1, "quality"]),
            "chlorides_row_2": float(demo.loc[2, "chlorides"]),
            "residual_sugar_row_3": float(demo.loc[3, "residual sugar"]),
        },
        "original_dataframe_unchanged": bool(df.equals(load_data())),
    }


def label_quality(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.select(
        [df["quality"].le(4), df["quality"].le(6)],
        ["low", "medium"], default="high"
    ), index=df.index, name="class")


def descriptive(df: pd.DataFrame) -> dict:
    stats = df.agg(["count", "mean", "median", "var", "min", "max"]).T
    stats["q1"] = df.quantile(0.25)
    stats["q3"] = df.quantile(0.75)
    labels = label_quality(df)
    group_means = df.groupby(labels)[FEATURES].mean()
    dedup = df.drop_duplicates()
    dedup_labels = label_quality(dedup)
    return {
        "statistics": stats.to_dict(orient="index"),
        "skewness": df.skew(numeric_only=True).to_dict(),
        "quality_counts": {str(k): int(v) for k, v in df["quality"].value_counts().sort_index().items()},
        "class_counts": {k: int(labels.eq(k).sum()) for k in COLORS},
        "class_feature_means": group_means.to_dict(orient="index"),
        "quality_correlations": df.corr(numeric_only=True)["quality"].to_dict(),
        "selected_predictor_correlations": {
            f"{a} | {b}": float(df[a].corr(df[b]))
            for a, b in [
                ("free sulfur dioxide", "total sulfur dioxide"),
                ("fixed acidity", "pH"), ("fixed acidity", "density"),
                ("alcohol", "density"), ("alcohol", "volatile acidity"),
                ("alcohol", "total sulfur dioxide"),
                ("residual sugar", "pH"), ("alcohol", "pH"),
            ]
        },
        "deduplicated_sensitivity": {
            "rows": len(dedup),
            "class_counts": {k: int(dedup_labels.eq(k).sum()) for k in COLORS},
            "alcohol_quality_correlation": float(dedup["alcohol"].corr(dedup["quality"])),
            "volatile_acidity_quality_correlation": float(
                dedup["volatile acidity"].corr(dedup["quality"])
            ),
        },
    }


def svg_text(x: float, y: float, value: str, size: int = 14,
             anchor: str = "start", fill: str = "#28333d", weight: int = 400) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" '
            f'font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
            f'font-weight="{weight}">{escape(str(value))}</text>')


def svg_doc(width: int, height: int, pieces: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
            + "".join(pieces) + "</svg>\n")


def save_svg(name: str, width: int, height: int, pieces: list[str]) -> None:
    FIGS.mkdir(exist_ok=True)
    (FIGS / name).write_text(svg_doc(width, height, pieces), encoding="utf-8")


def draw_quality_counts(df: pd.DataFrame) -> None:
    counts = df["quality"].value_counts().sort_index()
    p = [svg_text(40, 38, "Observed sensory quality scores", 23, weight=700),
         svg_text(40, 64, "n = 1,599; bars are coloured by the report's three-class grouping", 14)]
    base, scale = 365, 0.42
    p.append('<line x1="80" y1="365" x2="770" y2="365" stroke="#66737c"/>')
    for i, (score, count) in enumerate(counts.items()):
        x = 105 + i * 108
        cls = "low" if score <= 4 else "medium" if score <= 6 else "high"
        h = count * scale
        p.append(f'<rect x="{x}" y="{base-h:.1f}" width="70" height="{h:.1f}" fill="{COLORS[cls]}"/>')
        p.append(svg_text(x+35, base-h-9, str(count), 14, "middle", weight=700))
        p.append(svg_text(x+35, base+22, str(score), 15, "middle"))
    p.append(svg_text(430, 420, "Quality score", 15, "middle"))
    for i, cls in enumerate(COLORS):
        x = 195 + i * 175
        p.append(f'<rect x="{x}" y="451" width="16" height="16" fill="{COLORS[cls]}"/>')
        p.append(svg_text(x+23, 465, cls.title(), 14))
    save_svg("eda_quality_counts.svg", 850, 495, p)


def draw_histograms(df: pd.DataFrame) -> None:
    fields = ["alcohol", "volatile acidity", "residual sugar", "density"]
    p = [svg_text(32, 35, "Selected physicochemical distributions", 23, weight=700),
         svg_text(32, 60, "Full original dataset; 24 equal-width bins per panel", 14)]
    for i, field in enumerate(fields):
        col, row = i % 2, i // 2
        x0, y0 = 54 + col * 455, 100 + row * 255
        values = df[field].to_numpy()
        hist, edges = np.histogram(values, bins=24)
        max_count = int(hist.max())
        p.append(svg_text(x0, y0, field, 17, weight=700))
        p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#66737c"/>' %
                 (x0, y0+178, x0+380, y0+178))
        for j, count in enumerate(hist):
            bh = 140 * count / max_count
            bx = x0 + j * 15.7
            p.append(f'<rect x="{bx:.1f}" y="{y0+178-bh:.1f}" width="14.5" '
                     f'height="{bh:.1f}" fill="#728dab"/>')
        p.append(svg_text(x0, y0+201, f"{edges[0]:.3f}" if field == "density" else f"{edges[0]:.2f}", 12))
        p.append(svg_text(x0+380, y0+201, f"{edges[-1]:.3f}" if field == "density" else f"{edges[-1]:.2f}", 12, "end"))
        p.append(svg_text(x0+380, y0+19, f"max bin n={max_count}", 12, "end"))
    save_svg("eda_feature_distributions.svg", 950, 630, p)


def draw_class_profiles(df: pd.DataFrame) -> None:
    labels = label_quality(df)
    means = df.groupby(labels)[["alcohol", "volatile acidity", "sulphates"]].mean()
    p = [svg_text(35, 38, "Feature means by model class", 23, weight=700),
         svg_text(35, 63, "Descriptive group means on all rows; axes use feature-specific scales", 14)]
    fields = ["alcohol", "volatile acidity", "sulphates"]
    for i, field in enumerate(fields):
        y = 105 + i * 195
        vals = means.loc[list(COLORS), field]
        left = max(0.0, float(vals.min()) * 0.8)
        right = float(vals.max()) * 1.13
        p.append(svg_text(38, y, field, 17, weight=700))
        for j, cls in enumerate(COLORS):
            yy = y + 31 + j * 39
            p.append(svg_text(45, yy+4, cls.title(), 14))
            x = 150 + (float(vals[cls])-left)/(right-left)*470
            p.append(f'<line x1="150" y1="{yy}" x2="620" y2="{yy}" stroke="#e4e8eb"/>')
            p.append(f'<circle cx="{x:.1f}" cy="{yy}" r="8" fill="{COLORS[cls]}"/>')
            p.append(svg_text(640, yy+4, f"{vals[cls]:.3f}", 14))
    save_svg("eda_class_profiles.svg", 760, 700, p)


def heat_color(value: float) -> str:
    t = min(abs(value), 1.0)
    target = (56, 126, 171) if value >= 0 else (185, 91, 78)
    rgb = tuple(round(246 * (1-t) + c*t) for c in target)
    return "#%02x%02x%02x" % rgb


def draw_heatmap(df: pd.DataFrame) -> None:
    corr = df.corr(numeric_only=True)
    names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
             "chlorides", "free SO₂", "total SO₂", "density", "pH",
             "sulphates", "alcohol", "quality"]
    p = [svg_text(25, 34, "Pearson correlation matrix", 23, weight=700),
         svg_text(25, 59, "Blue: positive; red: negative; values rounded to two decimals", 14)]
    x0, y0, size = 184, 200, 49
    for j, name in enumerate(names):
        x = x0+j*size+size/2
        p.append(f'<text x="{x:.1f}" y="{y0-13}" transform="rotate(-55 {x:.1f} {y0-13})" '
                 f'font-family="Arial,sans-serif" font-size="11" text-anchor="start" '
                 f'fill="#28333d">{escape(name)}</text>')
        p.append(svg_text(x0-9, y0+j*size+31, name, 11, "end"))
    for i, col_i in enumerate(COLUMNS):
        for j, col_j in enumerate(COLUMNS):
            val = float(corr.loc[col_i, col_j])
            x, y = x0+j*size, y0+i*size
            p.append(f'<rect x="{x}" y="{y}" width="{size-1}" height="{size-1}" '
                     f'fill="{heat_color(val)}"/>')
            p.append(svg_text(x+size/2, y+30, f"{val:.2f}", 11, "middle",
                              "#ffffff" if abs(val) >= 0.58 else "#25323a"))
    save_svg("eda_correlation_heatmap.svg", 805, 825, p)


def draw_alcohol_quality_heatmap(df: pd.DataFrame) -> None:
    # Two-dimensional count map avoids overplotting the discrete quality axis.
    bins = np.arange(8.0, 15.51, 0.5)
    counts, _, _ = np.histogram2d(df["quality"], df["alcohol"],
                                  bins=[np.arange(2.5, 9.5, 1), bins])
    max_count = int(counts.max())
    x0, y0, cell_w, cell_h = 110, 105, 44, 51
    p = [svg_text(30, 36, "Wine counts by quality and alcohol", 23, weight=700),
         svg_text(30, 62, "Each cell counts wines in a 0.5-unit alcohol interval", 14)]
    for i, score in enumerate(range(3, 9)):
        p.append(svg_text(x0-17, y0+i*cell_h+32, str(score), 15, "end"))
        for j in range(len(bins)-1):
            n = int(counts[i, j])
            t = n / max_count
            c = tuple(round(245*(1-t) + v*t) for v in (52, 112, 143))
            fill = "#%02x%02x%02x" % c
            x, y = x0+j*cell_w, y0+i*cell_h
            p.append(f'<rect x="{x}" y="{y}" width="{cell_w-1}" height="{cell_h-1}" fill="{fill}"/>')
            if n:
                p.append(svg_text(x+cell_w/2, y+31, str(n), 12, "middle",
                                  "white" if t > 0.58 else "#28333d"))
    for j, edge in enumerate(bins[:-1]):
        if j % 2 == 0:
            p.append(svg_text(x0+j*cell_w+cell_w/2, y0+6*cell_h+19, f"{edge:.0f}", 12, "middle"))
    p.append(svg_text(440, 475, "Alcohol (dataset units)", 15, "middle"))
    p.append(svg_text(30, 270, "Quality", 14))
    p.append(svg_text(735, 510, f"Darkest cell = {max_count} wines", 12, "end"))
    save_svg("eda_alcohol_quality_heatmap.svg", 780, 535, p)


def draw_alcohol_ecdf(df: pd.DataFrame) -> None:
    labels = label_quality(df)
    xmin, xmax = 8.0, 15.0
    x0, y0, w, h = 90, 85, 610, 325
    p = [svg_text(35, 36, "Alcohol distributions by model class", 23, weight=700),
         svg_text(35, 61, "Empirical cumulative share at or below each alcohol level", 14)]
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        y = y0+h*(1-frac)
        p.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+w}" y2="{y:.1f}" stroke="#e3e9ed"/>')
        p.append(svg_text(x0-12, y+4, f"{frac:.2f}", 12, "end"))
    for xval in [8, 9, 10, 11, 12, 13, 14, 15]:
        x = x0+w*(xval-xmin)/(xmax-xmin)
        p.append(svg_text(x, y0+h+22, str(xval), 12, "middle"))
    for cls in COLORS:
        values = np.sort(df.loc[labels.eq(cls), "alcohol"].to_numpy())
        points = [f"{x0+w*(v-xmin)/(xmax-xmin):.1f},{y0+h*(1-(i+1)/len(values)):.1f}"
                  for i, v in enumerate(values)]
        p.append(f'<polyline points="{x0},{y0+h} '+" ".join(points)+
                 f'" fill="none" stroke="{COLORS[cls]}" stroke-width="2.4"/>')
    p.append(svg_text(x0+w/2, 465, "Alcohol (dataset units)", 15, "middle"))
    for i, cls in enumerate(COLORS):
        x = 140+i*175
        p.append(f'<line x1="{x}" y1="495" x2="{x+25}" y2="495" stroke="{COLORS[cls]}" stroke-width="3"/>')
        p.append(svg_text(x+32, 499, f"{cls.title()} (n={labels.eq(cls).sum()})", 13))
    save_svg("eda_alcohol_ecdf.svg", 760, 530, p)


def main() -> None:
    before_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    df = load_data()
    results = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "sha256": before_hash,
        "audit": audit(df),
        "cleaning_demo": cleaning_demonstration(df),
        "descriptive": descriptive(df),
    }
    draw_quality_counts(df)
    draw_histograms(df)
    draw_class_profiles(df)
    draw_heatmap(df)
    draw_alcohol_quality_heatmap(df)
    draw_alcohol_ecdf(df)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before_hash
    (ROOT / "eda_results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"sha256": before_hash, "audit": results["audit"],
                      "cleaning_demo": results["cleaning_demo"],
                      "class_counts": results["descriptive"]["class_counts"],
                      "quality_correlations": results["descriptive"]["quality_correlations"],
                      "selected_predictor_correlations": results["descriptive"]["selected_predictor_correlations"],
                      "class_feature_means": results["descriptive"]["class_feature_means"]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
