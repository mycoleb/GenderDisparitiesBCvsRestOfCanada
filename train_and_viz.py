import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

import shap


def normalize_str(s: str) -> str:
    return str(s).strip().lower()


def pick_heatmap_features(
    df: pd.DataFrame,
    exclude: List[str],
    max_cardinality: int = 18,
) -> Tuple[Optional[str], Optional[str]]:
    candidates = []
    for c in df.columns:
        if c in exclude:
            continue
        ser = df[c]
        nunique = ser.nunique(dropna=True)
        if nunique < 2:
            continue

        is_cat = (ser.dtype == "object") or str(ser.dtype).startswith("category")
        is_low_int = pd.api.types.is_integer_dtype(ser) and nunique <= max_cardinality
        if (is_cat or is_low_int) and nunique <= max_cardinality:
            candidates.append((c, nunique))

    candidates.sort(key=lambda x: x[1])
    if len(candidates) >= 2:
        return candidates[0][0], candidates[1][0]
    return None, None


def build_model(model_type, numeric_cols, categorical_cols, random_state=42):
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])

    if model_type == "logreg":
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    else:
        clf = GradientBoostingClassifier(random_state=random_state)

    return Pipeline([("pre", pre), ("clf", clf)])


def get_feature_names(model):
    pre = model.named_steps["pre"]
    names = []
    for name, trans, cols in pre.transformers_:
        if hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
            ohe = trans.named_steps["onehot"]
            names.extend(ohe.get_feature_names_out(cols))
        else:
            names.extend(cols)
    return names


def shap_plots(model, X, out_prefix, label):
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]
    Xt = pre.transform(X)
    names = get_feature_names(model)

    if isinstance(clf, GradientBoostingClassifier):
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt)
    else:
        bg = Xt[:min(200, Xt.shape[0])]
        explainer = shap.LinearExplainer(clf, bg)
        sv = explainer.shap_values(Xt)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, Xt, feature_names=names, show=False)
    plt.title(f"SHAP Summary — {label}")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_shap_beeswarm.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, Xt, feature_names=names, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {label}")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_shap_bar.png", dpi=200)
    plt.close()


def probability_heatmap(df, prob_col, f1, f2, outpath, title):
    pivot = df.pivot_table(index=f1, columns=f2, values=prob_col, aggfunc="mean")
    plt.figure(figsize=(10, 7))
    plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(label="Mean predicted probability")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(title)
    plt.xlabel(f2)
    plt.ylabel(f1)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="employed_post")
    ap.add_argument("--gender_col", default="gender")
    ap.add_argument("--women_value", default="female")
    ap.add_argument("--men_value", default="male")
    ap.add_argument("--province_col", default="province")
    ap.add_argument("--bc_value", default="british columbia")
    ap.add_argument("--model", choices=["logreg", "gb"], default="gb")
    args = ap.parse_args()

    out = Path("outputs")
    out.mkdir(exist_ok=True)

    df = pd.read_csv(args.csv)
    df["_y"] = pd.to_numeric(df[args.target], errors="coerce").fillna(0).astype(int)
    df["_gender"] = df[args.gender_col].map(normalize_str)
    df["_prov"] = df[args.province_col].map(normalize_str)

    features = [c for c in df.columns if c not in {
        args.target, "_y", "_gender", "_prov", args.gender_col, args.province_col
    }]

    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in features if c not in num]

    f1, f2 = pick_heatmap_features(df, exclude=[args.target, "_y", "_gender", "_prov"])

    def train(sub, label):
        X = sub[features]
        y = sub["_y"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y)
        model = build_model(args.model, num, cat)
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        print(label, "AUC:", roc_auc_score(yte, p))
        return model

    women = df[df["_gender"] == normalize_str(args.women_value)].copy()
    men = df[df["_gender"] == normalize_str(args.men_value)].copy()

    wm = train(women, "Women")
    mm = train(men, "Men")

    women["_p"] = wm.predict_proba(women[features])[:, 1]

    bc = women[women["_prov"] == normalize_str(args.bc_value)]
    roc = women[women["_prov"] != normalize_str(args.bc_value)]

    probability_heatmap(women, "_p", f1, f2,
        out / "women_all_canada_heatmap.png",
        "Women — All Canada")

    probability_heatmap(bc, "_p", f1, f2,
        out / "women_bc_heatmap.png",
        "Women — British Columbia")

    probability_heatmap(roc, "_p", f1, f2,
        out / "women_rest_of_canada_heatmap.png",
        "Women — Rest of Canada")

    shap_plots(wm, women[features], out / "women_model", "Women")
    shap_plots(mm, men[features], out / "men_model", "Men")


if __name__ == "__main__":
    main()
