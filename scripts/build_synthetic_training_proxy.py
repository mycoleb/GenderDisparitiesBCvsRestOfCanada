#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def pick_col(cols: list[str], candidates: list[str]) -> str | None:
    ncols = {norm(c): c for c in cols}
    for cand in candidates:
        if norm(cand) in ncols:
            return ncols[norm(cand)]
    for cand in candidates:
        for c in cols:
            if norm(cand) in norm(c):
                return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/14100287.csv")
    ap.add_argument("--out_csv", default="data/model_microdata.csv")
    ap.add_argument("--bc_value", default="British Columbia")
    ap.add_argument("--n_per_cell", type=int, default=300)
    ap.add_argument("--max_ref_dates", type=int, default=8, help="Keep most recent N REF_DATE values to reduce size")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise SystemExit(f"Missing input CSV: {in_path}")

    cols = list(pd.read_csv(in_path, nrows=0).columns)

    col_geo = pick_col(cols, ["GEO", "Geography"])
    col_gender = pick_col(cols, ["Gender", "Sex"])
    col_ref = pick_col(cols, ["REF_DATE", "Reference period"])
    col_char = pick_col(cols, ["Labour force characteristics", "Characteristics"])
    col_val = pick_col(cols, ["VALUE", "Value"])
    col_age = pick_col(cols, ["Age group", "Age"])
    col_uom = pick_col(cols, ["UOM", "Unit of measure"])

    missing = [name for name, col in {
        "GEO": col_geo,
        "Gender/Sex": col_gender,
        "REF_DATE": col_ref,
        "Labour force characteristics": col_char,
        "VALUE": col_val,
    }.items() if col is None]

    if missing:
        raise SystemExit(f"Missing required columns: {missing}\nColumns seen: {cols}")

    usecols = [col_ref, col_geo, col_gender, col_char, col_val]
    if col_age and col_age not in usecols:
        usecols.append(col_age)
    if col_uom and col_uom not in usecols:
        usecols.append(col_uom)

    df = pd.read_csv(in_path, usecols=usecols, low_memory=False)

    # Reduce size: keep only the most recent REF_DATE values
    ref_vals = sorted(df[col_ref].dropna().unique())
    if args.max_ref_dates and len(ref_vals) > args.max_ref_dates:
        keep = set(ref_vals[-args.max_ref_dates:])
        df = df[df[col_ref].isin(keep)].copy()

    # Filter to employment rate rows
    mask = df[col_char].astype(str).str.contains("employment rate", case=False, na=False)
    if col_uom is not None:
        mask_pct = df[col_uom].astype(str).str.contains("%|percent|percentage", case=False, na=False)
        if mask_pct.any():
            mask = mask & mask_pct

    sub = df.loc[mask].copy()
    if sub.empty:
        top = df[col_char].dropna().astype(str).str.lower().value_counts().head(40)
        raise SystemExit("No rows matched 'employment rate'. Top characteristic values:\n" + top.to_string())

    sub[col_val] = pd.to_numeric(sub[col_val], errors="coerce")
    sub = sub.dropna(subset=[col_val])

    # percent -> probability
    if sub[col_val].max() > 1.5:
        sub["p_employed"] = (sub[col_val] / 100.0).clip(0, 1)
    else:
        sub["p_employed"] = sub[col_val].clip(0, 1)

    rng = np.random.default_rng(42)
    n = int(args.n_per_cell)

    rows = []
    for _, r in sub.iterrows():
        p = float(r["p_employed"])
        y = rng.binomial(1, p, size=n)

        province = str(r[col_geo]).strip()
        gender = str(r[col_gender]).strip()
        age = str(r[col_age]).strip() if col_age is not None else "Unknown"

        base = {
            "ref_date": r[col_ref],
            "province": province,
            "gender": gender,
            "age_group": age,
            "is_bc": 1 if province == args.bc_value else 0,
        }
        for yi in y:
            rows.append({**base, "employed": int(yi)})

    out = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Done. Wrote: {out_path} (rows={len(out):,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
