#!/usr/bin/env python3
"""
dataset_feasibility_audit.py -- "measure before you build" for tabular ML projects.

Answers, from the actual CSV and in about a minute: is there anything here worth
building on, and which traps would silently inflate the results?

  1. PROVENANCE   fingerprints of generated data (heuristic flags, never a verdict)
  2. LEAKAGE      single-feature scan; near-deterministic features; post-outcome names
  3. SIGNAL       holdout + CV vs baseline vs a label-permutation null at THIS sample size
  4. ADEQUACY     events-per-variable
  5. STRUCTURE    random vs grouped (entity) vs forward-in-time validation; survivorship
  6. FAIRNESS     group base rates, selection-rate ratio, per-group AUC (screen only)
  7. BENCHMARK    observed effect size vs a published upper bound you supply

Usage examples
  python dataset_feasibility_audit.py placement.csv --target Placement --id Student_ID \
         --sensitive Gender,Branch
  python dataset_feasibility_audit.py retention.csv --target Dropout --group Student_ID \
         --time Term --sensitive Gender
  # re-run after removing a suspected leaky column:
  python dataset_feasibility_audit.py data.csv --target y --drop suspicious_col

Only the dataset author's disclosure can actually prove data is synthetic. Read the
Kaggle description for "synthetic", "simulated" or "generated" first.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
VERBOSE = True
REPORT: dict = {"gates": {}, "details": {}}

POST_OUTCOME_HINT = re.compile(
    r"(salary|package|ctc|offer|company|employer|joined|final|result|status|placed|"
    r"outcome|graduat|withdraw|dropout|retain|churn|total_terms|n_terms|terms_enrolled)", re.I)


def say(s: str = "") -> None:
    if VERBOSE:
        print(s)


def gate(name: str, status: str, detail: str) -> None:
    REPORT["gates"][name] = {"status": status, "detail": detail}
    say(f"  [{status:4}] {name}: {detail}")



# ----------------------------------------------------------------------------- modelling helpers
def split_cols(df, cols):
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    return num, [c for c in cols if c not in num]


def prep(num, cat, linear=False):
    parts = []
    if linear:
        if num:
            parts.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num))
        if cat:
            parts.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                               OneHotEncoder(handle_unknown="ignore")), cat))
    else:
        if num:
            parts.append(("num", "passthrough", num))
        if cat:
            parts.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                                encoded_missing_value=-1), cat))
    return ColumnTransformer(parts)


def make_model(task, kind, num, cat, shallow=False):
    if kind == "linear":
        est = Ridge(alpha=1.0) if task == "regression" else LogisticRegression(max_iter=1000)
        return make_pipeline(prep(num, cat, linear=True), est)
    depth, iters = (3, 50) if shallow else (4, 120)
    cls = HistGradientBoostingRegressor if task == "regression" else HistGradientBoostingClassifier
    return make_pipeline(prep(num, cat), cls(max_depth=depth, max_iter=iters, early_stopping=False,
                                             random_state=SEED))


def fit_score(model, Xtr, ytr, Xte, yte, task):
    model.fit(Xtr, ytr)
    if task == "regression":
        return r2_score(yte, model.predict(Xte))
    p = model.predict_proba(Xte)
    if task == "binary":
        return roc_auc_score(yte, p[:, 1])
    return roc_auc_score(yte, p, multi_class="ovr", average="weighted", labels=np.arange(p.shape[1]))


def cv_scores(model, X, y, task, groups=None, n_splits=5):
    if groups is not None:
        it = GroupKFold(n_splits=n_splits).split(X, y, groups)
    elif task == "regression":
        it = KFold(n_splits, shuffle=True, random_state=SEED).split(X)
    else:
        it = StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(X, y)
    out = [fit_score(clone(model), X.iloc[a], y[a], X.iloc[b], y[b], task) for a, b in it]
    return float(np.mean(out)), float(np.std(out))


def prep_target(raw):
    s = pd.Series(raw)
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
        return s.astype(float).values, "regression"
    y = LabelEncoder().fit_transform(s.astype(str))
    return y, ("binary" if len(np.unique(y)) == 2 else "multiclass")


# ----------------------------------------------------------------------------- 1. provenance
def sequential_id_cols(df):
    hits = []
    for c in df.columns:
        s = df[c]
        if s.nunique() < 0.999 * len(df):
            continue
        num = s.astype(str).str.extract(r"(\d+)\s*$")[0].dropna()
        if len(num) < 0.99 * len(df):
            continue
        d = np.diff(np.sort(num.astype(int).values))
        if (d == 1).mean() > 0.99:
            hits.append(c)
    return hits


def partition_pairs(df, cat_cols, num_cols, max_levels=12):
    hits = []
    for c in cat_cols:
        if not 2 <= df[c].nunique() <= max_levels:
            continue
        for n in num_cols:
            g = df.groupby(c)[n].agg(["min", "max"]).sort_values("min")
            if len(g) >= 2 and (g["min"].values[1:] > g["max"].values[:-1]).all():
                hits.append((c, n))
    return hits


def textbook_marginals(df, num_cols):
    tested, hits = 0, []
    for c in num_cols:
        s = df[c].dropna()
        if s.nunique() < 30 or len(s) < 500:
            continue
        tested += 1
        s = s.sample(min(len(s), 5000), random_state=SEED)
        z = (s - s.min()) / (s.max() - s.min() + 1e-12)
        if stats.normaltest(s).pvalue > 0.05:
            hits.append((c, "normal"))
        elif stats.kstest(z, "uniform").pvalue > 0.05:
            hits.append((c, "uniform"))
    return tested, hits


def section_provenance(df, feats_num, feats_cat):
    say("\n== 1. PROVENANCE: does this look generated? (heuristic flags) ==")
    flags = []
    if int(df.isna().sum().sum()) == 0 and len(df) >= 1000:
        flags.append(("zero_missing_cells", "weak", f"0 missing cells in {len(df):,} rows x {df.shape[1]} cols"))
    for c in sequential_id_cols(df):
        flags.append(("sequential_ids", "weak", f"'{c}' is perfectly sequential"))
    for c, n in partition_pairs(df, feats_cat, [x for x in feats_num if df[x].nunique() >= 8]):
        flags.append(("perfect_partition", "strong",
                      f"levels of '{c}' occupy DISJOINT ranges of '{n}' (derived column or generator rule)"))
    tested, hits = textbook_marginals(df, feats_num)
    if tested >= 3 and len(hits) / tested >= 0.5:
        flags.append(("textbook_marginals", "moderate",
                      f"{len(hits)}/{tested} continuous columns pass a normal/uniform test at large n"))
    for f in flags:
        say(f"    flag[{f[1]:8}] {f[0]}: {f[2]}")
    strong = sum(f[1] == "strong" for f in flags)
    other = len(flags) - strong
    level = "HIGH" if strong else ("MEDIUM" if other >= 2 else "LOW")
    REPORT["details"]["provenance_flags"] = flags
    gate("provenance", "WARN" if level != "LOW" else "PASS",
         f"generated-data likelihood {level} ({len(flags)} flags). If HIGH/MEDIUM: results describe the "
         f"generator, not people; label every finding as simulation-based.")


# ----------------------------------------------------------------------------- 2. leakage
def section_leakage(df, feats, y, task, max_rows):
    say("\n== 2. LEAKAGE: single-feature scan (shallow trees, 5-fold) ==")
    idx = np.arange(len(df))
    if len(df) > max_rows:
        idx = np.random.default_rng(SEED).choice(len(df), max_rows, replace=False)
    d, yy = df.iloc[idx], y[idx]
    scores = {}
    for f in feats:
        num, cat = split_cols(d, [f])
        scores[f] = cv_scores(make_model(task, "hgb", num, cat, shallow=True), d[[f]], yy, task)[0]
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    metric = "R^2" if task == "regression" else "AUC"
    for f, s in ranked[:5]:
        say(f"    {f:32} single-feature {metric} = {s:.3f}")
    top_f, top = ranked[0]
    hint = [f for f in feats if POST_OUTCOME_HINT.search(f)]
    if hint:
        say(f"    name-based hint (review by hand: may be recorded AFTER the outcome): {hint}")
    REPORT["details"]["single_feature_scores"] = dict(ranked)
    if top >= 0.98:
        gate("leakage", "FAIL", f"'{top_f}' alone reaches {metric} {top:.3f}: it (nearly) defines the target. Drop it.")
    elif top >= 0.90:
        gate("leakage", "WARN", f"'{top_f}' alone reaches {metric} {top:.3f}: near-deterministic; confirm it is "
                                f"observable BEFORE the outcome.")
    else:
        gate("leakage", "PASS", f"strongest single feature '{top_f}' {metric} {top:.3f}"
             + (f"; review name hints {hint}" if hint else ""))


# ----------------------------------------------------------------------------- 3/4. signal + adequacy
def section_signal(df, feats, y, task, max_rows):
    say("\n== 3. SIGNAL: holdout + CV vs a label-permutation null at this sample size ==")
    num, cat = split_cols(df, feats)
    X = df[feats]
    Xd, Xh, yd, yh = train_test_split(X, y, test_size=0.2, random_state=SEED,
                                      stratify=None if task == "regression" else y)
    metric = "R^2" if task == "regression" else "AUC"
    cv_h = cv_scores(make_model(task, "hgb", num, cat), Xd, yd, task)
    cv_l = cv_scores(make_model(task, "linear", num, cat), Xd, yd, task)
    ho_h = fit_score(make_model(task, "hgb", num, cat), Xd, yd, Xh, yh, task)
    ho_l = fit_score(make_model(task, "linear", num, cat), Xd, yd, Xh, yh, task)
    base = 0.0 if task == "regression" else 0.5
    say(f"    baseline {metric}: {base}")
    say(f"    linear    CV {cv_l[0]:.3f} +/- {cv_l[1]:.3f}   HOLDOUT {ho_l:.3f}")
    say(f"    boosting  CV {cv_h[0]:.3f} +/- {cv_h[1]:.3f}   HOLDOUT {ho_h:.3f}")
    say(f"    boosting minus linear (holdout): {ho_h - ho_l:+.3f}  (is the complex model earning its keep?)")
    # permutation null
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(Xd), min(len(Xd), 4000), replace=False)
    Xs, ys = Xd.iloc[sub], yd[sub]
    null = [cv_scores(make_model(task, "hgb", num, cat, shallow=True), Xs, rng.permutation(ys), task, n_splits=3)[0]
            for _ in range(30)]
    n95 = float(np.percentile(null, 95))
    say(f"    label-shuffled null (same pipeline, n={len(Xs):,}): mean {np.mean(null):.3f}, 95th pct {n95:.3f}")
    REPORT["details"]["signal"] = dict(cv_hgb=cv_h, cv_linear=cv_l, holdout_hgb=ho_h, holdout_linear=ho_l,
                                       null_mean=float(np.mean(null)), null_p95=n95)
    if ho_h <= n95 + 0.01:
        gate("signal", "FAIL", f"holdout {metric} {ho_h:.3f} is indistinguishable from shuffled labels ({n95:.3f})")
    elif task != "regression" and ho_h > 0.95:
        gate("signal", "WARN", f"holdout AUC {ho_h:.3f} is implausibly high for observational data: "
                               f"suspect leakage or a generated target")
    else:
        gate("signal", "PASS", f"holdout {metric} {ho_h:.3f} vs shuffled-null p95 {n95:.3f}")
    if task != "regression":
        counts = np.bincount(y)
        n_cols = len(num) + sum(df[c].nunique() for c in cat)
        epv = counts.min() / max(n_cols, 1)
        gate("adequacy", "FAIL" if epv < 10 else ("WARN" if epv < 20 else "PASS"),
             f"events-per-variable {epv:.1f} (rarest class {counts.min():,} / ~{n_cols} encoded features)")


# ----------------------------------------------------------------------------- 5. structure
def section_structure(df, feats, y, task, group, time_col):
    say("\n== 5. STRUCTURE: does random validation overstate performance? ==")
    num, cat = split_cols(df, feats)
    X = df[feats]
    metric = "R^2" if task == "regression" else "AUC"
    if group:
        g = df[group].values
        rows = df.groupby(group).size()
        say(f"    {group}: {rows.shape[0]:,} entities, {rows.mean():.1f} rows each (min {rows.min()}, max {rows.max()})")
        rand = cv_scores(make_model(task, "hgb", num, cat), X, y, task)[0]
        grp = cv_scores(make_model(task, "hgb", num, cat), X, y, task, groups=g)[0]
        say(f"    random-row CV {metric} {rand:.3f}   vs   GROUPED-by-{group} CV {grp:.3f}   inflation {rand - grp:+.3f}")
        const = [f for f in feats if (df.groupby(group)[f].nunique(dropna=False) <= 1).mean() > 0.95]
        if const:
            say(f"    entity-constant features (memorisation risk under random splits): {const}")
        n_per = df.groupby(group)[group].transform("size")
        traj = []
        for f in num:
            gf = df.groupby(group)[f].mean()
            rho = stats.spearmanr(gf.values, df.groupby(group).size().reindex(gf.index).values).correlation
            if rho == rho and abs(rho) > 0.9:
                traj.append((f, round(float(rho), 3)))
        if traj:
            say(f"    features that track trajectory LENGTH (label leak in panels): {traj}")
        if task == "binary":
            ever = pd.Series(y, index=df.index).groupby(df[group]).max()
            n_obs = df.groupby(group).size().reindex(ever.index)
            if ever.nunique() == 2:
                say(f"    survivorship: #periods observed predicts 'ever positive' with AUC "
                    f"{roc_auc_score(ever, -n_obs):.3f} (any whole-trajectory feature leaks)")
        REPORT["details"]["group"] = dict(random=rand, grouped=grp, constant_features=const, trajectory_features=traj)
        infl = rand - grp
        gate("structure_group", "FAIL" if infl > 0.05 else ("WARN" if infl > 0.02 else "PASS"),
             f"random-vs-grouped inflation {infl:+.3f}. " +
             ("Random row splits are invalid here; use GroupKFold." if infl > 0.02 else "Groups behave."))
    if time_col:
        t = df[time_col]
        t = pd.to_datetime(t, errors="coerce").astype("int64") if t.dtype == object else t
        cut = np.nanpercentile(t, 70)
        tr, te = np.where(t <= cut)[0], np.where(t > cut)[0]
        if len(te) > 50:
            fwd = fit_score(make_model(task, "hgb", num, cat), X.iloc[tr], y[tr], X.iloc[te], y[te], task)
            rnd = cv_scores(make_model(task, "hgb", num, cat), X, y, task)[0]
            say(f"    random CV {metric} {rnd:.3f}   vs   train-early/test-late {fwd:.3f}   drift {rnd - fwd:+.3f}")
            REPORT["details"]["time"] = dict(random=rnd, forward=fwd)
            d = rnd - fwd
            gate("structure_time", "FAIL" if d > 0.08 else ("WARN" if d > 0.03 else "PASS"),
                 f"random-vs-forward gap {d:+.3f}")


# ----------------------------------------------------------------------------- 6. fairness
def section_fairness(df, feats, y, sensitive):
    say("\n== 6. FAIRNESS SCREEN (binary target; holdout; threshold = top base-rate share) ==")
    num, cat = split_cols(df, feats)
    tr, te = train_test_split(np.arange(len(df)), test_size=0.25, random_state=SEED, stratify=y)
    m = make_model("binary", "hgb", num, cat).fit(df.iloc[tr][feats], y[tr])
    p = m.predict_proba(df.iloc[te][feats])[:, 1]
    thr = np.quantile(p, 1 - y.mean())
    worst_di, worst_tpr = 1.0, 0.0
    for s in sensitive:
        sub = df.iloc[te][s].astype(str).values
        rows = []
        for lv in np.unique(sub):
            k = sub == lv
            if k.sum() < 30:
                continue
            yt, sel = y[te][k], p[k] >= thr
            tpr = sel[yt == 1].mean() if (yt == 1).any() else np.nan
            auc = roc_auc_score(yt, p[k]) if len(np.unique(yt)) == 2 else np.nan
            rows.append((lv, int(k.sum()), yt.mean(), sel.mean(), tpr, auc))
        if len(rows) >= 2:
            say(f"    {s}:")
            for lv, n, br, sr, tpr, auc in rows:
                say(f"       {lv:18} n={n:5}  base rate {br:.2f}  selected {sr:.2f}  TPR {tpr:.2f}  AUC {auc:.2f}")
            srs = [r[3] for r in rows]
            di = min(srs) / max(max(srs), 1e-9)
            tp = [r[4] for r in rows if r[4] == r[4]]
            worst_di = min(worst_di, di)
            worst_tpr = max(worst_tpr, max(tp) - min(tp) if tp else 0)
    gate("fairness", "WARN" if (worst_di < 0.8 or worst_tpr > 0.10) else "PASS",
         f"min selection-rate ratio {worst_di:.2f} (four-fifths rule = 0.80), max TPR gap {worst_tpr:.2f}. "
         f"Base-rate differences may be real OR an artefact; investigate before any deployment.")


# ----------------------------------------------------------------------------- 7. benchmark
def section_benchmark(df, spec):
    say("\n== 7. LITERATURE BENCHMARK: is the effect size plausible? ==")
    feat, targ, r2max = spec.split(":")
    r = stats.pearsonr(df[feat].astype(float), df[targ].astype(float))[0]
    ratio = r * r / float(r2max)
    say(f"    observed r = {r:+.3f} -> r^2 = {r * r:.3f}; published upper bound r^2 = {float(r2max)} "
        f"-> {ratio:.0f}x the benchmark")
    gate("benchmark", "WARN" if ratio > 10 else "PASS",
         f"{feat} vs {targ}: {ratio:.0f}x the published upper bound" + (
             " -- effects this large are rare in observational data; suspect generated data" if ratio > 10 else ""))


def run_feasibility_audit(
    df: pd.DataFrame,
    target: str | None = None,
    id_cols: list[str] | None = None,
    group_col: str | None = None,
    time_col: str | None = None,
    sensitive_cols: list[str] | None = None,
    drop_cols: list[str] | None = None,
    benchmark_spec: str = "",
    max_rows: int = 20000,
    verbose: bool = True,
) -> dict:
    """
    Programmatic entry point to run the 7-check feasibility audit.

    Returns:
        dict: {"gates": {...}, "details": {...}}
    """
    global REPORT, VERBOSE
    VERBOSE = verbose
    REPORT = {"gates": {}, "details": {}}

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    say(f"Auditing DataFrame: {len(df):,} rows x {df.shape[1]} cols")

    id_list = id_cols or []
    drop_list = drop_cols or []
    sens_list = sensitive_cols or []

    ids = id_list + sequential_id_cols(df)
    excl = set(ids) | {x for x in [group_col, time_col, target] if x} | set(drop_list)
    feats = [c for c in df.columns if c not in excl and df[c].nunique() > 1]
    fn, fc = split_cols(df, feats)
    say(f"Features used: {len(feats)} ({len(fn)} numeric, {len(fc)} categorical); excluded: {sorted(excl)}")
    section_provenance(df, fn, fc)

    if target:
        y, task = prep_target(df[target].values)
        say(f"\nTarget '{target}': task={task}, " + (f"class counts {np.bincount(y).tolist()}" if task != 'regression' else ""))
        section_leakage(df, feats, y, task, max_rows)
        section_signal(df, feats, y, task, max_rows)
        if group_col or time_col:
            section_structure(df, feats, y, task, group_col, time_col)
        if sens_list and task == "binary":
            section_fairness(df, feats, y, sens_list)
    if benchmark_spec:
        section_benchmark(df, benchmark_spec)

    say("\n== GATE SUMMARY ==")
    for k, v in REPORT["gates"].items():
        say(f"  {v['status']:4}  {k}")

    return REPORT


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--target")
    ap.add_argument("--id", default="", help="identifier column(s), comma separated (excluded from features)")
    ap.add_argument("--group", help="entity column for panel data (excluded from features)")
    ap.add_argument("--time", help="time/period column (excluded from features)")
    ap.add_argument("--sensitive", default="", help="comma separated columns for the fairness screen")
    ap.add_argument("--drop", default="", help="columns to exclude from features (e.g. suspected leakers)")
    ap.add_argument("--benchmark", default="", help="FEATURE:TARGET:MAX_R2 from published literature")
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    ids = [c for c in a.id.split(",") if c]
    drops = [c for c in a.drop.split(",") if c]
    sens = [c for c in a.sensitive.split(",") if c]

    rep = run_feasibility_audit(
        df=df,
        target=a.target,
        id_cols=ids,
        group_col=a.group,
        time_col=a.time,
        sensitive_cols=sens,
        drop_cols=drops,
        benchmark_spec=a.benchmark,
        max_rows=a.max_rows,
        verbose=True,
    )

    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(rep, fh, indent=2, default=str)


if __name__ == "__main__":
    main()

