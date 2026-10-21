from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
ID_COLS = ["batch_id", "instructor_id", "course_id"]
OUTCOME_COLS = ["completion_rate", "dropout_rate", "avg_score_improvement", "avg_quiz_score"]
ENGAGEMENT_COLS = ["avg_watch_time", "assignment_submission_rate", "forum_activity_rate"]
FEEDBACK_COLS = ["avg_feedback_score", "feedback_response_rate"]
NUMERIC_COLS = OUTCOME_COLS + ENGAGEMENT_COLS + FEEDBACK_COLS
FEATURE_SOURCES = ENGAGEMENT_COLS + ["feedback_response_rate"]
FEATURES_MEANS_ONLY = [f"{c}_mean" for c in FEATURE_SOURCES]
CONTEXT_FEATURES = ["n_batches", "n_courses", "course_diversity", "course_baseline_mean"]
TIER_ORDER = ["Low", "Medium", "High"]
SCORE_SCENARIOS = [
    "Chosen: 70% outcomes / 30% satisfaction",
    "Equal weights 50 / 50",
    "Outcomes only",
    "Satisfaction only",
    "Add engagement (50/20/30)",
    "Drop quiz score from outcomes",
    "No course adjustment (global z only)",
]


@dataclass
class DataQuality:
    frame: pd.DataFrame
    notes: list[str]
    source_rows: int


def validate_data(raw: pd.DataFrame) -> DataQuality:
    """Apply the notebook schema/range checks and report excluded upload rows."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"fedback_response_rate": "feedback_response_rate"})
    notes: list[str] = []
    source_rows = len(df)
    missing = sorted(set(ID_COLS + NUMERIC_COLS) - set(df.columns))
    if missing:
        raise ValueError("Missing required notebook columns: " + ", ".join(missing))
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[ID_COLS + NUMERIC_COLS].isna().any(axis=1)
    if bad.any():
        notes.append(f"Excluded {int(bad.sum())} row(s) with missing IDs or metric values.")
        df = df.loc[~bad].copy()
    ranges = {c: (0, 1) for c in ["completion_rate", "dropout_rate", "avg_watch_time", "assignment_submission_rate", "forum_activity_rate", "feedback_response_rate"]}
    ranges.update({"avg_feedback_score": (1, 5)})
    for col, (low, high) in ranges.items():
        invalid = ~df[col].between(low, high)
        if invalid.any():
            notes.append(f"Excluded {int(invalid.sum())} row(s) with {col} outside [{low}, {high}].")
            df = df.loc[~invalid].copy()
    if df["batch_id"].duplicated().any():
        raise ValueError("batch_id must be unique, as required by the notebook.")
    if df.duplicated().any():
        raise ValueError("Exact duplicate rows found. Remove duplicates before analysis, as required by the notebook checks.")
    if df.empty:
        raise ValueError("No valid rows remain after the notebook schema checks.")
    if notes:
        notes.insert(0, "Uploaded data did not pass every notebook assertion; invalid rows were excluded for this session.")
    return DataQuality(df, notes, source_rows)


def zscore(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=series.index, name=series.name)
    return (series - series.mean()) / sd


def course_centred_z(frame: pd.DataFrame, col: str) -> pd.Series:
    sd = frame[col].std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=frame.index)
    return (frame[col] - frame.groupby("course_id")[col].transform("mean")) / sd


def eb_shrunk_mean(frame: pd.DataFrame, col: str, group: str = "instructor_id") -> tuple[pd.Series, float]:
    """Notebook's method-of-moments empirical-Bayes mean and shrinkage constant."""
    grouped = frame.groupby(group)[col]
    n, means, mu = grouped.size(), grouped.mean(), frame[col].mean()
    sigma2 = grouped.var(ddof=1).mean()
    tau2 = max(float(means.var(ddof=1) - (sigma2 / n).mean()), 1e-9) if len(means) > 1 else 1e-9
    k = float(sigma2 / tau2) if pd.notna(sigma2) else 0.0
    return mu + (n / (n + k)) * (means - mu), k


def make_score_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the notebook's exact batch-level score and sensitivity alternatives."""
    df = frame.copy()
    df["z_improvement"] = course_centred_z(df, "avg_score_improvement")
    df["z_quiz"] = course_centred_z(df, "avg_quiz_score")
    df["z_completion"] = course_centred_z(df, "completion_rate")
    df["z_dropout_inv"] = -course_centred_z(df, "dropout_rate")
    df["z_retention"] = (df["z_completion"] + df["z_dropout_inv"]) / 2
    df["z_feedback"] = course_centred_z(df, "avg_feedback_score")
    df["pillar_outcomes"] = zscore(.40 * df.z_improvement + .35 * df.z_retention + .25 * df.z_quiz)
    df["pillar_satisfaction"] = zscore(df.feedback_response_rate * df.z_feedback)
    eng_raw = sum(course_centred_z(df, c) for c in ENGAGEMENT_COLS) / len(ENGAGEMENT_COLS)
    df["pillar_engagement"] = zscore(eng_raw)
    outcomes_no_quiz = zscore(.5 * df.z_improvement + .5 * df.z_retention)
    no_course_outcomes = .40 * zscore(df.avg_score_improvement) + .35 * (zscore(df.completion_rate) - zscore(df.dropout_rate)) / 2 + .25 * zscore(df.avg_quiz_score)
    no_course_sat = df.feedback_response_rate * zscore(df.avg_feedback_score)
    df["score_chosen"] = .70 * df.pillar_outcomes + .30 * df.pillar_satisfaction
    df["score_equal"] = .50 * df.pillar_outcomes + .50 * df.pillar_satisfaction
    df["score_outcomes_only"] = df.pillar_outcomes
    df["score_satisfaction_only"] = df.pillar_satisfaction
    df["score_add_engagement"] = .50 * df.pillar_outcomes + .20 * df.pillar_engagement + .30 * df.pillar_satisfaction
    df["score_drop_quiz"] = .70 * outcomes_no_quiz + .30 * df.pillar_satisfaction
    df["score_global_only"] = .70 * zscore(no_course_outcomes) + .30 * zscore(no_course_sat)
    return df


SCENARIO_COL = {
    SCORE_SCENARIOS[0]: "score_chosen",
    SCORE_SCENARIOS[1]: "score_equal",
    SCORE_SCENARIOS[2]: "score_outcomes_only",
    SCORE_SCENARIOS[3]: "score_satisfaction_only",
    SCORE_SCENARIOS[4]: "score_add_engagement",
    SCORE_SCENARIOS[5]: "score_drop_quiz",
    SCORE_SCENARIOS[6]: "score_global_only",
}


def build_instructor_table(scored: pd.DataFrame, score_scenario: str = SCORE_SCENARIOS[0]) -> tuple[pd.DataFrame, pd.DataFrame, float, tuple[float, float]]:
    df = scored.copy()
    counts = df.groupby("instructor_id").size()
    eligible = counts[counts >= 5].index
    df = df[df.instructor_id.isin(eligible)].copy()
    if df.empty:
        raise ValueError("No instructor has the notebook minimum of five batches.")
    score_col = SCENARIO_COL[score_scenario]
    df["batch_score"] = df[score_col]
    inst = pd.DataFrame({"n_batches": df.groupby("instructor_id").size()})
    inst["n_courses"] = df.groupby("instructor_id").course_id.nunique()
    inst["course_diversity"] = inst.n_courses / inst.n_batches
    inst["low_confidence"] = inst.n_batches < 10
    inst["score_raw_mean"] = df.groupby("instructor_id").batch_score.mean()
    inst["effectiveness_score"], k_score = eb_shrunk_mean(df, "batch_score")
    for col in FEATURE_SOURCES:
        inst[f"{col}_mean"], _ = eb_shrunk_mean(df, col)
        inst[f"{col}_std"] = df.groupby("instructor_id")[col].std(ddof=1)
    # Notebook's course baseline is leave-one-instructor-out and uses the uncentred composite.
    out = .40 * zscore(df.avg_score_improvement) + .35 * (zscore(df.completion_rate) - zscore(df.dropout_rate)) / 2 + .25 * zscore(df.avg_quiz_score)
    # Keep the notebook's transform structure explicitly for one-row-per-batch peer baselines.
    work = df[["instructor_id", "course_id"]].copy()
    work["outcome_uncentred"] = out
    total = work.groupby("course_id")["outcome_uncentred"].transform("sum")
    n_total = work.groupby("course_id")["outcome_uncentred"].transform("count")
    own = work.groupby(["instructor_id", "course_id"])["outcome_uncentred"].transform("sum")
    own_n = work.groupby(["instructor_id", "course_id"])["outcome_uncentred"].transform("count")
    work["course_baseline"] = (total - own) / (n_total - own_n)
    inst["course_baseline_mean"] = work.groupby("instructor_id").course_baseline.mean()
    q25, q75 = inst.effectiveness_score.quantile([.25, .75])
    inst["tier"] = pd.cut(inst.effectiveness_score, [-np.inf, q25, q75, np.inf], labels=TIER_ORDER).astype(str)
    inst["tier_int"] = inst.tier.map({tier: i for i, tier in enumerate(TIER_ORDER)}).astype(int)
    inst["effectiveness_index"] = 50 + 15 * zscore(inst.effectiveness_score)
    # Notebook: 300 bootstrap samples per instructor, RNG seeded once at 42.
    rng = np.random.default_rng(RANDOM_STATE)
    mu_batch = df.batch_score.mean()
    confidence: dict[Any, float] = {}
    for instructor, group in df.groupby("instructor_id", sort=True):
        vals = group.batch_score.to_numpy()
        n = len(vals)
        means = rng.choice(vals, size=(300, n), replace=True).mean(axis=1)
        shrunk = mu_batch + (n / (n + k_score)) * (means - mu_batch)
        tiers = pd.cut(shrunk, [-np.inf, q25, q75, np.inf], labels=TIER_ORDER).astype(str)
        confidence[instructor] = float(np.mean(tiers == inst.loc[instructor, "tier"]))
    inst["tier_confidence"] = pd.Series(confidence)
    inst.index.name = "instructor_id"
    return inst.reset_index(), df, k_score, (float(q25), float(q75))


def make_lr(C: float = 1.0):
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE))])


def candidates() -> dict[str, Any]:
    return {
        "Dummy (majority class)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression (C=0.1)": make_lr(.1),
        "Logistic Regression (C=1)": make_lr(1.0),
        "Logistic Regression (C=10)": make_lr(10.0),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=.05, subsample=.8, random_state=RANDOM_STATE),
    }


def metric_row(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "qw_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def train_and_evaluate(inst: pd.DataFrame) -> dict[str, Any]:
    """Notebook's split, training-only feature filter, repeated CV, one-SE rule and evaluation."""
    if len(inst) < 20 or inst.tier.nunique() != 3:
        raise ValueError("The notebook model-selection workflow needs enough eligible instructors in all three tiers.")
    feature_sources = FEATURES_MEANS_ONLY + [f"{c}_std" for c in FEATURE_SOURCES] + CONTEXT_FEATURES
    X_full = inst.set_index("instructor_id")[feature_sources].copy()
    y_full = inst.set_index("instructor_id").tier_int.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=.25, stratify=y_full, random_state=RANDOM_STATE)
    fc = X_train.corr()
    assoc = X_train.apply(lambda col: abs(col.corr(y_train, method="spearman")))
    to_drop: set[str] = set()
    for i, a in enumerate(fc.columns):
        for b in fc.columns[i + 1:]:
            if abs(fc.loc[a, b]) > .90 and a not in to_drop and b not in to_drop:
                to_drop.add(a if assoc[a] < assoc[b] else b)
    candidate_features = [f for f in feature_sources if f not in to_drop]
    feature_sets = {"all_12": candidate_features, "means_4": FEATURES_MEANS_ONLY}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro", "qw_kappa": make_scorer(cohen_kappa_score, weights="quadratic")}
    models = candidates()
    cv_scores: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for fs, cols in feature_sets.items():
        for name, model in models.items():
            cv_scores[(fs, name)] = cross_validate(model, X_train[cols], y_train, cv=cv, scoring=scoring, n_jobs=-1)
    rows = []
    for (fs, name), result in cv_scores.items():
        if name.startswith("Dummy"):
            continue
        f1s = result["test_macro_f1"]
        rows.append({"features": fs, "model": name, "n_features": len(feature_sets[fs]), "mean_f1": float(f1s.mean()), "se": float(f1s.std() / np.sqrt(len(f1s))), "is_tree": not name.startswith("Logistic")})
    sel = pd.DataFrame(rows)
    best = sel.loc[sel.mean_f1.idxmax()]
    near = sel[sel.mean_f1 >= best.mean_f1 - best.se].sort_values(["n_features", "is_tree", "mean_f1"], ascending=[True, True, False])
    selected = near.iloc[0]
    final_name = selected.model
    final_features = feature_sets[selected.features]
    final_model = clone(models[final_name]).fit(X_train[final_features], y_train)
    pred_test = final_model.predict(X_test[final_features])
    dummy = clone(models["Dummy (majority class)"]).fit(X_train[final_features], y_train).predict(X_test[final_features])
    holdout = {"Final model": metric_row(y_test, pred_test), "Dummy baseline": metric_row(y_test, dummy)}
    y_oof = cross_val_predict(clone(models[final_name]), X_full[final_features], y_full, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), method="predict_proba")
    y_oof_pred = y_oof.argmax(axis=1)
    oof = metric_row(y_full, y_oof_pred)
    # Notebook's validation-fold permutation importance on all instructors.
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=RANDOM_STATE)
    imp_runs = []
    for tr, va in rskf.split(X_full[final_features], y_full):
        model = clone(models[final_name]).fit(X_full.iloc[tr][final_features], y_full.iloc[tr])
        pi = permutation_importance(model, X_full.iloc[va][final_features], y_full.iloc[va], scoring="f1_macro", n_repeats=10, random_state=RANDOM_STATE)
        imp_runs.append(pi.importances_mean)
    importance = pd.DataFrame(imp_runs, columns=final_features)
    importance_summary = pd.DataFrame({"mean_drop_in_macroF1": importance.mean(), "std": importance.std()}).sort_values("mean_drop_in_macroF1", ascending=False)
    # Logistic coefficient interpretation on the full instructor set, matching the notebook's interpretation-only fit.
    interp = clone(models[final_name]).fit(X_full[final_features], y_full)
    coef_table = None
    if hasattr(interp, "named_steps") and hasattr(interp.named_steps["clf"], "coef_"):
        coef_table = pd.DataFrame(interp.named_steps["clf"].coef_, index=TIER_ORDER, columns=final_features).T
    # Notebook's low-tier threshold table from OOF probabilities.
    low_idx = 0
    y_is_low = (y_full.to_numpy() == low_idx).astype(int)
    probs_low = y_oof[:, low_idx]
    threshold_rows = []
    for threshold in [.2, .3, .4, .5, .6, .7]:
        flag = probs_low >= threshold
        threshold_rows.append({"threshold": threshold, "flagged": int(flag.sum()), "precision": float(y_is_low[flag].mean()) if flag.any() else np.nan, "recall": float(y_is_low[flag].sum() / y_is_low.sum()) if y_is_low.sum() else np.nan})
    tier_by_id = y_full.map(dict(enumerate(TIER_ORDER)))
    oof_table = pd.DataFrame({"instructor_id": X_full.index, "observed_tier": tier_by_id.values, "predicted_tier": [TIER_ORDER[i] for i in y_oof_pred], "probability_low": probs_low})
    test_table = pd.DataFrame({"instructor_id": X_test.index, "observed_tier": y_test.map(dict(enumerate(TIER_ORDER))).values, "predicted_tier": [TIER_ORDER[i] for i in pred_test]})
    cv_table = pd.DataFrame([{ "feature_set": fs, "model": name, "metric": metric, "mean": float(np.mean(res[f"test_{metric}"])), "std": float(np.std(res[f"test_{metric}"])) } for (fs, name), res in cv_scores.items() for metric in scoring])
    return {
        "train_ids": X_train.index.tolist(), "test_ids": X_test.index.tolist(),
        "candidate_features": candidate_features, "dropped_features": sorted(to_drop),
        "cv_table": cv_table, "cv_raw": cv_scores, "selected_features": final_features,
        "selected_model": final_name, "selection_rule": "Best mean CV macro-F1, then choose the smallest model within one standard error; ties prefer a non-tree model.",
        "holdout_metrics": holdout, "oof_metrics": oof,
        "holdout_confusion": confusion_matrix(y_test, pred_test, labels=[0, 1, 2]),
        "oof_confusion": confusion_matrix(y_full, y_oof_pred, labels=[0, 1, 2]),
        "oof_normalized_confusion": confusion_matrix(y_full, y_oof_pred, labels=[0, 1, 2], normalize="true"),
        "low_thresholds": pd.DataFrame(threshold_rows), "oof_table": oof_table,
        "holdout_table": test_table, "importance": importance_summary,
        "coefficients": coef_table, "permutation_runs": importance,
    }


def notebook_feature_table(inst: pd.DataFrame) -> pd.DataFrame:
    cols = FEATURES_MEANS_ONLY + [f"{c}_std" for c in FEATURE_SOURCES] + CONTEXT_FEATURES
    return inst.set_index("instructor_id")[cols].copy()
