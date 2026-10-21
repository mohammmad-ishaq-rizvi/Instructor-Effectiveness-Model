from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from instructor_model import (
    CONTEXT_FEATURES,
    FEATURES_MEANS_ONLY,
    NUMERIC_COLS,
    SCORE_SCENARIOS,
    TIER_ORDER,
    build_instructor_table,
    make_score_scenarios,
    notebook_feature_table,
    train_and_evaluate,
    validate_data,
)

st.set_page_config(page_title="Instructor Effectiveness Modeling", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container{padding-top:1.25rem;max-width:1480px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#101b35,#17294b)}
[data-testid="stSidebar"] *{color:#edf3ff}
div[data-testid="stMetric"]{background:#f4f7fb;border:1px solid #e4eaf2;padding:14px 16px;border-radius:14px}
.hero{padding:1.35rem 1.6rem;border-radius:18px;color:white;background:linear-gradient(115deg,#142a50,#265a84 58%,#4b9b9b);margin-bottom:1rem}
.hero h1{margin:0 0 .35rem 0;font-size:2rem}.hero p{margin:0;color:#e2f0f7}
</style>
<div class="hero"><h1>Instructor Effectiveness Modeling</h1><p>Interactive view of the supplied notebook analysis, scoring method, and model evaluation</p></div>
""", unsafe_allow_html=True)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "instructor_effectiveness_dataset.csv"
RAW_REQUIRED = ["batch_id", "instructor_id", "course_id"] + NUMERIC_COLS
MODEL_RESULT_KEY = "notebook_model_result"


def read_upload(uploaded) -> tuple[pd.DataFrame, str]:
    if uploaded is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError("Place instructor_effectiveness_dataset.csv beside app.py or upload a dataset.")
        return pd.read_csv(DATA_PATH), DATA_PATH.name
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded), uploaded.name
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded), uploaded.name
    raise ValueError("Upload a CSV or Excel file.")


def map_columns(raw: pd.DataFrame) -> pd.DataFrame:
    norm = raw.copy()
    norm.columns = [str(c).strip().lower() for c in norm.columns]
    norm = norm.rename(columns={"fedback_response_rate": "feedback_response_rate"})
    st.sidebar.markdown("### Dataset column mapping")
    st.sidebar.caption("Map uploaded headers to the notebook schema. Matching headers are selected automatically.")
    required = ["batch_id", "instructor_id", "course_id"] + NUMERIC_COLS
    mapping: dict[str, str] = {}
    with st.sidebar.expander("Map source columns", expanded=False):
        for target in required:
            options = [""] + list(norm.columns)
            default = target if target in norm.columns else ""
            selected = st.selectbox(target, options, index=options.index(default), key=f"column_{target}")
            if selected:
                mapping[selected] = target
    return norm.rename(columns=mapping)


def score_all_scenarios(valid: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, float, tuple[float, float]]]:
    scored = make_score_scenarios(valid)
    return {name: build_instructor_table(scored, name) for name in SCORE_SCENARIOS}


@st.cache_data(show_spinner=False, max_entries=3)
def cached_model_fit(instructor_table: pd.DataFrame) -> dict:
    return train_and_evaluate(instructor_table)


with st.sidebar:
    st.markdown("## Notebook sections")
    page = st.radio("Section", ["Overview & EDA", "Effectiveness score", "Instructor analytics", "Model selection", "Evaluation", "Model insights", "Data management & exports"], label_visibility="collapsed")
    st.divider()
    upload = st.file_uploader("Use another dataset", type=["csv", "xlsx", "xls"], help="Uploads replace the bundled CSV for this session.")
    st.caption("Random seed 42 · five-batch modelling minimum · fewer than ten batches flagged low confidence")

try:
    source_raw, source_name = read_upload(upload)
    mapped = map_columns(source_raw)
    quality = validate_data(mapped)
    raw_df = quality.frame
    if quality.notes:
        for note in quality.notes:
            st.sidebar.warning(note)
except Exception as exc:
    st.error(str(exc))
    st.info("Expected fields follow the notebook schema: batch/instructor/course IDs and nine outcome, engagement, and feedback metrics.")
    st.stop()

source_token = f"{source_name}:{pd.util.hash_pandas_object(raw_df, index=True).sum()}"
if st.session_state.get("source_token") != source_token:
    st.session_state.pop(MODEL_RESULT_KEY, None)
    st.session_state["source_token"] = source_token

try:
    scenarios = score_all_scenarios(raw_df)
except Exception as exc:
    st.error(f"Could not calculate the notebook score: {exc}")
    st.stop()

scenario = SCORE_SCENARIOS[0]
if page == "Effectiveness score":
    scenario = st.selectbox("Notebook score definition", SCORE_SCENARIOS, index=0, help="The default exactly matches the notebook. Other choices are its Section 5.5 sensitivity scenarios.")
inst, batches, k_score, (q25, q75) = scenarios[scenario]
inst["tier"] = pd.Categorical(inst["tier"], categories=TIER_ORDER, ordered=True)

if page == "Overview & EDA":
    st.subheader("Notebook analysis overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Batches", f"{len(raw_df):,}")
    c2.metric("Instructors", f"{raw_df.instructor_id.nunique():,}")
    c3.metric("Courses", f"{raw_df.course_id.nunique():,}")
    c4.metric("Instructors flagged low confidence", f"{int(inst.low_confidence.sum())}")
    st.markdown("#### Dataset structure and quality")
    left,right=st.columns([1.1,.9])
    with left:
        counts=raw_df.groupby("instructor_id").size().rename("batches").reset_index()
        st.plotly_chart(px.histogram(counts,x="batches",nbins=20,title="Batches per instructor"),use_container_width=True)
    with right:
        st.dataframe(pd.DataFrame({"column":raw_df.columns,"dtype":[str(raw_df[c].dtype) for c in raw_df.columns],"missing":raw_df.isna().sum().values,"unique":raw_df.nunique(dropna=True).values}),hide_index=True,use_container_width=True)
    st.markdown("#### Batch-level metric distributions")
    metric=st.selectbox("Metric",NUMERIC_COLS,index=0)
    st.plotly_chart(px.histogram(raw_df,x=metric,nbins=35,marginal="box",title=f"Distribution of {metric}"),use_container_width=True)
    st.markdown("#### Correlations")
    corr=raw_df[NUMERIC_COLS].corr()
    st.plotly_chart(px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,title="Batch-level Pearson correlation matrix"),use_container_width=True)
    st.write("Notebook checks: completion and dropout are near complements; instructor-level aggregation is central because many batches belong to each instructor. The data contain no timestamp, so time trends are not available.")
    st.markdown("#### Summary statistics")
    st.dataframe(raw_df[NUMERIC_COLS].describe().T,use_container_width=True)

elif page == "Effectiveness score":
    st.subheader("Notebook score and sensitivity analysis")
    st.write("The selected view is the notebook's custom observed score. Engagement is excluded from the chosen score; the model prediction is shown separately in the model sections.")
    st.info("Chosen definition: course-centred learning outcomes (70%; improvement 40%, retention 35%, quiz 25%) plus response-rate-adjusted satisfaction (30%). Instructor scores use empirical-Bayes shrinkage; tiers use this dataset's 25th and 75th percentiles.")
    a,b,c,d=st.columns(4)
    a.metric("Shrinkage constant k",f"{k_score:.2f} batches")
    b.metric("Low / Medium cutoff",f"{q25:.3f}")
    c.metric("Medium / High cutoff",f"{q75:.3f}")
    d.metric("Median tier confidence",f"{inst.tier_confidence.median():.0%}")
    score_hist=px.histogram(inst,x="effectiveness_score",nbins=25,color="tier",color_discrete_map={"Low":"#e03131","Medium":"#f08c00","High":"#2f9e44"},title=f"Instructor score: {scenario}")
    score_hist.add_vline(x=q25,line_dash="dash",line_color="red");score_hist.add_vline(x=q75,line_dash="dash",line_color="green")
    st.plotly_chart(score_hist,use_container_width=True)
    st.plotly_chart(px.scatter(inst,x="score_raw_mean",y="effectiveness_score",color="n_batches",hover_name="instructor_id",title="Raw mean and empirical-Bayes shrunk score"),use_container_width=True)
    st.markdown("#### Notebook sensitivity alternatives")
    summary=[]
    chosen_scores=scenarios[SCORE_SCENARIOS[0]][0].set_index("instructor_id").effectiveness_score
    chosen_tiers=scenarios[SCORE_SCENARIOS[0]][0].set_index("instructor_id").tier.astype(str)
    for name,(alt,_,_,_) in scenarios.items():
        alt=alt.set_index("instructor_id")
        s=alt.effectiveness_score.reindex(chosen_scores.index)
        summary.append({"definition":name,"spearman_vs_chosen":s.corr(chosen_scores,method="spearman"),"same_tier_percent":100*(alt.tier.astype(str).reindex(chosen_tiers.index)==chosen_tiers).mean()})
    st.dataframe(pd.DataFrame(summary).round(3),use_container_width=True,hide_index=True)
    st.dataframe(inst[["instructor_id","n_batches","score_raw_mean","effectiveness_score","effectiveness_index","tier","tier_confidence","low_confidence"]].sort_values("effectiveness_score",ascending=False),use_container_width=True,hide_index=True)

elif page == "Instructor analytics":
    st.subheader("Instructor-level analysis")
    courses=sorted(raw_df.course_id.astype(str).unique())
    selected_courses=st.multiselect("Courses",courses,default=courses)
    selected_tiers=st.multiselect("Tiers",TIER_ORDER,default=TIER_ORDER)
    filtered=batches[batches.course_id.astype(str).isin(selected_courses)]
    i_filtered=inst[inst.tier.astype(str).isin(selected_tiers)]
    if filtered.empty or i_filtered.empty:
        st.info("No records match these filters.")
    else:
        agg=filtered.groupby("instructor_id").agg(n_batches=("batch_id","count"),avg_completion=("completion_rate","mean"),avg_dropout=("dropout_rate","mean"),avg_improvement=("avg_score_improvement","mean"),avg_quiz=("avg_quiz_score","mean"),avg_feedback=("avg_feedback_score","mean"),avg_assignment=("assignment_submission_rate","mean"),avg_forum=("forum_activity_rate","mean")).reset_index()
        agg=agg.merge(i_filtered[["instructor_id","tier","effectiveness_score","tier_confidence","n_courses"]],on="instructor_id",how="inner")
        measure=st.selectbox("Compare outcome",["avg_completion","avg_improvement","avg_quiz","avg_feedback","avg_dropout"])
        st.plotly_chart(px.scatter(agg,x="avg_assignment",y=measure,color="tier",size="n_batches",hover_name="instructor_id",hover_data=["avg_forum","tier_confidence","n_courses"],color_discrete_map={"Low":"#e03131","Medium":"#f08c00","High":"#2f9e44"},title=f"Assignment submission and {measure}"),use_container_width=True)
        st.dataframe(agg.sort_values("effectiveness_score",ascending=False),use_container_width=True,hide_index=True)
        st.caption("Course and instructor comparisons are based on selected observed batches; no dates are available for trend analysis.")

elif page == "Model selection":
    st.subheader("Notebook model selection")
    st.write("Runs the notebook pipeline on the active dataset: 75/25 stratified instructor split, training-only near-duplicate filtering, repeated stratified 5-fold CV repeated 5 times, both notebook feature sets and all notebook candidate algorithms. The final choice follows the one-standard-error rule.")
    st.caption("This can take a few minutes because it reproduces the notebook's 25-fold Random Forest and Gradient Boosting comparisons. It runs only when requested and caches the result for the active data.")
    if st.button("Run notebook model selection and evaluation",type="primary"):
        try:
            with st.spinner("Running the notebook's model comparison and evaluation..."):
                st.session_state[MODEL_RESULT_KEY]=cached_model_fit(inst.drop(columns=[]))
        except Exception as exc:
            st.error(f"Notebook model workflow could not run: {exc}")
    if MODEL_RESULT_KEY in st.session_state:
        result=st.session_state[MODEL_RESULT_KEY]
        st.success(f"Selected model: {result['selected_model']} · features: {', '.join(result['selected_features'])}")
        st.caption(result["selection_rule"])
        cv=result["cv_table"]
        cvplot=cv[(cv.metric=="macro_f1")].copy()
        cvplot["model_set"]=cvplot["model"]+" · "+cvplot["feature_set"]
        st.plotly_chart(px.box(cvplot,x="mean",y="model_set",color="feature_set",title="Training CV macro-F1 by model and notebook feature set"),use_container_width=True)
        st.dataframe(cv.pivot_table(index=["feature_set","model"],columns="metric",values="mean").round(3),use_container_width=True)
        st.write("Features removed as near-duplicates in training data:",result["dropped_features"] or "none")

elif page == "Evaluation":
    st.subheader("Notebook evaluation")
    if MODEL_RESULT_KEY not in st.session_state:
        st.info("Run the model-selection workflow first. No notebook performance values are copied into a new dataset's results.")
    else:
        r=st.session_state[MODEL_RESULT_KEY]
        st.markdown("#### Held-out 25% test metrics")
        hold=pd.DataFrame(r["holdout_metrics"]).T
        st.dataframe(hold.style.format("{:.3f}"),use_container_width=True)
        st.plotly_chart(px.imshow(r["holdout_confusion"],x=TIER_ORDER,y=TIER_ORDER,text_auto=True,color_continuous_scale="Blues",labels={"x":"Predicted","y":"Observed","color":"Count"},title="Holdout confusion matrix"),use_container_width=True)
        st.markdown("#### Five-fold out-of-fold results on all instructors")
        m1,m2,m3,m4=st.columns(4)
        for col,(name,value) in zip([m1,m2,m3,m4],r["oof_metrics"].items()):col.metric(name.replace("_"," ").title(),f"{value:.3f}")
        st.plotly_chart(px.imshow(r["oof_normalized_confusion"],x=TIER_ORDER,y=TIER_ORDER,text_auto=".2f",color_continuous_scale="Greens",labels={"x":"Predicted","y":"Observed","color":"Recall"},title="Row-normalized out-of-fold confusion matrix"),use_container_width=True)
        st.markdown("#### Low-tier precision and recall thresholds")
        st.plotly_chart(px.line(r["low_thresholds"],x="threshold",y=["precision","recall"],markers=True,title="OOF Low probability threshold trade-off"),use_container_width=True)
        st.dataframe(r["low_thresholds"].round(3),use_container_width=True,hide_index=True)
        wrong=r["oof_table"].query("observed_tier != predicted_tier")
        far=(wrong.observed_tier.eq("Low") & wrong.predicted_tier.eq("High") | wrong.observed_tier.eq("High") & wrong.predicted_tier.eq("Low")).sum()
        st.write(f"Out-of-fold tier errors: {len(wrong)}; Low↔High errors: {far}.")
        st.dataframe(r["holdout_table"],use_container_width=True,hide_index=True)

elif page == "Model insights":
    st.subheader("Notebook feature interpretation")
    if MODEL_RESULT_KEY not in st.session_state:
        st.info("Run model selection first to calculate actual validation-fold feature importance and model interpretation.")
    else:
        r=st.session_state[MODEL_RESULT_KEY]
        imp=r["importance"].reset_index(names="feature")
        st.plotly_chart(px.bar(imp.sort_values("mean_drop_in_macroF1"),x="mean_drop_in_macroF1",y="feature",orientation="h",error_x="std",title="Permutation importance (mean macro-F1 drop across validation folds)"),use_container_width=True)
        if r["coefficients"] is not None:
            cf=r["coefficients"].rename_axis("feature").reset_index()
            cf=cf.melt(id_vars="feature",var_name="tier",value_name="standardized_coefficient")
            st.plotly_chart(px.bar(cf,x="standardized_coefficient",y="feature",color="tier",barmode="group",title="Standardized logistic-regression coefficients"),use_container_width=True)
        else:
            st.info("The selected tree model has no coefficient table; use permutation importance above.")
        table=notebook_feature_table(inst)
        outcomes_only=build_instructor_table(make_score_scenarios(raw_df),SCORE_SCENARIOS[2])[0].set_index("instructor_id").effectiveness_score
        full_score=inst.set_index("instructor_id").effectiveness_score
        rank=pd.DataFrame({"with_full_score":table.apply(lambda c:c.corr(full_score,method="spearman")),"with_outcomes_only":table.apply(lambda c:c.corr(outcomes_only,method="spearman"))}).sort_values("with_full_score",ascending=True).reset_index(names="feature")
        st.plotly_chart(px.bar(rank.melt(id_vars="feature",var_name="score",value_name="spearman"),x="spearman",y="feature",color="score",barmode="group",title="Notebook model-free Spearman relationships"),use_container_width=True)
        st.caption("Permutation importance and coefficients describe model associations, not causal effects.")

elif page == "Data management & exports":
    st.subheader("Data management and reports")
    c1,c2,c3=st.columns(3)
    c1.metric("Uploaded/source rows",f"{quality.source_rows:,}")
    c2.metric("Valid rows",f"{len(raw_df):,}")
    c3.metric("Exact duplicate rows",f"{int(raw_df.duplicated().sum())}")
    st.dataframe(pd.DataFrame({"column":source_raw.columns,"dtype":[str(source_raw[c].dtype) for c in source_raw.columns],"missing":source_raw.isna().sum().values,"unique":source_raw.nunique(dropna=True).values}),use_container_width=True,hide_index=True)
    st.markdown("#### Active dataset preview")
    st.dataframe(raw_df.head(100),use_container_width=True,hide_index=True)
    st.download_button("Download validated notebook-schema data",raw_df.to_csv(index=False).encode("utf-8"),"validated_instructor_effectiveness_data.csv","text/csv")
    st.download_button("Download batch scores (chosen notebook definition)",batches.to_csv(index=False).encode("utf-8"),"notebook_batch_scores.csv","text/csv")
    st.download_button("Download instructor scores and confidence",inst.to_csv(index=False).encode("utf-8"),"notebook_instructor_scores.csv","text/csv")
    if MODEL_RESULT_KEY in st.session_state:
        r=st.session_state[MODEL_RESULT_KEY]
        st.download_button("Download CV comparison",r["cv_table"].to_csv(index=False).encode("utf-8"),"notebook_cv_comparison.csv","text/csv")
        st.download_button("Download holdout predictions",r["holdout_table"].to_csv(index=False).encode("utf-8"),"notebook_holdout_predictions.csv","text/csv")
        st.download_button("Download out-of-fold predictions",r["oof_table"].to_csv(index=False).encode("utf-8"),"notebook_oof_predictions.csv","text/csv")
