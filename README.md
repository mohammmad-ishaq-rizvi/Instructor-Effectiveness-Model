# Instructor Effectiveness Modeling

This is the Streamlit app for the work in `Instructor_Effectiveness_Modeling.ipynb` and its PDF export. The notebook and its supplied dataset remain the source of truth. The app reproduces the notebook's calculations and interactive visualizations; it does not modify either reference file or substitute a new scoring methodology.

## Files

- `Instructor_Effectiveness_Modeling.ipynb` - original analysis notebook (unchanged).
- `Instructor_Effectiveness_Modeling.pdf` - supplied notebook export (unchanged).
- `instructor_effectiveness_dataset.csv` - notebook's 2,000 batch rows.
- `instructor_model.py` - notebook-derived scoring, aggregation, feature engineering, model selection, and evaluation.
- `app.py` - Streamlit navigation and interactive dashboard.
- `requirements.txt` - Python dependencies.
- `.streamlit/config.toml` - dashboard styling and hosted-app settings.
- `.gitignore` - ignores local environments, notebook checkpoints, editor files, logs, and secrets.

## Run locally

Use Python 3.10-3.12.

```powershell
cd C:\Users\D2D\PycharmProjects\Edtech_model
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app loads `instructor_effectiveness_dataset.csv` from this folder by default. You can upload a CSV or Excel dataset in the sidebar and map its column names to the notebook schema. The model requires the notebook identifiers and numeric signals. Invalid rows are reported and excluded; duplicate batch IDs and duplicate records are rejected.

## Notebook-faithful workflow

- **Overview & EDA:** dataset dimensions, summary statistics, metric distributions, missingness, and correlations.
- **Effectiveness score:** course-centred metric z-scores; outcomes/satisfaction pillars; response-rate reliability adjustment; empirical-Bayes instructor shrinkage; five-batch minimum; ten-batch low-confidence flag; 25th/75th percentile tiers; and the notebook's 300-resample tier-confidence estimate.
- **Score sensitivity:** the seven definitions already evaluated in notebook Section 5.5. The chosen 70/30 score is the default.
- **Instructor analytics:** course and tier filters with batch outcomes and engagement comparisons.
- **Model selection:** 75/25 stratified holdout; training-only near-duplicate filtering; `all_12` and `means_4` feature sets; the notebook's Dummy, three Logistic Regression, Random Forest, and Gradient Boosting candidates; 5-fold stratified CV repeated 5 times; and the notebook's one-standard-error model-selection rule.
- **Evaluation:** held-out metrics and baseline, five-fold out-of-fold predictions, confusion matrices, low-tier threshold precision/recall, and ordinal error distance.
- **Model insights:** repeated validation-fold permutation importance, standardized Logistic Regression coefficients when applicable, and feature/score rank correlations.
- **Data management & exports:** data quality view and CSV exports of validated data, batch scores, instructor scores, CV results, holdout predictions, and out-of-fold predictions.

The app calculates metrics from the active dataset when you run model selection. It does not copy the PDF's example metrics into new runs. Model outputs are screening estimates, not instructor verdicts. The notebook's limits still apply: the dataset is clipped, does not contain timestamps, and does not measure learner prior ability or motivation.

## GitHub and deployment

1. Review the dataset before publishing. The bundled CSV is included to reproduce the notebook, but do not push it if it contains information you are not authorized to publish.
2. This folder currently sits inside the existing `C:\Users\D2D\PycharmProjects` Git worktree, which has no GitHub remote configured. From that parent worktree, stage only this project (including the notebook and PDF if you want them in GitHub), so the other PyCharm projects are not included:

   ```bash
   cd C:\Users\D2D\PycharmProjects
   git add Edtech_model/app.py Edtech_model/instructor_model.py Edtech_model/requirements.txt Edtech_model/README.md Edtech_model/.gitignore Edtech_model/.streamlit/config.toml
   git add Edtech_model/Instructor_Effectiveness_Modeling.ipynb Edtech_model/Instructor_Effectiveness_Modeling.pdf Edtech_model/instructor_effectiveness_dataset.csv
   git commit -m "Add notebook-faithful instructor dashboard"
   git remote add origin <your-github-repository>
   git push -u origin HEAD
   ```

3. In Streamlit Community Cloud, create an app from that repository, select the branch, and set `app.py` as the entry point. The notebook remains in the repository for reference but is not changed or required as a runtime artifact.
4. Uploads and Streamlit session state are temporary on hosted instances. No secrets are needed for the bundled dataset. Keep any later service credentials in Streamlit secrets and out of Git.

The dashboard is prepared for deployment; it has not been pushed to GitHub or published to a hosting service because no remote is configured for the existing worktree.
