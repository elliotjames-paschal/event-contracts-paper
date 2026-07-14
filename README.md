# Event Contracts Paper — Data & Scripts

Replication data and analysis scripts for **"The Design and Quality of Event Contracts: Evidence from Prediction Markets"** (Hall & Paschal, working paper — do not cite or circulate).

This repository contains only the inputs that feed the paper: the graded-market datasets and the scripts that generate every figure and table. The manuscript itself and the upstream data-collection pipeline live elsewhere.

## Layout

```
data/                          Input datasets (read-only; never modified by scripts)
  full_grades.jsonl              Full graded sample of market rules
  pilot_v07_results.json         Pilot grading run (v0.7)
  ab_adjudication_key.csv        A/B adjudication key
  adjudication_final_verdicts.csv  Final adjudication verdicts
  disputed_kalshi_events.json    Disputed Kalshi events
  disputed_audit_results.json    10-axis ratings of disputed markets
  fetched/uma_disputes_enriched.json  Polymarket UMA disputes
scripts/
  dispute_regression.py          Shared regression machinery (logit + L2, AUC)
src/parser/
  schema.py                      Resolution Spec pydantic models (grade tiers)
paper/
  data/                          Paper-side derived data
    labeled_disputes.json          Output of label_disputes.py
    spec_issue_sample_100.json     100-item spec-issue sample
  prompts/grading_system_prompt.txt  System prompt used for grading
  scripts/                       Figure/table generators (one per artifact group)
  figures/                       Output directory (generated, not committed)
  tables/                        Output directory (generated, not committed)
```

## Reproducing figures and tables

Scripts locate data relative to the repo root, so run them from anywhere:

```bash
python paper/scripts/generate_figures.py
python paper/scripts/generate_tables.py
python paper/scripts/generate_regression_table.py
python paper/scripts/generate_section6.py
python paper/scripts/generate_grade_agreement.py
python paper/scripts/generate_dispute_forest_figure.py
```

Outputs are written to `paper/figures/` and `paper/tables/`.

`paper/scripts/label_disputes.py` regenerates `paper/data/labeled_disputes.json` from the raw dispute files; it calls the OpenAI API (requires `OPENAI_API_KEY` in the environment or a `.env` file) and caches responses under `paper/data/.cache/`.

## Requirements

Python 3.11+ with `numpy`, `matplotlib`, and `pydantic`. The regression and Section 6 scripts additionally use `scikit-learn`, `xgboost`, `interpret`, and `shap`; `label_disputes.py` needs `openai` and `python-dotenv`.
