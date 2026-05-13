# Roadmap — Local Platform

Scope: **local implementation only** (`aipc_local.yaml` + `src/local_platform/`).
Structure follows the three pipeline stages defined in [`pipeline_definitions.yaml`](../../../config/pipeline_definitions.yaml).
Each row is a single operation instance identified by its `id` in [`aipc_local.yaml`](../metadata/aipc_local.yaml).

---

## How to read this table

| Column | Meaning |
|---|---|
| **id** | Operation instance key in `aipc_local.yaml` |
| **type** | Canonical operation type from `pipeline_definitions.yaml` |
| **req** | `requirement_dimension` — `baseline` or `fairness` |
| **method** | `method_name` declared in the YAML spec (the contract) |
| **impl** | Actual function present in source; `—` if absent |
| **status** | See legend below |

## Status legend

| Status | Meaning |
|---|---|
| `done` | Method exists, logic is complete, YAML `inputs`/`outputs` are fully defined |
| `in progress` | Method exists but has incomplete logic, bugs, or YAML `inputs`/`outputs` are missing |
| `planned` | YAML entry is present; function body is a stub (`pass`) or the method is absent from source |
| `backlog` | Neither a complete YAML entry nor a source implementation exists |

---

## Stage 1 — `data_preparation`

Canonical operations: `data_profiling` · `data_validation` · `data_preprocessing` · `data_documentation`

| id | type | req | method (YAML spec) | impl (source) | status | notes |
|---|---|---|---|---|---|---|
| `data_profiling` | `data_profiling` | baseline | `data_drift_detection` | `data_drift_detection` | `in progress` | Evidently drift logic present; YAML `method_name` is wrong — should point to a profiling function, not drift detection; see [spec gap #8](#spec-gaps) |
| `data_profiling_custom` | `data_profiling` | baseline | `data_profiling_custom` | `data_profiling_custom` | `done` | All 8 profiling actions implemented; JSON report saved to `artifacts/report/` |
| `data_drift_detection` | `data_validation` | baseline | `data_drift_status` | `data_drift_status` | `done` | Evidently `DataDriftPreset`; returns boolean `Status` |
| `data_preprocessing_step1` | `data_preprocessing` | baseline | `load_data` | `load_data` | `in progress` | Loads CSV, drops NaN; split and encoding live in separate functions not yet wired to this op's YAML `outputs` |
| `data_preprocessing_reweighing` | `data_preprocessing` | fairness | `bias_mitigation_pre_reweighing` | `preprocess_reweighing` | `in progress` | Reweighing logic implemented; `Reweighing` import commented out; function name in source does not match YAML `method_name` |
| `data_documentation` | `data_documentation` | baseline | *(no YAML entry)* | `data_card_generation` | `backlog` | Function is a stub (`pass`); no YAML op entry — see [spec gap #1](#spec-gaps) |

---

## Stage 2 — `modelling`

Canonical operations: `feature_engineering` · `model_training` · `model_evaluation` · `model_validation` · `model_documentation`

| id | type | req | method (YAML spec) | impl (source) | status | notes |
|---|---|---|---|---|---|---|
| `feature_engineering` | `feature_engineering` | baseline | *(no YAML entry)* | `permutation_feature_importance` | `in progress` | Core permutation loop present; references undefined variables `y_test`, `group_a_test`, `group_b_test`; no YAML op entry — see [spec gap #2](#spec-gaps) |
| `model_train` | `model_training` | baseline | `train_model` | `train_model` | `done` | RidgeClassifier; saves `model_baseline.pickle`; train/test split applied |
| `model_train_reweighing` | `model_training` | fairness | `bias_mitigation_in_process_train` | `bias_mitigation_in_process_train` | `in progress` | In-process training with sample weights implemented; `mlflow` import commented out at module level; reads `metrics_accuracy.csv` via hardcoded path instead of input `Report` artifact |
| `model_evaluation_accuracy` | `model_evaluation` | baseline | `model_evaluation_accuracy` | `model_evaluation_accuracy` | `done` | Overall accuracy on test set; saves CSV report |
| `model_evaluation_accuracy_demographic_groups` | `model_evaluation` | baseline | `model_evaluation_accuracy_demographic_groups` | `model_evaluation_accuracy_demographic_groups` | `in progress` | Per-group accuracy loop implemented; `report` used on line 284 but not declared as a function parameter |
| `model_evaluation_equality_of_outcome` | `model_evaluation` | fairness | `model_evaluation_equality_of_outcome` | *(absent)* | `planned` | YAML entry present; no function in `modelling.py`; must compute statistical parity + disparate impact |
| `model_evaluation_equality_of_opportunity` | `model_evaluation` | fairness | `model_evaluation_equality_of_opportunity` | *(absent)* | `planned` | YAML entry present; no function in `modelling.py`; must compute equal opportunity difference + average odds difference |
| `model_validation_baseline` | `model_validation` | baseline | `model_validation_baseline` | `model_validation_baseline` | `planned` | YAML entry present; function body is a stub (`pass`) |
| `model_validation_fairness` | `model_validation` | fairness | `model_validation_fairness` | *(absent)* | `planned` | YAML entry present; no function in `modelling.py`; must apply thresholds from `model_evaluation_fairness_metrics` config |
| `model_documentation` | `model_documentation` | baseline | *(no YAML entry)* | *(absent)* | `backlog` | `docs/ModelCard.md` exists but generation is not automated; no YAML op entry — see [spec gap #3](#spec-gaps) |

---

## Stage 3 — `operationalization`

Canonical operations: `model_deployment` · `model_monitoring` · `production_data_monitoring` · `system_monitoring` · `pre_inference_transformations` · `post_inference_transformations`

| id | type | req | method (YAML spec) | impl (source) | status | notes |
|---|---|---|---|---|---|---|
| `model_deploy` | `model_deployment` | baseline | `model_deploy` | `model_deploy` | `done` | KServe `KFServer` + `RESTConfig` client; `configs/kserve.yaml` present |
| `model_monitor` | `model_monitoring` | baseline | `model_monitor` | `model_monitor` | `in progress` | FastAPI proxy captures request/response/latency; uses `print` instead of real storage; YAML `inputs`/`outputs` are empty — see [spec gap #4](#spec-gaps) |
| `production_data_monitor` | `production_data_monitoring` | baseline | `production_data_monitor` | `data_drift_detection_evidently` | `planned` | Listed in `excluded_operations`; target function is a stub (`pass`); YAML `inputs`/`outputs` empty — see [spec gap #5](#spec-gaps) |
| `system_monitor` | `system_monitoring` | baseline | *(no method in YAML)* | *(absent)* | `backlog` | Listed in `excluded_operations`; no implementation; Prometheus/Grafana integration not yet scoped — see [spec gap #6](#spec-gaps) |
| *(excluded)* | `pre_inference_transformations` | baseline | *(no YAML entry)* | `pre_inference_transformation` | `backlog` | Listed in `excluded_operations`; function is a stub (`pass`) |
| `post_inference_eq_odds` | `post_inference_transformations` | fairness | *(no YAML entry)* | `post_processing_fairness` | `in progress` | Equalized Odds fully implemented in `operationalization.py`; no YAML op entry — see [spec gap #7](#spec-gaps) |

---

## Spec Gaps

Operations that are implemented or partially implemented but lack a correct, complete entry in `aipc_local.yaml`. Resolve these to keep the YAML the authoritative specification.

| # | Gap | Required action |
|---|---|---|
| 1 | `data_documentation` has no `operations` entry | Add op: `id: data_documentation`, `type: data_documentation`, `method_name: data_card_generation`, output `documentation: data_card_hiring` |
| 2 | `feature_engineering` has no `operations` entry | Add op: `id: feature_engineering`, `type: feature_engineering`, `method_name: permutation_feature_importance`, output `report: feature_importance_report` |
| 3 | `model_documentation` has no `operations` entry | Add op: `id: model_documentation`, `type: model_documentation`, `method_name: model_card_generation`, output `documentation: model_card_hiring` |
| 4 | `model_monitor` has empty `inputs` and `outputs` | Define `inputs: [service: model_deploy_service, data: hiring_data_testing]`, `outputs: [logs: prediction_logs]` |
| 5 | `production_data_monitor` has empty `inputs` and `outputs` | Define `inputs: [data: hiring_data_testing]`, `outputs: [report: data_drift_report]`; remove from `excluded_operations` when ready |
| 6 | `system_monitor` has empty `inputs`, `outputs`, and no method | Define method, `inputs: [logs: prediction_logs]`, `outputs: [report: system_health_report]`; keep in `excluded_operations` until scoped |
| 7 | `post_inference_transformations` has no `operations` entry | Add op: `id: post_inference_eq_odds`, `type: post_inference_transformations`, `requirement_dimension: fairness`, `method_name: post_processing_fairness` |
| 8 | `data_profiling` entry has wrong `method_name` | Change `method_name: data_drift_detection` → `method_name: data_profiling`; the current pointer confuses profiling (YData) with drift detection (Evidently) |

---

## Implementation Priority

Ordered to unblock downstream operations first.

| Priority | id | Fix required |
|---|---|---|
| 1 | `data_preprocessing_step1` | Wire `split_train_valid_test_data` + `preprocess_train_data` into this op; align YAML `outputs` |
| 2 | `data_preprocessing_reweighing` | Uncomment `Reweighing` import; rename source function to `bias_mitigation_pre_reweighing` |
| 3 | `model_evaluation_accuracy_demographic_groups` | Add `report: Report` as function parameter |
| 4 | `model_evaluation_equality_of_outcome` | Implement: wrap `statistical_parity` + `disparate_impact` calls already present in `modelling.py` |
| 5 | `model_evaluation_equality_of_opportunity` | Implement: wrap `average_odds_diff` + equal opportunity difference calls |
| 6 | `model_validation_baseline` | Implement threshold check against `model_train` config; return `Status` |
| 7 | `model_validation_fairness` | Implement threshold check against `model_evaluation_fairness_metrics` config; return `Status` |
| 8 | `model_train_reweighing` | Uncomment `mlflow` import; replace hardcoded `metrics_accuracy.csv` path with input `Report` artifact |
| 9 | `model_monitor` | Replace `print(log_entry)` with MLflow or file-based logging; fill YAML `inputs`/`outputs` (spec gap #4) |
| 10 | `production_data_monitor` | Implement Evidently drift report; fill YAML `inputs`/`outputs` (spec gap #5); remove from `excluded_operations` |
