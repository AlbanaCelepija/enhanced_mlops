# Tech Stack

## Overview

The tabular AI product is organized into five functional layers. Each layer has a designated set of libraries chosen for modularity, reproducibility, and production readiness.

---

## Stage 1 — Data

| Concern | Library | Version | Role |
|---|---|---|---|
| Data profiling | `ydata-profiling` | latest | Automated EDA reports (statistics, correlations, missing values) |
| Synthetic data | `ydata-synthetic` | latest | Generate synthetic samples for augmentation or testing |
| Fairness metrics & mitigation | `holisticai` | 1.0.14 | Reweighing, Grid Search Reduction, Equalized Odds, full fairness metric suite |
| Drift detection | `evidently` | latest | Detect distribution shifts between training and production data |
| Data manipulation | `pandas`, `numpy` | latest | Tabular data processing and feature transformations |


**Fairness techniques implemented:**

| Stage | Technique | Source |
|---|---|---|
| Pre-processing | Reweighing | Kamiran & Calders, 2012 |
---

## Stage 2 — Modelling

| Concern | Library | Version | Role |
|---|---|---|---|
| Core ML | `scikit-learn` | 1.6.1 | Ridge Classifier, preprocessing pipelines, cross-validation |
| Gradient boosting | `lightgbm` | 3.3.0 | Alternative boosted tree classifier |
| Fairness metrics & mitigation | `holisticai` | 1.0.14 | Reweighing, Grid Search Reduction, Equalized Odds, full fairness metric suite |
| Fairness (alternative) | `fairlearn` | latest | Microsoft fairness toolkit for additional constraints and metrics |
| Feature importance | `interpret` | latest | Global and local explainability (EBM, SHAP) |
| Local explanations | `lime` | latest | LIME post-hoc explanations per prediction |
| AI explainability | `aix360` | latest | IBM AIX360 contrastive and rule-based explanations |

**Fairness techniques implemented:**

| Stage | Technique | Source |
|---|---|---|
| In-processing | Grid Search Reduction | Agarwal et al., 2018 |
---


## Stage 3 — Operationalisation

| Concern | Library / Tool | Role |
|---|---|---|
| Model serving | `kserve` | Kubernetes-native InferenceService REST API |
| MLflow serving | `mlserver`, `mlserver-mlflow` | MLflow model server integration |
| Monitoring API | `fastapi` | REST endpoint for real-time prediction logging |
| Drift monitoring | `evidently` | Report data and prediction drift in production |

**Fairness techniques implemented:**

| Stage | Technique | Source |
|---|---|---|
| Post-processing | Equalized Odds | Hardt et al., 2016 |

---


## Platform Targets

| Environment | Stack |
|---|---|
| **Local** | Python, scikit-learn, MLflow, FastAPI — runs on a single machine for development and testing |
| **Digital Hub** | `digitalhub_runtime_python` decorators  — runs on enterprise Kubernetes cluster |

---

## Configuration Format

All pipeline stages and operations are declared in YAML:

| File | Purpose |
|---|---|
| `metadata/aipc_local.yaml` | Full pipeline spec for local execution |
| `metadata/aipc_dh.yaml` | Full pipeline spec for Digital Hub execution |

---

## Language & Runtime

- **Language:** Python 3.10+
- **Package management:** `pip` + `requirements.txt`
- **Testing:** `pytest`
- **Notebook exploration:** Jupyter
