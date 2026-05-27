<div align="center">

# 🔬 MLOps Experiments Practice

**Hands-on MLflow experimentation — from manual logging to autolog, hyperparameter search with parent/child runs, DagsHub integration, and understanding full MLflow deployment architectures.**

</div>

***

> 🏫 **Credits + Ownership**: Inspired by Vikash Das's MLOps classes — but I ran it, broke it, debugged it, and pushed it further with my own variations. Model registration gave me a headache (tracked as a known issue below). Everything else was genuinely amazing to learn.

***

## 📋 Table of Contents

- [What This Project Is](#-what-this-project-is)
- [Experiments Overview](#-experiments-overview)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [MLflow Autologging](#-mlflow-autologging)
- [Hyperparameter Tuning with Parent/Child Runs](#-hyperparameter-tuning-with-parentchild-runs)
- [DagsHub Integration](#-dagshub-integration)
- [MLflow Deployment Architectures](#-mlflow-deployment-architectures)
- [Key Learnings & Known Issues](#-key-learnings--known-issues)

***

## 🔍 What This Project Is

A focused practice project to go deep on **MLflow's experiment tracking ecosystem**. The goal wasn't to build a production model — it was to understand every knob MLflow exposes:

- How `mlflow.autolog()` works and what it captures automatically
- How to structure **parent runs with nested child runs** for hyperparameter search
- How `GridSearchCV` + MLflow together log every single combination and surface the best one
- How **DagsHub** acts as a remote MLflow tracking server with zero infrastructure setup
- What the three **MLflow deployment architectures** actually look like and when to use each

The model is a **RandomForestClassifier** on breast cancer / wine datasets — the focus is entirely on the *observability and tracking layer* around the model.

***

## 🧪 Experiments Overview

| File | Dataset | Technique | Tracking Mode |
|------|---------|-----------|---------------|
| `file1.py` | Wine | RandomForest baseline | Manual `mlflow.log_*` |
| `file2.py` | Wine | RandomForest with artifacts | Manual logging + confusion matrix artifact |
| `autologging.py` | Wine | RandomForest | `mlflow.autolog()` |
| `hypertune.py` | Breast Cancer | GridSearchCV + RF | Parent/Child runs → DagsHub |
| `hypertuneautologging.py` | Breast Cancer | GridSearchCV + RF | Autolog + Parent/Child |

***

## 📁 Project Structure

```
MLOPsExperimentsPractice/
│
├── src/
│   ├── file1.py                   # Basic MLflow run — manual logging
│   ├── file2.py                   # Manual logging + confusion matrix artifact
│   ├── autologging.py             # mlflow.autolog() experiment
│   ├── hypertune.py               # GridSearchCV with parent/child run structure
│   └── hypertuneautologging.py    # GridSearchCV + autolog combined
│
├── mlartifacts/                   # Local MLflow artifact store
├── Confusion_matrix.png           # Logged artifact from autologging experiment
├── autolognotes.md                # Personal notes on autolog behaviour
└── README.md
```

***

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [DagsHub](https://dagshub.com) account (free) linked to this repo

### Installation

```bash
git clone https://github.com/TheHashiramaSenju/MLOPsExperimentsPractice.git
cd MLOPsExperimentsPractice

python -m venv venv
source venv/bin/activate

pip install mlflow dagshub scikit-learn pandas matplotlib seaborn
```

### Set DagsHub Credentials

```bash
export MLFLOW_TRACKING_USERNAME=your_dagshub_username
export MLFLOW_TRACKING_PASSWORD=your_dagshub_token
```

Or use `dagshub.init()` inside the script — it handles auth interactively on first run.

### Run Any Experiment

```bash
python src/autologging.py           # autolog demo
python src/hypertune.py             # parent/child GridSearch run
python src/hypertuneautologging.py  # combined
```

View results at: `https://dagshub.com/TheHashiramaSenju/MLOPsExperimentsPractice.mlflow`

Or locally (if using local tracking):

```bash
mlflow ui
# Open http://127.0.0.1:5000
```

***

## ⚡ MLflow Autologging

`mlflow.autolog()` is one of MLflow's most powerful features. A single line before training automatically captures:

- **Parameters** — all hyperparameters passed to the model constructor
- **Metrics** — training accuracy, validation scores
- **Model** — serialized model artifact
- **Dataset hash** — for reproducibility

```python
mlflow.autolog()  # That's it. One line.

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=100, n_estimators=20)
    rf.fit(X_train, y_train)
    # ↑ MLflow already logged n_estimators=20, max_depth=100, accuracy, the model, etc.
```

### What autolog does vs manual logging

| Aspect | Manual `mlflow.log_*` | `mlflow.autolog()` |
|--------|----------------------|---------------------|
| Parameters | You explicitly list each one | All constructor args captured |
| Metrics | You compute and log | Logged after `.fit()` automatically |
| Model | `mlflow.sklearn.log_model(...)` | Auto-saved |
| Artifacts | You call `log_artifact()` | Some frameworks include feature importance, etc. |
| Control | Full | Less — but enough for most cases |

> **Lesson learned**: `autolog()` must be called **before** `mlflow.start_run()`, not inside it. Calling it after the run context opens may cause some params to be missed.

***

## 🌳 Hyperparameter Tuning with Parent/Child Runs

This is the most interesting pattern learned in this project. When running `GridSearchCV`, you want to track **every combination** as a separate run, but also have a **single parent run** that shows the best result. MLflow's nested runs handle exactly this.

### The Pattern

```python
with mlflow.start_run() as parent:
    grid_search.fit(X_train, y_train)

    # Log each CV combination as a child run
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])

    # Log the winner on the parent
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", grid_search.best_score_)
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")
```

### How it looks in the MLflow UI

```
📁 Parent Run  [breast-cancer-RandomForest-hp]
│   ├── best_params: {n_estimators: 100, max_depth: 10}
│   └── best_accuracy: 0.967
│
├── 🔵 Child Run 1  → n_estimators=10,  max_depth=None  → acc=0.941
├── 🔵 Child Run 2  → n_estimators=10,  max_depth=10    → acc=0.953
├── 🔵 Child Run 3  → n_estimators=20,  max_depth=None  → acc=0.958
│   ...
└── 🔵 Child Run N  → n_estimators=100, max_depth=10   → acc=0.967  ← best
```

This gives you a **comparison table of all 42 combinations** (6 n_estimators × 7 max_depth values) and you can sort by accuracy in the UI to instantly find the winner.

### What also gets logged on the parent run

- Training dataset (as `mlflow.data.from_pandas`)
- Test dataset
- The source file itself (`mlflow.log_artifact(__file__)`)
- Author tag
- The best model artifact under `random_forest/`

***

## 🐶 DagsHub Integration

Instead of running a local MLflow server or self-managing a remote tracking server, **DagsHub** provides a hosted MLflow backend for free — tied directly to your GitHub repo.

```python
import dagshub
dagshub.init(repo_owner='thehashiramasenju', repo_name='mlopsexperimentspractice', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/TheHashiramaSenju/MLOPsExperimentsPractice.mlflow")
```

DagsHub acts as both:
- The **MLflow tracking server** (stores run metadata, params, metrics)
- The **artifact store** (stores model files, plots, datasets)

No Postgres, no S3 setup. Perfect for learning and solo projects.

***

## 🏗️ MLflow Deployment Architectures

This was one of the most valuable conceptual things learned. There are three standard MLflow architectures depending on your scale and infrastructure needs.

***

### 🖥️ Architecture 1 — Fully Local (Scenario 1)

Everything lives on your laptop. Great for solo experiments and learning.

```
┌─────────────────────────────────────────────────────┐
│                    YOUR MACHINE                     │
│                                                     │
│  ┌──────────────┐        ┌──────────────────────┐  │
│  │  Python      │        │  MLflow Tracking     │  │
│  │  Script      │──────▶ │  Server              │  │
│  │  (your code) │        │  (mlflow ui /        │  │
│  └──────────────┘        │   http://localhost:  │  │
│                          │   5000)              │  │
│                          └──────────┬───────────┘  │
│                                     │               │
│                          ┌──────────▼───────────┐  │
│                          │  ./mlruns/           │  │
│                          │  (SQLite backend)    │  │
│                          │                      │  │
│                          │  ./mlartifacts/      │  │
│                          │  (local filesystem)  │  │
│                          └──────────────────────┘  │
└─────────────────────────────────────────────────────┘

  Tracking URI:  http://127.0.0.1:5000
  Artifact URI:  ./mlartifacts   (relative path!)
  Backend store: ./mlruns/mlflow.db
```

**Use when**: Learning, solo projects, no collaboration needed.

***

### ☁️ Architecture 2 — Hybrid (Scenario 2)

Tracking server runs locally or on a VM, but artifacts go to cloud storage. Most common in small teams.

```
┌───────────────────────┐         ┌────────────────────────┐
│   YOUR MACHINE /      │         │   CLOUD STORAGE        │
│   TEAM VM             │         │   (GCS / S3 / Azure)   │
│                       │         │                        │
│  ┌────────────────┐   │         │  ┌──────────────────┐  │
│  │ Python Script  │   │  push   │  │  model.pkl       │  │
│  │ (training run) │───┼────────▶│  │  confusion.png   │  │
│  └───────┬────────┘   │artifacts│  │  source_file.py  │  │
│          │            │         │  └──────────────────┘  │
│          │ log params │         └────────────────────────┘
│          │ metrics    │
│          ▼            │         ┌────────────────────────┐
│  ┌────────────────┐   │         │   REMOTE DB            │
│  │ MLflow Server  │───┼────────▶│   (PostgreSQL /        │
│  │ (tracking URI) │   │metadata │    MySQL / SQLite)     │
│  └────────────────┘   │         └────────────────────────┘
└───────────────────────┘

  Tracking URI:  http://<server-ip>:5000
  Artifact URI:  gs://your-bucket/mlartifacts
  Backend store: postgresql://...
```

**Use when**: Team collaboration, artifacts too large for local disk, experiments need to persist.

***

### 🌐 Architecture 3 — Fully Remote / MLflow Server (Scenario 3)

Everything is remote. Training runs from anywhere (local, CI/CD, cloud VM) and everything is centralized. This is what DagsHub approximates for free.

```
┌────────────────────┐
│   YOUR LAPTOP /    │
│   CI/CD Pipeline / │
│   Cloud VM         │
│                    │
│  ┌──────────────┐  │
│  │ Python       │  │
│  │ Training     │  │
│  │ Script       │  │
│  └──────┬───────┘  │
└─────────┼──────────┘
          │
          │  HTTPS (params, metrics, tags)
          ▼
┌─────────────────────────────────────────────────┐
│              MLFLOW TRACKING SERVER             │
│           (Hosted / DagsHub / self-managed)     │
│                                                 │
│   ┌──────────────────┐  ┌─────────────────────┐│
│   │  Backend Store   │  │   Artifact Store    ││
│   │  (PostgreSQL /   │  │   (S3 / GCS /       ││
│   │   MySQL)         │  │    Azure Blob)      ││
│   │                  │  │                     ││
│   │  runs, params,   │  │  model.pkl, plots,  ││
│   │  metrics, tags   │  │  datasets, source   ││
│   └──────────────────┘  └─────────────────────┘│
└─────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│   MLflow UI         │
│   (browser view)    │
│   Compare runs,     │
│   register models,  │
│   view plots        │
└─────────────────────┘

  Tracking URI:  https://dagshub.com/<user>/<repo>.mlflow
                 OR https://your-mlflow-server.com
  Artifact URI:  Managed by the server
  Backend store: Managed by the server
```

**Use when**: Production, teams across machines, CI/CD integration, model registry needed.

***

### Architecture Comparison

| | Fully Local | Hybrid | Fully Remote |
|--|-------------|--------|--------------|
| **Setup complexity** | Zero | Medium | High (or use DagsHub) |
| **Collaboration** | ❌ Solo only | ✅ Team | ✅ Team + CI/CD |
| **Artifact storage** | Local disk | Cloud bucket | Cloud bucket (managed) |
| **Backend DB** | SQLite/flat files | Remote DB | Remote DB (managed) |
| **Model Registry** | ✅ Local | ✅ | ✅ Full support |
| **Best for** | Learning | Small teams | Production |

***

## 🧠 Key Learnings & Known Issues

### ✅ Things That Clicked

**`mlflow.autolog()` is genuinely flexible** — it knows what each framework exposes. For `sklearn`, it logs `n_estimators`, `max_depth`, training accuracy, the fitted model, and more — without you writing a single `log_param`.

**Parent/child run nesting** makes hyperparameter search readable. Without nesting, 42 GridSearchCV combinations flood the experiment view as 42 separate top-level rows. With nesting, you see 1 parent row and expand to see children — clean and comparable.

**DagsHub removes all infra friction** — no Postgres, no S3, no `mlflow server` command. Just `dagshub.init()` and your remote tracking URI. Ideal for this learning phase.

**`mlflow.log_input()`** for datasets is underrated — it hashes the training and test data and links them to the run, so you always know exactly what data was used.

**`mlflow.log_artifact(__file__)`** logs the current Python script itself as an artifact — so you can always reproduce what code produced a given run. Smart habit.

### 🔴 Known Issue — Model Registration

Attempting to register the best model to the **MLflow Model Registry** via DagsHub hit permission/configuration issues. The model was logged as an artifact (`mlflow.sklearn.log_model(...)`) successfully, but the **registered model** step (promoting it to `Staging` / `Production` stages) was not completed.

```python
# This works ✅
mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")

# This needs fixing ⚠️
mlflow.register_model(
    model_uri=f"runs:/{run_id}/random_forest",
    name="BreastCancerRF"
)
# → Hits DagsHub model registry config issue — to be resolved
```

**TODO**: Debug the model registry permissions on DagsHub and complete the registration + staging workflow.

### 🟡 `autolog()` Call Order Matters

```python
# ❌ Wrong — autolog called inside the run context, some params missed
with mlflow.start_run():
    mlflow.autolog()
    rf.fit(X_train, y_train)

# ✅ Correct — autolog before the run opens
mlflow.autolog()
with mlflow.start_run():
    rf.fit(X_train, y_train)
```

### 🟡 Commented-Out Manual Logging

In `autologging.py`, the manual `log_metric`, `log_param`, and `log_artifact` calls are commented out intentionally — they were the "before" state. `autolog()` replaces all of them. The comments serve as a reference for what autolog handles automatically.

### 🟢 Local vs Remote Tracking URI Switch

```python
# Local (for offline dev)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Remote DagsHub (for sharing / persistence)
mlflow.set_tracking_uri("https://dagshub.com/TheHashiramaSenju/MLOPsExperimentsPractice.mlflow")
```

Switching between these is the core of the hybrid architecture — same code, different destination.

***

<div align="center">

Genuinely one of the most satisfying things to learn. 🔥

**[TheHashiramaSenju](https://github.com/TheHashiramaSenju)**

</div>