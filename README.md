# Predicting Severe (Complicated) vs Non-Severe Pediatric Appendicitis — a Modern-Algorithm Benchmark

A fully reproducible benchmark that isolates the **severity** prediction task (complicated vs uncomplicated appendicitis) on the **Regensburg Pediatric Appendicitis** dataset and tests whether modern tabular classifiers can match or beat the published results of Marcinkevičs et al. (2024).


---

## Table of contents
1. [Introduction & objectives](#1-introduction--objectives)
2. [Dataset](#2-dataset)
3. [What was done (methodology)](#3-what-was-done-methodology)
4. [Results](#4-results)
5. [Discussion](#5-discussion)
6. [How to reproduce](#6-how-to-reproduce)
7. [Repository contents — everything to upload](#7-repository-contents--everything-to-upload)
8. [TabPFN v2 note](#8-tabpfn-v2-note)
9. [Optional: ultrasound image branch (MedSigLIP)](#9-optional-ultrasound-image-branch-medsiglip)
10. [Limitations & ethics](#10-limitations--ethics)
11. [Citation](#11-citation)
12. [License](#12-license)

---

## 1. Introduction & objectives

### Clinical background
Acute appendicitis is the most common abdominal surgical emergency in children. A key clinical distinction is between **uncomplicated** appendicitis (which may be managed conservatively) and **complicated** appendicitis — defined by **abscess formation, gangrene, or perforation** — which typically requires urgent surgery and carries higher morbidity. Reliably predicting *complicated* (severe) disease early, from data available before surgery, could support triage and treatment decisions.

### Reference study
This work uses the dataset and takes as its benchmark the severity results of:

> Marcinkevičs R., Reis Wolfertstetter P., Klimiene U., et al. (2024). *Interpretable and intervenable ultrasonography-based machine learning models for pediatric appendicitis.* **Medical Image Analysis** 91, 103042. https://doi.org/10.1016/j.media.2023.103042

That paper predicted diagnosis, management and **severity** from **abdominal ultrasound images** using concept-bottleneck and related models. Its **best severity result** was **AUROC ≈ 0.77, AUPR ≈ 0.58, Brier ≈ 0.15** (a radiomics + cost-sensitive Random Forest baseline) — and, notably, its deep/interpretable image models scored lower on severity (AUROC 0.66–0.74).

### Research question & objectives
**Question:** Can modern classifiers, trained on the dataset's **tabular** clinical/laboratory/score/ultrasound-finding data, match or exceed the paper's best **severity** performance?

**Objectives:**
1. Isolate the severity task (complicated vs uncomplicated) and reproduce the paper's evaluation metrics (AUROC, AUPR, Brier).
2. Build a strictly **leakage-safe** feature set (severity is *defined* by perforation/abscess/gangrene, which must not be used as inputs).
3. Benchmark a panel of modern algorithms — gradient boosting, balanced ensembles, a glass-box additive model, and (optionally) a tabular foundation model.
4. Compare against the published severity numbers and discuss what the comparison does and does not show.

---

## 2. Dataset

- **Name:** Regensburg Pediatric Appendicitis
- **Source:** UCI Machine Learning Repository, **ID 938** — https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis
- **Origin:** Children's Hospital St. Hedwig, Regensburg, Germany (patients 2016–2021).
- **Size:** 782 patients; **781** have a Severity label (one missing record dropped).
- **Modalities:** tabular data (demographics, physical examination, laboratory markers, clinical scores [Alvarado, Paediatric Appendicitis Score], and **expert-extracted abdominal ultrasound findings**) **plus** raw ultrasound images (hosted separately).
- **Targets:** `Diagnosis`, `Management`, `Severity`. **This benchmark uses `Severity`.**
- **Severity class balance:** **119 complicated (15.2%)** vs 662 uncomplicated — a small, imbalanced target.

### How the data is obtained (automatic)
The notebook downloads the **tabular** data programmatically via the `ucimlrepo` client (`fetch_ucirepo(id=938)`) on first run and caches it to `regensburg_appendicitis.csv`. No manual download is required.

> The raw **ultrasound image set** (several GB, hosted on Zenodo) is **not** used by this benchmark and is **not** included in the repo. It is only needed for the optional image branch (§9).

---

## 3. What was done (methodology)

Everything below is implemented in **`Appendicitis_Severity_Benchmark.ipynb`** and is fully reproducible.

### 3.1 Target definition
Binary severity, following the paper: **complicated (positive) vs uncomplicated (negative)**. `y = (Severity == "complicated")`.

### 3.2 Leakage control (the most important step)
Because *complicated* is **defined** by perforation, abscess or gangrene, any column encoding or revealing the outcome was removed from the predictors. **Excluded columns:**

| Excluded | Reason |
|---|---|
| `Perforation`, `Appendicular_Abscess`, `Abscess_Location` | Definitional (perforation / abscess) |
| `Perfusion` | Proxy for gangrene / necrosis |
| `Length_of_Stay` | Post-hoc outcome |
| `Diagnosis`, `Management` | The other target labels |
| `Gynecological_Findings`, `Lymph_Nodes_Location` | Sparse, unusable German free-text |

This leaves **46 predictors** (16 numeric, 30 categorical). Radiologist-assessed ultrasound *signs* observable pre-operatively (free fluid, surrounding tissue reaction, appendix wall layers, target sign, etc.) were **kept** — these are the same high-level "concepts" the source paper modelled, and are legitimate inputs.

### 3.3 Preprocessing
- Categorical features → one-hot encoded with an explicit **missing-value indicator** (`dummy_na=True`).
- Numeric features → **median-imputed within each CV fold** (no leakage across folds).
- Resulting design matrix: **118 columns**. The **same matrix is used for every model** (a fair, algorithm-only comparison).

### 3.4 Models
All configured for class imbalance, with light/sensible defaults (no aggressive tuning, to avoid optimistic bias on a small dataset):

| Model | Family | Imbalance handling |
|---|---|---|
| Elastic-net Logistic Regression | Linear (glass-box) | `class_weight="balanced"` |
| Random Forest (cost-sensitive) | Bagged trees | `balanced_subsample` |
| Balanced Random Forest | Bagged trees + undersampling | balanced bootstrap (imblearn) |
| XGBoost | Gradient boosting | `scale_pos_weight` |
| CatBoost | Gradient boosting | `auto_class_weights="Balanced"` |
| Explainable Boosting Machine (EBM) | Glass-box additive (GAM) | sample reweighting |
| *TabPFN v2 (optional)* | Tabular foundation model | in-context (see §8) |

### 3.5 Evaluation protocol
- **Repeated stratified cross-validation:** 5 repeats × stratified 5-fold (different seeds).
- For each repeat, **out-of-fold** predicted probabilities are collected for all samples and scored; metrics are averaged across repeats (**mean ± sd**). This mirrors the paper's 10 initialisations.
- **Metrics (same as the paper):** **AUROC** (discrimination), **AUPR** (precision–recall; important under imbalance), **Brier score** (calibration).
- All reported numbers are on **held-out** data.

### 3.6 Tooling
Python 3.14; scikit-learn, xgboost, catboost, imbalanced-learn, interpret (EBM), pandas/numpy, matplotlib. See [`requirements.txt`](requirements.txt).

---

## 4. Results

### 4.1 This work (severity; 5×5-fold CV, mean ± sd)

| Model | AUROC | AUPR | Brier (↓) |
|---|---|---|---|
| **Random Forest (cost-sensitive)** | **0.915 ± 0.003** | 0.635 ± 0.013 | 0.083 |
| Explainable Boosting Machine (EBM) | 0.914 ± 0.004 | **0.681 ± 0.016** | **0.077** |
| Balanced Random Forest | 0.914 ± 0.002 | 0.617 ± 0.007 | 0.116 |
| CatBoost | 0.907 ± 0.004 | 0.644 ± 0.014 | 0.088 |
| XGBoost | 0.906 ± 0.004 | 0.624 ± 0.014 | 0.091 |
| Elastic-net Logistic Regression | 0.891 ± 0.004 | 0.642 ± 0.017 | 0.103 |

Best discrimination: **cost-sensitive Random Forest (AUROC 0.915)**. Best precision–recall and calibration: the glass-box **EBM (AUPR 0.681, Brier 0.077)**. Lower Brier is better.

### 4.2 Comparison with Marcinkevičs et al. (2024) — severity

| Source / Model | Inputs | AUROC | AUPR | Brier (↓) |
|---|---|---|---|---|
| Marcinkevičs 2024 — Radiomics+RF (their best) | US images | 0.77 | 0.58 | 0.15 |
| Marcinkevičs 2024 — MVBM-LSTM (black-box) | US images | 0.74 | 0.58 | 0.22 |
| Marcinkevičs 2024 — ResNet-18 | US images | 0.73 | 0.52 | 0.18 |
| Marcinkevičs 2024 — CBM (interpretable) | US images | 0.68 | 0.44 | 0.23 |
| **This work — RF (cost-sensitive)** | **Tabular** | **0.915** | **0.635** | **0.083** |
| **This work — EBM (glass-box)** | **Tabular** | **0.914** | **0.681** | **0.077** |
| **This work — CatBoost** | **Tabular** | **0.907** | **0.644** | **0.088** |

Relative to the paper's best severity model, the tabular models improve **AUROC by ≈ +0.14** (0.77 → 0.91–0.92), **AUPR by ≈ +0.06–0.10** (0.58 → 0.62–0.68), and roughly **halve the Brier score** (0.15 → 0.077–0.09).

### 4.3 Figures (generated in the notebook)
- ROC, precision–recall and calibration curves for the best model (out-of-fold).
- AUROC bar chart for all models, with the paper's 0.77 best marked as a reference line.
- EBM global feature-importance plot (clinical/US drivers of predicted *complicated* appendicitis).

---

## 5. Discussion

All six modern classifiers exceeded the paper's best severity figures on every metric. Tree ensembles (Random Forest, EBM, CatBoost) clustered around **AUROC ≈ 0.91**; the **EBM** delivered the best precision–recall and calibration **while remaining a fully interpretable glass-box model**.

### ⚠️ The essential caveat — this is **not** a like-for-like comparison
Marcinkevičs et al. deliberately predicted severity from **ultrasound images only**. This benchmark uses the **tabular** clinical / laboratory / scoring / ultrasound-finding data. **A large part of the improvement therefore reflects richer, higher-level inputs** (CRP, white-cell counts, clinical scores, radiologist-extracted US findings) **rather than a better algorithm operating on the same data.** The correct reading is:

> *“Structured clinical data predicts complicated appendicitis substantially better than image-only models,”* **not** *“algorithm X beats their algorithm on identical inputs.”*

A fair head-to-head would require a **multimodal** model fusing the ultrasound images with the tabular data (scaffolded in §9).

### Why the tabular models do so well
- **Concepts as features:** the dataset's expert-extracted ultrasound findings are essentially the same concepts the paper's models had to *learn* from raw pixels; supplied directly, they are highly predictive.
- **Laboratory signal:** inflammatory markers (CRP, WBC, neutrophilia) and clinical scores carry strong complication signal that image-only models never see.
- **Model fit:** tree ensembles handle mixed types, non-linearity, interactions and missingness natively — well-suited to this heterogeneous, partially-missing table.

### Interpretability parity
The **EBM** — intrinsically interpretable — gave the best AUPR (0.681) and best calibration (Brier 0.077). This satisfies the same clinician-interpretability motivation behind the paper's concept-bottleneck approach, but with markedly higher discrimination and calibration on the tabular data.

### Limitations
- Different modality/inputs from the paper (above) — the headline comparison is **indicative**, not head-to-head.
- Small, imbalanced sample (119 positives); CV standard deviations were small, but **external/temporal validation is required** before any generalisation claim.
- Models used sensible defaults without extensive tuning — results are conservative, but also not tuned to test folds.
- Leakage control depends on correctly identifying definitional/outcome columns; revisit the exclusion list if the schema changes.
- TabPFN v2 results are not yet included (license gate, §8).

---

## 6. How to reproduce

```bash
# 1. Clone
git clone <your-repo-url>.git
cd <your-repo>

# 2. Environment (Python 3.10+; tested on 3.14)
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch and run the notebook (data auto-downloads on first run)
jupyter lab        # open Appendicitis_Severity_Benchmark.ipynb -> Run All
```

- **No manual data download needed** — the notebook fetches UCI ID 938 automatically (or reads the bundled `regensburg_appendicitis.csv` if present).
- **Runtime:** ~3–4 minutes on a laptop CPU for the full 5×5-fold benchmark (EBM is the slowest model).
- **Determinism:** fixed seeds throughout; results reproduce to the reported precision. To match the paper's protocol exactly, set `N_REPEATS = 10` in the Configuration cell.
- **Expected outputs:** the printed results table, `severity_benchmark_results.csv`, `severity_comparison_vs_paper.csv`, and the three figures.

### Headless re-execution (optional)
```bash
python -c "import nbformat; from nbclient import NotebookClient; \
nb=nbformat.read('Appendicitis_Severity_Benchmark.ipynb',as_version=4); \
NotebookClient(nb,timeout=900).execute(); \
nbformat.write(nb,'Appendicitis_Severity_Benchmark.ipynb')"
```

---

## 7. Repository contents — everything to upload

Upload **all** of the following so the repo is complete and self-contained:

| File | Required? | Description |
|---|---|---|
| `README.md` | ✅ | This document |
| `requirements.txt` | ✅ | Pinned dependencies |
| `Appendicitis_Severity_Benchmark.ipynb` | ✅ | The executed, end-to-end benchmark notebook (data → leakage-safe features → repeated CV → comparison → plots → EBM explanation; includes a ready-to-run TabPFN cell) |
| `regensburg_appendicitis.csv` | ➖ recommended | Cached copy of the UCI tabular data (regenerated automatically if absent — include it for exact, offline reproducibility) |
| `severity_benchmark_results.csv` | ➖ optional | Saved results table (regenerated by the notebook) |
| `severity_comparison_vs_paper.csv` | ➖ optional | Saved comparison-vs-paper table (regenerated by the notebook) |
| `LICENSE` | ✅ recommended | Add one (see §12) |
| `.gitignore` | ➖ recommended | See block below |

**Do NOT commit:**
- The Marcinkevičs et al. (2024) **PDF** — it is copyrighted; link to the DOI instead (it lives outside this repo).
- The raw **ultrasound image set** (large; hosted on Zenodo) — link to it rather than committing.
- Virtual environments / caches (`.venv/`, `__pycache__/`, `.ipynb_checkpoints/`).

Suggested **`.gitignore`**:
```gitignore
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
*.log
.DS_Store
```

---

## 8. TabPFN v2 note
**TabPFN v2** (a tabular foundation model, Hollmann et al., *Nature* 2025) is included as a **ready-to-run cell** in the notebook but is **not executed by default**: its pretrained weights require a **one-time, free licence acceptance** (interactive `import tabpfn` login, or setting the `TABPFN_TOKEN` environment variable). It is genuinely well-suited to this small, imbalanced dataset. To enable it:
```bash
pip install tabpfn        # also installs torch
# then accept the licence once (interactive) OR set TABPFN_TOKEN
```
The cell skips cleanly with an explanatory message if the licence is not yet accepted.

---

## 9. Optional: ultrasound image branch (MedSigLIP)
To attempt a **fair multimodal comparison** with the paper, extend the benchmark with the ultrasound images:
1. Download the image set (Zenodo; several GB).
2. Extract per-image embeddings with a medical foundation encoder (e.g. **`google/medsiglip-448`**, gated on Hugging Face; GPU recommended).
3. Aggregate per patient (mean or attention-based MIL) and **late-fuse** with the tabular model.

This is scaffolded (commented) in the notebook (§11 there) and is **not** run by default (requires HF gated access + GPU + the image download).

---

## 10. Limitations & ethics
- Educational/methodological benchmark — **not medical advice**, **not** a validated clinical tool.
- Indicative (not head-to-head) comparison due to the modality difference (§5).
- Small, imbalanced, single-centre dataset; external/temporal validation, calibration-in-use and decision-curve analysis are required before any clinical interpretation.
- Respect the dataset's licence and the source paper's copyright.

---

## 11. Citation

If you use this work, please cite the dataset/source study:

```bibtex
@article{marcinkevics2024appendicitis,
  title   = {Interpretable and intervenable ultrasonography-based machine learning models for pediatric appendicitis},
  author  = {Marcinkevi\v{c}s, Ri\v{c}ards and Reis Wolfertstetter, Patricia and Klimiene, Ugne and others},
  journal = {Medical Image Analysis},
  volume  = {91},
  pages   = {103042},
  year    = {2024},
  doi     = {10.1016/j.media.2023.103042}
}
```
Dataset: Regensburg Pediatric Appendicitis, UCI Machine Learning Repository ID 938 — https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis

---

## 12. License
No licence is set yet. For an open-source release, add a `LICENSE` file — **MIT** is a common permissive choice for the **code**. Note the **dataset** carries its own licence/terms (see the UCI page) and must be respected independently; this repo only redistributes a cached CSV for convenience.
