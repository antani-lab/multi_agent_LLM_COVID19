# Do autonomous agents improve chest radiograph interpretation? A leakage-audited evaluation

Code accompanying the manuscript **“Do autonomous agents improve chest radiograph interpretation? A leakage-audited evaluation.”** This repository implements a locally deployed, leakage-audited multi-agent workflow for modified Radiographic Assessment of Lung Edema (mRALE) pneumonia-severity estimation on frontal chest radiographs (CXRs), with exploratory prediction of reverse transcription polymerase chain reaction (RT-PCR)-confirmed SARS-CoV-2 infection.

## Authors

Zhaohui Liang; Sivaramakrishnan Rajaraman; Niccolo Marini; Zhiyun Xue; Sameer Antani

Division of Intramural Research, National Library of Medicine, National Institutes of Health, Bethesda, Maryland, USA. Correspondence: sameer.antani@nih.gov.

## Abstract

**Purpose:** To determine whether a locally deployed multi-agent system improves the estimation of pneumonia severity on chest radiographs (CXRs). Severity was quantified using the modified Radiographic Assessment of Lung Edema (mRALE) score. A secondary exploratory goal was to test whether artificial intelligence could predict SARS-CoV-2 infection from CXRs. Reverse transcription polymerase chain reaction (RT-PCR) served as the reference standard.

**Materials and methods:** This retrospective study included 2,581 CXRs from 2,064 studies organized into 1,762 independent groups. We used a leakage-audited, five-fold, group-stratified cross-validation framework. Models were trained separately for each fold. Inner-validation data were used for model selection and calibration, and outer test folds remained locked until evaluation. We compared conventional and multimodal agents, anatomy-aware inputs, deterministic fusion, learned stacking, and a locally deployed Qwen3.5-4B reasoner. Comparisons used grouped bootstrap confidence intervals and multiplicity-controlled paired tests.

**Results:** Softmax fusion achieved the lowest mRALE mean absolute error (MAE), 3.349 (95% confidence interval [CI], 3.167–3.544), and a quadratic weighted kappa (QWK) of 0.835. Ridge stacking (MAE, 3.518) outperformed the full large language model (LLM) reasoner (MAE, 4.014). The mean paired absolute-error reduction was 0.496 points (95% CI, 0.330–0.685). Anatomy-aware processing performed worse than direct whole-image input (MAE, 7.134 vs 3.661). For exploratory prediction of RT-PCR-confirmed SARS-CoV-2 infection, the highest area under the receiver operating characteristic curve was 0.665, with a specificity of 0.024 at the default threshold. Radiographic appearance was frequently discordant with virologic status. Among RT-PCR-positive images, 17.9% were radiographically clean. Among RT-PCR-negative images, 82.3% showed pulmonary opacity.

**Conclusion:** Simple fusion of locally deployed, task-specific models improved mRALE-based pneumonia severity estimation. LLM reasoning and anatomy-aware processing provided no additional benefit. Exploratory CXR-based prediction of RT-PCR-confirmed SARS-CoV-2 infection was unreliable and did not support image-only infection detection. Trustworthy development of agentic radiology systems requires leakage auditing, validation-only model selection, locked outer-fold evaluation, and transparent reporting of negative ablations.

## What is implemented

- Leakage-audited cohort construction and group-stratified five-fold partitions.
- Lung localization and anatomy-aware whole-, bilateral-, and unilateral-lung views.
- Conventional CXR models, frozen encoders, BiomedCLIP concept features, zero-shot multimodal models, and fold-specific LoRA adapters.
- A common registry for model predictions, uncertainty, availability, localization, and radiographic evidence.
- Mean, median, inverse-error, inverse-variance, and softmax fusion.
- Ridge, logistic, gradient-boosted, and gating-based learned integration.
- A locally deployed Qwen3.5-4B reasoning agent with strict structured-output checks.
- Patient-grouped confidence intervals, paired statistical tests, calibration, threshold transfer, subgroup analysis, domain-shift testing, rationale audits, and reproducibility gates.
- Interruption-safe execution for long Biowulf jobs.

The repository reports negative ablations as first-class results. It does not assume that an LLM reasoner or anatomy-aware processing must outperform simpler integration.

## Repository layout

```text
docs/                         Hardening and interruption/resume documentation
notebooks/
  stage_A/                    NB 00–04: environment, cohort, folds, external data, localization
  stage_B/                    NB 05–12: individual agents and anatomy-aware ablations
  stage_C/                    NB 13–16: registry, fusion, reasoning, sensitivity analyses
  stage_D/                    NB 17–22: statistics, validation, figures, and release audit
  legacy_training/            Earlier training notebooks retained for provenance
scripts/                      Repository validation and shared-module synchronization
src/multi_agent_cxr/          Importable metric, reasoning, and statistical utilities
tests/                        Lightweight unit and repository-integrity tests
legacy/initial_implementation Earlier public implementation retained from repository history
```

The Stage A–D notebooks are the manuscript-aligned workflow. The files under `legacy/` and `notebooks/legacy_training/` are retained for provenance and may contain historical paths or superseded experimental choices.

## Installation

Python 3.10 or later is recommended. GPU notebooks require a CUDA-capable PyTorch environment; the study used NVIDIA A100 GPUs on NIH Biowulf.

```bash
git clone https://github.com/StanleyLiangYork/multi_agent_LLM_COVID19.git
cd multi_agent_LLM_COVID19
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,test]"
```

Install PyTorch and, when used, bitsandbytes with versions compatible with the CUDA runtime on the execution host. MedGemma is gated on Hugging Face; accept its terms and authenticate with the Hugging Face client before running the relevant notebooks. Model weights, LoRA adapters, datasets, and generated results are not committed here.

## Expected inputs

Stage A NB 00 resolves the project paths and writes the path contract consumed by every later notebook. Configure its `PROJECT_ROOT`, dataset search directories, and any path rewrites before the first run. The three primary source files expected by the current workflow are:

```text
covid_midrc_dataset.csv
combined_cxr_harmony_train.jsonl
montgomery_cxr_images.csv
```

The corresponding MIDRC and Montgomery CXR images must be available locally under their applicable data-use terms. Never commit patient data, protected health information, credentials, restricted reports, retrieval indexes built from restricted text, model checkpoints, or generated result directories.

## Execution order

Run the numbered notebooks in order within each stage. Long, independent Stage B jobs may run concurrently when their Stage A prerequisites are complete. Each notebook documents its inputs, outputs, smoke-test settings, and final gate.

1. **Stage A, NB 00–04:** establish the environment, audit the cohort, generate leakage-free folds, prepare external cohorts, and cache anatomy-aware views.
2. **Stage B, NB 05–12:** generate out-of-fold predictions for individual conventional, encoder, multimodal, LoRA, and anatomy-aware agents.
3. **Stage C, NB 13–16:** build the common registry, fit fusion methods, run the local LLM reasoner, and evaluate prompt/decoding depth.
4. **Stage D, NB 17–22:** perform locked statistical comparisons, calibration and threshold analysis, external stress testing, interpretability audits, export tables/figures, and run the release audit.

Detailed per-notebook functions and dependencies are listed in [`notebooks/README.md`](notebooks/README.md) and the README inside each stage. Interruption and restoration behavior is documented in [`docs/RESUME.md`](docs/RESUME.md).

## Reproducibility contract

- Related images remain in the same group and never cross training/test boundaries.
- Every outer fold is evaluated only by a model or adapter that did not train on that fold.
- Early stopping, configuration selection, reliability estimation, calibration, fusion fitting, and threshold selection use only outer-training or inner-validation data.
- Outer-fold predictions are pooled only after each fold has been evaluated under its locked configuration.
- External cohorts do not contribute to training, tuning, calibration, fusion fitting, or selection.
- Random seeds, model revisions, prediction schemas, cache fingerprints, gate files, and artifact hashes are recorded.

## Shared Python modules and tests

The notebook-local helper modules are mirrored as the importable `multi_agent_cxr` package:

- `metrics`: prediction schema, mRALE and classification metrics, calibration, and invalid-output policy.
- `reasoner`: prompt rendering, strict response parsing, journal/resume logic, and local model loading.
- `statistics`: patient-level bootstrap, paired tests, multiplicity control, and reporting helpers.

Run the lightweight checks without downloading models or data:

```bash
python -m pytest
python scripts/validate_repository.py
python scripts/sync_shared_modules.py --check
```

## Data and model availability

This repository distributes code only. Obtain MIDRC and Montgomery data from their original sources and follow their access conditions. Foundation-model weights are obtained from their respective model providers. Fine-tuned adapters and large generated artifacts should be released separately when permitted, with immutable version identifiers and no restricted data.

## Research-use statement

This software is provided for research and reproducibility. It is not a medical device and is not intended for clinical diagnosis, triage, treatment decisions, or replacement of radiologist interpretation or virologic testing.

## Citation

Please cite the repository and associated manuscript when using this code:

> Liang Z, Rajaraman S, Marini N, Xue Z, Antani S. Do autonomous agents improve chest radiograph interpretation? A leakage-audited evaluation [software]. GitHub; 2026. https://github.com/StanleyLiangYork/multi_agent_LLM_COVID19

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff). Add the journal citation and DOI after publication.

## License

This project is released under the [MIT License](LICENSE).
