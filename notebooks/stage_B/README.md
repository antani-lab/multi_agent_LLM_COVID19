# Stage B — baseline models, adapted VLMs, and anatomy-aware mRALE

Stage B contains notebooks **NB 05–NB 12**. All eight notebooks are drafted. They build the
baseline suite, fine-tune the two principal multimodal models, characterize NV-Reason, and run
the anatomy-aware mRALE ablation.

This README describes the notebooks as they currently exist. It does not imply that cloud
checkpoints or completed results are present; every notebook must verify its own required inputs
and saved adapters at runtime.

## Recommended execution order

1. **NB 05, NB 06, and NB 08** — independent non-generative baselines; they may run in parallel.
2. **NB 07** — zero-shot MedGemma, Qwen3.5, and NV-Reason agents.
3. **NB 09** — MedGemma multitask LoRA sweep and five-fold training.
4. **NB 10** — matched Qwen3.5 multitask LoRA control; normally inherits NB 09's selected
   configuration.
5. **NB 11** — NV-Reason findings, answer-format scoring, and abstention analysis.
6. **NB 12** — anatomy-aware E4 grid using the five held-out NB 09 adapters.

Run the inexpensive smoke-test mode in each notebook before starting a full cloud job. Do not
mix smoke-test and full-run artifacts unless the notebook's cache fingerprint proves that they
use identical data, model, adapter, prompt, and decoding settings.

## Notebook map

| # | notebook | purpose | principal protocol arms | approximate GPU time |
| --- | --- | --- | --- | --- |
| 05 | `05_frozen_encoder_agents.ipynb` | Frozen CXformer probes and a shared per-lung ordinal head | E0d, E0e, E0f; agent A5 | ~1 h |
| 06 | `06_conventional_cxr_classifiers.ipynb` | End-to-end conventional CXR classifiers | E0g | ~10 h |
| 07 | `07_zeroshot_vlm_agents.ipynb` | Zero-shot VLM generation and token-probability scoring | E0a, E0b, E0c | ~8–12 h |
| 08 | `08_biomedclip_entity_agent.ipynb` | Named diagnostic entities and an entity-feature probe | E0i, E1-L6; agent A6 | ~0.5 h |
| 09 | `09_medgemma_multitask_lora_5fold.ipynb` | MedGemma multitask LoRA sweeps and five-fold training | E5-R/T/L/M, E8, E8d; agent A2 | ~65 h |
| 10 | `10_qwen35_multitask_lora_5fold.ipynb` | Matched Qwen3.5 LoRA control for medical-specialization RQ1 | E5/E8 control; agent A3 | ~55 h |
| 11 | `11_nvreason_agent_and_optional_lora.ipynb` | NV-Reason findings, scoring, and abstention analysis | A4 analysis; E5-S reserved | ~2–3 h, excluding future E5-S training |
| 12 | `12_anatomy_aware_mrale_5fold.ipynb` | Anatomy-aware mRALE decomposition and localization ablations | E4a–E4g; RQ4 | ~25 h |

## Shared Stage A inputs

All Stage B notebooks read paths and pinned model revisions from Stage A NB 00 and use the
regenerated study-group folds from Stage A NB 02. They must not fall back to the legacy
`multi_task_CV` split because that split placed some study groups on both sides of a fold.

Primary inputs are:

- `stage_a_paths.json` from NB 00;
- `midrc_folds_v2.csv` and `fold_definitions/` from NB 02;
- external-cohort manifests and membership checks from NB 03;
- `view_index.csv` with V0, V1, V2L, and V2R paths from NB 04;
- `prompt_templates.json` for the shared prompting contract.

Every internal prediction must be out of fold: an image is scored only by a model or adapter
that did not train on its Stage A `group_id`. Fold-ensemble averaging is allowed only for
external cohorts after the NB 03 membership guard passes.

### Strict JSONL warning

The current Stage A Harmony fold files contain non-standard `NaN` constants. Python's default
`json.loads` accepts them, but strict JSONL tools may reject them. Before using a different data
loader or publishing the files, regenerate them with missing values written as JSON `null` and
validate every line with a strict parser.

## Shared metrics and prediction schema

All notebooks import `cxr_metrics.py`. It owns:

- `PREDICTION_FIELDS` and `make_prediction_row` for per-image results;
- the invalid-output policy: an invalid COVID decision is incorrect with score 0.5, invalid
  total mRALE receives a 24-point penalty, and an invalid per-lung score receives a 12-point
  penalty;
- COVID metrics including AUROC, AUPRC, sensitivity, specificity, balanced accuracy, F1, MCC,
  Brier score, ECE, and score/generation agreement;
- mRALE metrics including penalized MAE, RMSE, correlations, QWK, within-tolerance accuracy,
  coverage, arithmetic consistency, and severity-band MAE;
- fold-level aggregation and 95% confidence intervals.

The module has a 39-check self-test suite. Run it after any edit:

```bash
cd /Users/liangz2/Documents/TETCI_resubmit/experiments/notebooks/stage_B
python3 cxr_metrics.py
```

Several of those checks are regression tests for a specific bug: `is not None` does not catch
NaN, and every notebook here moves prediction rows through a pandas DataFrame, which turns a
JSON `null` into `float('nan')`. NaN then passed the guards, reached `int()`, and crashed a
downstream notebook on behalf of an arm that had legitimately abstained. Use `cm.is_missing()`
rather than `is not None` for any prediction field.

The older README smoke assertions still hold and are included in the suite:

```bash
cd /Users/liangz2/Documents/TETCI_resubmit/experiments/notebooks/stage_B
python3 - <<'PY'
import cxr_metrics as cm

assert cm.mrale_metrics([
    {"gt_mrale_total": 20, "mrale_total": None}
])["mae"] == 24.0

m = cm.classification_metrics(
    ["Yes"] * 93 + ["No"] * 7,
    ["Yes"] * 100,
    [0.99] * 100,
)
assert abs(m["accuracy"] - 0.93) < 1e-9
assert abs(m["auroc"] - 0.5) < 1e-9
print("cxr_metrics smoke checks passed")
PY
```

`arm_summary.csv` is not currently column-identical across NB 05–NB 12. For example, some
notebooks use `coverage`, others use `mrale_coverage`, and generative arms add usability fields.
NB 13 and later consumers must normalize these summaries to a canonical schema rather than
concatenating them by position or assuming a fixed number of columns.

## NB 05 — frozen CXformer agent

NB 05 sets the inexpensive non-generative reference point.

| arm | view and head | question |
| --- | --- | --- |
| E0d | V0, linear probe | What is linearly decodable from frozen CXformer features? |
| E0e | V0, two-layer MLP | Does non-linearity improve the same representation? |
| E0f | V2L/V2R, shared CORAL head | Does per-lung decomposition help a frozen predictor? |

The shared E0f head is important: it prevents the regional arm from winning merely by doubling
its parameter count. Its result is an independent precursor to the E4c-versus-E4d test in NB 12.

Outputs under `stage_B/nb05_frozen_encoder/` include cached embeddings,
`predictions_frozen.jsonl`, fold metrics, `arm_summary.csv`, external predictions,
`run_config.json`, and `gate_nb05.json`.

Reproducibility note: a legacy embedding cache without a fingerprint should be deleted and
recomputed. Accepting an unversioned cache is unsafe after changing the encoder revision,
pooling, views, or preprocessing.

## NB 06 — conventional CXR classifiers

NB 06 trains DenseNet-121, ConvNeXt-Tiny, and ViT-B/16 end to end. DenseNet-121 should use
TorchXRayVision CXR weights. If `torchxrayvision` is unavailable, the notebook records an
ImageNet fallback in `run_config.json`; such a fallback must not be described as domain
pretrained in the manuscript.

The augmentation contract forbids horizontal flipping because right- and left-lung labels are
separate. Other forbidden operations include vertical flipping, elastic deformation, and
mixup/cutmix on mRALE targets.

Outputs under `stage_B/nb06_conventional/` include `predictions_conventional.jsonl`, per-model
cross-fold aggregates, checkpoints, training curves, `arm_summary.csv`, `usability.json`,
`run_config.json`, and `gate_nb06.json`.

## NB 07 — zero-shot VLMs and usable COVID scores

NB 07 compares:

| arm | model | role |
| --- | --- | --- |
| E0a | `google/medgemma-1.5-4b-it` | Medically tuned zero-shot VLM |
| E0b | `Qwen/Qwen3.5-4B` | Matched-scale general-purpose control |
| E0c | `nvidia/NV-Reason-CXR-3B` | Reasoning-tuned CXR findings model |

For COVID, the notebook performs free generation and exact teacher-forced sequence scoring of
the Yes/No candidates. The likelihood-derived score is eligible for ROC analysis only when it
agrees with the model's generated decision on decisive cases. An individual failing arm is
quarantined with `score_usable: false`; its hard-decision metrics remain available, but its
AUROC, AUPRC, DeLong, and calibration results must be omitted. The notebook blocks only when
every arm fails, indicating a systematic scoring problem.

NV-Reason uses a CheXpert-style findings output rather than the mRALE JSON schema. Under
zero-shot prompting, its mRALE output is therefore marked incompatible and omitted rather than
reported as a 24-point MAE. NB 11 analyzes this behavior and scores NV-Reason using its own
answer grammar.

Important outputs are `predictions_zeroshot.jsonl`, `token_probability_scores.parquet`,
`score_validation.csv`, `score_usability.json`, `usability.json`,
`nvreason_traces.jsonl`, `arm_summary.csv`, `run_config.json`, and `gate_nb07.json`.

## NB 08 — BiomedCLIP diagnostic-entity agent

NB 08 restores the BiomedCLIP component explicitly requested in referee 1.2. It provides named
finding scores and an interpretable entity-feature probe rather than another anonymous image
embedding.

Because the cohort has no entity-level ground truth, the gate uses pre-registered construct
validity checks: opacity entities should correlate positively with mRALE, clear-lung entities
should correlate negatively, Montgomery normals should score below severe MIDRC cases, and TB
images should show an appropriate differential entity pattern.

Outputs under `stage_B/nb08_biomedclip_entity/` include `entity_scores.parquet`,
`entity_findings.jsonl`, `predictions_entity_probe.jsonl`, `probe_coefficients.csv`,
`construct_validity.csv`, `arm_summary.csv`, `usability.json`, `run_config.json`, and
`gate_nb08.json`.

BiomedCLIP is loaded from an exact Hugging Face snapshot. NB 08 uses the revision registered by
NB 00 when available and otherwise falls back to the pinned commit recorded in its configuration.
OpenCLIP receives the downloaded snapshot through its `local-dir:` loader, so the recorded commit
is the snapshot actually used rather than metadata attached to an unpinned `main` download.

## NB 09 — MedGemma multitask LoRA

NB 09 is the principal medically tuned model. It runs fold-0 sweeps for LoRA rank, target
modules, optimization settings, multitask composition, and imbalance handling, then trains the
selected configuration across all five folds.

Gate G3 deliberately does not require reproduction of the old MAE 3.88 and balanced accuracy
0.616 because those values came from the superseded leaky folds. The regenerated folds establish
the new reference point.

Outputs under `stage_B/nb09_medgemma_lora/` include sweep tables,
`sweep_selection.json`, five `folds/fold_<k>/best_adapter/` directories, fold summaries,
`predictions_medgemma_lora.jsonl`, token-probability scores, `arm_summary.csv`, fold aggregates,
external predictions, `usability.json`, `run_config.json`, and `gate_nb09.json`.

### Operating points and resume behavior

The notebook now generates COVID scores on each fold's inner validation partition, selects the
Youden-J and 90%-sensitivity thresholds there, and applies the frozen fold-specific thresholds to
the corresponding outer test fold. `operating_points.csv` records per-fold rows and pooled
outer-test decisions without fitting a threshold on outer-test labels.

A completed fold now requires a non-empty `adapter_model.safetensors` or adapter binary as well
as `adapter_config.json`. Older completed adapters without inner-validation score files are
reloaded for evaluation; they are not retrained solely to produce the corrected operating points.

## NB 10 — matched Qwen3.5 LoRA control

NB 10 is intentionally parallel to NB 09 so that RQ1 isolates medically tuned versus
general-purpose pretraining at approximately matched scale. It normally reads NB 09's selected
hyperparameters instead of tuning Qwen more extensively.

Architecture-specific deviations—dynamic visual-token budget, Qwen's loader and LoRA target
discovery, and `enable_thinking=False`—are recorded in `run_config.json`. Trainable-parameter
counts can differ and must be reported beside performance.

NB 10 uses the same corrected per-fold inner-validation threshold selection and non-empty adapter
weight checks as NB 09.

Outputs follow NB 09's structure under `stage_B/nb10_qwen_lora/`, with the addition of
`rq1_comparison.csv`.

## NB 11 — NV-Reason findings and abstention

NB 11 restates RQ2 around the behavior the model can actually express:

> Does a reasoning-tuned CXR model contribute usable diagnostic evidence, and is its abstention
> behavior a surface artifact of output formatting or a deeper limitation?

It parses findings from NB 07 traces, measures abstention by severity with image-intensity
confound checks, and computes an answer-format score by comparing outputs in NV-Reason's own
grammar. The mapping from lung opacity to PCR status is an author-defined radiographic proxy,
not a virologic measurement, and must be labeled accordingly.

Important outputs include `nvreason_findings.jsonl`, the abstention tables,
`answer_format_scores.jsonl`, `predictions_nvreason.jsonl`, `arm_summary.csv`,
`usability.json`, and `rq2_restatement.json`.

### E5-S status

E5-S is **not implemented** in the current notebook. Setting `RUN_E5S_LORA=True` raises a clear
`NotImplementedError`; it does not train, save an adapter, or write a completion marker. Do not
report E5-S as a completed experiment. Any older `SCAFFOLD ONLY` summaries from a previous
notebook revision must be ignored or archived before a future implementation is run.

## NB 12 — anatomy-aware mRALE grid

NB 12 evaluates the same held-out MedGemma LoRA predictor under different image-view and
aggregation conditions. It requires the five NB 09 fold adapters and verifies them at runtime;
this README does not assert that they are currently present.

| arm | current implementation | question |
| --- | --- | --- |
| E4a | V0 whole image; parse four components and constrain total to right + left | Whole-image component baseline |
| E4b | V1 thorax crop; same constrained component output | Does thorax cropping alone help? |
| E4c | V0 plus lung-box coordinates in text; constrained total | Do coordinates alone help? |
| **E4d** | Separate V2L and V2R masked views; constrained total | **Anatomy-aware regional-view arm** |
| E4e | Separate regional views; average the totals independently reported from the two views | What changes when the arithmetic constraint is removed? |
| E4f | Manually supplied ground-truth boxes; constrained total | Oracle localization ceiling |
| E4g | Fixed heuristic PA boxes; constrained total | Localization floor |

E4a should not be described as a free monolithic-total predictor: the current code parses four
components and recomputes the total. E4e also needs cautious interpretation because it averages
whole-lung totals emitted from two single-lung masked inputs.

The load-bearing contrast is **E4d versus E4c**. It asks whether separate regional image views
provide value beyond supplying the same anatomy as text coordinates. A confidence interval that
crosses zero is a null result and must be reported as such.

### E4f status

E4f is disabled because no newly radiologist-scored box set is available. The implementation is
safe for a partial box CSV if an existing annotation source is later identified: it validates
the coordinate schema and laterality, creates the oracle-view directories, evaluates only the
annotated subset, and forms E4f contrasts only from shared annotated cases. Missing annotations
are not converted into invalid-prediction penalties. In the present project, report the absence
of an oracle ceiling as a limitation and use E4g only as a heuristic localization floor.

### Resume and inference provenance

`anatomy_aware_predictions.jsonl` is now protected by a run fingerprint containing the fold and
view-index hashes, model revision, prompts, decoding and geometry settings, selected arms,
ground-truth-box source, and hashes of every adapter weight file. A legacy or mismatched cache is
rejected rather than silently reused.

Each adapter must contain a non-empty `adapter_model.safetensors` or adapter binary, and its PEFT
configuration must target the expected base model. Paired E4 confidence intervals use a clustered
bootstrap over Stage A `group_id`.

Outputs under `stage_B/nb12_anatomy_aware/` include
`anatomy_aware_predictions.jsonl`, `e4_localization_ablation.csv`, per-fold metrics,
per-arm `cross_fold_aggregate_95ci_<arm>.csv` files, `e4_contrasts.csv`,
`regional_disagreement.csv`, `usability.json`, `run_config.json`, and `gate_nb12.json`.

## Gate and reporting rules

- Treat penalized mRALE MAE as the primary severity endpoint; always report valid coverage next
  to it.
- Do not report the invalid-output penalty as evidence of model ability when an output space is
  structurally incompatible with the task.
- Do not use a COVID probability score for AUROC, AUPRC, DeLong, or calibration when its
  `score_usable` flag is false.
- Report architecture deviations, trainable-parameter counts, model revisions, preprocessing,
  latency, and peak memory.
- Select checkpoints, hyperparameters, and operating thresholds using training/inner-validation
  data only.
- Keep internal evaluation strictly out of fold.
- A null E4c-versus-E4d result changes the novelty claim; it is not repaired by selecting a
  different test subset or reporting pooled means without the paired interval.

## Artifact flow into later stages

```text
Stage A NB 00 -> stage_a_paths.json ---------------------> NB 05-NB 12
Stage A NB 02 -> folds and Harmony JSONL ----------------> NB 05-NB 12
Stage A NB 04 -> V0/V1/V2L/V2R view_index.csv ----------> NB 05, NB 06, NB 08, NB 12

NB 05-NB 12 -> shared-schema prediction JSONL ----------> NB 13 agent registry
NB 08       -> entity_findings.jsonl --------------------> NB 13 and NB 15 reasoner
NB 12       -> E4 contrasts and regional predictions ----> NB 13/NB 14 analyses
NB 13       -> agent registry ---------------------------> NB 14 conventional fusion
NB 14       -> fused evidence ---------------------------> NB 15 reasoner
NB 07/09/10 -> usable COVID scores ----------------------> NB 17 statistics/DeLong
NB 07/09/10 -> usable COVID scores ----------------------> NB 18 ROC/PR/calibration
NB 07/11    -> NV-Reason traces/findings ----------------> NB 20 grounding and interpretability
```

Downstream code must join predictions by stable identifiers and validate fold and ground-truth
fields. It must not rely on row order or assume that all `arm_summary.csv` files have identical
columns.

## Final pre-run checklist

- Stage A gates pass and the strict-JSONL issue has been handled for the selected loader.
- Model revisions are pinned; confirm the recorded BiomedCLIP revision source.
- Every required image and NB 04 view exists.
- NB 09/NB 10 fold adapters contain non-empty weight files and matching configuration metadata.
- Smoke-test output is separated from full-run output.
- Old prediction caches are archived or removed whenever their full input fingerprint cannot be verified.
- Inner-validation COVID score files exist for every NB 09/NB 10 fold before quoting operating-point metrics.
- E5-S remains disabled; enabling it must raise until a real training implementation replaces it.
- E4f remains disabled because no radiologist-scored box set is available; report this limitation.
- NB 13 and later notebooks normalize summary schemas and honor `score_usable`,
  `mrale_usable`, and `report_in_table2` flags.
