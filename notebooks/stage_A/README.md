# Stage A — Foundation notebooks

Run in order on Biowulf. Every notebook ends with a gate assertion; a red gate blocks the
notebooks after it. Do not comment out an assertion — fix the cause, or change the config
deliberately and record it in the protocol-deviation log.

| # | notebook | GPU | approx. runtime | key output |
| --- | --- | --- | --- | --- |
| 00 | `00_environment_and_model_registry.ipynb` | yes (for load checks) | 20–40 min (model downloads) | `stage_a_paths.json`, `model_registry.csv` |
| 01 | `01_data_inventory_qc_and_labels.ipynb` | no | 5–15 min | `midrc_manifest.csv` |
| 02 | `02_folds_and_leakage_audit.ipynb` | no | 2–5 min | `fold_definitions/`, `cohort_composition_table1.csv` |
| 03 | `03_external_cohorts_prep.ipynb` | no | 2–10 min | `external_manifests/` |
| 04 | `04_preprocessing_and_lung_localization_cache.ipynb` | **yes** | 2–4 GPU-hours | `view_index.csv`, `views/` |

## Before you start

Copy the three source files to one of the directories in `DATASET_SEARCH_DIRS` (NB 00,
section 2). The default first choice is:

```
/data/liangz2/openi/midrc/tetci_resubmit/datasets/
    covid_midrc_dataset.csv
    combined_cxr_harmony_train.jsonl
    montgomery_cxr_images.csv
```

NB 00 writes `stage_a_paths.json`; NB 01–04 read it and never hard-code a path. All output
goes under `/data/liangz2/openi/midrc/tetci_resubmit/stage_A/`, so nothing overwrites the
earlier results in `multi_task_CV/`, `medgemma15_4b_multitask_lora_rank32_cv/`, or
`medgemma15_4b_cxr_bbox_lora_rank32/`.

Suggested smoke-test path: run NB 00 with `RUN_MODEL_LOAD_CHECK = False` on a login node to
resolve paths and pin revisions, then re-run it on a GPU node with the load check on. In
NB 04, set `MAX_IMAGES = 16` first, review the QC panel, then set it back to `None`.

## Two findings from the local dry run

**1. The inherited folds leak, and Stage B must be repointed.**

Auditing `multi_task_CV/` against study directories found:

- 135–153 study directories present in **both train and test of the same fold**
- 409 pairwise occurrences of a study directory shared between two test folds
- root cause: the `fold` column in `covid_midrc_dataset.csv` is assigned per image, and 386
  study directories hold 2 images while 55 hold 3 or more

NB 02 publishes this as `fold_leakage_audit.csv` and regenerates folds with
`StratifiedGroupKFold` grouped on study directory and stratified by PCR status × mRALE
severity band. The regenerated folds were dry-run locally and give zero group leakage:

| splitter | images/fold | PCR-negative/fold | prevalence range |
| --- | --- | --- | --- |
| sklearn (default) | 498–529 | 27–39 | 0.9253–0.9458 |
| greedy fallback | 516–517 | 33–34 | 0.9341–0.9362 |

Set `PREFER_SPLITTER = "greedy"` in NB 02 if you want the tighter balance of the scarce
negative class; both are leakage-free and both are deterministic under seed 42.

Consequence: **every previously reported internal number is superseded.** mRALE MAE 3.88 and
COVID balanced accuracy 0.616 came from the leaky folds. Regenerate the M5 baseline arm on
the new folds first and treat that as the reference point, per protocol gate G3.

Stage B config change:

```python
FOLD_DATA_DIR = Path(".../stage_A/nb02_folds/fold_definitions")
DIRECT_TASKS  = {"covid_classification", "mrale_prediction"}
AUXILIARY_TASKS = {"midrc_mrale_truthfulness"}   # confirm against NB 02's printed train_tasks
```

**2. The qualitative mRALE vocabulary differs between the CSV and the Harmony files.**

| numeric | `covid_midrc_dataset.csv` | legacy Harmony JSONL (what the models learned) |
| --- | --- | --- |
| extent 0 | `""` | `"No involvement"` |
| extent 1 | `"<=25%"` | `"<25%"` |
| extent 2 | `"25-50%"` | `"25-50%"` |
| extent 3 | `"51-75%"` | `">50-75%"` |
| extent 4 | `">75%"` | `">75%"` |
| density 0 | `""` | `"No opacity"` |

Building records from the CSV vocabulary would change the supervision target of ~70% of
mRALE records while leaving every numeric field correct — a silent accuracy drop with no
obvious cause. NB 02 therefore harvests the vocabulary and the prompt templates from the
legacy fold files at runtime. Verified locally: the rebuilt `ground_truth` strings match the
legacy strings for all 2,581 mRALE records exactly.

## If NB 01's gate fails on "conflicting PCR labels"

Fixed in the current version. The original code merged cross-study near-duplicate pairs into
shared `group_id`s unconditionally, which manufactured mixed-label groups and then failed its
own gate on them. At study-directory level there are **zero** PCR conflicts, so every conflict
was self-inflicted.

Three changes:

1. **`NEAR_DUPLICATE_MERGE_POLICY`** (NB 01 section 1), default `"conservative"`: a merge is
   declined when it would put a PCR-positive and a PCR-negative image in one group. Such a
   pair is either an annotation error or a hash collision; forcing it into one stratum helps
   neither. Alternatives are `"aggressive"`, `"sha256_only"`, and `"none"`.
2. **Section 5b** measures whether the hash is trustworthy here. Chest radiographs are
   visually homogeneous, so a 64-bit dhash on an 8×8 grid can collide between different
   patients. The cell samples 200k random pairs, reports the null Hamming distribution and the
   expected number of chance pairs, and tells you outright whether to switch to
   `"sha256_only"`. It also characterises the cross-study pairs on independent evidence
   (byte-identical SHA-256, identical dimensions, label and mRALE agreement).
3. **The gate now separates two things it previously conflated.** A conflict inside one study
   directory is a real source-data defect and blocks — one acquisition cannot have two PCR
   results. A conflict created by near-duplicate merging warns instead, because NB 02 resolves
   it by majority vote for *stratification only*; per-image labels are untouched and all
   metrics use them.

Also fixed while in there: the near-duplicate prefilter bucketed on the top 32 bits, which
silently missed pairs whose differences fell in the high half of the hash. It now uses 4 bands
of 16 bits, which by the pigeonhole principle gives **complete** recall at Hamming ≤ 3. Expect
the reported pair count to change.

New outputs: `near_duplicate_merge_log.csv`, `near_duplicate_label_conflicts.csv`,
`group_label_conflicts.csv`, and `hash_diagnostics` inside `inventory_summary.json`.

What to do: re-run NB 01 from section 5. Read the section 5b verdict. If it says the hash is
weakly discriminative, set `NEAR_DUPLICATE_MERGE_POLICY = "sha256_only"` and re-run from
section 6 — that keeps only byte-identical merges, which need no perceptual-hash assumption
at all.

## If NB 03's gate fails on the membership guard

Fixed, and it confirms the NB 01 diagnosis. The guard reported that Montgomery (X1) shared 8
perceptual hashes with MIDRC and blocked. Two reasons that was wrong:

- **Every hit was at Hamming exactly 3** — the threshold boundary. Genuine duplicates cluster
  at 0–1. Across 356,178 cross-cohort comparisons, 8 hits is an empirical collision rate of
  2.2 × 10⁻⁵, roughly **9.5 × 10⁹ times** what a uniform 64-bit hash model predicts. A dhash
  is not uniform over frontal chest radiographs; they all share the same gross anatomy, so the
  hash space is concentrated and unrelated images collide at small distances.
- **Overlap is impossible a priori.** Montgomery is a pre-COVID tuberculosis screening set from
  Montgomery County, Maryland; MIDRC is 2020+ COVID imaging. There is no mechanism by which
  they could share an image.

The guard now **grades** evidence instead of pooling it:

| tier | signal | effect |
| --- | --- | --- |
| HARD | identical filename, file SHA-256, or DICOM study UID | blocks |
| HARD | perceptual match at Hamming 0, or a near match corroborated by identical bytes or identical dimensions | blocks |
| SOFT | uncorroborated perceptual match at Hamming 1–3 | listed for review only |

Two safeguards keep the relaxation from becoming a loophole:

- **Study-UID evidence is pattern-checked** — a parent directory counts as study identity only
  if it is a dotted-numeric DICOM UID and not a generic name like `images/`. Otherwise any
  cohort using a common directory name would appear to collide.
- **A wholesale-overlap floor still blocks** — if more than 25% of a cohort falls within the
  review threshold, that is a re-export, not stray collisions, and the gate fails regardless of
  corroboration.

X4's de-duplication now applies the same graded rule, so its remainder logic no longer discards
clean cases on chance collisions. New output: `membership_guard_detail.csv` with the nearest
internal neighbour, distance, and corroboration flags for every flagged external image, plus
per-cohort nearest-neighbour distance statistics printed inline.

The three X2/X3/X4 warnings in your output are expected — those cohorts aren't downloaded yet.

## Protocol decision needing your sign-off

NB 02 sets `INCLUDE_MONTGOMERY_IN_TRAINING = False` and `INCLUDE_NORMALITY_TASK = False`,
following protocol section 3.3: every Montgomery case is COVID-negative, so training on them
creates a source-label confound. The legacy folds put 110 Montgomery `normality_classification`
records into each training file.

Consequence: the `normality_classification` task disappears from the regenerated folds, and
Stage B trains two direct tasks instead of three. If you would rather keep the normality task,
flip both flags in NB 02 section 1 — but then X1's role as an untouched external
specificity cohort is compromised, and referee 1.5 / 2e are only partly answered.

## Artifact contracts

```
NB 00 → stage_a_paths.json ──────────────→ NB 01, 02, 03, 04 (all paths, model revisions)
NB 01 → midrc_manifest.csv ──────────────→ NB 02 (group_id, labels), NB 04 (image list)
        near_duplicate_pairs.csv ────────→ NB 02 (straddle check)
        image_hashes.csv ────────────────→ NB 03 (membership guard, X4 de-duplication)
NB 02 → fold_definitions/*.jsonl ────────→ Stage B (training), NB 03 (membership guard)
        prompt_templates.json ───────────→ NB 03 (identical prompting)
        cohort_composition_table1.csv ───→ NB 03 (appends external rows) → Table 1
NB 03 → external_manifests/*.csv ────────→ NB 04 (localize external cohorts)
        external_manifests/*.jsonl ──────→ Stage C/D (E9 evaluation)
NB 04 → view_index.csv ──────────────────→ Stage B/C (V0/V1/V2L/V2R paths, E4 arms)
        localization_metrics.csv ────────→ manuscript, reported with every E4 number
```

## Known gaps to plan for

- **E4f (oracle box ceiling) has no data source yet.** It needs ground-truth lung boxes on a
  100–200 image subset. Without it, a negative E4 result cannot be attributed between "the
  anatomy-aware idea is wrong" and "the localizer is not good enough" — which is the
  difference between a reportable negative finding and an uninterpretable one.
- **X2/X3/X4 are not downloaded.** NB 03 runs and skips them with recorded reasons. To add
  one: place images under the declared root, write a `labels.csv`, adjust `COLUMN_MAPS`, and
  re-run. No other cell changes.
- **Grouping is at study level, not patient level.** No patient identifier exists in the
  source CSV. If MIDRC case identifiers can be exported, add a `patient_id` column and NB 01
  will prefer it automatically. State the limitation in the manuscript either way.
- **`google/cxr-foundation` was withdrawn** (protocol D2a) as redundant with the MedGemma-1.5
  vision tower. The registry holds four models, and the study no longer has any
  TensorFlow/Keras dependency. `m42-health/CXformer-base` is now the sole frozen-encoder
  agent and carries RQ3 alone — it is marked `required: False` in NB 00 only because its
  loader is the least standard of the four (may need `trust_remote_code` or `timm`). If it
  fails the load check, resolve it before Stage B rather than proceeding without it.
