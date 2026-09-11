# Interruption safety — what survives a crash, and what to re-run

Every notebook that costs more than a few minutes can now be interrupted and resumed. Re-run the
notebook from the top: completed work is detected and skipped, and nothing is recomputed.

| # | notebook | cost | max work lost on interruption |
| --- | --- | --- | --- |
| 00 | environment & model registry | 20–40 min | nothing (Hugging Face cache) |
| 01 | data inventory & QC | 5–15 min | one image (per-image JSONL) |
| 02 | folds & leakage audit | 2–5 min | whole run — but it is minutes |
| 03 | external cohorts | 2–10 min | whole run — but it is minutes |
| 04 | lung-localization cache | 2–4 GPU-h | **one image** (per-image JSONL) |
| 05 | frozen encoder agents | ~1 GPU-h | **512 images** (atomic cache flush) |
| 06 | conventional classifiers | ~10 GPU-h | **one epoch**; completed folds skipped |
| 07 | zero-shot VLM agents | ~20 GPU-h | **one (arm, image, task)**; stage markers |
| 08 | BiomedCLIP entity agent | ~0.5 GPU-h | **512 images** (atomic cache flush) |
| 09 | MedGemma multitask LoRA | ~65 GPU-h | **one sweep config** or **one trainer checkpoint**; completed folds skipped |
| 10 | Qwen3.5 multitask LoRA | ~55 GPU-h | same as NB 09; inherits its selected configuration |
| 11 | NV-Reason agent | ~8 GPU-h | **one image** (per-image JSONL) |
| 12 | anatomy-aware mRALE | ~30 GPU-h | **one (arm, fold)**; completed arms skipped |
| 13 | agent output registry | ~2 min | whole run — but it is minutes |
| 14 | conventional fusion | ~5 min | whole run — but it is minutes |
| 15 | LLM reasoner aggregation | ~60–120 GPU-h | **one image**, per (configuration, fold) journal |
| 16 | decoding/prompt sensitivity | ~10–20 GPU-h | **one image**, per-arm journal |
| 17 | statistics & paired comparisons | ~10–30 min | whole run — but the bootstrap index cache survives |
| 18 | calibration, ROC, thresholds | ~5–15 min | whole run — minutes |
| 19 | external validation | ~5 min | whole run — minutes |
| 20 | interpretability & failure cases | ~5–20 min | whole run — panels re-render |
| 21 | figures, tables, exports | ~2 min | whole run — minutes |
| 22 | reproducibility & reference audit | ~2–10 min | whole run — hashing dominates |

## The three mechanisms

**Per-item journals.** NB 01, 04, and 07 append one JSON line per unit of work and reload it on
start. NB 07 keys on `(arm, image_key, task)`, so it resumes mid-arm.

**Per-fold / per-epoch checkpoints.** NB 06 writes `resume.pt` after every epoch (model,
optimizer, scheduler, best-so-far state, history) and skips folds whose `fold_summary.json`
already exists. NB 09 uses HF `get_last_checkpoint` to resume a partially trained fold and skips
completed folds. NB 09's Phase-A sweep appends each config's result to `sweep_results.jsonl`
before the next config starts.

**Stage markers and journals with input fingerprints.** NB 07 records a fingerprint of each section's inputs
in `stage_status.json`; a section whose inputs are unchanged is skipped and its artifacts
reloaded. NB 08's embedding cache carries a fingerprint of the entity phrases, prompt templates,
view, and model revision. NB 15/16 fingerprint the complete model/adapter loader spec, evaluated
row set, readable image paths/view rule, rendered prompt identity, reliability context, decoding
settings, and COVID logits-scoring probe. Retired `.stale-*` journals are never read as active
fold results.

**Thirty-six-hour allocations.** NB 15 and NB 16 stop softly after 35 hours, between images,
leaving approximately one hour before Biowulf's hard limit. The expected
`WalltimeCheckpoint` means the journals are safe: resubmit and re-run from the top. If the node
is killed before that boundary, startup removes only a malformed final JSONL fragment and
regenerates that one image; it never silently skips an invalid record in the middle of a journal.

## Two properties that matter more than the resume itself

**Config-change invalidation.** A skip only fires when the saved work was produced under an
**identical configuration**. Change a learning rate, a prompt phrase, or an augmentation setting
and the affected work is recomputed rather than silently reused. This is the half of resume that
is easy to get wrong: a resume that reuses stale work is worse than no resume, because the
resulting table looks fine.

**Atomic writes.** Caches are written to `.tmp` and renamed. A crash *during* a cache write would
otherwise leave a truncated file that reads as valid on the next run — a failure mode that is very
hard to notice later.

## Forcing recomputation

| notebook | flag |
| --- | --- |
| 01 | delete `image_hashes.jsonl` |
| 04 | `OVERWRITE_EXISTING_VIEWS = True`, or delete `anatomy_localizations.jsonl` |
| 05 / 08 | delete the `.npz` cache (or change any scored setting — the fingerprint handles it) |
| 06 | `FORCE_RETRAIN = True`, or `RESUME_MID_FOLD = False` for fold-level only |
| 07 | `FORCE_REGENERATE`, `FORCE_RESCORE`, `FORCE_RECOMPUTE_METRICS`, `RESET_ARMS=[...]` |
| 09 | delete a fold directory, or `FORCE_FINAL_CONFIG` to change the configuration |
| 15 / 16 | `FORCE_RECOMPUTE_CONFIGS = [...]` for named configurations, `FORCE_RECOMPUTE_ALL` for everything |
| 17 | delete `nb17_statistics/bootstrap_indices.npz` to redraw the replicates |

## Practical notes for Biowulf

- **Re-run from the top.** Skips are cheap and the resume logic lives in the cells themselves,
  not in kernel state. Do not try to resume by running a subset of cells — NB 07's gate is
  self-healing precisely because that turned out to be the natural thing to do and it broke.
- **NB 09 is the one to watch.** Budget for at least one interruption across ~65 GPU-hours. Phase
  A (sweeps, ~25 GPU-h) and Phase B (five folds, ~40 GPU-h) resume independently.
- **Disk.** NB 09 keeps two full 4B checkpoints per fold while training and deletes nothing
  automatically; sweeps no longer save checkpoints at all. NB 06 keeps one `resume.pt` per
  in-progress fold and removes it when the fold completes.
- **NB 15 journals are keyed per (configuration, fold).** The fingerprint includes the
  reliability numbers, which are fold-specific by design; one journal per configuration would
  look "changed" at every fold boundary and retire the previous fold's work. Five folds would
  run and one would survive, with a plausible-looking MAE over a fifth of the data. NB 16 uses
  per-arm journals because all of E6 is confined to one fold-0 inner-validation evaluation pool.
- **NB 16 will not resume against an unverified reference.** It first requires prompt hashes from
  NB 15, exact loader-spec agreement, and at least 20 shared reference-arm cases. Missing evidence
  is a failure, not a warning.
- **Check what will be skipped before submitting.** Each notebook prints its resume state near the
  top (`Resuming with N …`, `Previously completed stages: …`, `already complete under this
  config`). If it says it will redo something you expected to be done, find out why before
  spending the queue slot.

## Stage D has no resume machinery, on purpose

Nothing in NB 17–22 costs more than about half an hour, so per-item journals would be
complexity without benefit. Re-run from the top.

All six notebooks nevertheless carry a 35-hour soft deadline for Biowulf's 36-hour maximum,
checked between code sections. Reaching it raises `WalltimeCheckpoint` before another section
starts. This is a safety alarm rather than a normal expected event: the correct recovery is to
resubmit and rerun the affected short notebook from the top. Bootstrap and JSON control files
use durable temporary-file replacement, so a hard interruption cannot leave a partial file that
looks complete.

The single piece of persisted state is `nb17_statistics/bootstrap_indices.npz`: 2000
patient-level replicates drawn once and reused by every arm, every paired difference and every
band in Figures 3 and 4. NB 17 may create or replace it when the locked patient list changes.
NB 18 and NB 19 open it read-only and require NB 17's fingerprint; a mismatch is a blocking
error with the remedy to re-run NB 17, never a downstream redraw.

NB 22 is also deliberately non-resumable, but it preserves the text below `## Added by hand` in
`protocol_deviations.md`. Re-running the release audit refreshes the automatic section without
erasing human decisions, and writes `artifact_manifest.csv` last so the manifest includes NB 22's
own gate, checklist, environment, and audit outputs.
