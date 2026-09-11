# Stage C — fusion and reasoning (E1, E2, E3, E6, E7)

Stage B produced agents. Stage C decides whether combining them is worth doing, and by what
method. Four notebooks, run in order, plus one shared module.

| # | notebook | protocol | cost | reads | writes |
| --- | --- | --- | --- | --- | --- |
| 13 | `13_agent_output_registry.ipynb` | registry + reliability | ~2 min | Stage B predictions | `agent_registry.parquet`, `agent_reliability_inner.csv` |
| 14 | `14_conventional_fusion_baselines.ipynb` | E7a–E7e, mRALE + COVID | ~5 min | NB 13 | `fusion_predictions.jsonl`, `e7_fusion_metrics.csv`, `e7f_target.json` |
| 15 | `15_llm_reasoner_aggregation.ipynb` | E1, E2, E3, E7f | **~60–120 GPU-h** | NB 13, NB 14 | `e1_roster_metrics.csv`, `e2_reasoner_metrics.csv`, `e3_metrics_aware.csv`, `reasoning_traces.jsonl`, `e7f_vs_e7d.json` |
| 16 | `16_decoding_prompt_and_depth_sensitivity.ipynb` | E6 | ~10–20 GPU-h | NB 15 | `e6_sensitivity_grid.csv`, `prompt_brittleness.csv`, `latency_vs_accuracy.csv` |

`stage_c_reasoner.py` sits beside them and owns prompt rendering, constrained generation, JSON
parsing, self-consistency aggregation, and journal handling. Run `python stage_c_reasoner.py`
to execute its 65 self-tests — do that after any edit, before submitting a job.

---

## The one comparison that decides the paper

NB 14 fits learned stacking for both mRALE and COVID on the same endpoint-specific agent outputs
the reasoner sees, and writes both targets to `e7f_target.json`. NB 15 compares against those
targets on the **same images**; NB 17 is the single source of truth for paired confidence
intervals, DeLong tests, multiplicity correction, and any superiority/equivalence claim.

If it does, the paper claims that reasoning improves accuracy. If it does not, protocol risk K1
applies and the claim becomes "equivalent accuracy with auditable, clinician-readable
justification" — still publishable, but it has to be said plainly. `e7f_vs_e7d.json` carries the
verdict in words as well as numbers so it cannot be quietly dropped from the write-up.

NB 14 prints `TARGET FOR NB 15` when it finishes. Do not weaken that baseline to create a gap.

---

## Leakage: three boundaries, all enforced in code

**1. Reliability is fold-relative.** "Inner validation" is not a property of an image; it is a
relation between an image and a fold. Fold *k*'s inner-validation set is drawn from folds ≠ *k*,
so those same images are test data for whichever fold actually held them out. NB 13 therefore
emits one indicator column per fold (`inner_fold_0` … `inner_fold_4`) rather than a single
`split` column, and its gate asserts that no row flagged `inner_fold_k` belongs to fold *k*.

> A single `split` column defined as "is this image in its own fold's inner set" is
> *identically false*, because `inner_by_fold[k]` never contains fold-*k* images. A notebook
> built on one runs to completion with every fitting set empty and every fold silently skipped.
> That bug was in the first draft of NB 13 and NB 14 and was caught by a dry run, not by review.

**2. Arm selection is fold-specific.** NB 13 emits `selected_for_fold_0` …
`selected_for_fold_4`. The arm applied to outer fold *k* is selected only from fold *k*'s inner
validation rows. Pooled reliability is descriptive only; it is never used to choose an arm for
an outer fold.

**3. Reasoner selection never sees its own evaluation rows.** NB 15 picks the reasoner (E2) and the
metrics-awareness setting (E3) on fold 0's inner-validation pool. That pool is split **in half by
patient group**: one half computes the reliability numbers shown in the prompt, the other is
scored. Otherwise a metrics-aware setting would be reading statistics derived from its own
evaluation set.

**4. E6 never touches a test fold.** Sensitivity analysis is exploration. NB 16 runs on the same
evaluation half NB 15 selected on; running 25 configurations against held-out folds and then
reporting the headline arm would turn the test set into a development set.

---

## Running order and what each gate blocks on

```
NB 13  →  NB 14  →  NB 15  →  NB 16
```

| notebook | blocking conditions |
| --- | --- |
| 13 | no row flagged `inner_fold_k` belongs to fold *k*; no fold's fitting set is empty; registry non-empty |
| 14 | fitting and application sets disjoint in every fold |
| 15 | every final E1 roster configuration has strict complete-schema rate ≥ 0.99 and logits-based COVID score coverage ≥ 0.99 with score/decision agreement ≥ 0.995; low-quality E2/E3 screening candidates are quarantined and recorded as negative output-integrity results; agent-output provenance recomputes correctly for **every** final answer; an image placeholder was rendered for every model |
| 16 | E6-Q resolved (exactly one JSON object, no stray control tokens, strict-schema rate ≥ 0.99); reference prompt verified byte-for-byte; the greedy reference arm reproduces ≥ 95% of NB 15's answers on at least 20 shared images |

Warnings are not failures, and several of them are meant to be acted on in the manuscript rather
than in the code — the E7f verdict, a flat E3 ladder, inconclusive leave-one-agent-out effects, prompt
brittleness. A negative result that is reported is a result; one that is suppressed is a
problem.

---

## Interruption safety

Every generating configuration writes **one JSON line per image** to its own journal, so the
most an interruption costs is one image. Re-run the notebook from the top: completed
configurations are skipped, a partial one resumes at the next unscored image.

On Biowulf, NB 15 and NB 16 use a **35-hour soft stop** for the 36-hour allocation limit. The
deadline is checked only between images: the current image is journaled and synced first, then
the notebook raises an expected `WalltimeCheckpoint`. Resubmit and run from the top. A hard kill
before the soft stop can still cost the current image only; a truncated final JSONL record is
repaired automatically, while corruption in the middle of a journal remains a blocking error.

| notebook | journal layout | forcing recomputation |
| --- | --- | --- |
| 15 | `journals/<config>__fold<k>.jsonl` | `FORCE_RECOMPUTE_CONFIGS = ["E2_R_qwen_base"]`, or `FORCE_RECOMPUTE_ALL` |
| 16 | `journals/<arm>.jsonl` | same two flags |

**NB 15 journals are keyed per configuration and fold.** The fingerprint includes the reliability
numbers, and those are fold-specific by design. A single journal per configuration therefore
looks "changed" at every fold boundary and retires the previous fold's work: five folds run, one
fold survives, and the arm reports a fifth of its data with a plausible-looking MAE. That bug
was in the first draft of NB 15 and was caught by a dry run. NB 16 runs only on the fold-0
selection pool, so its journals are keyed per sensitivity arm.

**The fingerprint covers everything that changes the answer** — the complete loader spec and
resolved adapter, model revision, evaluated row set, image paths/view rule, rendered prompt
identity, roster, E3 setting, decoding parameters, sample count, reliability figures, conflict
threshold, and COVID token-scoring probe. Change a prompt phrase and only the affected journals
are invalidated. Retired
journals are renamed `<name>.stale-<fingerprint>.jsonl` rather than deleted, so a previous run
stays inspectable.

**Verify resume the first time you use it.** Interrupt a run, restart it, and confirm the pending
count *drops*. A resume that has never been exercised is not a resume — NB 11 shipped one that
printed "resuming with 2,431 cached" and then rescored everything, because it compared a string
against tuple keys. `stage_c_reasoner.journal_keys()` returns plain strings specifically to
remove that trap.

---

## Compute notes for Biowulf

- **NB 15 is the expensive one.** 22 selection configurations plus (rosters × folds) evaluation
  runs. Start with `SMOKE_TEST = True` (16 images per configuration) to verify wiring, then set
  `E1_MAX_IMAGES_PER_FOLD` to a few hundred for a first pass, then `None` for the full run.
- **Duplicate rosters are removed automatically.** With A1 absent, `E1-4_add_A6` is the same
  roster as `E1-full`, and `E1-3_add_A5` the same as `E1-L_minus_A6`. NB 15 runs each once and
  reports the alias, rather than generating two identical columns and inviting someone to read
  the sampling noise between them as an effect.
- **Models load once per (reasoner, fold)**, not once per configuration. With twelve E1 rosters
  that is the difference between 5 and 60 model loads.
- **Old kernel.** Biowulf's 4.18 kernel is below accelerate's minimum and the documented failure
  is DataLoader *worker* deadlock. Stage C does no batched data loading, so it is not exposed —
  but if you add any, force `num_workers = 0`.
- **A missing LoRA adapter raises.** It does not fall back to the base model. A silent fallback
  would put a differently-trained model into the E2 table under the LoRA label, and nothing
  about the output would look wrong.
- **COVID scoring adds two forward passes per image.** The probability is the normalized
  likelihood of the `Yes` and `No` continuations, not a self-reported confidence. The generated
  confidence/vote is retained as provenance only, and the reported decision is thresholded from
  the logits-based score so the two cannot contradict each other.
- **Image view is V0-first.** The whole radiograph is used when readable; V1 thorax is only a
  fallback. A missing path is blocking because silently dropping cases changes the denominator.

---

## Reading the outputs

`nb13_registry/`
- `agent_registry.csv|parquet` — one row per (image, agent, arm) with per-fold membership flags
- `agent_reliability_inner.csv` — `scope = inner_validation_for_fold` rows are what E3 shows the
  reasoner and what selects that fold's arm; pooled rows are for descriptive display only
- `agent_availability.csv` — which Stage B notebooks have not run yet, recorded rather than
  omitted, because an absent agent cannot otherwise be told apart from one that contributed
  nothing

`nb14_fusion/`
- `e7_fusion_metrics.csv` — every mRALE and COVID fusion arm, identified by `endpoint`; mRALE rows
  include `n_clipped_to_scale` (ridge and
  GBT are unbounded and routinely emit values just below zero; an mRALE of −0.3 is an artefact,
  and the count makes the clipping visible)
- `e7f_target.json` — what NB 15 must beat

`nb15_reasoner/`
- `e1_roster_metrics.csv` — the roster ablation, with `also_known_as` for deduplicated arms and
  fold-level mean ± 95% CI
- `e1_paired_contrasts.csv` — endpoint-specific leave-one-out deltas with cluster-bootstrap
  intervals over patient groups. An interval straddling zero is **inconclusive**, not proof of no
  contribution; removal requires the predeclared equivalence-margin analysis in NB 17
- `reasoning_traces.jsonl` — per case: what was shown, what was answered, what was cited. The
  provenance audit recomputes the shown values against the registry rather than trusting the
  field, and counts fabricated citations (an agent named that was never supplied) and the
  override rate (answers outside the range the agents spanned). A reasoner that never leaves that
  range is doing arithmetic, and the paper should not describe it as reading the image

`nb16_sensitivity/`
- `e6q_output_hygiene.csv` — the precondition: one JSON object per generation, no stray control
  tokens
- `prompt_brittleness.csv` — MAE standard deviation across paraphrases within each variant. If
  that is comparable to the difference *between* variants, the variants cannot be ranked
- `latency_vs_accuracy.csv` — with a `pareto_frontier` flag. Arms off the frontier are worse on
  both axes and should not be recommended whatever their headline MAE

---

## If something goes wrong

| symptom | first thing to check |
| --- | --- |
| every fold skipped, fitting sets empty | re-run NB 13; the registry predates the `inner_fold_*` columns |
| `valid_rate` near 0.000 | image binding. `run_config.json → image_template_variant`; a value starting `NO_PLACEHOLDER` means the model never saw the radiograph |
| "resuming with N cached" then it rescores everything | the fingerprint changed. Compare `journals/<name>.fingerprint.json` against the retired `.stale-*` file |
| NB 16 refuses to start on a prompt mismatch | `evidence_block` / `build_prompt` has drifted from NB 15. Reconcile them; do not relax the check |
| NB 16's reference arm disagrees with NB 15 | adapter path, message schema, or dtype. Both runs are greedy on an identical prompt and should agree almost exactly |
| a metric looks impossible | check `usability.json` — an arm can be valid for one endpoint and not another |
| last gate lists low-schema E2/E3 candidates | expected screening quarantine, not a failed final run; inspect `raw_output`, retain the validity rate, and do not repair prose or `<think>/<answer>` text into strict JSON |
