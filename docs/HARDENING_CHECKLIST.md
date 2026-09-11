# Hardening checklist for any new notebook

Derived from bugs actually found in NB 05–11 during this project — not a generic style guide.
Apply it to `11_prepare_external_cxr_mimic.ipynb` (which is not in the synced folder, so I could
not review it directly) and to every notebook added from here.

## 1. Resume must actually fire — verify it, don't assume it

The most dangerous bug in this project was a resume that *looked* like it worked.

```python
seen = cm.load_jsonl_by_key(path, ["image_key"])   # keys are TUPLES: ("MIDRC::a.png",)
if str(row["image_key"]) not in seen:              # comparing a STRING -> never matches
```

This prints a reassuring `Resuming with 2,431 cached…` and then rescores everything. Found in
NB 11; worth grepping for anywhere you use `load_jsonl_by_key`.

```python
if (str(row["image_key"]),) not in seen:           # correct
```

**Test it:** run the loop, interrupt it, re-run, and confirm the pending count *drops*. A resume
that has never been exercised is not a resume.

## 2. The cache key must cover everything that changes the result

NB 08's entity cache keyed on entity *names* but not on the positive/negative *phrases* — so
following the gate's own advice to rewrite a prompt silently returned stale scores, and the fix
appeared not to work. NB 07 had the same class of problem with chat-template drift.

Fingerprint every input that affects the output — prompts, thresholds, model revision, view,
preprocessing — store it beside the cache, and invalidate on mismatch:

```python
FINGERPRINT = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
```

**For a MIMIC prep notebook specifically:** the fingerprint should include the report-section
filter, the view-position filter, the tokenizer/embedding model revision, and any de-identification
or text-normalisation rules. Changing how `findings` is extracted must invalidate the corpus.

## 3. Atomic writes

A crash *during* a cache write leaves a truncated file that reads as valid on the next run — a
corruption you discover weeks later as an inexplicable metric. Write to `.tmp`, then `replace()`.
NB 05 and NB 08 were flushing non-atomically until this was fixed.

## 4. Flush periodically, not only at the end

NB 05 and NB 08 wrote their caches with a single call *after* the encode loop. Interrupt at 90%
and everything is lost, while the missing file looks like work never started. Flush every N items
(512 worked well) and also flush before re-raising on an error.

## 5. Multi-level resume for long jobs

NB 09/10 have three levels, and the middle one was missing at first:

| level | mechanism | what it saves |
| --- | --- | --- |
| item | append-per-item JSONL | one image |
| epoch | `resume.pt` / `get_last_checkpoint` | one epoch |
| **stage** | **adapter stamped with its config** | **a completed training phase whose *evaluation* was interrupted** |

That middle case cost ~8 GPU-hours per fold in NB 09: training finished, the summary is written
only after generation, so an interrupt during generation discarded valid training. Use a
**three-state** check (`complete` / `trained` / `none`), not a boolean.

## 6. Skips must verify configuration, not just existence

Every skip compares the saved config against the current one and retrains on mismatch. A resume
that reuses stale work is worse than no resume, because the resulting table looks fine.

## 7. Put failure reasons inside the exception

`assert not failures, f"{len(failures)} issue(s)"` produces a traceback that says nothing. Three
debugging rounds in NB 07 were spent guessing at reasons that were printed above the traceback and
never pasted.

```python
if failures:
    detail = "\n".join(f"  [{i+1}] {m}" for i, m in enumerate(failures))
    raise AssertionError(f"NB xx gate failed with {len(failures)} issue(s):\n{detail}")
```

## 8. Preflight before anything expensive

Assert the pipeline is doing what you think **before** committing GPU-hours:

- NB 07: is the image actually bound? (`<image>` placeholder present, `pixel_values` emitted).
  A 20-hour run completed with `valid_rate = 0.000` because it wasn't.
- NB 08: do different images produce different embeddings?
- NB 09/10: does the collated batch carry image tensors, and did the response-only mask leave any
  supervised tokens?

**For MIMIC prep:** before ingesting ~370k studies, verify on ~20 that the report parser finds
`FINDINGS`/`IMPRESSION`, that the view filter keeps PA/AP and drops laterals, and that the
subject/study IDs join correctly to the image paths. Failures here are silent and produce a corpus
that looks populated.

## 9. Environment shims

- `torch_dtype` → `dtype`: inspect the signature and pass whichever is accepted.
- `eval_strategy` vs `evaluation_strategy`: renamed in transformers 4.46; the wrong one raises
  `TypeError` at construction and kills a job seconds after the queue allocated the node.
- **Biowulf's kernel is 4.18, below accelerate's 5.5 minimum.** The documented failure is
  DataLoader *worker* deadlock — the classic silent overnight death. Force
  `dataloader_num_workers=0` and `pin_memory=False` on old kernels.

## 10. Cross-notebook reads

NB 10 was built by transforming NB 09's cells, and a global `NB09_DIR → NB10_DIR` rename rewrote
the line that was supposed to **read** NB 09's artifact. Audit every cross-notebook path after any
bulk edit, and make the error distinguish "not produced yet" from "wrong path" — the remedies are
hours apart.

## 11. Stamp usability onto the outputs

An arm can be valid for one endpoint and invalid for another (NV-Reason: usable findings, no
mRALE). Write `usability.json` **and** stamp `report_in_table2` onto `arm_summary.csv`, so a row
cannot be lifted into a table without its health flags.

## 12. A relation is not a property — the Stage C lesson

NB 13's first draft carried a single `split` column answering "is this image inner validation?".
There is no fold-free answer: fold *k*'s inner-validation set is drawn from folds ≠ *k*, so the
same images are test data for whichever fold held them out. The column, defined as *is this
image in its own fold's inner set*, was **identically false**.

Nothing crashed. NB 14 ran, printed a table, and skipped every fold because each fitting set
was empty. NB 15 would have inherited it.

```python
registry["split"] = ...          # wrong: collapses a relation into a property
registry["inner_fold_2"] = ...   # right: one indicator per fold
```

The same shape of error produced the per-fold journal bug in NB 15 (see item 5): a cache key
that omits a dimension the value depends on. Ask of every derived column and every cache key:
**what does this actually depend on?** If the answer includes something not in the key, it is a
bug that will not announce itself.

## 13. Dry-run the logic before it reaches the queue

Both Stage C bugs above were found by executing the notebooks against a synthetic cohort with
stubbed `torch`/`transformers` and a fake generator, in seconds, on a laptop-class machine. Both
would have survived code review — they produce clean output and plausible numbers.

Anything with fold logic, cache keys, or cross-notebook reads is worth 20 minutes of synthetic
data before it is worth a GPU queue slot.

## 14. Duplicated logic needs an equality assertion, not a comment

NB 16 rebuilds NB 15's prompt. Rather than asserting the equivalence in a comment, it hashes the
reference prompt and compares against the hashes NB 15 recorded in its own journal for the same
images, and refuses to run on a mismatch. Its gate additionally requires that the greedy
reference arm reproduce ≥ 95% of NB 15's answers — both runs are greedy on an identical prompt,
so anything less means one of the notebooks is not running the system it claims to.

Where duplication is genuinely unavoidable, make the copy testable. Where it is avoidable, put
the code in a module beside the notebooks (`cxr_metrics.py`, `stage_c_reasoner.py`) with
self-tests that run in one command.

## 15. Select an arm relative to the fold where it will be applied

Pooled inner-validation rows contain every outer fold's test cases. Choosing one global agent
arm from that pool therefore lets fold *k* influence the arm later evaluated on fold *k*. NB 13
now writes `selected_for_fold_0` … `selected_for_fold_4`; NB 14–16 consume the appropriate
selector for each fit/application context. `selected_for_outer_fold` is only a convenience for a
row evaluated on its own held-out fold. Never use the pooled reliability table for model choice.

## 16. Evidence-only agents must remain agents

A4 and A6 can contribute findings without a numeric mRALE prediction. Requiring
`dropna(subset=AGENTS)` silently removes every image once such an agent joins the roster. Define
eligibility by the presence of one selected row per protocol agent, then build endpoint-specific
numeric intersections for conventional fusion. The reasoner prompt should explicitly show an
abstention or findings-only report instead of dropping that agent.

## 17. Parse strictly; do not repair a model into validity

Out-of-range, fractional, incomplete, or formula-inconsistent JSON is an invalid output. Do not
clip `40` to `24`, derive a missing total, or round a fractional category and then count the row
as valid. `stage_c_reasoner.parse_reasoner_output` requires the full component/regional/total and
COVID schema, checks `extent × density = regional` and `right + left = total`, and leaves the
affected endpoint missing so the registered invalid-output penalty applies.

Self-consistency needs the same care: independent medians of extent, density, and score can make
an impossible aggregate even when every sample is valid. Choose the valid sample nearest the
median total (a coherent medoid) and retain all regional fields from that sample.

## 18. A self-reported confidence is not a continuous model score

For COVID AUROC/AUPRC/calibration, derive `P(positive)` from normalized `Yes`/`No` continuation
likelihoods. Store generated confidence or self-consistency vote share as provenance only, and
derive the reported decision from the logits score at 0.5. Gate on score coverage and
score/decision agreement. Otherwise a model can emit confident prose with no calibrated score,
and the metric will look more rigorous than its input.

## 19. NaN is truthy, so `primary or fallback` is unsafe for table paths

After reading a CSV, a missing path is commonly `numpy.nan`; `nan or fallback_path` returns the
NaN rather than the fallback. Validate that a candidate is a non-empty string and exists, trying
V0 before V1. Missing Stage C image paths are blocking because silently dropping them changes the
registered denominator.

## 20. Exploratory preconditions are still blocking preconditions

NB 16 is exploratory, but E6-Q output hygiene and reference reproduction determine whether its
latency, token, and sensitivity numbers mean what they claim. Missing NB 15 prompt hashes, zero
JSON objects, repeated objects, stray control tokens, strict-schema validity below 0.99, or fewer
than 20 shared reference cases are failures. Calling these warnings permits a full expensive grid
to run on an unidentified system.

## 21. A shared random cache has one writer and read-only consumers

NB 17 owns `bootstrap_indices.npz`. NB 18 and NB 19 open it with `read_only=True` and NB 17's
expected fingerprint. A consumer that redraws or replaces a mismatch silently changes the
uncertainty analysis after the headline table was locked. Re-run the owner; never let a consumer
repair it.

## 22. Threshold transfer requires scored inner-validation rows

Fold membership is not a model score. For outer fold *k*, select the threshold only from that
fold's locked, scored inner-validation rows for the same arm, then apply it to outer-fold test
rows. Pooled OOF scores contain outer test data and are not a substitute. Record the source per
arm/fold and fail when it is missing.

## 23. Finalize multiplicity only after every notebook contributes

NB 17 cannot finalize F1–F5 because NB 18 adds fixed-operating-point McNemar tests and NB 19 adds
external F5 tests. NB 19 reconstructs every `PValue`, applies Holm within complete families, and
writes `paired_comparisons_final.csv`. Publication tables read that final artifact only.

## 24. Trace table cells by identity, not numeric coincidence

A global lookup for a displayed value is unsafe: `0.500`, `1`, and common counts occur in many
arms and metrics. Resolve the table row by stable identifiers (arm, endpoint, family, cohort,
subgroup), then accept only values from that source row. An equal rounded value elsewhere is not
provenance.

## 25. Preserve human work in regenerated artifacts

NB 20 preserves human coding columns by `image_key` only when the full case-table fingerprint also
matches, and refuses to overwrite them if the sampled cases or the predictions/traces behind their
panels changed. NB 22 regenerates only the automatic part of `protocol_deviations.md` and keeps
everything below `## Added by hand`. Every manually completed artifact needs an explicit merge
contract before it is safe to re-run.

## 26. Write a release manifest last and exclude only itself

An early manifest omits outputs created later in the same notebook. Use an in-memory snapshot for
safety checks, write the checklist, audits, run configuration, and gate, then rebuild and write
the authoritative manifest. Exclude the manifest itself to avoid self-hash recursion.

## 27. Existence is not release readiness

A CSV can exist with missing intervals, provisional p-values, or no rows; a gate JSON can exist
with `passed: false`. Validate content: final adjusted p-value metadata, confidence intervals,
immutable revisions, explicit seeds, passing Stage D gates, completed dual coding, and a non-empty
bibliography audit. NB 22 treats these as blocking.

## 28. Separate controlled working storage from the publication allowlist

Do not call an entire experiment tree a release bundle. Checkpoint and cache names may correctly
mention restricted datasets while remaining inside controlled storage. Inventory those paths for
governance, build an explicit allowlist of files intended for publication, and make the blocking
restricted-data check operate on that allowlist. This avoids both false release failures and the
more serious error of publishing a DUA-controlled derivative.

## 29. Bind the final gate to upstream identities, not gate filenames

A `passed: true` file can survive beside artifacts from a later rerun. The release gate must
require exactly one gate for each upstream notebook and compare the values that identify the run:
locked reference arm, shared bootstrap fingerprint, final multiplicity-family counts, and the
qualitative case-table and selection fingerprints. NB 22 verifies this NB 17–21 chain before it
certifies the package.

## 30. Do not conflate analysis integrity with camera-ready completion

A missing bibliography, dual-human coding sheet, or final language sign-off means the manuscript
is not ready to submit; it does not mean the numerical analysis diverged. Emit two explicit
states: a hard analysis gate for corrupted or stale computational artifacts, and a camera-ready
state for the full release checklist. During analysis, preserve release blockers in machine-
readable output without aborting. For the final submission run, enable strict camera-ready mode
so those same blockers become failures. Never silently downgrade them to warnings.

## MIMIC-specific items not covered above

1. **DUA compliance is a gate, not a comment.** MIMIC images, reports, identifiers, and FAISS
   indexes must never leave access-controlled storage. Assert the output directory is on the
   permitted filesystem and refuse to write otherwise.
2. **De-duplication against the internal cohort** — same three keys as NB 03 (filename, study UID,
   perceptual hash), with the same graded-evidence policy. Note the lesson from NB 03: a bare
   perceptual-hash match at Hamming ≤ 3 is **not** evidence on chest radiographs; require
   corroboration.
3. **MedGemma was pretrained partly on MIMIC-CXR.** Any retrieval result using MedGemma as either
   the embedding model or the reasoner is confounded, and the protocol already requires disclosing
   this. Record which model produced the embeddings in the corpus manifest so the confound is
   visible at analysis time rather than argued about later.
4. **MIMIC has no mRALE labels.** It is a retrieval/evidence corpus, not an external severity
   validation set. Make that structural: do not emit an `mrale_total` ground-truth field at all,
   so no downstream notebook can accidentally score MAE against it.
