# Stage D — analysis and delivery (statistics, calibration, external, interpretability, exports)

Stage C produced arms. Stage D decides what may be said about them, and produces the artifacts
the manuscript is assembled from. Six notebooks, run in order, plus one shared module.

| # | notebook | protocol | cost | reads | writes |
| --- | --- | --- | --- | --- | --- |
| 17 | `17_statistics_and_paired_comparisons.ipynb` | §8 in full | ~10–30 min | Stage B + NB 14–16 gated predictions | `all_metrics_with_ci.csv`, `paired_comparisons.csv`, `endpoint_usability.csv`, `multiplicity_families.json` |
| 18 | `18_calibration_roc_and_thresholds.ipynb` | R1.3, E8d | ~5–15 min | NB 09–10, 14–15, 17 | `roc_pr_curves.csv`, `calibration.csv`, `operating_points.csv`, `operating_point_comparisons.csv`, Fig. 3–4 |
| 19 | `19_external_validation_and_domain_shift.ipynb` | E9 (F5) | ~5 min | NB 03, 17, 18 | `external_metrics.csv`, `subgroup_metrics.csv`, `f5_external_comparisons.csv`, `paired_comparisons_final.csv`, `multiplicity_families_final.json` |
| 20 | `20_interpretability_and_failure_cases.ipynb` | E10 | ~5–20 min | NB 04, 13, 15, 17 | `case_panels/*.png`, `failure_taxonomy.csv`, `grounding_metrics.csv` |
| 21 | `21_figures_tables_and_exports.ipynb` | §11, R1.8, R1.9 | ~2 min | passing gates from NB 17–20 plus NB 00, 02–03, 08–16 | Tables 1–11/S1–S3, E9c Table 8b, fingerprinted Figure 6 manifest, vector plots, `supplementary_results.xlsx` |
| 22 | `22_reproducibility_and_reference_audit.ipynb` | R1.6, R1.10, R2.c | ~2–10 min | passing NB 17–21 outputs, model registry, manuscript/bibliography | fingerprint-bound `release_checklist.csv/.md`, last-built `artifact_manifest.csv`, explicit `release_allowlist.csv`, restricted-path inventories, `model_references.csv`, `reference_audit.csv`, deviation log, environment, run configuration, gate |

`stage_d_stats.py` sits beside them and owns the patient-level bootstrap, DeLong, McNemar,
Wilcoxon, Holm–Bonferroni, TOST equivalence, the p-value metadata contract and the single
number-formatting function. Run `python stage_d_stats.py` for its 84 self-tests — do that after
any edit, before submitting a job.

---

## The five commitments Stage D exists to enforce

**1. Bootstrap indices are drawn once** (§8.3). One patient-level index matrix — 2000
replicates, seed 42 — cached to `nb17_statistics/bootstrap_indices.npz` and handed to every arm
and every comparison. If each arm resampled independently, a paired difference would carry the
sum of two independent sampling errors: intervals that *look* conservative but are not, because
the extra width hides real differences. NB 18 and NB 19 reload the same cache **read-only** and
require NB 17's fingerprint; a downstream notebook cannot overwrite it.

**2. The unit of analysis is the patient** (§8.1). Several radiographs per patient are not
independent observations; resampling images would narrow every interval in the direction that
manufactures significance. `PatientBootstrap` resamples patients and expands to their images by
weight.

**3. A p-value cannot exist without its metadata** (§8.8). `sd.PValue` refuses to construct
without a test name, the paired unit, its family, the family size and its adjustment status.
NB 17 produces provisional F1–F4 records, NB 18 adds fixed-operating-point F4 comparisons, and
NB 19 reconstructs the records and performs the final F1–F5 Holm pass consumed by NB 21.

**4. Upstream gates and endpoint-specific usability are binding.** NB 17 refuses to ingest
NB 14–16 unless their gates passed. An arm quarantined for mRALE or COVID remains excluded from
that endpoint's inferential comparisons even if its other endpoint is usable.

**5. Denominators are locked before comparison.** NB 14's separate mRALE and COVID rows are
merged by `(arm, image_key)` without losing endpoint identity. Duplicate endpoint rows, stale
cohort keys, declared-count mismatches, and missing rows in a paired comparison block the gate
instead of silently shrinking the denominator.

---

## Multiplicity families (§8.6)

| family | comparison set | adjustment |
| --- | --- | --- |
| F1 | E0 baseline suite vs the full framework | Holm |
| F2 | E1 leave-one-agent-out | Holm |
| F3 | E4 localization | Holm |
| F4 | E7 fusion, including the decisive E7f vs E7d | Holm |
| F5 | E9 external | Holm |
| EXPLORATORY | E5/E6 sensitivity | **none, by declaration** |

The exploratory set is deliberately excluded. Adjusting it would let a sensitivity result borrow
confirmatory standing, which is the opposite of what declaring it exploratory means —
`sd.PValue` raises if anyone tries.

Family size is set from actual membership, not from a number typed in advance, so a family that
ran short is adjusted for what it contains.

---

## Two distinctions the code will not let you blur

**Absence of evidence is not equivalence.** A leave-one-agent-out interval that straddles zero
is inconclusive. Deleting the agent is a claim of *equivalence*, so NB 17 runs a two-one-sided
test against a margin declared before any result is seen (0.5 mRALE points, 0.02 AUROC) and
returns one of three verdicts — `different`, `equivalent`, `inconclusive`. Only `equivalent`
licenses removing a module.

**A constant score is not a measurement.** Protocol 7.4 substitutes p = 0.5 for an *individual*
invalid COVID score, so a mostly-scoring arm keeps its AUROC. An arm that emits no score at all
is a different case: substituting 0.5 everywhere hands it a constant vector and an AUROC of
exactly 0.500, which looks like a result. NB 17 records NaN plus a `covid_endpoint_note`, and
NB 18 excludes the arm from ROC and calibration using the same 50% coverage threshold — so the
two notebooks cannot disagree about which arms have a P2 endpoint.

---

## Running order and what each gate blocks on

```
Stage C  →  NB 17  →  NB 18  →  NB 19 ─┐
                 └────────→  NB 20 ─────┼→ NB 21 → NB 22
                                        ┘
```

| notebook | blocking conditions |
| --- | --- |
| 17 | NB 14–16 gates passed; exact locked reference and stacking arms are endpoint-usable; endpoint denominators match; every provisional p-value passes §8.8; repeated-seed status is explicit; one shared bootstrap matrix |
| 18 | every fold has fold-specific inner-validation scores; no threshold is selected on its test rows; the NB 17 bootstrap is read-only; ROC and PR bands plus fixed-OP McNemar rows exist |
| 19 | cohort membership and de-duplication guards pass; external intervals exist; F5 is present; every confirmatory p-value has the final Holm adjustment |
| 20 | re-executing the sampling rule reproduces the same selection hash; the taxonomy is not described as dual-coded until two human codings exist |
| 21 | NB 17–20 gates and fingerprints agree; every numeric cell traces within its identity-matched upstream row; the complete §11 table map and E9c transferred-threshold headline exist; final Holm-family counts agree; every selected NB 20 panel is exported; Table 10 contains every required operational field |
| 22 | analysis mode: exactly one passing gate per NB 17–21; reference arm, bootstrap, final family counts and NB 20 selection fingerprints agree; core artifacts, model provenance and release allowlist are valid. Camera-ready mode additionally blocks on repeated seeds, bibliography/software-reference audit, acronym first-use, dual coding and other release items |

Warnings are not failures in NBs 17–21 when they describe a result or a manuscript caveat. NB 22
is the camera-ready gate: unfinished dual coding, missing release artifacts, failed upstream
gates, unpinned models, missing seeds, and an unaudited bibliography are blocking there.

---

## What each notebook is really for

### NB 17 — internal-cohort intervals and provisional paired tests

Downstream notebooks read `all_metrics_with_ci.csv`; NB 19 combines NB 17's provisional tests,
NB 18's fixed-operating-point tests, and F5 into `paired_comparisons_final.csv`. NB 21 reads that
final file only.

Each mRALE comparison produces **two** tests (§8.5) and reports both: a patient-level paired
bootstrap and a Wilcoxon signed-rank on per-patient mean errors. They should agree; where they
do not, the disagreement is recorded, because it usually means the difference is carried by a
few patients rather than by a distribution shift — which changes what the result means.

AUROC comparisons get DeLong *and* a clustered bootstrap. DeLong assumes independent
observations and is anti-conservative when a patient contributes several radiographs, so both
appear and disagreements are flagged rather than resolved silently.

NB 16's completed E6 grid is also analysed here against its locked greedy reference on the
inner-validation subset. Those effect sizes, intervals, and paired p-values are labelled
`EXPLORATORY` and are never added to a Holm-adjusted confirmatory family.

NB 17's `decisive_comparison.json` is explicitly provisional. NB 19 writes
`decisive_comparison_final.json` after NB 18 has contributed the remaining F4 tests, so the
manuscript's central verdict uses the final family size rather than an early adjustment.

### NB 18 — thresholds that mean something

A threshold is selected on fold *k*'s **scored** inner-validation rows written by the owning
prediction notebook and applied to fold *k*'s test rows. Registry membership alone is not a
score. Choosing the threshold that maximises Youden's J *on the test
set* and reporting what it achieves there is circular: it reports the best of ~200 thresholds as
if it were one. The gate fails on any overlap.

Every operating point carries its full confusion matrix. Precision and F1 without TP/FP/TN/FN is
how the specificity collapse hid in the rejected version.

The calibration panel compares each arm's Brier score against the Brier score of predicting the
cohort prevalence for every case. An arm that loses that comparison has probabilities that carry
no usable information, whatever its AUROC.

### NB 19 — external validation, with the metrics each cohort can support

The dispatch refuses to compute what a cohort cannot support: no MAE on X3 (RALO is a different
rubric on a different scale — protocol E9d), no AUROC or sensitivity on X1 (no positives). Those
are not reminders in a comment; the code does not compute them, so they cannot leak into a table.

**E9b is the arm to read first.** If the false-positive rate on tuberculous opacity is materially
higher than on normal radiographs, the detection head is responding to opacity rather than to
COVID. That is the mechanism behind the internal detection result, and it belongs in the
discussion rather than a limitations footnote.

E9c transfers NB 18's median fold-specific inner-validation threshold as the **internal**
operating point and reports any locally re-tuned point beside it as an explicit upper bound. If
five fold-specific external predictions exist, the primary E9c result uses the predeclared fold-0
adapter; the five-adapter ensemble is reserved for E9g. If Stage B stored only a fixed precomputed
ensemble, NB 19 preserves that provenance instead of silently calling it a single model. Thus
neither external labels nor file order can choose the headline adapter. NB 19 verifies NB 18's
gate, arm contract, two F4 rows, and shared bootstrap fingerprint before use, supplies
patient-clustered intervals, and finalizes multiplicity family F5.

### NB 20 — panels chosen by rule, not by appearance

The E10 sampling rule is executed verbatim, seeded, and stamped with a hash covering the rule,
the locked case-table fingerprint, and the resulting case list. Ties among largest errors are
resolved by image key. The gate re-executes the rule and fails if the selection does not reproduce
— so the panels in the paper are provably the panels the rule selects from the audited inputs.

`pcr_label_vs_imaging_mismatch` is reported narrowly. A PCR-positive image with mRALE 0 is not an
mRALE-opacity error when the model also scores 0, but its COVID decision can still be wrong. The
mRALE annotation alone cannot prove an irreducible ceiling for every possible image feature.

Dual coding is a protocol requirement. The notebook produces the coding sheet with an automatic
first pass and refuses to report a κ until two *human* codings exist — κ between an algorithm and
itself is 1.0 and means nothing. Human entries are preserved only when both the case keys and the
case-table fingerprint match; stale generated panels are removed on rerun. Grounding is
negation-aware, reads structured finding objects correctly, and limits annotation-based
contradiction checks to opacity claims that mRALE can actually adjudicate.

### NB 21 — no hand-typed numbers

Every numeric cell is looked up only in the identity-matched row of its owning locked artifact,
and `traceability.csv` records that source. A coincidentally equal rounded value in another arm or
metric no longer passes. NB 21 also exports the full protocol map: Tables 1–11 and S1–S3, plus
Table 8b for X2 at the threshold transferred unchanged from internal inner validation. The
locally re-tuned X2 point remains an upper bound and is not substituted into the headline table.

The export gate requires passing NB 17–20 gates, the frozen bootstrap fingerprint, NB 20's case
selection and case-table fingerprints, and NB 19's final multiplicity-family counts. Figure 6
copies only the exact case panels named by the locked NB 20 selection and records them in
`figure6_case_panel_manifest.csv`; stale panels are never copied by directory glob. Figure 7 is
labelled as an automatic first pass until `dual_coding_kappa.json` confirms two complete human
codings with all disagreements resolved.

Table 10 normalizes the operational schemas actually written upstream: parameter counts in
absolute or millions units, peak memory in GiB, fold training time in seconds, and latency/token
counts from NB 15's fingerprinted outer-test journals or NB 16's sensitivity artifact. Reading
the journals is an export-only aggregation and does not load a model. Missing values remain
visible as dashes; an entirely absent required measurement class blocks NB 21, while arm-specific
gaps are reported as warnings rather than replaced with invented values.

`sd.format_number` is the only place a float becomes a string. Best/second-best highlighting is
computed from each metric's own direction (MAE lower, AUROC higher), so the bold cell is chosen
by the data.

### NB 22 — the last thing before submission

NB 22 treats the stage roots as controlled working storage and creates a separate publication
allowlist from the required results plus NB 21 exports. Restricted-name working artifacts are
inventoried, but only a restricted path entering that allowlist blocks release. This avoids both
publishing a DUA-controlled derivative and falsely treating every checkpoint or cache as part of
the submission package.

The reproducibility scan reads structured model-ID fields instead of searching narrative notes
for product names. It rejects known closed-model IDs, requires an immutable Stage A revision for
every registered model and a seed in every run configuration, and records the environment. It
also requires exactly one passing gate for each NB 17–21 and binds the chain with NB 17's locked
reference arm and bootstrap fingerprint, NB 19's final multiplicity-family counts, and NB 20's
selection and case-table fingerprints as repeated by NB 21.

The reference audit reads BibTeX, BBL, text, Markdown and Word manuscript sources. It distinguishes
**replace** (a preprint without a peer-reviewed DOI), **verify** (a DOI is present but the entry
still reads as a preprint), and **software artifact** (a model card cited for what the software
is, not as clinical evidence). Both `replace` and `verify` block release. A software entry passes
only when its citation or the pinned model registry supplies an immutable revision hash.

An empty audit is not a passing audit: if no bibliography is found, the checklist says
`NOT AUDITED`, and an empty model registry reads `NO MODEL REGISTRY` rather than READY. Manuscripts
outside the cloud project can be named explicitly in `EXTRA_REFERENCE_SOURCES` at the top of
NB 22. Confidence intervals are required only for finite endpoint estimates, so an explicitly
unusable endpoint with a NaN estimate does not masquerade as training divergence.

The notebook has two intentional modes. Its default `ENFORCE_CAMERA_READY = False` runs the
**analysis gate**: computational corruption, stale fingerprints, missing core artifacts,
restricted release paths, closed models, unpinned revisions, and undeclared stochastic seeds
still fail. Repeated-seed experiments, reference review, acronym first-use, dual human coding and
author sign-offs remain visible as camera-ready blockers but do not crash an otherwise valid
analysis run. For the final submission run, set `ENFORCE_CAMERA_READY = True`; the same pending
items then become assertion failures. A fingerprinted greedy inference configuration such as
NB 12 is explicitly classified as deterministic rather than falsely reported as a missing-seed
training run. The Excel workbook is required only when NB 21 says it created one; the per-table
CSVs remain the complete supported fallback when `openpyxl` is unavailable. The manifest is
written last and therefore hashes NB 22's final gate, checklist, audits, environment and run
configuration.

---

## Cross-notebook contracts

| consumer | reads | what breaks if it is stale |
| --- | --- | --- |
| NB 18 | `bootstrap_indices.npz` | bands in Fig. 3 would describe a different resampling from the intervals in Table 2 |
| NB 18 | fold-specific inner-validation score records from NB 09/10/14/15 | registry membership without scores cannot select a threshold |
| NB 19 | NB 18's `operating_points.csv` | E9c could only report the re-tuned upper bound |
| NB 20 | NB 15's `reasoning_traces.jsonl` | evidence grounding (E10c) cannot be computed |
| NB 21 | NB 19's final comparisons plus every table-owning artifact | provisional p-values or value-only trace matches could enter the paper |
| NB 22 | NB 17–21 gates/configurations plus NB 19/NB 20 fingerprints | a passing gate from a stale run could otherwise certify unrelated exports |

Each cross-notebook read distinguishes "not produced yet" from "wrong path" in its error
message. The remedies are hours apart, and NB 10 once spent an afternoon on the second while
reading the first.

---

## Practical notes

- **Re-run from the top.** Stage D is cheap; there is no resume machinery because nothing here
  costs more than half an hour. The bootstrap index cache is the only persisted state, and it is
  fingerprinted on the patient list.
- **Thirty-six-hour allocations.** Every Stage D notebook has the same 35-hour soft deadline as
  Stage C, checked before each subsequent code section. It should never fire under the expected
  2–30 minute runtimes; if it does, the upstream data volume or environment has changed enough
  to deserve investigation. Resubmit and rerun that short notebook from the top. The shared
  bootstrap cache and JSON state are written durably via temporary-file replacement.
- **Order matters.** NB 18 and NB 20 follow NB 17; NB 19 follows NB 18; NB 21 follows NB 19 and
  NB 20. Running NB 21 first
  produces a confusing cascade of "not produced yet" errors rather than a wrong answer, which is
  the intended behaviour.
- **NB 18 has an explicit Stage C producer contract.** NB 14 and NB 15 must save
  `folds/fold_<k>/inner_validation_score_records.jsonl` for the locked stacking and reasoner arms.
  If those files are absent, NB 18 stops and names the missing arm/fold; pooled OOF scores are not
  substituted because they were produced by different outer-fold models.
- **matplotlib is optional but expected.** Without it the CSVs are complete and NB 21 says so;
  with it, plotted Figures 3–5 and 7 are written as PDF and SVG with `pdf.fonttype = 42` so text
  stays selectable (R1.9). Figure 6 remains PNG because its content is radiographic imagery;
  NB 21 copies no raster version of an NB 18 plot.
- **`openpyxl` is optional.** Without it the per-table CSVs are the complete supplement.
- **A failing gate is information.** NB 19's gate fails when no external predictions exist —
  that is the correct outcome, because R1.5 asked for external validation and an empty result
  set is the honest way to discover it has not been produced.

---

## If something goes wrong

| symptom | first thing to check |
| --- | --- |
| NB 18's AUROCs disagree with NB 17 | the two are reading different row sets; compare `arm_availability.csv` against what NB 18 discovered |
| an arm shows AUROC exactly 0.500 | it has no continuous score; NB 17 should record NaN and a `covid_endpoint_note` — if it does not, the coverage threshold was bypassed |
| NB 21 reports untraceable numbers | a value was computed in NB 21 instead of read from the notebook that owns it, or a new artifact needs adding to `LOCKED_SOURCES` |
| NB 20's selection hash changes between runs | something in the case table is non-deterministic; the rule itself is seeded, so look at the inputs |
| "family size disagrees with membership" | a `PValue` was created after `apply_holm_within_families` ran |
| intervals look implausibly narrow | check the bootstrap is resampling patients, not images — `bootstrap.n_patients` should be the patient count, not the image count |
