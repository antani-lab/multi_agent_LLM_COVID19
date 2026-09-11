# Notebook workflow

The manuscript-aligned workflow contains 23 numbered notebooks across four stages. Run them from the repository's `notebooks/` directory, or preserve the same relative directory layout so the shared modules can be discovered.

## Stage A — cohort and preprocessing

| NB | Notebook | Function |
| --- | --- | --- |
| 00 | `stage_A/00_environment_and_model_registry.ipynb` | Capture software/hardware, resolve data paths, and pin model revisions. |
| 01 | `stage_A/01_data_inventory_qc_and_labels.ipynb` | Build the image manifest; audit labels, quality, hashes, and duplicate groups. |
| 02 | `stage_A/02_folds_and_leakage_audit.ipynb` | Generate group-stratified folds and verify zero group leakage. |
| 03 | `stage_A/03_external_cohorts_prep.ipynb` | Prepare external manifests and membership/de-duplication checks. |
| 04 | `stage_A/04_preprocessing_and_lung_localization_cache.ipynb` | Generate and cache whole-image and lung-localized views. |

## Stage B — individual agents

| NB | Notebook | Function |
| --- | --- | --- |
| 05 | `stage_B/05_frozen_encoder_agents.ipynb` | Fit frozen-CXformer probes and a shared per-lung ordinal head. |
| 06 | `stage_B/06_conventional_cxr_classifiers.ipynb` | Train DenseNet-121, ConvNeXt-Tiny, and ViT-B/16 baselines. |
| 07 | `stage_B/07_zeroshot_vlm_agents.ipynb` | Evaluate zero-shot MedGemma, Qwen3.5, and NV-Reason models. |
| 08 | `stage_B/08_biomedclip_entity_agent.ipynb` | Extract BiomedCLIP entity scores and fit entity-feature probes. |
| 09 | `stage_B/09_medgemma_multitask_lora_5fold.ipynb` | Tune and train fold-specific MedGemma multitask LoRA adapters. |
| 10 | `stage_B/10_qwen35_multitask_lora_5fold.ipynb` | Train the matched Qwen3.5-4B multitask LoRA control. |
| 11 | `stage_B/11_nvreason_agent_and_optional_lora.ipynb` | Evaluate NV-Reason findings, output validity, and abstention. |
| 12 | `stage_B/12_anatomy_aware_mrale_5fold.ipynb` | Run anatomy-aware mRALE and localization ablations. |

## Stage C — registry, fusion, and reasoning

| NB | Notebook | Function |
| --- | --- | --- |
| 13 | `stage_C/13_agent_output_registry.ipynb` | Normalize out-of-fold agent outputs and inner-validation reliability. |
| 14 | `stage_C/14_conventional_fusion_baselines.ipynb` | Evaluate deterministic fusion, learned stacking, boosting, and gating. |
| 15 | `stage_C/15_llm_reasoner_aggregation.ipynb` | Run the local Qwen3.5-4B reasoner and prespecified agent ablations. |
| 16 | `stage_C/16_decoding_prompt_and_depth_sensitivity.ipynb` | Evaluate prompt schema, decoding, and reasoning-depth sensitivity. |

## Stage D — locked analysis and export

| NB | Notebook | Function |
| --- | --- | --- |
| 17 | `stage_D/17_statistics_and_paired_comparisons.ipynb` | Compute patient-grouped intervals and provisional paired comparisons. |
| 18 | `stage_D/18_calibration_roc_and_thresholds.ipynb` | Evaluate calibration, ROC/PR curves, and validation-selected thresholds. |
| 19 | `stage_D/19_external_validation_and_domain_shift.ipynb` | Perform external stress tests and finalize multiplicity-controlled tests. |
| 20 | `stage_D/20_interpretability_and_failure_cases.ipynb` | Select deterministic failure cases and generate rationale-audit materials. |
| 21 | `stage_D/21_figures_tables_and_exports.ipynb` | Export manuscript tables, figures, captions, and traceability records. |
| 22 | `stage_D/22_reproducibility_and_reference_audit.ipynb` | Audit artifacts, model provenance, seeds, references, and release readiness. |

Read each stage README before execution. Every notebook ends with a gate; a failed gate indicates that downstream results should not be treated as valid. Long-running notebooks support checkpointing or per-item journals as described in [`../docs/RESUME.md`](../docs/RESUME.md).

The four files in `legacy_training/` predate the fully audited Stage A–D pipeline and are retained only for provenance.
