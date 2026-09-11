"""
Shared prediction schema, metric definitions, and I/O helpers for Stage B / C / D.

WHY THIS IS A MODULE AND NOT COPY-PASTED CELLS
----------------------------------------------
Manuscript Table 2 compares a frozen linear probe, three end-to-end CNN/ViT baselines, three
zero-shot VLMs, two LoRA-adapted VLMs, and the full framework. That comparison is only valid
if every arm's MAE, AUROC, and invalid-output penalty are computed by the same code. Copying a
metrics cell into eight notebooks guarantees drift, and the drift would be invisible.

Protocol references: Section 7 (metrics and endpoints), Section 7.4 (invalid-output policy),
Section 8.2 (fold-level aggregation convention).

Import from a notebook in this directory:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd()))
    import cxr_metrics as cm
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

__all__ = [
    "PREDICTION_FIELDS", "make_prediction_row", "EXTENT_RANGE", "DENSITY_RANGE",
    "MRALE_TOTAL_MAX", "MRALE_LUNG_MAX", "INVALID_TOTAL_PENALTY", "INVALID_LUNG_PENALTY",
    "severity_band", "SEVERITY_BANDS",
    "read_jsonl", "write_jsonl", "append_jsonl", "load_jsonl_by_key", "write_json",
    "classification_metrics", "mrale_metrics", "localization_free_metrics",
    "quadratic_weighted_kappa", "expected_calibration_error", "brier_score",
    "student_t_975", "aggregate_over_folds", "flatten_numeric", "json_safe",
]

# --------------------------------------------------------------------------------------
# Label geometry (protocol Section 3.2)
# --------------------------------------------------------------------------------------

EXTENT_RANGE = (0, 4)          # integers 0-4
DENSITY_RANGE = (0, 3)         # integers 0-3
MRALE_LUNG_MAX = 12            # extent * density
MRALE_TOTAL_MAX = 24           # right + left

# Invalid-output policy, protocol Section 7.4. Applied identically to every arm.
INVALID_TOTAL_PENALTY = 24.0
INVALID_LUNG_PENALTY = 12.0
INVALID_COVID_SCORE = 0.5      # uninformative score, so AUROC is neither helped nor hurt

SEVERITY_BANDS = [(0, 0, "none"), (1, 10, "mild"), (11, 18, "moderate"), (19, 24, "severe")]


def is_missing(value):
    """
    True for None and for any non-finite numeric value (NaN or +/- infinity).

    `value is not None` is not enough. Every one of these notebooks moves prediction rows
    through a pandas DataFrame at some point, and pandas turns a JSON `null` into `float('nan')`
    on the way out. NaN then passes an `is not None` guard, reaches `int()`, and raises
    "cannot convert float NaN to integer" -- which is how an arm that legitimately abstains
    (NV-Reason abstains on roughly a third of severe cases) takes down a downstream notebook
    that had nothing to do with it.
    """
    if value is None:
        return True
    try:
        return not bool(np.isfinite(value))
    except (TypeError, ValueError):
        return False


def _is_bounded_number(value, minimum, maximum):
    """Whether value is a finite scalar inside an inclusive numeric interval."""
    if is_missing(value):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and minimum <= number <= maximum


def severity_band(total):
    if is_missing(total):
        return None
    for low, high, name in SEVERITY_BANDS:
        if low <= total <= high:
            return name
    return "out_of_range"


# --------------------------------------------------------------------------------------
# Prediction schema
# --------------------------------------------------------------------------------------
# One row per (image, agent, arm, task). NB 12 assembles these into the agent registry, so
# any new agent must emit exactly these fields. Extra fields are allowed under "extra".

PREDICTION_FIELDS = [
    # identity
    "image_key", "cohort", "subcohort", "filename", "held_out_fold",
    "agent", "arm", "view", "task",
    # COVID prediction
    "covid_pred", "covid_score",
    # mRALE prediction
    "mrale_total", "mrale_right", "mrale_left",
    "extent_right", "density_right", "extent_left", "density_left",
    # continuous / soft mRALE (protocol 7.2)
    "mrale_total_expected", "mrale_uncertainty",
    # ground truth, copied for self-contained files (registry re-joins to be safe)
    "gt_covid", "gt_mrale_total", "gt_mrale_right", "gt_mrale_left",
    "gt_extent_right", "gt_density_right", "gt_extent_left", "gt_density_left",
    # integrity and provenance
    "valid", "parse_error", "raw_output", "seconds",
    "model_id", "model_revision", "extra",
]


def make_prediction_row(**kwargs):
    row = OrderedDict((field, kwargs.pop(field, None)) for field in PREDICTION_FIELDS)
    if kwargs:
        extra = row.get("extra") or {}
        if not isinstance(extra, dict):
            extra = {"value": extra}
        extra.update(kwargs)
        row["extra"] = extra
    if row["valid"] is None:
        row["valid"] = row["parse_error"] is None
    return row


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------

def read_jsonl(path):
    rows = []
    path = Path(path)
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl_by_key(path, key_fields):
    rows = {}
    path = Path(path)
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows[tuple(str(row[field]) for field in key_fields)] = row
            except Exception:
                continue
    return rows


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if (math.isnan(number) or math.isinf(number)) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


# --------------------------------------------------------------------------------------
# Classification metrics (protocol 7.1 P2, 7.3)
# --------------------------------------------------------------------------------------

def brier_score(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    return float(np.mean((y_score - y_true) ** 2))


def expected_calibration_error(y_true, y_score, n_bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    error = 0.0
    for index in range(n_bins):
        low, high = edges[index], edges[index + 1]
        mask = (y_score > low) & (y_score <= high) if index > 0 else (y_score >= low) & (y_score <= high)
        if not mask.any():
            continue
        error += mask.sum() / total * abs(y_true[mask].mean() - y_score[mask].mean())
    return float(error)


def classification_metrics(y_true, y_pred, y_score=None, positive_label="Yes",
                           n_calibration_bins=10):
    """
    y_true : ground-truth labels, "Yes"/"No" or 0/1. None entries are dropped.
    y_pred : predicted labels. None counts as INCORRECT (protocol 7.4).
    y_score: continuous P(positive). None entries get INVALID_COVID_SCORE.

    Returns a flat dict. Coverage is always reported next to the valid-only metrics.
    """
    def to_binary(value):
        if is_missing(value):
            return None
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            return int(value)
        return 1 if str(value).strip().lower() in {"yes", "1", "true", "positive", "pos"} else 0

    truth, predicted, scores = [], [], []
    for index, label in enumerate(y_true):
        binary_truth = to_binary(label)
        if binary_truth is None:
            continue                                   # no reference standard: excluded
        truth.append(binary_truth)
        raw_prediction = y_pred[index] if index < len(y_pred) else None
        predicted.append(to_binary(raw_prediction))    # None preserved as invalid
        raw_score = None if y_score is None else (
            y_score[index] if index < len(y_score) else None)
        scores.append(INVALID_COVID_SCORE if is_missing(raw_score) else float(raw_score))

    n = len(truth)
    metrics = {"n": n, "positive_label": positive_label}
    if n == 0:
        return metrics

    truth_array = np.asarray(truth, dtype=int)
    valid_mask = np.asarray([item is not None for item in predicted], dtype=bool)
    metrics["valid_predictions"] = int(valid_mask.sum())
    metrics["valid_rate"] = float(valid_mask.mean())

    # Invalid predictions count as incorrect: assign them the opposite of the truth.
    resolved = np.asarray(
        [item if item is not None else 1 - truth_array[i] for i, item in enumerate(predicted)],
        dtype=int,
    )

    tp = int(((resolved == 1) & (truth_array == 1)).sum())
    tn = int(((resolved == 0) & (truth_array == 0)).sum())
    fp = int(((resolved == 1) & (truth_array == 0)).sum())
    fn = int(((resolved == 0) & (truth_array == 1)).sum())
    metrics.update({"tp": tp, "tn": tn, "fp": fp, "fn": fn})

    def safe_divide(numerator, denominator):
        return float(numerator / denominator) if denominator else float("nan")

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    metrics["accuracy"] = safe_divide(tp + tn, n)
    metrics["sensitivity"] = sensitivity
    metrics["specificity"] = specificity
    metrics["precision"] = precision
    metrics["recall"] = sensitivity
    metrics["balanced_accuracy"] = (
        float(np.nanmean([sensitivity, specificity]))
        if not (math.isnan(sensitivity) and math.isnan(specificity)) else float("nan")
    )
    metrics["f1"] = safe_divide(2 * tp, 2 * tp + fp + fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    metrics["mcc"] = float((tp * tn - fp * fn) / denominator) if denominator else float("nan")
    metrics["prevalence"] = float(truth_array.mean())

    score_array = np.asarray(scores, dtype=float)
    both_classes_present = len(set(truth_array.tolist())) == 2
    if both_classes_present:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            metrics["auroc"] = float(roc_auc_score(truth_array, score_array))
            metrics["auprc"] = float(average_precision_score(truth_array, score_array))
        except Exception as exc:
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")
            metrics["score_error"] = f"{type(exc).__name__}: {exc}"
        metrics["brier"] = brier_score(truth_array, score_array)
        metrics["ece"] = expected_calibration_error(
            truth_array, score_array, n_bins=n_calibration_bins)
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
        metrics["brier"] = float("nan")
        metrics["ece"] = float("nan")
        metrics["single_class_warning"] = True

    # Does the continuous score reproduce the hard decision? Protocol 7.2 requires >= 0.995
    # agreement before the score is used for ROC/DeLong.
    if valid_mask.any():
        score_decision = (score_array >= 0.5).astype(int)
        hard = np.asarray([item if item is not None else -1 for item in predicted], dtype=int)
        comparable = hard >= 0
        metrics["score_decision_agreement"] = float(
            (score_decision[comparable] == hard[comparable]).mean()
        ) if comparable.any() else float("nan")

    return metrics


# --------------------------------------------------------------------------------------
# mRALE metrics (protocol 7.1 P1, 7.3, 7.4)
# --------------------------------------------------------------------------------------

def quadratic_weighted_kappa(y_true, y_pred, min_rating=None, max_rating=None):
    # Convert to float first. Casting NaN/inf directly to int produces NumPy's minimum-int
    # sentinel, which then becomes a huge negative matrix index. QWK is a valid-only metric,
    # so non-finite and explicitly out-of-range pairs are excluded defensively here.
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same number of ratings")
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if min_rating is not None:
        valid &= (y_true >= min_rating) & (y_pred >= min_rating)
    if max_rating is not None:
        valid &= (y_true <= max_rating) & (y_pred <= max_rating)
    y_true = np.rint(y_true[valid]).astype(int)
    y_pred = np.rint(y_pred[valid]).astype(int)
    if len(y_true) == 0:
        return float("nan")
    if min_rating is None:
        min_rating = int(min(y_true.min(), y_pred.min()))
    if max_rating is None:
        max_rating = int(max(y_true.max(), y_pred.max()))
    n_ratings = max_rating - min_rating + 1
    if n_ratings < 2:
        return float("nan")

    observed = np.zeros((n_ratings, n_ratings), dtype=float)
    for actual, predicted in zip(y_true, y_pred):
        observed[actual - min_rating, predicted - min_rating] += 1
    observed /= observed.sum()

    truth_hist = np.bincount(y_true - min_rating, minlength=n_ratings).astype(float)
    pred_hist = np.bincount(y_pred - min_rating, minlength=n_ratings).astype(float)
    expected = np.outer(truth_hist / truth_hist.sum(), pred_hist / pred_hist.sum())

    indices = np.arange(n_ratings)
    weights = (indices[:, None] - indices[None, :]) ** 2 / (n_ratings - 1) ** 2
    denominator = (weights * expected).sum()
    if denominator == 0:
        return float("nan")
    return float(1.0 - (weights * observed).sum() / denominator)


def _penalised_errors(truth, prediction, penalty, minimum=None, maximum=None):
    """Absolute errors with the protocol 7.4 penalty substituted for invalid predictions."""
    errors, valid_errors, n_valid = [], [], 0
    for reference, predicted in zip(truth, prediction):
        if is_missing(reference):
            continue
        invalid = is_missing(predicted)
        if not invalid and minimum is not None and maximum is not None:
            invalid = not _is_bounded_number(predicted, minimum, maximum)
        if invalid:
            errors.append(float(penalty))
        else:
            error = abs(float(predicted) - float(reference))
            errors.append(error)
            valid_errors.append(error)
            n_valid += 1
    return np.asarray(errors), np.asarray(valid_errors), n_valid


def mrale_metrics(records, prefix=""):
    """
    records: iterable of dicts with keys
        gt_mrale_total, mrale_total          (required)
        gt_mrale_right, mrale_right          (optional)
        gt_mrale_left,  mrale_left           (optional)
        gt_extent_right, extent_right, ...   (optional, per component)
        mrale_total_expected                 (optional continuous score)
    Rows whose ground truth is None are excluded, matching the classification convention.
    """
    rows = [row for row in records if not is_missing(row.get("gt_mrale_total"))]
    metrics = {f"{prefix}n": len(rows)}
    if not rows:
        return metrics

    truth_total = [row["gt_mrale_total"] for row in rows]
    pred_total = [row.get("mrale_total") for row in rows]

    errors, valid_errors, n_valid = _penalised_errors(
        truth_total, pred_total, INVALID_TOTAL_PENALTY, 0, MRALE_TOTAL_MAX)
    metrics[f"{prefix}valid_predictions"] = n_valid
    metrics[f"{prefix}coverage"] = float(n_valid / len(rows))
    # PRIMARY ENDPOINT P1: penalised MAE over every held-out image.
    metrics[f"{prefix}mae"] = float(errors.mean())
    metrics[f"{prefix}rmse"] = float(np.sqrt((errors ** 2).mean()))
    metrics[f"{prefix}mae_valid_only"] = (
        float(valid_errors.mean()) if n_valid else float("nan"))
    metrics[f"{prefix}rmse_valid_only"] = (
        float(np.sqrt((valid_errors ** 2).mean())) if n_valid else float("nan"))

    paired = [(reference, predicted) for reference, predicted in zip(truth_total, pred_total)
              if _is_bounded_number(reference, 0, MRALE_TOTAL_MAX)
              and _is_bounded_number(predicted, 0, MRALE_TOTAL_MAX)]
    if len(paired) >= 2:
        reference_array = np.asarray([item[0] for item in paired], dtype=float)
        predicted_array = np.asarray([item[1] for item in paired], dtype=float)
        metrics[f"{prefix}exact_accuracy"] = float((predicted_array == reference_array).mean())
        metrics[f"{prefix}within1_accuracy"] = float(
            (np.abs(predicted_array - reference_array) <= 1).mean())
        metrics[f"{prefix}within2_accuracy"] = float(
            (np.abs(predicted_array - reference_array) <= 2).mean())
        residual = reference_array - predicted_array
        total_variance = ((reference_array - reference_array.mean()) ** 2).sum()
        metrics[f"{prefix}r2"] = float(
            1 - (residual ** 2).sum() / total_variance) if total_variance > 0 else float("nan")
        try:
            from scipy.stats import pearsonr, spearmanr
            if reference_array.std() > 0 and predicted_array.std() > 0:
                metrics[f"{prefix}pearson_r"] = float(pearsonr(reference_array, predicted_array)[0])
                metrics[f"{prefix}spearman_rho"] = float(spearmanr(reference_array, predicted_array)[0])
            else:
                metrics[f"{prefix}pearson_r"] = float("nan")
                metrics[f"{prefix}spearman_rho"] = float("nan")
        except Exception:
            metrics[f"{prefix}pearson_r"] = float("nan")
            metrics[f"{prefix}spearman_rho"] = float("nan")
        metrics[f"{prefix}qwk"] = quadratic_weighted_kappa(
            np.rint(reference_array).astype(int), np.rint(predicted_array).astype(int),
            min_rating=0, max_rating=MRALE_TOTAL_MAX)

    # Severity-stratified MAE (protocol 7.3). Reported for every band, including empty ones,
    # so a missing band is visible rather than silently absent from the table.
    for _, _, band in SEVERITY_BANDS:
        band_rows = [row for row in rows if severity_band(row["gt_mrale_total"]) == band]
        if not band_rows:
            metrics[f"{prefix}mae_band_{band}"] = float("nan")
            metrics[f"{prefix}n_band_{band}"] = 0
            continue
        band_errors, _, _ = _penalised_errors(
            [row["gt_mrale_total"] for row in band_rows],
            [row.get("mrale_total") for row in band_rows],
            INVALID_TOTAL_PENALTY,
            0,
            MRALE_TOTAL_MAX,
        )
        metrics[f"{prefix}mae_band_{band}"] = float(band_errors.mean())
        metrics[f"{prefix}n_band_{band}"] = len(band_rows)

    # Severity-band classification view.
    band_truth = [severity_band(row["gt_mrale_total"]) for row in rows]
    band_pred = [None if not _is_bounded_number(row.get("mrale_total"), 0, MRALE_TOTAL_MAX)
                 else severity_band(row.get("mrale_total")) for row in rows]
    comparable = [(t, p) for t, p in zip(band_truth, band_pred) if p is not None]
    if comparable:
        # band_accuracy is a VALID-ONLY metric: rows with no usable prediction have no band.
        # Protocol 7.4 forbids reporting one without its coverage, so the coverage is emitted
        # here rather than left for each notebook to remember.
        metrics[f"{prefix}band_coverage"] = float(len(comparable) / len(rows))
        metrics[f"{prefix}band_accuracy"] = float(
            np.mean([t == p for t, p in comparable]))
        band_names = [name for _, _, name in SEVERITY_BANDS]
        f1_scores = []
        for name in band_names:
            tp = sum(1 for t, p in comparable if t == name and p == name)
            fp = sum(1 for t, p in comparable if t != name and p == name)
            fn = sum(1 for t, p in comparable if t == name and p != name)
            if tp + fp + fn == 0:
                continue
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1_scores.append(2 * precision * recall / (precision + recall)
                             if precision + recall else 0.0)
        metrics[f"{prefix}band_macro_f1"] = float(np.mean(f1_scores)) if f1_scores else float("nan")

    # Per-lung MAE with the 12-point penalty.
    for side in ["right", "left"]:
        truth_side = [row.get(f"gt_mrale_{side}") for row in rows]
        pred_side = [row.get(f"mrale_{side}") for row in rows]
        if all(is_missing(item) for item in truth_side):
            continue
        side_errors, _, side_valid = _penalised_errors(
            truth_side, pred_side, INVALID_LUNG_PENALTY, 0, MRALE_LUNG_MAX)
        if len(side_errors):
            metrics[f"{prefix}mae_{side}"] = float(side_errors.mean())
            metrics[f"{prefix}coverage_{side}"] = float(
                side_valid / max(1, sum(1 for item in truth_side
                                        if not is_missing(item))))

    # Per-component exact accuracy.
    for component in ["extent_right", "density_right", "extent_left", "density_left"]:
        component_range = EXTENT_RANGE if component.startswith("extent_") else DENSITY_RANGE
        pairs = [(row.get(f"gt_{component}"), row.get(component)) for row in rows]
        pairs = [(t, p) for t, p in pairs
                 if _is_bounded_number(t, *component_range)
                 and _is_bounded_number(p, *component_range)]
        if pairs:
            metrics[f"{prefix}{component}_accuracy"] = float(
                np.mean([int(t) == int(p) for t, p in pairs]))

    # Arithmetic consistency: does the reported total equal right + left?
    consistency = [
        row for row in rows
        if _is_bounded_number(row.get("mrale_total"), 0, MRALE_TOTAL_MAX)
        and _is_bounded_number(row.get("mrale_right"), 0, MRALE_LUNG_MAX)
        and _is_bounded_number(row.get("mrale_left"), 0, MRALE_LUNG_MAX)
    ]
    if consistency:
        metrics[f"{prefix}formula_consistency"] = float(np.mean([
            int(row["mrale_total"]) == int(row["mrale_right"]) + int(row["mrale_left"])
            for row in consistency
        ]))

    # Continuous (soft) total, when the arm provides one.
    soft = [(row["gt_mrale_total"], row["mrale_total_expected"]) for row in rows
            if _is_bounded_number(row.get("mrale_total_expected"), 0, MRALE_TOTAL_MAX)]
    if len(soft) >= 2:
        reference_array = np.asarray([item[0] for item in soft], dtype=float)
        soft_array = np.asarray([item[1] for item in soft], dtype=float)
        metrics[f"{prefix}mae_expected"] = float(np.abs(soft_array - reference_array).mean())
        try:
            from scipy.stats import spearmanr
            metrics[f"{prefix}spearman_rho_expected"] = float(
                spearmanr(reference_array, soft_array)[0])
        except Exception:
            pass

    return metrics


def localization_free_metrics(rows, prefix="output."):
    """Output-integrity metrics that apply to every generative arm (protocol 7.3)."""
    total = len(rows)
    metrics = {f"{prefix}n": total}
    if not total:
        return metrics
    metrics[f"{prefix}valid_rate"] = float(np.mean([bool(row.get("valid")) for row in rows]))
    errors = Counter(
        str(row.get("parse_error")).split(":")[0]
        for row in rows if row.get("parse_error")
    )
    metrics[f"{prefix}n_parse_errors"] = int(sum(errors.values()))
    metrics[f"{prefix}parse_error_types"] = dict(errors)
    seconds = [row.get("seconds") for row in rows
               if not is_missing(row.get("seconds"))]
    if seconds:
        metrics[f"{prefix}median_seconds"] = float(np.median(seconds))
        metrics[f"{prefix}p95_seconds"] = float(np.percentile(seconds, 95))
    return metrics


# --------------------------------------------------------------------------------------
# Fold aggregation (protocol 8.2) -- matches the convention already used by the tested
# notebooks' cross_validation_aggregate_95ci.csv so old and new tables line up.
# --------------------------------------------------------------------------------------

_T_CRITICAL_975 = {
    1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582, 6: 2.446912,
    7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139, 11: 2.200985, 12: 2.178813,
    13: 2.160369, 14: 2.144787, 15: 2.131450, 16: 2.119905, 17: 2.109816, 18: 2.100922,
    19: 2.093024, 20: 2.085963, 21: 2.079614, 22: 2.073873, 23: 2.068658, 24: 2.063899,
    25: 2.059539, 26: 2.055529, 27: 2.051831, 28: 2.048407, 29: 2.045230, 30: 2.042272,
}


def student_t_975(degrees_of_freedom):
    return _T_CRITICAL_975.get(degrees_of_freedom, 1.959964)


def flatten_numeric(prefix, value, output):
    if isinstance(value, dict):
        for key, item in value.items():
            flatten_numeric(f"{prefix}.{key}" if prefix else str(key), item, output)
    elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            output[prefix] = number


def aggregate_over_folds(per_fold_metrics, skip_count_metrics=True):
    """
    per_fold_metrics: {fold -> flat metric dict}
    Returns a list of rows: metric, n_folds, mean, sample_std, standard_error,
    t_critical_95, ci95_margin, ci95_lower, ci95_upper.
    """
    def is_count(name):
        leaf = name.rsplit(".", 1)[-1]
        return (leaf in {"n", "valid_predictions", "tp", "tn", "fp", "fn", "epoch", "step",
                         "n_parse_errors"}
                or leaf.startswith("n_band_"))

    flattened = {}
    for fold, metrics in per_fold_metrics.items():
        row = {}
        flatten_numeric("", metrics, row)
        flattened[fold] = row

    all_metrics = sorted({key for row in flattened.values() for key in row})
    rows = []
    for metric in all_metrics:
        if skip_count_metrics and is_count(metric):
            continue
        values = [row[metric] for row in flattened.values()
                  if metric in row and math.isfinite(row[metric])]
        n_folds = len(values)
        if n_folds == 0:
            continue
        mean_value = sum(values) / n_folds
        if n_folds >= 2:
            variance = sum((value - mean_value) ** 2 for value in values) / (n_folds - 1)
            sample_std = math.sqrt(variance)
            standard_error = sample_std / math.sqrt(n_folds)
            t_critical = student_t_975(n_folds - 1)
            margin = t_critical * standard_error
        else:
            sample_std = standard_error = t_critical = margin = float("nan")
        rows.append({
            "metric": metric, "n_folds": n_folds, "mean": mean_value,
            "sample_std": sample_std, "standard_error": standard_error,
            "t_critical_95": t_critical, "ci95_margin": margin,
            "ci95_lower": mean_value - margin, "ci95_upper": mean_value + margin,
        })
    return rows


# --------------------------------------------------------------------------------------
# Self-tests. Run with `python3 cxr_metrics.py`.
#
# Added when `is_missing` was introduced: every notebook in this project moves prediction rows
# through a pandas DataFrame at some point, and pandas turns JSON `null` into `float('nan')`.
# NaN passed the old `is not None` guards, reached `int()`, and crashed a downstream notebook.
# The NaN cases below are the regression tests for that.
# --------------------------------------------------------------------------------------

def _self_test():
    checks, failures = 0, []

    def ok(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(message)

    nan = float("nan")

    # ---- is_missing ---------------------------------------------------------------------
    ok(is_missing(None), "None is missing")
    ok(is_missing(nan), "NaN is missing")
    ok(is_missing(np.nan), "numpy NaN is missing")
    ok(is_missing(float("inf")) and is_missing(float("-inf")),
       "positive and negative infinity are invalid/missing")
    ok(not is_missing(0), "zero is a value, not a missing marker")
    ok(not is_missing(0.0), "0.0 is a value")
    ok(not is_missing("Yes"), "a string is not missing")
    ok(not is_missing(False), "False is a value")

    # ---- severity bands -------------------------------------------------------------------
    ok(severity_band(0) == "none", "0 is the none band")
    ok(severity_band(10) == "mild" and severity_band(11) == "moderate", "mild/moderate edge")
    ok(severity_band(18) == "moderate" and severity_band(19) == "severe",
       "moderate/severe edge")
    ok(severity_band(nan) is None, "NaN has no band")

    # ---- invalid-output policy (protocol 7.4) ---------------------------------------------
    ok(mrale_metrics([{"gt_mrale_total": 20, "mrale_total": None}])["mae"]
       == INVALID_TOTAL_PENALTY, "an invalid total takes the 24-point penalty")
    ok(mrale_metrics([{"gt_mrale_total": 20, "mrale_total": nan}])["mae"]
       == INVALID_TOTAL_PENALTY, "a NaN total takes the penalty too, and does not crash")
    ok(mrale_metrics([{"gt_mrale_total": 20, "mrale_total": None}])["coverage"] == 0.0,
       "coverage reports the invalid row")
    mixed = mrale_metrics([{"gt_mrale_total": 4, "mrale_total": 4},
                           {"gt_mrale_total": 4, "mrale_total": nan}])
    ok(abs(mixed["mae"] - 12.0) < 1e-9, "penalty is averaged with the valid rows")
    ok(abs(mixed["mae_valid_only"] - 0.0) < 1e-9, "valid-only MAE excludes the penalty")
    ok(abs(mixed["coverage"] - 0.5) < 1e-9, "coverage travels with the valid-only metric")
    nonfinite = mrale_metrics([
        {"gt_mrale_total": 4, "mrale_total": float("inf")},
        {"gt_mrale_total": 4, "mrale_total": 99},
        {"gt_mrale_total": 4, "mrale_total": 4},
    ])
    ok(abs(nonfinite["coverage"] - 1 / 3) < 1e-9,
       "infinite and out-of-range totals are invalid rather than QWK inputs")
    ok(abs(nonfinite["mae"] - 16.0) < 1e-9,
       "infinite and out-of-range totals receive the invalid-output penalty")

    # ---- NaN survives every sub-metric ------------------------------------------------------
    nan_rows = [
        {"gt_mrale_total": 6, "mrale_total": nan, "mrale_right": nan, "mrale_left": nan,
         "gt_mrale_right": 3, "gt_mrale_left": 3, "extent_right": nan, "gt_extent_right": 2,
         "mrale_total_expected": nan, "seconds": nan},
        {"gt_mrale_total": 4, "mrale_total": 4, "mrale_right": 2, "mrale_left": 2,
         "gt_mrale_right": 2, "gt_mrale_left": 2, "extent_right": 2, "gt_extent_right": 2,
         "mrale_total_expected": 4.0, "seconds": 1.0},
    ]
    try:
        metrics = mrale_metrics(nan_rows)
        ok(abs(metrics["formula_consistency"] - 1.0) < 1e-9,
           "formula consistency skips NaN rows rather than crashing")
        ok(abs(metrics["mae_right"] - INVALID_LUNG_PENALTY / 2) < 1e-9,
           "per-lung penalty is 12 for a NaN side")
        ok(metrics["extent_right_accuracy"] == 1.0, "component accuracy skips NaN")
        # Both rows (gt 6 and gt 4) fall in the mild band, one penalised at 24 and one exact.
        ok(abs(metrics["mae_band_mild"] - INVALID_TOTAL_PENALTY / 2) < 1e-9,
           "band MAE averages the penalty with the valid row in the same band")
        ok(abs(metrics["band_coverage"] - 0.5) < 1e-9,
           "band accuracy is valid-only, so its coverage is emitted beside it")
    except Exception as error:
        ok(False, f"NaN rows must not raise: {type(error).__name__}: {error}")

    ok(localization_free_metrics([{"valid": True, "seconds": nan},
                                  {"valid": False, "seconds": 2.0,
                                   "parse_error": "ValueError: x"}])["output.valid_rate"] == 0.5,
       "output integrity tolerates NaN seconds")

    # ---- classification --------------------------------------------------------------------
    metrics = classification_metrics(["Yes"] * 93 + ["No"] * 7, ["Yes"] * 100, [0.99] * 100)
    ok(abs(metrics["accuracy"] - 0.93) < 1e-9, "accuracy on a prevalence-heavy cohort")
    ok(abs(metrics["auroc"] - 0.5) < 1e-9,
       "a constant score is uninformative however high the accuracy")
    perfect = classification_metrics(["No", "No", "Yes", "Yes"],
                                     ["No", "No", "Yes", "Yes"], [0.1, 0.2, 0.8, 0.9])
    ok(abs(perfect["auroc"] - 1.0) < 1e-9, "perfect separation")
    ok(perfect["tp"] == 2 and perfect["tn"] == 2, "confusion counts are reported")
    invalid = classification_metrics(["Yes", "No"], [None, "No"], [None, 0.1])
    ok(invalid["accuracy"] == 0.5, "an invalid prediction counts as incorrect")
    nan_scores = classification_metrics(["Yes", "No"], [nan, "No"], [nan, 0.1])
    ok(nan_scores["accuracy"] == 0.5, "a NaN prediction counts as incorrect, and does not crash")
    ok(classification_metrics([], [], [])["n"] == 0, "an empty cohort is handled")

    # ---- QWK ---------------------------------------------------------------------------------
    ok(abs(quadratic_weighted_kappa([0, 1, 2], [0, 1, 2], 0, 24) - 1.0) < 1e-9,
       "identical ratings give kappa 1")
    ok(quadratic_weighted_kappa([0, 24], [24, 0], 0, 24) < 0, "reversed ratings are negative")
    ok(math.isnan(quadratic_weighted_kappa([1, 2], [float("nan"), float("inf")], 0, 24)),
       "QWK drops non-finite pairs without producing an integer sentinel")
    ok(abs(quadratic_weighted_kappa([0, 1, 2], [0, 99, 2], 0, 24) - 1.0) < 1e-9,
       "QWK drops explicitly out-of-range pairs")

    # ---- fold aggregation --------------------------------------------------------------------
    aggregate = {row["metric"]: row for row in
                 aggregate_over_folds({f: {"mae": value} for f, value in
                                       enumerate([3.0, 3.2, 3.4, 3.6, 3.8])})}
    ok(abs(aggregate["mae"]["mean"] - 3.4) < 1e-9, "fold mean")
    ok(aggregate["mae"]["n_folds"] == 5, "fold count")
    ok(abs(aggregate["mae"]["t_critical_95"] - student_t_975(4)) < 1e-9,
       "t critical for four degrees of freedom")
    ok("n" not in aggregate, "count metrics are excluded from aggregation")

    # ---- round trip ---------------------------------------------------------------------------
    row = make_prediction_row(image_key="MIDRC::a.png", mrale_total=6, extra_field="kept")
    ok(row["valid"] is True, "validity is inferred from the absence of a parse error")
    ok(row["extra"]["extra_field"] == "kept", "unknown keys land in extra")
    ok(list(row)[:4] == PREDICTION_FIELDS[:4], "field order is stable across notebooks")

    print(f"cxr_metrics self-test: {checks - len(failures)}/{checks} passed")
    for message in failures:
        print("  FAILED:", message)
    return not failures


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(0 if _self_test() else 1)
