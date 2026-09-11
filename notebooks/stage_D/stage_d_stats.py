"""
Shared statistics machinery for Stage D (NB 17-22).

Protocol section 8 in one place. Three properties this module exists to guarantee, none of
which survives being reimplemented per notebook:

1.  **Bootstrap indices are drawn once and reused.** Protocol 8.3. If each arm resampled
    independently, paired differences would carry the sum of two independent sampling errors
    and every paired interval would be too wide -- which looks conservative and is not, because
    it hides real differences. `PatientBootstrap` draws one index matrix, caches it to disk with
    a fingerprint of the patient list, and hands the same replicates to every arm.

2.  **The unit of analysis is the patient.** Protocol 8.1. This cohort has multiple radiographs
    per patient; resampling images would produce intervals that are too narrow, in the direction
    that manufactures significance.

3.  **A p-value cannot exist without its metadata.** Protocol 8.8. `PValue` refuses to
    construct without a test name, the paired unit, its family, the family size, and whether it
    is adjusted. There is no code path that emits a bare float.

Import-light and dependency-light: numpy is required, scipy is used when present and has a
tested fallback when it is not. Run the self-tests with

    python stage_d_stats.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BOOTSTRAP_REPLICATES = 2000        # protocol 8.3, raised from 1000 for paired-difference CIs
BOOTSTRAP_SEED = 42
CONFIDENCE = 0.95

# Pre-declared multiplicity families (protocol 8.6). Nothing else may be Holm-adjusted, and
# anything outside these families is exploratory and must say so in its own row.
FAMILIES = OrderedDict([
    ("F1", "E0 baseline suite vs full framework"),
    ("F2", "E1 leave-one-agent-out family"),
    ("F3", "E4 localization family"),
    ("F4", "E7 fusion family"),
    ("F5", "E9 external family"),
    ("EXPLORATORY", "E5/E6 sensitivity — unadjusted, effect sizes and CIs, never confirmatory"),
])


class WalltimeCheckpoint(RuntimeError):
    """Expected Stage D stop raised between notebook sections near an allocation limit."""


def make_session_deadline(max_session_hours):
    """Return a monotonic deadline, or None when the soft wall-time guard is disabled."""
    if max_session_hours is None:
        return None
    hours = float(max_session_hours)
    return None if hours <= 0 else time.monotonic() + hours * 3600.0


def check_session_deadline(deadline, context="Stage D analysis"):
    """Refuse to start another section after the configured soft wall-time."""
    if deadline is not None and time.monotonic() >= float(deadline):
        raise WalltimeCheckpoint(
            f"{context}: the configured soft wall-time has been reached. Stage D notebooks "
            "are deterministic and short; resubmit and rerun this notebook from the top."
        )


# ======================================================================================
# Patient-level bootstrap -- drawn once, reused everywhere
# ======================================================================================

class PatientBootstrap:
    """
    One set of patient-level resampling indices, shared by every arm and every comparison.

    `patients` is the ordered list of patient (group) identifiers. Replicate *b* is a vector of
    positions into that list, drawn with replacement. An arm's statistic for replicate *b* is
    computed over the images belonging to the drawn patients, so two arms evaluated on the same
    images see the same patients in the same replicate and their difference is properly paired.

    Cached to disk. The cache key covers the patient list, the replicate count and the seed, so
    adding a patient invalidates it rather than silently reusing indices that no longer address
    the same people.
    """

    def __init__(self, patients, n_replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
                 cache_path=None, log=print, read_only=False, expected_fingerprint=None):
        self.patients = [str(p) for p in patients]
        self.n_patients = len(self.patients)
        self.n_replicates = int(n_replicates)
        self.seed = int(seed)
        if self.n_patients == 0:
            raise ValueError("PatientBootstrap needs at least one patient")
        self.fingerprint = hashlib.sha256(
            json.dumps({"patients": self.patients, "n": self.n_replicates,
                        "seed": self.seed}, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        self.read_only = bool(read_only)
        if expected_fingerprint is not None and self.fingerprint != str(expected_fingerprint):
            raise RuntimeError(
                f"bootstrap patient fingerprint {self.fingerprint} does not match the locked "
                f"fingerprint {expected_fingerprint}. A downstream notebook may not redraw or "
                "replace NB 17's bootstrap cache.")
        self.position = {patient: index for index, patient in enumerate(self.patients)}
        self.indices = self._load_or_draw(cache_path, log)

    def _load_or_draw(self, cache_path, log):
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.is_file():
                payload = np.load(cache_path, allow_pickle=False)
                if str(payload["fingerprint"]) == self.fingerprint:
                    log(f"Bootstrap indices reused from {cache_path.name} "
                        f"({self.n_replicates} replicates, fingerprint {self.fingerprint})")
                    return payload["indices"]
                if self.read_only:
                    raise RuntimeError(
                        f"bootstrap cache {cache_path} has fingerprint "
                        f"{str(payload['fingerprint'])}, expected {self.fingerprint}. "
                        "The locked cache is read-only; rerun NB 17 instead of replacing it.")
                log(f"Bootstrap cache {cache_path.name} was drawn for a different patient set; "
                    "redrawing. Reusing it would resample patients who are no longer in the "
                    "cohort.")
            elif self.read_only:
                raise FileNotFoundError(
                    f"locked bootstrap cache {cache_path} does not exist. Run NB 17 first; "
                    "downstream notebooks are not allowed to create it.")
        elif self.read_only:
            raise ValueError("read_only=True requires cache_path")
        rng = np.random.default_rng(self.seed)
        indices = rng.integers(0, self.n_patients,
                               size=(self.n_replicates, self.n_patients), dtype=np.int32)
        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # np.savez_compressed appends ".npz" unless the name already ends with it, so the
            # temporary file has to keep that suffix or the rename below chases a path numpy
            # never wrote.
            temporary = cache_path.with_name(cache_path.stem + ".tmp.npz")
            np.savez_compressed(temporary, indices=indices, fingerprint=self.fingerprint)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(cache_path)
            log(f"Bootstrap indices drawn ONCE and cached to {cache_path.name} "
                f"({self.n_replicates} x {self.n_patients}, seed {self.seed})")
        return indices

    def patient_positions(self, patient_ids):
        """Map an array of per-observation patient ids to positions in the patient list."""
        try:
            return np.array([self.position[str(p)] for p in patient_ids], dtype=np.int64)
        except KeyError as error:
            raise KeyError(
                f"patient {error.args[0]!r} is not in the bootstrap's patient list. Every arm "
                "must be resampled over the same people, so build the bootstrap from the union "
                "of patients before computing any interval.") from error

    def replicate_weights(self, patient_positions, replicate):
        """
        How many times each observation appears in replicate `replicate`.

        Weights rather than index expansion: a patient drawn three times contributes each of
        their images three times, and weights express that without materialising a large index
        array for every statistic.
        """
        counts = np.bincount(self.indices[replicate], minlength=self.n_patients)
        return counts[patient_positions]

    def resample_statistic(self, patient_positions, statistic, n_replicates=None):
        """
        Apply `statistic(weights)` across replicates and return the resulting array.

        `statistic` receives an integer weight per observation and returns a scalar, or NaN when
        the replicate is degenerate (for example an AUROC replicate containing one class only).
        NaN replicates are dropped by `percentile_interval` rather than propagated.
        """
        total = self.n_replicates if n_replicates is None else int(n_replicates)
        out = np.empty(total, dtype=float)
        counts_matrix = np.zeros(self.n_patients, dtype=np.int64)
        for b in range(total):
            counts_matrix[:] = np.bincount(self.indices[b], minlength=self.n_patients)
            out[b] = statistic(counts_matrix[patient_positions])
        return out


def percentile_interval(draws, confidence=CONFIDENCE):
    """Percentile interval over finite replicates, with the discard count reported."""
    draws = np.asarray(draws, dtype=float)
    finite = draws[np.isfinite(draws)]
    n_discarded = int(draws.size - finite.size)
    if finite.size < 2:
        return {"ci_low": float("nan"), "ci_high": float("nan"),
                "n_replicates": int(finite.size), "n_discarded": n_discarded}
    alpha = (1.0 - confidence) / 2.0
    low, high = np.percentile(finite, [100 * alpha, 100 * (1 - alpha)])
    return {"ci_low": float(low), "ci_high": float(high),
            "n_replicates": int(finite.size), "n_discarded": n_discarded}


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return float("nan")
    return float((values * weights).sum() / total)


def weighted_auroc(labels, scores, weights):
    """
    AUROC under integer observation weights, via the rank-sum identity with tie correction.

    Returns NaN when a replicate contains only one class -- which happens, and must not be
    silently scored as 0.5.
    """
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = weights > 0
    if not keep.any():
        return float("nan")
    labels, scores, weights = labels[keep], scores[keep], weights[keep]
    n_pos = weights[labels == 1].sum()
    n_neg = weights[labels == 0].sum()
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    scores, labels, weights = scores[order], labels[order], weights[order]
    # Mid-ranks under weights: each tie group shares the average of the ranks it spans.
    ranks = np.empty(scores.size, dtype=float)
    start = 0
    cumulative = np.concatenate([[0.0], np.cumsum(weights)])
    while start < scores.size:
        stop = start
        while stop + 1 < scores.size and scores[stop + 1] == scores[start]:
            stop += 1
        group_start, group_end = cumulative[start], cumulative[stop + 1]
        ranks[start:stop + 1] = (group_start + group_end + 1.0) / 2.0
        start = stop + 1
    rank_sum_pos = float((ranks[labels == 1] * weights[labels == 1]).sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ======================================================================================
# Hypothesis tests
# ======================================================================================

def _normal_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilcoxon_signed_rank(differences):
    """
    Two-sided Wilcoxon signed-rank on paired differences (protocol 8.5).

    Zeros are dropped (Wilcoxon's own convention). Uses scipy when available; otherwise a
    normal approximation with tie and continuity correction, which agrees with scipy to about
    1e-3 on samples above ~30 pairs. Returns (statistic, p, method).
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    nonzero = differences[differences != 0]
    if nonzero.size < 1:
        return float("nan"), float("nan"), "undefined (no non-zero differences)"
    try:
        from scipy.stats import wilcoxon
        result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        return float(result.statistic), float(result.pvalue), "wilcoxon (scipy, exact/normal)"
    except Exception:
        pass

    n = nonzero.size
    order = np.argsort(np.abs(nonzero), kind="mergesort")
    magnitudes = np.abs(nonzero)[order]
    ranks = np.empty(n, dtype=float)
    start = 0
    while start < n:
        stop = start
        while stop + 1 < n and magnitudes[stop + 1] == magnitudes[start]:
            stop += 1
        ranks[start:stop + 1] = (start + stop + 2) / 2.0
        start = stop + 1
    signs = np.sign(nonzero)[order]
    w_plus = float(ranks[signs > 0].sum())
    mean = n * (n + 1) / 4.0
    tie_correction = 0.0
    start = 0
    while start < n:
        stop = start
        while stop + 1 < n and magnitudes[stop + 1] == magnitudes[start]:
            stop += 1
        size = stop - start + 1
        tie_correction += size ** 3 - size
        start = stop + 1
    variance = (n * (n + 1) * (2 * n + 1) - tie_correction / 2.0) / 24.0
    if variance <= 0:
        return w_plus, float("nan"), "undefined (zero variance)"
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(variance)
    return w_plus, float(2 * _normal_sf(z)), "wilcoxon (normal approximation, tie-corrected)"


def _midrank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=float)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(x.size, dtype=float)
    out[order] = ranks
    return out


def delong_roc_test(labels, scores_a, scores_b, positive_label=1):
    """
    DeLong's test for two correlated ROC curves (protocol 8.5), via the Sun & Xu fast algorithm.

    Returns AUROCs, their difference, the standard error, z and a two-sided p.

    **DeLong assumes independent observations.** Where a patient contributes several
    radiographs it is anti-conservative, so NB 17 reports it beside the patient-level paired
    bootstrap and flags any disagreement rather than choosing one silently.
    """
    labels = np.asarray(labels)
    positive = np.asarray([1 if l == positive_label or l is True or l == 1 else 0
                           for l in labels], dtype=int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    keep = np.isfinite(scores_a) & np.isfinite(scores_b)
    positive, scores_a, scores_b = positive[keep], scores_a[keep], scores_b[keep]

    pos_mask = positive == 1
    m, n = int(pos_mask.sum()), int((~pos_mask).sum())
    if m < 2 or n < 2:
        return {"auroc_a": float("nan"), "auroc_b": float("nan"), "delta": float("nan"),
                "standard_error": float("nan"), "z": float("nan"), "p": float("nan"),
                "n_positive": m, "n_negative": n,
                "note": "DeLong needs at least two cases per class"}

    aurocs, v01, v10 = [], [], []
    for scores in (scores_a, scores_b):
        x, y = scores[pos_mask], scores[~pos_mask]
        tx, ty, tz = _midrank(x), _midrank(y), _midrank(np.concatenate([x, y]))
        auroc = (tz[:m].sum() / (m * n) - (m + 1) / (2.0 * n))
        aurocs.append(float(auroc))
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)

    v01 = np.vstack(v01)
    v10 = np.vstack(v10)
    s01 = np.cov(v01) if v01.shape[0] > 1 else np.array([[np.var(v01, ddof=1)]])
    s10 = np.cov(v10) if v10.shape[0] > 1 else np.array([[np.var(v10, ddof=1)]])
    s = s01 / m + s10 / n
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ s @ contrast)
    delta = aurocs[0] - aurocs[1]
    if variance <= 0:
        p = 1.0 if delta == 0 else 0.0
        return {"auroc_a": aurocs[0], "auroc_b": aurocs[1], "delta": float(delta),
                "standard_error": 0.0, "z": float("inf") if delta else 0.0, "p": float(p),
                "n_positive": m, "n_negative": n,
                "note": "degenerate variance; scores may be identical"}
    standard_error = math.sqrt(variance)
    z = delta / standard_error
    return {"auroc_a": aurocs[0], "auroc_b": aurocs[1], "delta": float(delta),
            "standard_error": standard_error, "z": float(z),
            "p": float(2 * _normal_sf(abs(z))), "n_positive": m, "n_negative": n,
            "note": "assumes independent observations; see the clustered bootstrap beside it"}


def mcnemar_test(correct_a, correct_b, continuity_correction=True):
    """
    McNemar on paired binary decisions at a fixed operating point (protocol 8.5).

    Uses the exact binomial when the discordant count is small, where the chi-square
    approximation is unreliable -- which is exactly the regime a specificity comparison on a
    small external cohort lands in.
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    b = int((correct_a & ~correct_b).sum())     # a right, b wrong
    c = int((~correct_a & correct_b).sum())     # b right, a wrong
    n_discordant = b + c
    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "statistic": float("nan"),
                "p": 1.0, "method": "no discordant pairs"}
    if n_discordant < 25:
        # Exact two-sided binomial(b; n, 0.5).
        tail = sum(math.comb(n_discordant, k) for k in range(0, min(b, c) + 1))
        p = min(1.0, 2.0 * tail / (2 ** n_discordant))
        return {"b": b, "c": c, "n_discordant": n_discordant, "statistic": float(min(b, c)),
                "p": float(p), "method": "mcnemar (exact binomial)"}
    numerator = abs(b - c) - (1.0 if continuity_correction else 0.0)
    statistic = max(numerator, 0.0) ** 2 / n_discordant
    p = math.erfc(math.sqrt(statistic / 2.0)) if statistic > 0 else 1.0
    return {"b": b, "c": c, "n_discordant": n_discordant, "statistic": float(statistic),
            "p": float(min(1.0, p)),
            "method": "mcnemar (chi-square, continuity-corrected)"
            if continuity_correction else "mcnemar (chi-square)"}


def holm_bonferroni(p_values, alpha=0.05):
    """
    Holm-Bonferroni step-down within one family (protocol 8.6).

    Returns adjusted p-values in the input order, plus rejection flags. Adjusted values are
    made monotone, so a later comparison can never be reported as more significant than an
    earlier one it was outranked by.
    """
    p_values = list(p_values)
    n = len(p_values)
    if n == 0:
        return [], []
    order = sorted(range(n), key=lambda i: (math.inf if p_values[i] is None
                                            or not math.isfinite(p_values[i])
                                            else p_values[i]))
    adjusted = [None] * n
    running = 0.0
    for rank, index in enumerate(order):
        p = p_values[index]
        if p is None or not math.isfinite(p):
            adjusted[index] = None
            continue
        candidate = min(1.0, (n - rank) * p)
        running = max(running, candidate)
        adjusted[index] = running
    rejected = [a is not None and a <= alpha for a in adjusted]
    return adjusted, rejected


def tost_equivalence(differences, margin, confidence=0.90):
    """
    Two one-sided tests for equivalence on paired differences.

    NB 15 marks a leave-one-agent-out arm whose interval straddles zero as *inconclusive*. That
    is absence of evidence, not evidence of equivalence, and the distinction decides whether an
    agent may be deleted from the system. Equivalence requires the (1-2*alpha) interval to sit
    entirely inside +/- `margin`, with the margin declared before looking.

    The 90% interval is the conventional companion to alpha = 0.05 TOST.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    n = differences.size
    if n < 3:
        return {"n": n, "mean": float("nan"), "equivalent": False,
                "note": "too few pairs for an equivalence claim"}
    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / math.sqrt(n))
    if standard_error == 0:
        equivalent = abs(mean) < margin
        return {"n": n, "mean": mean, "standard_error": 0.0, "margin": float(margin),
                "ci_low": mean, "ci_high": mean, "equivalent": bool(equivalent),
                "p_lower": 0.0 if equivalent else 1.0, "p_upper": 0.0 if equivalent else 1.0,
                "note": "zero variance"}
    alpha = (1.0 - confidence) / 2.0
    try:
        from scipy.stats import t as student_t
        critical = float(student_t.ppf(1.0 - alpha, df=n - 1))
        p_lower = float(student_t.sf((mean + margin) / standard_error, df=n - 1))
        p_upper = float(student_t.sf(-(mean - margin) / standard_error, df=n - 1))
        method = "paired TOST with Student-t reference distribution"
    except Exception:
        critical = _normal_quantile(1.0 - alpha)
        p_lower = _normal_sf((mean + margin) / standard_error)
        p_upper = _normal_sf(-(mean - margin) / standard_error)
        method = "paired TOST with normal approximation (scipy unavailable)"
    ci_low, ci_high = mean - critical * standard_error, mean + critical * standard_error
    # TOST rejects non-equivalence only when BOTH one-sided tests reject.
    return {"n": n, "mean": mean, "standard_error": standard_error, "margin": float(margin),
            "ci_low": float(ci_low), "ci_high": float(ci_high),
            "p_lower": float(p_lower), "p_upper": float(p_upper),
            "p_tost": float(max(p_lower, p_upper)),
            "equivalent": bool(ci_low > -margin and ci_high < margin),
            "method": method,
            "note": f"{confidence:.0%} interval inside +/-{margin} means equivalent"}


def _normal_quantile(p):
    """Acklam's inverse normal CDF; adequate well past the precision any p-value needs."""
    if not 0 < p < 1:
        return float("nan")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def cohens_kappa(coder_a, coder_b, categories=None):
    """Cohen's kappa for the dual-coded failure taxonomy (E10b)."""
    coder_a, coder_b = list(coder_a), list(coder_b)
    if len(coder_a) != len(coder_b) or not coder_a:
        return {"kappa": float("nan"), "n": len(coder_a), "note": "unequal or empty codings"}
    categories = sorted(set(coder_a) | set(coder_b)) if categories is None else list(categories)
    index = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    matrix = np.zeros((k, k), dtype=float)
    for a, b in zip(coder_a, coder_b):
        matrix[index[a], index[b]] += 1
    n = matrix.sum()
    observed = np.trace(matrix) / n
    expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum()) / (n * n)
    if expected >= 1.0:
        return {"kappa": float("nan"), "n": int(n), "observed_agreement": float(observed),
                "note": "expected agreement is 1; kappa undefined"}
    kappa = (observed - expected) / (1 - expected)
    return {"kappa": float(kappa), "n": int(n), "observed_agreement": float(observed),
            "expected_agreement": expected,
            "interpretation": kappa_strength(kappa)}


def kappa_strength(kappa):
    if not math.isfinite(kappa):
        return "undefined"
    for threshold, label in [(0.0, "poor (worse than chance)"), (0.20, "slight"),
                             (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")]:
        if kappa < threshold or (threshold == 0.0 and kappa < 0):
            return label
    return "almost perfect"


# ======================================================================================
# p-values cannot exist without their metadata (protocol 8.8)
# ======================================================================================

@dataclass
class PValue:
    """
    A p-value and everything the manuscript must state beside it.

    Protocol 8.8: no p-value appears without its test name, the paired unit, the number of
    comparisons in its family, and whether it is adjusted. Making those constructor arguments
    means there is no way to produce a bare float and decide the rest later.
    """
    value: float
    test: str
    paired_unit: str
    family: str
    family_size: int
    adjusted: bool = False
    adjusted_value: float = None
    effect: float = None
    effect_name: str = None
    ci_low: float = None
    ci_high: float = None
    n: int = None
    note: str = ""

    def __post_init__(self):
        if not self.test or not str(self.test).strip():
            raise ValueError("a p-value needs its test name (protocol 8.8)")
        if not self.paired_unit or not str(self.paired_unit).strip():
            raise ValueError("a p-value needs its paired unit (protocol 8.1/8.8)")
        if self.family not in FAMILIES:
            raise ValueError(
                f"family {self.family!r} is not pre-declared. Allowed: {list(FAMILIES)}. "
                "Adding a family after seeing results is exactly what multiplicity control "
                "exists to prevent.")
        if self.family_size is None or int(self.family_size) < 1:
            raise ValueError("a p-value needs the size of its family (protocol 8.6/8.8)")
        if self.adjusted and self.adjusted_value is None:
            raise ValueError("adjusted=True requires the adjusted value")
        if self.family == "EXPLORATORY" and self.adjusted:
            raise ValueError(
                "exploratory comparisons are reported unadjusted by declaration (protocol 8.6); "
                "adjusting one and calling it exploratory is having it both ways")

    @property
    def reportable(self):
        """The value the manuscript prints, with adjustment applied where it exists."""
        return self.adjusted_value if self.adjusted else self.value

    def sentence(self):
        """A manuscript-ready phrase carrying every mandatory element."""
        p = self.reportable
        rendered = "n/a" if p is None or not math.isfinite(p) else (
            "< 0.001" if p < 0.001 else f"= {p:.3f}")
        pieces = [f"{self.test}", f"paired by {self.paired_unit}"]
        if self.family == "EXPLORATORY":
            pieces.append("exploratory, unadjusted")
        else:
            pieces.append(f"family {self.family}, {self.family_size} comparison"
                          f"{'s' if self.family_size != 1 else ''}, "
                          + ("Holm-adjusted" if self.adjusted else "unadjusted"))
        detail = "; ".join(pieces)
        effect = ""
        if self.effect is not None and math.isfinite(self.effect):
            effect = f"{self.effect_name or 'difference'} {self.effect:+.4f}"
            if self.ci_low is not None and math.isfinite(self.ci_low):
                effect += f" [{self.ci_low:+.4f}, {self.ci_high:+.4f}]"
            effect += "; "
        return f"{effect}p {rendered} ({detail})"

    def as_row(self):
        row = asdict(self)
        row["reportable_p"] = self.reportable
        row["sentence"] = self.sentence()
        return row


def apply_holm_within_families(p_values, alpha=0.05):
    """
    Holm-adjust each pre-declared family in place and return the same objects.

    EXPLORATORY is deliberately skipped: protocol 8.6 declares those unadjusted, and adjusting
    them would let an exploratory result borrow confirmatory standing.
    """
    by_family = {}
    for item in p_values:
        by_family.setdefault(item.family, []).append(item)
    for family, members in by_family.items():
        if family == "EXPLORATORY":
            for item in members:
                item.family_size = len(members)
                item.adjusted = False
                item.adjusted_value = None
            continue
        adjusted, _ = holm_bonferroni([m.value for m in members], alpha=alpha)
        for item, value in zip(members, adjusted):
            item.family_size = len(members)
            item.adjusted = value is not None
            item.adjusted_value = value
    return p_values


# ======================================================================================
# Number formatting -- ONE function, used by every table (protocol NB 21)
# ======================================================================================

def format_number(value, kind="metric", digits=None):
    """
    The single number-formatting function for the manuscript.

    Every table in NB 21 routes through here, so a metric cannot appear with three decimals in
    one table and two in another. `kind` selects the convention; `digits` overrides it.
    """
    if value is None:
        return "--"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "--"
    conventions = {"metric": 3, "mae": 3, "auroc": 3, "kappa": 3, "count": 0,
                   "percent": 1, "seconds": 2, "delta": 3, "p": 3}
    places = conventions.get(kind, 3) if digits is None else int(digits)
    if kind == "p":
        return "< 0.001" if value < 0.001 else f"{value:.3f}"
    if kind == "percent":
        return f"{value * 100:.{places}f}%" if abs(value) <= 1 else f"{value:.{places}f}%"
    if kind == "count":
        return f"{int(round(value)):,}"
    if kind == "delta":
        return f"{value:+.{places}f}"
    return f"{value:.{places}f}"


def format_interval(point, low, high, kind="metric"):
    """Point estimate with its interval, in the one house style."""
    if low is None or high is None or not (math.isfinite(float(low))
                                           and math.isfinite(float(high))):
        return format_number(point, kind)
    return (f"{format_number(point, kind)} "
            f"[{format_number(low, kind)}, {format_number(high, kind)}]")


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, default=str))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def provenance_stamp(notebook, extra=None):
    stamp = {"written_utc": datetime.now(timezone.utc).isoformat(), "notebook": notebook,
             "bootstrap_replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
             "confidence": CONFIDENCE, "families": dict(FAMILIES)}
    stamp.update(extra or {})
    return stamp


# ======================================================================================
# Self-tests
# ======================================================================================

def _self_test():
    checks, failures = 0, []

    def ok(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(message)

    # ---- PatientBootstrap ------------------------------------------------------------------
    patients = [f"P{i:03d}" for i in range(50)]
    boot = PatientBootstrap(patients, n_replicates=200, log=lambda *a: None)
    ok(boot.indices.shape == (200, 50), "index matrix has replicate x patient shape")
    boot2 = PatientBootstrap(patients, n_replicates=200, log=lambda *a: None)
    ok(np.array_equal(boot.indices, boot2.indices), "same seed and patients give same indices")
    boot3 = PatientBootstrap(patients + ["P999"], n_replicates=200, log=lambda *a: None)
    ok(boot3.fingerprint != boot.fingerprint, "adding a patient changes the fingerprint")

    observation_patients = [patients[i % 50] for i in range(120)]
    positions = boot.patient_positions(observation_patients)
    ok(positions.shape == (120,), "observation positions map one-to-one")
    weights = boot.replicate_weights(positions, 0)
    ok(weights.shape == (120,), "weights are per observation")
    ok(int(weights.sum()) > 0, "replicate weights are not all zero")
    try:
        boot.patient_positions(["NOT_A_PATIENT"])
        ok(False, "unknown patient must raise")
    except KeyError:
        ok(True, "unknown patient raises with an explanation")

    # A patient drawn k times contributes each of their images k times.
    single = PatientBootstrap(["A", "B"], n_replicates=1, log=lambda *a: None)
    single.indices = np.array([[0, 0]], dtype=np.int32)     # draw A twice, B never
    pos = single.patient_positions(["A", "A", "B"])
    w = single.replicate_weights(pos, 0)
    ok(list(w) == [2, 2, 0], "weights repeat every image of a resampled patient")

    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "boot.npz"
        first = PatientBootstrap(patients, n_replicates=100, cache_path=cache,
                                 log=lambda *a: None)
        ok(cache.is_file(), "bootstrap indices cached to disk")
        second = PatientBootstrap(patients, n_replicates=100, cache_path=cache,
                                  log=lambda *a: None)
        ok(np.array_equal(first.indices, second.indices), "cached indices reused verbatim")
        readonly = PatientBootstrap(patients, n_replicates=100, cache_path=cache,
                                    read_only=True, expected_fingerprint=first.fingerprint,
                                    log=lambda *a: None)
        ok(np.array_equal(first.indices, readonly.indices),
           "read-only consumers reuse the exact locked bootstrap")
        try:
            PatientBootstrap(patients[:40], n_replicates=100, cache_path=cache,
                             read_only=True, expected_fingerprint=first.fingerprint,
                             log=lambda *a: None)
            ok(False, "read-only fingerprint mismatch must raise")
        except (ValueError, RuntimeError):
            ok(True, "read-only fingerprint mismatch raises rather than overwriting")
        third = PatientBootstrap(patients[:40], n_replicates=100, cache_path=cache,
                                 log=lambda *a: None)
        ok(third.indices.shape == (100, 40), "different patient set redraws rather than reuses")

    # ---- percentile_interval ----------------------------------------------------------------
    draws = np.linspace(0, 1, 1001)
    interval = percentile_interval(draws)
    ok(abs(interval["ci_low"] - 0.025) < 1e-6 and abs(interval["ci_high"] - 0.975) < 1e-6,
       "percentile interval brackets the middle 95%")
    with_nan = np.concatenate([draws, [np.nan] * 10])
    interval = percentile_interval(with_nan)
    ok(interval["n_discarded"] == 10, "degenerate replicates are counted, not propagated")
    ok(math.isnan(percentile_interval([np.nan, np.nan])["ci_low"]),
       "an all-NaN set yields NaN rather than a fabricated interval")

    # ---- weighted statistics ----------------------------------------------------------------
    ok(abs(weighted_mean([1, 2, 3], [1, 1, 1]) - 2.0) < 1e-12, "unweighted mean")
    ok(abs(weighted_mean([1, 2, 3], [3, 0, 0]) - 1.0) < 1e-12, "weights select observations")
    ok(math.isnan(weighted_mean([1, 2], [0, 0])), "zero total weight yields NaN")

    labels = [0, 0, 1, 1]
    ok(abs(weighted_auroc(labels, [0.1, 0.2, 0.8, 0.9], [1, 1, 1, 1]) - 1.0) < 1e-12,
       "perfect separation gives AUROC 1")
    ok(abs(weighted_auroc(labels, [0.9, 0.8, 0.2, 0.1], [1, 1, 1, 1]) - 0.0) < 1e-12,
       "reversed separation gives AUROC 0")
    ok(abs(weighted_auroc(labels, [0.5] * 4, [1, 1, 1, 1]) - 0.5) < 1e-12,
       "all-tied scores give AUROC 0.5")
    ok(math.isnan(weighted_auroc([1, 1, 1], [0.1, 0.2, 0.3], [1, 1, 1])),
       "one-class replicate gives NaN, not 0.5")
    ok(abs(weighted_auroc([0, 1, 1], [0.1, 0.8, 0.9], [2, 1, 1])
           - weighted_auroc([0, 0, 1, 1], [0.1, 0.1, 0.8, 0.9], [1, 1, 1, 1])) < 1e-12,
       "a weight of 2 equals a duplicated observation")

    try:
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        s = rng.random(200) + y * 0.4
        ok(abs(weighted_auroc(y, s, np.ones(200)) - roc_auc_score(y, s)) < 1e-9,
           "weighted AUROC matches sklearn on unit weights")
    except ImportError:
        pass

    # ---- Wilcoxon ---------------------------------------------------------------------------
    rng = np.random.default_rng(1)
    shifted = rng.normal(0.6, 1.0, 80)
    _, p_shift, _ = wilcoxon_signed_rank(shifted)
    ok(p_shift < 0.01, "Wilcoxon detects a clear shift")
    centred = np.concatenate([np.linspace(-1, 1, 81)])
    _, p_centred, _ = wilcoxon_signed_rank(centred)
    ok(p_centred > 0.5, "Wilcoxon does not reject a symmetric sample")
    _, p_zero, method = wilcoxon_signed_rank(np.zeros(10))
    ok(math.isnan(p_zero) and "undefined" in method, "all-zero differences are undefined")

    # ---- DeLong -----------------------------------------------------------------------------
    rng = np.random.default_rng(2)
    y = np.concatenate([np.zeros(120), np.ones(120)])
    good = np.concatenate([rng.normal(0, 1, 120), rng.normal(1.6, 1, 120)])
    poor = np.concatenate([rng.normal(0, 1, 120), rng.normal(0.2, 1, 120)])
    result = delong_roc_test(y, good, poor)
    ok(result["auroc_a"] > result["auroc_b"], "DeLong orders the better model first")
    ok(result["p"] < 0.001, "DeLong detects a large AUROC gap")
    identical = delong_roc_test(y, good, good.copy())
    ok(abs(identical["delta"]) < 1e-12, "identical scores give zero difference")
    ok(identical["p"] > 0.99 or identical["standard_error"] == 0,
       "identical scores are not significant")
    try:
        from sklearn.metrics import roc_auc_score
        ok(abs(result["auroc_a"] - roc_auc_score(y, good)) < 1e-9,
           "DeLong's AUROC matches sklearn")
    except ImportError:
        pass
    degenerate = delong_roc_test([1, 1, 1, 1], [1, 2, 3, 4], [4, 3, 2, 1])
    ok(math.isnan(degenerate["p"]), "single-class input yields NaN rather than a p-value")

    # ---- McNemar ----------------------------------------------------------------------------
    a = [True] * 40 + [False] * 60
    b = [True] * 10 + [False] * 30 + [True] * 50 + [False] * 10
    result = mcnemar_test(a, b)
    ok(result["n_discordant"] == result["b"] + result["c"], "discordant count is b + c")
    ok("mcnemar" in result["method"], "method recorded")
    small = mcnemar_test([True, False, True], [False, False, True])
    ok("exact" in small["method"], "small discordant counts use the exact binomial")
    tied = mcnemar_test([True, False], [True, False])
    ok(tied["p"] == 1.0 and tied["n_discordant"] == 0, "no discordant pairs gives p = 1")
    big_a = [True] * 100 + [False] * 100
    big_b = [False] * 60 + [True] * 40 + [False] * 100
    big = mcnemar_test(big_a, big_b)
    ok("chi-square" in big["method"] and big["p"] < 0.001,
       "large discordant counts use the corrected chi-square")

    # ---- Holm -------------------------------------------------------------------------------
    adjusted, rejected = holm_bonferroni([0.01, 0.02, 0.03, 0.04])
    ok(adjusted == sorted(adjusted), "Holm adjusted values are monotone")
    ok(abs(adjusted[0] - 0.04) < 1e-12, "smallest p multiplied by the family size")
    ok(all(a >= p for a, p in zip(adjusted, [0.01, 0.02, 0.03, 0.04])),
       "adjustment never decreases a p-value")
    adjusted, _ = holm_bonferroni([0.001])
    ok(abs(adjusted[0] - 0.001) < 1e-12, "a family of one is unchanged")
    adjusted, rejected = holm_bonferroni([0.6, 0.7])
    ok(not any(rejected), "large p-values are not rejected")
    adjusted, _ = holm_bonferroni([0.01, float("nan")])
    ok(adjusted[1] is None, "a missing p-value stays missing rather than becoming 1.0")
    ok(holm_bonferroni([]) == ([], []), "an empty family is handled")

    # ---- TOST -------------------------------------------------------------------------------
    tight = tost_equivalence(np.random.default_rng(3).normal(0.0, 0.05, 200), margin=0.5)
    ok(tight["equivalent"], "a tightly centred difference is equivalent within a wide margin")
    ok("Student-t" in tight["method"], "TOST records the finite-sample Student-t method")
    wide = tost_equivalence(np.random.default_rng(4).normal(0.0, 3.0, 30), margin=0.1)
    ok(not wide["equivalent"], "a noisy difference is not equivalent within a narrow margin")
    offset = tost_equivalence(np.random.default_rng(5).normal(1.0, 0.05, 200), margin=0.5)
    ok(not offset["equivalent"], "a real offset is not equivalence")
    ok(not tost_equivalence([0.1, 0.2], margin=1.0)["equivalent"],
       "two pairs cannot support an equivalence claim")

    # ---- Cohen's kappa ----------------------------------------------------------------------
    perfect = cohens_kappa(["a", "b", "c", "a"], ["a", "b", "c", "a"])
    ok(abs(perfect["kappa"] - 1.0) < 1e-12, "identical codings give kappa 1")
    ok(cohens_kappa(["a", "b"], ["b", "a"])["kappa"] < 0, "systematic disagreement is negative")
    ok(cohens_kappa([], [])["n"] == 0, "empty codings handled")
    ok(kappa_strength(0.85) == "almost perfect", "kappa strength labels")
    ok(kappa_strength(0.5) == "moderate", "kappa strength mid range")

    # ---- PValue metadata --------------------------------------------------------------------
    p = PValue(value=0.004, test="paired bootstrap", paired_unit="patient", family="F2",
               family_size=6, effect=-0.31, effect_name="MAE difference",
               ci_low=-0.55, ci_high=-0.08, n=2400)
    ok("paired by patient" in p.sentence(), "sentence names the paired unit")
    ok("family F2" in p.sentence(), "sentence names the family")
    ok("unadjusted" in p.sentence(), "sentence states adjustment status")
    ok("MAE difference" in p.sentence(), "sentence carries the effect size")
    for bad in [dict(test=""), dict(paired_unit=""), dict(family="F9"), dict(family_size=0)]:
        arguments = dict(value=0.01, test="t", paired_unit="patient", family="F1",
                         family_size=3)
        arguments.update(bad)
        try:
            PValue(**arguments)
            ok(False, f"invalid metadata must raise: {bad}")
        except ValueError:
            ok(True, f"invalid metadata raises: {list(bad)[0]}")
    try:
        PValue(value=0.01, test="t", paired_unit="patient", family="F1", family_size=3,
               adjusted=True)
        ok(False, "adjusted without a value must raise")
    except ValueError:
        ok(True, "adjusted without a value raises")
    try:
        PValue(value=0.01, test="t", paired_unit="patient", family="EXPLORATORY",
               family_size=3, adjusted=True, adjusted_value=0.03)
        ok(False, "adjusted exploratory must raise")
    except ValueError:
        ok(True, "an adjusted exploratory p-value raises")

    family = [PValue(value=v, test="paired bootstrap", paired_unit="patient", family="F2",
                     family_size=1) for v in [0.01, 0.02, 0.6]]
    exploratory = [PValue(value=0.02, test="paired bootstrap", paired_unit="patient",
                          family="EXPLORATORY", family_size=1)]
    apply_holm_within_families(family + exploratory)
    ok(all(item.family_size == 3 for item in family), "family size set from membership")
    ok(family[0].adjusted and family[0].adjusted_value >= 0.01, "F2 members are Holm-adjusted")
    ok(not exploratory[0].adjusted, "exploratory members stay unadjusted")
    ok(exploratory[0].reportable == 0.02, "an unadjusted p reports its raw value")
    ok(family[0].reportable == family[0].adjusted_value, "an adjusted p reports the adjustment")

    # ---- formatting -------------------------------------------------------------------------
    ok(format_number(0.123456) == "0.123", "metrics use three decimals")
    ok(format_number(None) == "--", "missing values render as a dash")
    ok(format_number(float("nan")) == "--", "NaN renders as a dash")
    ok(format_number(1234, kind="count") == "1,234", "counts are grouped")
    ok(format_number(0.0004, kind="p") == "< 0.001", "small p-values render as a bound")
    ok(format_number(-0.25, kind="delta") == "-0.250", "deltas keep their sign")
    ok(format_number(0.25, kind="delta") == "+0.250", "positive deltas show a plus")
    ok(format_number(0.5, kind="percent") == "50.0%", "fractions render as percentages")
    ok(format_interval(3.68, 3.32, 4.09) == "3.680 [3.320, 4.090]", "interval house style")
    ok(format_interval(3.68, None, None) == "3.680", "a point estimate without an interval")

    # ---- durable writes and allocation boundary --------------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "state.json"
        write_json_atomic(target, {"ready": True})
        ok(json.loads(target.read_text(encoding="utf-8"))["ready"],
           "atomic JSON state round-trips")
        ok(not target.with_suffix(".json.tmp").exists(),
           "atomic JSON temporary file is replaced")
    check_session_deadline(time.monotonic() + 60, "future deadline")
    try:
        check_session_deadline(time.monotonic() - 1, "test deadline")
        deadline_stopped = False
    except WalltimeCheckpoint:
        deadline_stopped = True
    ok(deadline_stopped, "soft wall-time stops between Stage D sections")

    print(f"stage_d_stats self-test: {checks - len(failures)}/{checks} passed")
    for message in failures:
        print("  FAILED:", message)
    return not failures


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
