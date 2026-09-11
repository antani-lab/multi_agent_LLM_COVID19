"""
Shared reasoner machinery for Stage C (NB 15, NB 16, and anything later that generates).

This exists for the same reason `cxr_metrics.py` does: NB 10 was built by transforming NB 09's
cells, and a bulk rename silently rewrote a path that was supposed to READ NB 09's output. A
second copy of prompt rendering, JSON parsing and journal handling would be the same accident
waiting to happen, except that its failure mode -- two notebooks parsing model output slightly
differently -- would show up as a metric difference nobody could explain.

Everything here is import-light. `torch` and `transformers` are imported lazily inside the
functions that need them, so the pure logic (JSON balance detection, parsing, clipping, journal
fingerprinting) can be unit-tested on a machine with neither installed:

    python stage_c_reasoner.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

MRALE_MIN, MRALE_MAX = 0, 24
LUNG_MIN, LUNG_MAX = 0, 12

DEFAULT_SYSTEM_PROMPT = (
    "You are a thoracic radiologist scoring a frontal chest radiograph for the modified "
    "Radiographic Assessment of Lung Edema (mRALE) score and for COVID-19.\n"
    "For EACH lung: extent is 0-4 (0 none, 1 <25%, 2 25-50%, 3 50-75%, 4 >75%) and density "
    "is 0-3 (0 none, 1 hazy, 2 moderate, 3 dense). That lung's score is extent x density, so "
    "0-12. The total is right + left, so 0-24.\n"
    "Answer ONLY with a single JSON object and nothing else."
)

DEFAULT_RESPONSE_SCHEMA = (
    '{"extent_right": int 0-4, "density_right": int 0-3, "extent_left": int 0-4, '
    '"density_left": int 0-3, "mrale_right": int 0-12, "mrale_left": int 0-12, '
    '"mrale_total": int 0-24, "covid_positive": "Yes" or "No", '
    '"covid_confidence": float 0-1 for the stated call, '
    '"agents_used": list of agent names you relied on, '
    '"rationale": string of at most 30 words}')

IMAGE_PLACEHOLDER_TOKENS = ["<start_of_image>", "<image_soft_token>", "<image>", "<img>",
                            "<|image_pad|>", "<|vision_start|>"]

MESSAGE_SCHEMAS = ["bare", "valued", "file_uri", "system_as_string",
                   "system_folded_into_user", "manual_vision_tokens"]
TOKEN_SCORE_PROBE_VERSION = "covid_yes_no_prefix_v1"


# ======================================================================================
# JSON structure -- pure functions, unit-tested below
# ======================================================================================

def scan_json_objects(text):
    """
    Walk `text` once and report top-level JSON object structure.

    Returns (n_complete_objects, first_object_end_index, depth_at_end). String contents and
    escapes are respected, so a brace inside a rationale does not confuse the count.

    This is the measurement behind protocol item E6-Q: the tested notebooks let the model keep
    generating after its object closed, so it emitted the same object several times plus a
    stray control token. Parsing recovered the first copy, which is why the metrics were valid
    and the problem went unnoticed -- but token counts, latency and any self-consistency arm
    were measuring repetition rather than the answer.
    """
    depth, in_string, escape = 0, False, False
    complete, first_end = 0, -1
    for index, character in enumerate(text):
        if escape:
            escape = False
            continue
        if character == "\\":
            escape = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            if depth > 0:
                depth -= 1
                if depth == 0:
                    complete += 1
                    if first_end < 0:
                        first_end = index
    return complete, first_end, depth


def first_object_is_closed(text):
    """True once the first top-level object has closed. Drives the stopping criterion."""
    start = text.find("{")
    if start < 0:
        return False
    complete, _, _ = scan_json_objects(text[start:])
    return complete >= 1


CONTROL_TOKEN_PATTERN = re.compile(r"<\|?[a-zA-Z_][\w|]{0,24}\|?>")


def text_outside_first_object(text):
    """Return all text before and after the first complete JSON object."""
    text = str(text)
    start = text.find("{")
    if start < 0:
        return text
    complete, end, _ = scan_json_objects(text[start:])
    if complete < 1 or end < 0:
        return text
    absolute_end = start + end + 1
    return text[:start] + text[absolute_end:]


def control_tokens_outside_json(text):
    """Detect stray model/control tokens outside the first answer object."""
    return sorted(set(CONTROL_TOKEN_PATTERN.findall(text_outside_first_object(text))))


def extract_json_object(text):
    """Recover the first top-level JSON object, tolerating fences and <think> blocks."""
    cleaned = re.sub(r"<think>.*?</think>", " ", str(text).strip(), flags=re.DOTALL)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(obj, dict):
        raise ValueError("Top-level value is not an object")
    return obj


def clipped_int(value, low, high):
    """Return (value_in_range_or_None, was_clipped)."""
    if value is None or isinstance(value, bool):
        return None, False
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None, False
    clipped = max(low, min(high, number))
    return clipped, clipped != number


def strict_int(value, low, high):
    """Parse an integer without repairing an invalid model output."""
    if value is None or isinstance(value, bool):
        return None, "missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, "not_numeric"
    if not math.isfinite(number) or not number.is_integer():
        return None, "not_integer"
    number = int(number)
    if not low <= number <= high:
        return None, f"out_of_range_{low}_{high}"
    return number, None


def parse_reasoner_output(text, roster=()):
    """
    Parse one reasoner answer into prediction fields.

    Returns (fields, parse_error, notes). A `parse_error` is NOT a missing value to be imputed:
    callers must mark the row invalid so `cxr_metrics` applies the same fixed penalty every
    other arm receives. Otherwise an arm can buy accuracy by declining to answer hard cases.
    """
    notes = {}
    try:
        obj = extract_json_object(text)
    except Exception as error:
        return {}, f"{type(error).__name__}: {error}", notes

    fields, violations = {}, []
    component_limits = {"extent_right": 4, "density_right": 3,
                        "extent_left": 4, "density_left": 3}
    for name, high in component_limits.items():
        fields[name], error = strict_int(obj.get(name), 0, high)
        if error:
            violations.append(f"{name}:{error}")

    for side in ["right", "left"]:
        name = f"mrale_{side}"
        fields[name], error = strict_int(obj.get(name), LUNG_MIN, LUNG_MAX)
        if error:
            violations.append(f"{name}:{error}")
        extent, density = fields.get(f"extent_{side}"), fields.get(f"density_{side}")
        if extent is not None and density is not None and fields[name] is not None:
            expected = extent * density
            if fields[name] != expected:
                violations.append(f"{name}:formula_mismatch_expected_{expected}")

    fields["mrale_total"], error = strict_int(obj.get("mrale_total"), MRALE_MIN, MRALE_MAX)
    if error:
        violations.append(f"mrale_total:{error}")
    if fields.get("mrale_right") is not None and fields.get("mrale_left") is not None \
            and fields.get("mrale_total") is not None:
        expected_total = fields["mrale_right"] + fields["mrale_left"]
        notes["formula_consistent"] = fields["mrale_total"] == expected_total
        if not notes["formula_consistent"]:
            violations.append(f"mrale_total:formula_mismatch_expected_{expected_total}")

    covid = obj.get("covid_positive")
    if isinstance(covid, bool):
        covid = "Yes" if covid else "No"
    if isinstance(covid, str):
        covid = covid.strip().capitalize()
    fields["covid_pred"] = covid if covid in {"Yes", "No"} else None
    if fields["covid_pred"] is None:
        violations.append("covid_positive:missing_or_invalid")

    try:
        confidence = float(obj.get("covid_confidence"))
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None and 0.0 <= confidence <= 1.0 and fields["covid_pred"]:
        # The model states confidence in ITS OWN call; the metric wants P(positive).
        fields["covid_score"] = confidence if fields["covid_pred"] == "Yes" \
            else 1.0 - confidence
    else:
        fields["covid_score"] = None
        violations.append("covid_confidence:missing_or_out_of_range")
        if confidence is not None:
            notes["confidence_out_of_range"] = confidence

    cited = obj.get("agents_used")
    cited = [str(a) for a in cited] if isinstance(cited, list) else []
    notes["agents_cited"] = cited
    notes["agents_fabricated"] = sorted(
        {a for a in cited if a not in set(roster)
         and a.strip().upper() not in {"NONE", "IMAGE", "N/A", ""}})
    notes["rationale"] = str(obj.get("rationale", ""))[:400]
    notes["schema_violations"] = violations
    notes["complete_fields"] = not violations

    # Do not repair schema violations into apparently valid scores. Protocol 7.4 assigns the
    # fixed invalid-output penalty; leaving the affected endpoint as None makes that policy
    # structural and prevents an out-of-range 40 from becoming a deceptively accurate 24.
    mrale_violations = [v for v in violations if v.startswith(
        ("extent_", "density_", "mrale_"))]
    covid_violations = [v for v in violations if v.startswith("covid_")]
    if mrale_violations:
        for name in [*component_limits, "mrale_right", "mrale_left", "mrale_total"]:
            fields[name] = None
    if covid_violations:
        fields["covid_pred"], fields["covid_score"] = None, None
    error = ("SchemaError: " + "; ".join(violations)) if violations else None
    return fields, error, notes


# ======================================================================================
# Self-consistency aggregation (E6-C)
# ======================================================================================

def aggregate_samples(samples):
    """
    Combine several sampled answers into one, per protocol E6-C: median for mRALE, majority
    vote for COVID. Invalid samples are dropped; if none survive the result is invalid, which
    is the honest outcome rather than silently returning the one parseable sample.
    """
    usable = [s for s in samples if s.get("mrale_total") is not None]
    if not usable:
        return {}, "ValueError: no sample produced a usable answer", {"n_samples": len(samples)}

    # Choose the valid sample nearest the median TOTAL and keep all of its regional fields.
    # Taking independent medians of extent, density and their products can create an impossible
    # aggregate (for example, right != extent_right * density_right) even when every sampled
    # answer was internally valid. E6-C uses odd n={1,3,5}, so the median total is represented
    # by at least one sample; this medoid rule also stays coherent for an accidental even n.
    totals = sorted(float(s["mrale_total"]) for s in usable)
    middle = len(totals) // 2
    median_total = (totals[middle] if len(totals) % 2
                    else (totals[middle - 1] + totals[middle]) / 2.0)
    representative = min(enumerate(usable),
                         key=lambda item: (abs(float(item[1]["mrale_total"]) - median_total),
                                           item[0]))[1]
    fields = {key: representative.get(key) for key in
              ["mrale_total", "mrale_right", "mrale_left",
               "extent_right", "density_right", "extent_left", "density_left"]}

    votes = [s.get("covid_pred") for s in usable if s.get("covid_pred") in {"Yes", "No"}]
    if votes:
        yes = sum(1 for v in votes if v == "Yes")
        fields["covid_pred"] = "Yes" if yes * 2 > len(votes) else "No"
        # Vote share is a better-behaved score than an average of self-reported confidences.
        fields["covid_score"] = yes / len(votes)
    else:
        fields["covid_pred"], fields["covid_score"] = None, None

    sample_totals = [float(s["mrale_total"]) for s in usable]
    mean = sum(sample_totals) / len(sample_totals)
    variance = sum((t - mean) ** 2 for t in sample_totals) / len(sample_totals)
    notes = {"n_samples": len(samples), "n_usable": len(usable),
             "sample_totals": sample_totals, "sample_sd": math.sqrt(variance),
             "median_total_target": median_total,
             "covid_vote_share": fields.get("covid_score")}
    return fields, None, notes


# ======================================================================================
# Journals -- per-item resume with configuration fingerprinting
# ======================================================================================

def fingerprint(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


class WalltimeCheckpoint(RuntimeError):
    """Expected stop raised only between journaled images near the allocation limit."""


def make_session_deadline(max_session_hours):
    """Return a monotonic deadline, or None when the soft wall-time guard is disabled."""
    if max_session_hours is None:
        return None
    hours = float(max_session_hours)
    return None if hours <= 0 else time.monotonic() + hours * 3600.0


def check_session_deadline(deadline, context="Stage C generation"):
    """Stop at a safe item boundary rather than during the next generation or journal write."""
    if deadline is not None and time.monotonic() >= float(deadline):
        raise WalltimeCheckpoint(
            f"{context}: the configured soft wall-time has been reached. Every completed "
            "image is durable in its journal. End this allocation, resubmit, and rerun the "
            "notebook from the top; cached images will be skipped."
        )


def repair_jsonl_tail(path, log=print):
    """Remove only a malformed final JSONL record left by an interrupted append.

    Interior corruption is blocking: silently skipping it could change a study denominator.
    A malformed final record is different—it is precisely the one item whose append was
    interrupted, so truncating back to its byte offset makes the next append safe.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False

    file_size = path.stat().st_size
    with path.open("rb+") as handle:
        line_number = 0
        while True:
            start = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            line_number += 1
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
                if "image_key" not in row:
                    raise ValueError("missing image_key")
            except Exception as exc:
                if handle.tell() != file_size:
                    raise ValueError(
                        f"Invalid interior JSONL record at {path}:{line_number}; refusing to "
                        "silently alter the journal."
                    ) from exc
                handle.seek(start)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                log(f"    repaired interrupted final record in {path.name}; that one image "
                    "will be regenerated")
                return True
    return False


def open_journal(journal_dir, name, config_fingerprint, force=False, log=print):
    """
    Return (journal_path, stamp_path). Retires the journal if the configuration changed.

    A resume that reuses stale work is worse than no resume, because the resulting table looks
    fine. The stale file is renamed rather than deleted so the previous run remains inspectable.
    """
    journal_dir = Path(journal_dir)
    journal_dir.mkdir(parents=True, exist_ok=True)
    path = journal_dir / f"{name}.jsonl"
    stamp_path = journal_dir / f"{name}.fingerprint.json"

    if force:
        for candidate in (path, stamp_path):
            if candidate.is_file():
                candidate.unlink()
        log(f"    forced recompute: {name}")
    elif path.is_file():
        stored = (json.loads(stamp_path.read_text(encoding="utf-8")).get("fingerprint")
                  if stamp_path.is_file() else None)
        if stored != config_fingerprint:
            label = stored or "unstamped"
            retired = journal_dir / f"{name}.stale-{label}.jsonl"
            path.replace(retired)
            log(f"    configuration changed since {label}; retired {retired.name} and "
                "starting fresh (reusing it would report answers from a different prompt)")

    if path.is_file():
        repair_jsonl_tail(path, log=log)

    write_json_atomic(stamp_path, {"fingerprint": config_fingerprint, "config": name,
                                   "written_utc": datetime.now(timezone.utc).isoformat()})
    return path, stamp_path


def write_json_atomic(path, payload):
    """Write via .tmp + replace. A crash mid-write must not leave a file that reads as valid."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, default=str))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def journal_keys(path):
    """
    Image keys already present in a journal, as a set of plain strings.

    NB 11 shipped a resume that compared a string against tuple keys, so it never matched: it
    printed a reassuring "resuming with N cached" and then rescored everything. Returning bare
    strings removes that trap at the source.
    """
    path = Path(path)
    if not path.is_file():
        return set()
    repair_jsonl_tail(path, log=lambda *_: None)
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                keys.add(str(json.loads(line)["image_key"]))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL record at {path}:{line_number}") from exc
    return keys


def active_fold_journals(journal_dir, config_name):
    """Return only exact active `<config>__fold<integer>.jsonl` journals.

    A broad glob also matches retired files such as
    `<config>__fold0.stale-deadbeef.jsonl`, silently mixing answers produced by different
    prompts. Match the complete filename instead.
    """
    journal_dir = Path(journal_dir)
    pattern = re.compile(rf"^{re.escape(str(config_name))}__fold(\d+)\.jsonl$")
    matched = []
    if not journal_dir.is_dir():
        return matched
    for path in journal_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_file():
            matched.append((int(match.group(1)), path))
    return [path for _, path in sorted(matched)]


# ======================================================================================
# Model loading and generation -- lazy imports, so the above stays testable
# ======================================================================================

def resolve_dtype():
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def model_load_kwargs(dtype, **extra):
    """`torch_dtype` was renamed `dtype`; pass whichever this transformers accepts."""
    import inspect
    from transformers import PreTrainedModel
    try:
        params = inspect.signature(PreTrainedModel.from_pretrained).parameters
        key = "dtype" if "dtype" in params else "torch_dtype"
    except Exception:
        key = "torch_dtype"
    return {key: dtype, **extra}


def model_device(model):
    import torch
    for parameter in model.parameters():
        if parameter.device.type not in {"meta", "cpu"}:
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vision_token_string(processor):
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    for attribute in ["image_token", "boi_token"]:
        token = getattr(processor, attribute, None) or getattr(tokenizer, attribute, None)
        if isinstance(token, str) and token:
            if "<|vision_start|>" in vocab and "<|vision_end|>" in vocab:
                return f"<|vision_start|>{token}<|vision_end|>"
            return token
    if "<|image_pad|>" in vocab:
        return "<|vision_start|><|image_pad|><|vision_end|>"
    for candidate in ["<start_of_image>", "<image_soft_token>", "<image>", "<img>"]:
        if candidate in vocab:
            return candidate
    return None


def build_messages(schema, system_prompt, user_prompt, image_path, processor):
    path = str(image_path)
    if schema == "bare":
        return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": user_prompt}]}]
    if schema == "valued":
        return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": path},
                                             {"type": "text", "text": user_prompt}]}]
    if schema == "file_uri":
        return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": f"file://{path}"},
                                             {"type": "text", "text": user_prompt}]}]
    if schema == "system_as_string":
        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image", "image": path},
                                             {"type": "text", "text": user_prompt}]}]
    if schema == "system_folded_into_user":
        return [{"role": "user", "content": [
            {"type": "image", "image": path},
            {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"}]}]
    if schema == "manual_vision_tokens":
        token = vision_token_string(processor)
        if not token:
            raise ValueError("no vision token discoverable for this model")
        return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [
                    {"type": "text", "text": f"{token}\n{user_prompt}"}]}]
    raise ValueError(f"unknown message schema {schema}")


def apply_chat_template(processor, spec, messages):
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    if spec.get("disable_thinking"):
        kwargs["enable_thinking"] = False
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def render_prompt(processor, spec, system_prompt, user_prompt, image_path, variant_registry):
    """
    Render the chat prompt, ENSURING it contains this model's image placeholder.

    `{"type": "image"}` with no value suffices for MedGemma and Qwen3.5, but some templates
    (notably Qwen2.5-VL derivatives such as NV-Reason-CXR-3B) emit nothing for it. The prompt
    then contains no vision token, the image is never attended to, and the model politely asks
    for an image while every JSON parse fails. That produced a 20-hour NB 07 run at
    valid_rate = 0.000. Variants are tried until one yields a placeholder, and the winner is
    recorded in `variant_registry` so the choice is auditable and reused.
    """
    key = spec.get("_label", spec.get("model_id", "model"))
    pinned = variant_registry.get(key)
    pinned = pinned if pinned in MESSAGE_SCHEMAS else None
    order = ([pinned] + [s for s in MESSAGE_SCHEMAS if s != pinned]) if pinned \
        else list(MESSAGE_SCHEMAS)

    first_text, errors = None, {}
    for name in order:
        try:
            text = apply_chat_template(processor, spec, build_messages(
                name, system_prompt, user_prompt, image_path, processor))
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        if first_text is None:
            first_text = text
        if any(token in text for token in IMAGE_PLACEHOLDER_TOKENS):
            variant_registry[key] = name
            return text
    if first_text is None:
        raise RuntimeError(f"Could not render a prompt for {key}: {errors}")
    variant_registry[key] = f"NO_PLACEHOLDER({errors})"
    return first_text


def load_cxr(path, max_pixels=1_400_000):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        if image.width * image.height > max_pixels:
            scale = (max_pixels / (image.width * image.height)) ** 0.5
            image = image.resize((max(1, int(image.width * scale)),
                                  max(1, int(image.height * scale))), Image.BICUBIC)
        return image.copy()


def prepare_inputs(processor, prompt_text, image_path, target_device, dtype):
    inputs = processor(text=prompt_text, images=load_cxr(image_path), return_tensors="pt")
    return {key: (value.to(device=target_device, dtype=dtype) if value.is_floating_point()
                  else value.to(target_device)) for key, value in inputs.items()}


def json_stopping_criteria(tokenizer, prompt_length):
    """Stop as soon as the first top-level JSON object closes (protocol E6-Q)."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    class BalancedJsonStop(StoppingCriteria):
        def __init__(self, tokenizer, prompt_length):
            self.tokenizer, self.prompt_length = tokenizer, prompt_length

        def __call__(self, input_ids, scores, **kwargs):
            text = self.tokenizer.decode(input_ids[0, self.prompt_length:],
                                         skip_special_tokens=True)
            return first_object_is_closed(text)

    return StoppingCriteriaList([BalancedJsonStop(tokenizer, prompt_length)])


def load_reasoner(label, spec, fold, model_revisions=None, log=print):
    """
    Load a reasoner, applying its per-fold LoRA adapter if the spec declares one.

    A missing adapter raises. Falling back to the base model would put a differently-trained
    model into the results table under the LoRA label -- the kind of error that survives to
    publication because nothing about the output looks wrong.
    """
    from transformers import AutoProcessor
    model_revisions = model_revisions or {}
    model_id = spec["model_id"]
    revision = model_revisions.get(model_id)
    kwargs = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision

    processor = AutoProcessor.from_pretrained(model_id, **kwargs)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    loader = spec.get("loader", "auto")
    if loader == "image_text_to_text":
        from transformers import AutoModelForImageTextToText as ModelClass
    elif loader == "multimodal_lm":
        try:
            from transformers import AutoModelForMultimodalLM as ModelClass
        except Exception:
            from transformers import AutoModelForImageTextToText as ModelClass
    else:
        try:
            from transformers import AutoModelForImageTextToText as ModelClass
        except Exception:
            from transformers import AutoModelForVision2Seq as ModelClass

    dtype = resolve_dtype()
    model = ModelClass.from_pretrained(model_id, device_map="auto", low_cpu_mem_usage=True,
                                       **model_load_kwargs(dtype), **kwargs)

    adapter_note = "base model, no adapter"
    if spec.get("adapter"):
        from peft import PeftModel
        adapter_dir = Path(str(spec["adapter"]).format(fold=fold))
        if not (adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"{label} expects a fold-{fold} LoRA adapter at {adapter_dir}, which does not "
                "exist. Run the corresponding Stage B notebook, or drop this candidate. "
                "Falling back to the base model would mislabel the arm.")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
        adapter_note = f"LoRA adapter fold {fold}"

    model.config.use_cache = True
    model.eval()
    log(f"    loaded {label}: {model_id} ({adapter_note}, dtype {dtype})")
    return model, processor, revision


def release(*objects):
    import gc
    import torch
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_json(model, processor, spec, image_path, user_prompt, variant_registry,
                  system_prompt=DEFAULT_SYSTEM_PROMPT, max_new_tokens=384,
                  temperature=0.0, top_p=1.0, use_json_stop=True, seed=None):
    """
    One generation. Returns (text, n_completion_tokens, n_prompt_tokens, n_json_objects).

    `temperature = 0` means greedy (`do_sample=False`), which is what the primary results use;
    the E6-D grid exists to show the conclusion is not an artefact of one lucky setting.
    """
    import torch
    with torch.inference_mode():
        prompt_text = render_prompt(processor, spec, system_prompt, user_prompt, image_path,
                                    variant_registry)
        target_device = model_device(model)
        inputs = prepare_inputs(processor, prompt_text, image_path, target_device,
                                resolve_dtype())
        prompt_length = inputs["input_ids"].shape[-1]

        generation = {"max_new_tokens": max_new_tokens,
                      "pad_token_id": processor.tokenizer.pad_token_id,
                      "eos_token_id": processor.tokenizer.eos_token_id,
                      "use_cache": True}
        if temperature and temperature > 0:
            if seed is not None:
                torch.manual_seed(seed)
            generation.update({"do_sample": True, "temperature": float(temperature),
                               "top_p": float(top_p)})
        else:
            generation["do_sample"] = False
        if use_json_stop:
            generation["stopping_criteria"] = json_stopping_criteria(
                processor.tokenizer, prompt_length)

        output_ids = model.generate(**inputs, **generation)
        completion = output_ids[0, prompt_length:]
        text = processor.decode(completion, skip_special_tokens=True).strip()
        n_objects, _, _ = scan_json_objects(text)
        return text, int(completion.shape[-1]), int(prompt_length), n_objects


def covid_token_probability(model, processor, spec, image_path, user_prompt,
                            variant_registry, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """Return P(COVID=Yes) from normalized Yes/No token-sequence likelihoods.

    The generated JSON's self-reported confidence is retained as provenance, but it is not a
    calibrated model score. Protocol 7.2 requires a score derived from model logits. This
    probe holds the image, evidence block and chat template fixed and scores the two allowed
    continuations after an explicit COVID-only JSON prefix. Multi-token variants are handled
    by summing their conditional log likelihoods.
    """
    import torch

    probe = (str(user_prompt) + "\n\nProbability-scoring probe: answer only with the JSON "
             "field covid_positive. Begin exactly with {\"covid_positive\":\"")
    prompt_text = render_prompt(processor, spec, system_prompt, probe, image_path,
                                variant_registry)
    prefix = prompt_text + '{"covid_positive":"'
    target_device = model_device(model)
    dtype = resolve_dtype()

    with torch.inference_mode():
        prefix_inputs = prepare_inputs(processor, prefix, image_path, target_device, dtype)
        prefix_length = int(prefix_inputs["input_ids"].shape[-1])
        scores = {}
        for label in ["Yes", "No"]:
            full_inputs = prepare_inputs(processor, prefix + label + '"}', image_path,
                                         target_device, dtype)
            total_length = int(full_inputs["input_ids"].shape[-1])
            if total_length <= prefix_length:
                scores[label] = float("nan")
                continue
            logits = model(**full_inputs).logits.float()
            log_probs = torch.log_softmax(
                logits[0, prefix_length - 1:total_length - 1], dim=-1)
            targets = full_inputs["input_ids"][0, prefix_length:total_length]
            scores[label] = float(
                log_probs.gather(-1, targets[:, None]).squeeze(-1).sum())

    yes, no = scores["Yes"], scores["No"]
    if not (math.isfinite(yes) and math.isfinite(no)):
        return None, scores
    maximum = max(yes, no)
    probability = math.exp(yes - maximum) / (
        math.exp(yes - maximum) + math.exp(no - maximum))
    return float(probability), scores


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

    # --- scan_json_objects / E6-Q ---------------------------------------------------------
    ok(scan_json_objects('{"a": 1}')[0] == 1, "single object counted once")
    ok(scan_json_objects('{"a": 1}{"a": 1}')[0] == 2, "repeated object counted twice")
    ok(scan_json_objects('{"a": "}"}')[0] == 1, "brace inside a string must not count")
    ok(scan_json_objects('{"a": "\\\\"}')[0] == 1, "escaped backslash handled")
    ok(scan_json_objects('{"a": {"b": 1}}')[0] == 1, "nesting counts as one object")
    ok(scan_json_objects('{"a": 1')[2] == 1, "unterminated object reports depth 1")
    ok(scan_json_objects('')[0] == 0, "empty text has no objects")
    ok(not first_object_is_closed('{"a": 1'), "open object is not closed")
    ok(first_object_is_closed('prefix {"a": 1} suffix'), "closed object detected after prefix")
    ok(not first_object_is_closed('no json here'), "no object means not closed")
    ok(control_tokens_outside_json('{"a": 1}<unused94>model') == ["<unused94>"],
       "trailing control token detected after a valid JSON object")
    ok(control_tokens_outside_json('<think>x</think>{"a": 1}') == ["<think>"],
       "control token before the JSON object detected")
    ok(control_tokens_outside_json('{"a": "<not_control_inside_json>"}') == [],
       "token-shaped text inside JSON is not a trailing-control violation")

    # --- extract_json_object ---------------------------------------------------------------
    ok(extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}, "fenced JSON recovered")
    ok(extract_json_object('<think>reasoning</think>{"a": 2}') == {"a": 2},
       "think block stripped")
    ok(extract_json_object('{"a": 1}{"a": 2}') == {"a": 1}, "first object wins")
    try:
        extract_json_object("no object")
        ok(False, "missing object must raise")
    except ValueError:
        ok(True, "missing object raises")
    try:
        extract_json_object("[1, 2, 3]")
        ok(False, "array must raise")
    except ValueError:
        ok(True, "array raises")

    # --- clipped_int -----------------------------------------------------------------------
    ok(clipped_int(3, 0, 4) == (3, False), "in-range value untouched")
    ok(clipped_int(9, 0, 4) == (4, True), "above range clipped and flagged")
    ok(clipped_int(-2, 0, 4) == (0, True), "below range clipped and flagged")
    ok(clipped_int("2", 0, 4) == (2, False), "numeric string accepted")
    ok(clipped_int(None, 0, 4) == (None, False), "None stays None")
    ok(clipped_int("abc", 0, 4) == (None, False), "non-numeric string rejected")
    ok(clipped_int(True, 0, 4) == (None, False), "bool is not a score")
    ok(clipped_int(2.6, 0, 4) == (3, False), "float rounded")

    # --- parse_reasoner_output --------------------------------------------------------------
    good_object = {"extent_right": 2, "density_right": 2, "extent_left": 1,
                   "density_left": 1, "mrale_right": 4, "mrale_left": 1,
                   "mrale_total": 5, "covid_positive": "Yes", "covid_confidence": 0.8,
                   "agents_used": ["A2"], "rationale": "bilateral haze"}
    good = json.dumps(good_object)
    fields, error, notes = parse_reasoner_output(good, roster=["A2", "A6"])
    ok(error is None, "well-formed answer parses")
    ok(fields["mrale_total"] == 5, "total read")
    ok(abs(fields["covid_score"] - 0.8) < 1e-9, "confidence maps to P(positive) for Yes")
    ok(notes["formula_consistent"] is True, "consistent formula flagged true")
    ok(notes["agents_fabricated"] == [], "cited agent in roster is not fabricated")

    negative = json.dumps({**good_object, "covid_positive": "No", "covid_confidence": 0.9})
    fields, error, _ = parse_reasoner_output(negative)
    ok(abs(fields["covid_score"] - 0.1) < 1e-9, "confidence inverted for a No call")

    incomplete = dict(good_object)
    incomplete.pop("mrale_total")
    fields, error, notes = parse_reasoner_output(json.dumps(incomplete))
    ok(error is not None and fields["mrale_total"] is None,
       "missing total is invalid rather than silently derived")
    ok("mrale_total:missing" in notes.get("schema_violations", []),
       "missing field violation recorded")

    out_of_range = {**good_object, "mrale_total": 40}
    fields, error, _ = parse_reasoner_output(json.dumps(out_of_range))
    ok(error is not None and fields["mrale_total"] is None,
       "out-of-range total receives invalid-output handling, not clipping")

    fields, error, notes = parse_reasoner_output(
        json.dumps({**good_object, "agents_used": ["A2", "A9"]}), roster=["A2"])
    ok(notes["agents_fabricated"] == ["A9"], "agent outside the roster flagged as fabricated")

    _, error, _ = parse_reasoner_output("the patient appears unwell")
    ok(error is not None, "prose is a parse error, not a zero prediction")

    fields, error, _ = parse_reasoner_output(json.dumps({"covid_positive": "Yes"}))
    ok(error is not None, "an answer with no mRALE is an error")

    fields, error, notes = parse_reasoner_output(
        json.dumps({**good_object, "covid_confidence": 7}))
    ok(fields["covid_score"] is None, "out-of-range confidence discarded")
    ok(notes.get("confidence_out_of_range") == 7, "out-of-range confidence recorded")
    ok(error is not None, "out-of-range confidence is a schema error")

    fields, _, _ = parse_reasoner_output(json.dumps({**good_object, "covid_positive": True}))
    ok(fields["covid_pred"] == "Yes", "boolean covid call accepted")

    fields, _, _ = parse_reasoner_output(json.dumps({**good_object,
                                                     "covid_positive": "yes"}))
    ok(fields["covid_pred"] == "Yes", "lower-case covid call normalised")

    inconsistent = {**good_object, "mrale_total": 9}
    fields, error, notes = parse_reasoner_output(json.dumps(inconsistent))
    ok(notes["formula_consistent"] is False and error is not None,
       "inconsistent formula is invalid, not merely flagged")
    ok(fields["mrale_total"] is None, "inconsistent total receives invalid-output handling")

    # --- aggregate_samples / E6-C ------------------------------------------------------------
    samples = [{"mrale_total": 4, "covid_pred": "Yes"},
               {"mrale_total": 8, "covid_pred": "Yes"},
               {"mrale_total": 6, "covid_pred": "No"}]
    fields, error, notes = aggregate_samples(samples)
    ok(error is None and fields["mrale_total"] == 6, "median over samples")
    ok(fields["covid_pred"] == "Yes", "majority vote")
    ok(abs(fields["covid_score"] - 2 / 3) < 1e-9, "vote share used as the score")
    ok(notes["n_usable"] == 3, "usable sample count recorded")

    fields, error, _ = aggregate_samples([{"mrale_total": 4}, {"mrale_total": 6}])
    ok(fields["mrale_total"] == 4, "even sample count chooses a deterministic coherent medoid")

    coherent = aggregate_samples([
        {"mrale_total": 2, "mrale_right": 2, "mrale_left": 0,
         "extent_right": 2, "density_right": 1, "extent_left": 0, "density_left": 0},
        {"mrale_total": 6, "mrale_right": 6, "mrale_left": 0,
         "extent_right": 3, "density_right": 2, "extent_left": 0, "density_left": 0},
        {"mrale_total": 9, "mrale_right": 9, "mrale_left": 0,
         "extent_right": 3, "density_right": 3, "extent_left": 0, "density_left": 0},
    ])[0]
    ok(coherent["mrale_right"] == coherent["extent_right"] * coherent["density_right"],
       "self-consistency aggregate preserves regional formula coherence")

    _, error, _ = aggregate_samples([{"mrale_total": None}, {}])
    ok(error is not None, "no usable sample is an error, not a silent fallback")

    tied = aggregate_samples([{"mrale_total": 2, "covid_pred": "Yes"},
                              {"mrale_total": 2, "covid_pred": "No"}])[0]
    ok(tied["covid_pred"] == "No", "a tied vote does not become positive")

    # --- journals ----------------------------------------------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        path, stamp = open_journal(directory, "cfg", "aaa", log=lambda *a: None)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"image_key": "MIDRC::a.png", "v": 1}) + "\n")
            handle.write(json.dumps({"image_key": "MIDRC::b.png", "v": 2}) + "\n")
        ok(journal_keys(path) == {"MIDRC::a.png", "MIDRC::b.png"},
           "journal keys read back as plain strings")
        ok("MIDRC::a.png" in journal_keys(path),
           "membership test works with a bare string (the NB 11 bug)")

        path2, _ = open_journal(directory, "cfg", "aaa", log=lambda *a: None)
        ok(len(journal_keys(path2)) == 2, "unchanged fingerprint keeps the journal")

        with path2.open("ab") as handle:
            handle.write(b'{"image_key":"MIDRC::interrupted')
        repaired = repair_jsonl_tail(path2, log=lambda *a: None)
        ok(repaired and journal_keys(path2) == {"MIDRC::a.png", "MIDRC::b.png"},
           "an interrupted final append is truncated without losing completed rows")
        with path2.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"image_key": "MIDRC::c.png", "v": 3}) + "\n")
        ok("MIDRC::c.png" in journal_keys(path2),
           "the first append after tail repair remains readable")

        interior = directory / "interior.jsonl"
        interior.write_text('{"image_key":"ok"}\nnot-json\n{"image_key":"later"}\n',
                            encoding="utf-8")
        try:
            repair_jsonl_tail(interior, log=lambda *a: None)
            interior_blocked = False
        except ValueError:
            interior_blocked = True
        ok(interior_blocked, "interior journal corruption is blocking, not silently skipped")

        path3, _ = open_journal(directory, "cfg", "bbb", log=lambda *a: None)
        ok(journal_keys(path3) == set(), "changed fingerprint retires the journal")
        ok((directory / "cfg.stale-aaa.jsonl").is_file(),
           "retired journal kept for inspection, not deleted")

        fold0 = directory / "arm__fold0.jsonl"
        fold1 = directory / "arm__fold1.jsonl"
        fold0.write_text("", encoding="utf-8")
        fold1.write_text("", encoding="utf-8")
        (directory / "arm__fold0.stale-old.jsonl").write_text("", encoding="utf-8")
        ok(active_fold_journals(directory, "arm") == [fold0, fold1],
           "active fold reader excludes retired stale journals")

        path4, _ = open_journal(directory, "cfg", "bbb", force=True, log=lambda *a: None)
        ok(journal_keys(path4) == set(), "forced recompute clears the journal")
        ok(journal_keys(directory / "missing.jsonl") == set(),
           "absent journal reads as empty rather than raising")

        write_json_atomic(directory / "x.json", {"a": 1})
        ok(json.loads((directory / "x.json").read_text())["a"] == 1, "atomic write round-trips")
        ok(not (directory / "x.json.tmp").is_file(), "temporary file removed")

    check_session_deadline(time.monotonic() + 60, "future deadline")
    try:
        check_session_deadline(time.monotonic() - 1, "test deadline")
        deadline_stopped = False
    except WalltimeCheckpoint:
        deadline_stopped = True
    ok(deadline_stopped, "soft wall-time stops at a safe boundary")

    ok(fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1}),
       "fingerprint is key-order independent")
    ok(fingerprint({"a": 1}) != fingerprint({"a": 2}), "fingerprint changes with content")

    print(f"stage_c_reasoner self-test: {checks - len(failures)}/{checks} passed")
    for message in failures:
        print("  FAILED:", message)
    return not failures


if __name__ == "__main__":
    import sys
    sys.exit(0 if _self_test() else 1)
