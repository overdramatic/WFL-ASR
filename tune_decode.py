"""Sweep the post-hoc decode parameters on the validation split.

The forced aligner and the free decoder are driven by a handful of hand-tuned
numbers that live under `postprocess` in the config: self_loop_penalty,
forward_penalty and skip_penalty for the forced-alignment path, viterbi_bias
for the no-transcript path, plus the hard-silence threshold. None of them
affect training -- they only affect DECODING. So one forward pass per
validation file is enough to score any number of combinations: this script
runs the encoder once, caches the logits, and then sweeps the rest for free.

The reported PER uses the reference .lab as the transcript (an oracle
transcript), which is the honest way to measure the aligner: it isolates
boundary placement and duration modelling from transcription accuracy.

Usage:
    python tune_decode.py -c checkpoints_x/config.yaml -ckpt checkpoints_x/model.ckpt
    python tune_decode.py ... --mode free --out best_decode.yaml
"""

import itertools
import json
import os

import numpy as np
import soundfile as sf
import torch
import yaml

from decode import (
    apply_hard_silence,
    continuous_segments,
    forced_align_viterbi,
    load_phoneme_types,
    viterbi_decode,
)
from infer import encode_audio
from model import BIOPhonemeTagger
from utils import (
    boundary_counts,
    collapse_repeated,
    decode_bio_tags,
    load_phoneme_list,
    merge_adjacent_segments,
    phone_error_rate,
)

DEFAULT_TOLERANCES = [20, 40]

DEFAULT_GRIDS = {
    "forced": {
        "self_loop_penalty": [-3.0, -4.0, -4.6, -5.5, -7.0],
        "forward_penalty": [-0.2, -0.6, -1.2],
        "skip_penalty": [-1.0, -2.3, -4.0, -8.0],
    },
    "free": {
        "viterbi_bias": [1.0, 2.0, 3.0, 5.0, 8.0, 12.0],
    },
}


def load_model(config_path, checkpoint, device):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    save_dir = cfg["output"]["save_dir"]
    labels = load_phoneme_list(os.path.join(save_dir, "phonemes.txt"))
    model = BIOPhonemeTagger(cfg, labels).to(device)
    model.eval()

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    state = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(
        {k[6:] if k.startswith("model.") else k: v for k, v in state.items()}
    )
    return cfg, labels, model


def validation_samples(cfg, labels):
    """The validation split declared by data_split.json, with its references."""
    save_dir = cfg["output"]["save_dir"]
    with open(os.path.join(save_dir, "dataset.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)
    split_path = os.path.join(save_dir, "data_split.json")
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        val_paths = set(split["val_paths"])
    else:
        val_paths = {s["wav_path"] for s in dataset[-cfg["data"]["num_val_files"]:]}
    return [s for s in dataset if s["wav_path"] in val_paths]


def encode_validation_set(cfg, model, samples, device, cache_dir=None, id2lang=None):
    """One forward pass per validation file; the decode is what gets swept."""
    cached = []
    for i, sample in enumerate(samples):
        path = sample["wav_path"]
        cache_path = (
            os.path.join(cache_dir, f"{i:03d}.npz") if cache_dir else None
        )

        if cache_path and os.path.exists(cache_path):
            blob = np.load(cache_path, allow_pickle=True)
            logits = blob["logits"]
            offsets = blob["offsets"]
            audio, sr = blob["audio"], int(blob["sr"])
        else:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            with torch.no_grad():
                logits, offsets, _ = encode_audio(
                    model, audio, sr, cfg, device, sample["lang_id"]
                )
            logits = logits.numpy()
            offsets = offsets.numpy() if offsets is not None else np.zeros((0, 2))
            if cache_path:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez_compressed(
                    cache_path, logits=logits, offsets=offsets,
                    audio=audio, sr=np.int64(sr),
                )

        duration = len(audio) / sr
        reference = continuous_segments(
            [(s, e, ph) for s, e, ph in sample["phoneme_segments"]], duration
        )
        phones = collapse_repeated([ph for _, _, ph in reference])
        cached.append({
            "name": os.path.basename(path),
            "logits": logits,
            "offsets": offsets,
            "audio": audio,
            "sr": sr,
            "duration": duration,
            "reference": reference,
            "phones": phones,
            # Carried through so the sweep resolves types per language, exactly
            # as train.py and infer.py do. Cached logits are language-specific,
            # so this survives a --cache-dir re-run unchanged.
            "lang": (id2lang or {}).get(sample["lang_id"]),
        })
        print(f"  [{i + 1}/{len(samples)}] {os.path.basename(path)} "
              f"({duration:.1f}s, {len(phones)} phonemes)")
    return cached


# Per-class values are swept as flat scalars rather than as a dict, because
# --grid only parses floats. Each name maps onto one decoder argument and one
# phonetic class, so the assembled dict is exactly what infer.py reads back from
# the config. Sweeping the scalar alone stays possible by simply not naming any
# of these.
#
# NOTE THE SIGN, it differs by mode and that is not a typo:
#   - the forced path takes COSTS, so its names are negative and stay so;
#   - the free path takes BONUSES (see decode.viterbi_decode), so a negative
#     value there is rejected with an error rather than applied.
# `--grid vowel_penalty=-2.5` is therefore a forced-path sweep, and on the free
# path it has to be `--grid vowel_penalty=2.5`.
PER_CLASS_PARAMS = {
    # forced path: forced_align_viterbi(self_loop_penalty_by_class=...)
    "self_loop_penalty_vowel": ("self_loop_penalty_by_class", "vowel"),
    "self_loop_penalty_sonorant": ("self_loop_penalty_by_class", "sonorant"),
    # obstruent was one class before phoneme_map.yaml split it in two; the name
    # still resolves, as an alias for both of them.
    "self_loop_penalty_obstruent": ("self_loop_penalty_by_class", "obstruent"),
    "self_loop_penalty_fricative": ("self_loop_penalty_by_class", "fricative"),
    "self_loop_penalty_stop": ("self_loop_penalty_by_class", "stop"),
    # free path: viterbi_decode(penalty_by_class=...). Positive values, because
    # that path takes a stay BONUS. The name keeps "penalty" for continuity
    # with the forced path and with older sweeps, so read the sign above before
    # assuming.
    "vowel_penalty": ("penalty_by_class", "vowel"),
    "sonorant_penalty": ("penalty_by_class", "sonorant"),
    "obstruent_penalty": ("penalty_by_class", "obstruent"),
    "fricative_penalty": ("penalty_by_class", "fricative"),
    "stop_penalty": ("penalty_by_class", "stop"),
    # per TYPE, for when a class is too coarse -- pt/rr is a trill (long) while
    # es/rr is a liquid, and the map can tell them apart even though the
    # duration class cannot.
    "semivowel_penalty": ("penalty_by_class", "semivowel"),
    "trill_penalty": ("penalty_by_class", "trill"),
    "nasal_penalty": ("penalty_by_class", "nasal"),
    "liquid_penalty": ("penalty_by_class", "liquid"),
    "rhotic_penalty": ("penalty_by_class", "rhotic"),
    "affricate_penalty": ("penalty_by_class", "affricate"),
    "aspirate_penalty": ("penalty_by_class", "aspirate"),
}


def split_per_class(params):
    """Pull the flat per-class scalars out of `params` into real dicts.

    Returns (by_class, params_without_them), where by_class has one entry per
    decoder argument it found values for. Absent classes are simply not listed,
    and decode.resolve_penalties fills them from the scalar default, so
    sweeping a single class is a valid thing to do.

    `params` is not mutated -- the caller still wants the flat names for the
    printed row and for the emitted config snippet.
    """
    by_class = {}
    rest = {}
    for key, value in params.items():
        target = PER_CLASS_PARAMS.get(key)
        if target is None:
            rest[key] = value
            continue
        arg, phone = target
        by_class.setdefault(arg, {})[phone] = value
    return by_class, rest


def decode_sample(entry, cfg, labels, mode, params, no_use_offset, phoneme_types=None):
    """Reproduce what infer.py does for this decoder, with swept parameters."""
    frame_duration = cfg["data"]["frame_duration"]
    offsets = None if no_use_offset else entry["offsets"]
    if offsets is not None and offsets.size == 0:
        offsets = None

    id2label = {i: lab for i, lab in enumerate(labels)}
    by_class, kwargs = split_per_class(params)
    try:
        if mode == "forced":
            tags = forced_align_viterbi(
                torch.from_numpy(entry["logits"]),
                id2label,
                entry["phones"],
                phoneme_types=phoneme_types,
                lang=entry.get("lang"),
                **kwargs,
            )
        else:
            tags = viterbi_decode(
                torch.from_numpy(entry["logits"]),
                id2label,
                viterbi_bias=kwargs["viterbi_bias"],
                penalty_by_class=by_class.get("penalty_by_class"),
                phoneme_types=phoneme_types,
                lang=entry.get("lang"),
            )
    except (ValueError, KeyError) as exc:
        return None, str(exc)

    segments = continuous_segments(
        decode_bio_tags(tags, frame_duration, offsets), entry["duration"]
    )
    merge_mode = cfg.get("postprocess", {}).get("merge_segments", "right")
    if merge_mode != "none":
        segments = merge_adjacent_segments(segments, merge_mode)

    # infer.py only applies the amplitude-based silence heuristic when there is
    # no transcript: on top of forced alignment it overrides boundaries the
    # model already placed.
    if mode == "free" and params.get("silence_threshold") is not None:
        segments = apply_hard_silence(
            segments,
            entry["audio"],
            entry["sr"],
            threshold=params["silence_threshold"],
            min_duration=params["min_silence_duration"],
            silence_phoneme=params.get("silence_phoneme", "SP"),
        )
    return continuous_segments(segments, entry["duration"]), None


def score(entries, cfg, labels, mode, params, tolerances, no_use_offset, phoneme_types=None):
    errors = ref_phones = 0
    files = 0
    matched = {tol: 0 for tol in tolerances}
    abs_err = matched_n = n_ref = n_hyp = 0

    for entry in entries:
        segments, err = decode_sample(
            entry, cfg, labels, mode, params, no_use_offset, phoneme_types
        )
        if segments is None:
            continue
        files += 1
        reference = collapse_repeated([ph for _, _, ph in entry["reference"]])
        prediction = collapse_repeated([ph for _, _, ph in segments])
        e, n = phone_error_rate(reference, prediction)
        errors += e
        ref_phones += n

        m, err_sum, err_n, r, h = boundary_counts(
            entry["reference"], segments, tolerances
        )
        for tol, count in m.items():
            matched[tol] += count
        abs_err += err_sum
        matched_n += err_n
        n_ref += r
        n_hyp += h

    if not files or not ref_phones:
        return None
    per = 100.0 * errors / ref_phones
    denom = n_ref + n_hyp
    return {
        "per": per,
        "files": files,
        "boundary_mae_ms": 1000.0 * abs_err / matched_n if matched_n else float("nan"),
        "boundary_f1": {
            tol: (2.0 * matched[tol] / denom if denom else float("nan"))
            for tol in tolerances
        },
    }


def parse_grid(text):
    """--grid self_loop_penalty=-4.6,-6 --grid forward_penalty=-0.6"""
    grid = {}
    for item in text or []:
        key, _, values = item.partition("=")
        grid[key] = [float(v) for v in values.split(",")]
    return grid


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True, help="config.yaml")
    parser.add_argument("--checkpoint", "-ckpt", required=True, help="model .ckpt")
    parser.add_argument(
        "--mode", choices=["forced", "free", "both"], default="both",
        help="forced = with transcript (uses the .lab as oracle), free = no transcript",
    )
    parser.add_argument(
        "--grid", action="append", default=[],
        help="override the sweep grid. WATCH THE SIGN, it differs by mode: the forced "
             "path sweeps COSTS (negative), the free path sweeps stay BONUSES "
             "(positive), because in viterbi_decode the value on an I- tag is "
             "the only thing deciding how long a phoneme lasts. A negative "
             "value on the free path is rejected with an error, not applied. "
             "e.g. --grid viterbi_bias=1,2,5 (free scalar) or "
             "--grid self_loop_penalty=-4.6,-6 (forced scalar). The per-class "
             "values are swept as flat scalars, because this only parses "
             "floats: --grid self_loop_penalty_vowel=-5,-4,-3 (forced) and "
             "--grid vowel_penalty=1.79,1.6 (free). Classes left unnamed "
             "fall back to the scalar. The four duration classes are "
             "vowel/sonorant/fricative/stop (obstruent still works as an alias "
             "for the last two), and every type in phoneme_map.yaml can be swept "
             "on its own, e.g. --grid nasal_penalty=2.5 on the free path.",
    )
    parser.add_argument(
        "--phoneme-map", default=None,
        help="phoneme_map.yaml giving each phoneme its type (default: "
             "postprocess.phoneme_map from the config, then phoneme_map.yaml)",
    )
    parser.add_argument("--tolerance", type=int, action="append", default=[])
    parser.add_argument("--no-use-offset", action="store_true")
    parser.add_argument(
        "--cache-dir", default=None,
        help="cache the encoder output here so re-runs skip the GPU",
    )
    parser.add_argument("--out", default=None, help="write the best config here")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, labels, model = load_model(args.config, args.checkpoint, device)
    tolerances = sorted(args.tolerance or DEFAULT_TOLERANCES)

    # Same lookup order as infer.py, so a sweep grades the decoder that will
    # actually run rather than one class for everything.
    map_path = args.phoneme_map or cfg.get("postprocess", {}).get("phoneme_map")
    if not map_path:
        map_path = os.path.join(
            os.path.dirname(os.path.abspath(args.config)), "phoneme_map.yaml"
        )
    if os.path.isfile(map_path):
        phoneme_types = load_phoneme_types(map_path)
        print(f"Phoneme map: {map_path} ({len(phoneme_types)} symbols)")
    else:
        print(
            f">>> No phoneme map at {map_path}; every phoneme falls back to the "
            f"'sonorant' class. Per-class sweeps will not differentiate."
        )
        phoneme_types = None

    # lang_id -> lang name, so types resolve per language.
    id2lang = {}
    langs_path = os.path.join(cfg["output"]["save_dir"], "langs.txt")
    if os.path.exists(langs_path):
        with open(langs_path, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) == 2 and parts[1].strip().isdigit():
                    id2lang[int(parts[1])] = parts[0]

    samples = validation_samples(cfg, labels)
    if not samples:
        raise SystemExit("No validation samples found.")
    print(f"Validation files: {len(samples)} | device: {device}")

    print("Running the encoder once per file...")
    entries = encode_validation_set(
        cfg, model, samples, device, cache_dir=args.cache_dir, id2lang=id2lang
    )

    modes = ["forced", "free"] if args.mode == "both" else [args.mode]
    overrides = parse_grid(args.grid)
    results = {}

    for mode in modes:
        grid = dict(DEFAULT_GRIDS[mode])
        grid.update(overrides)
        if mode == "free":
            grid.setdefault("silence_threshold", [None, 0.005, 0.01])
            grid.setdefault("min_silence_duration", [0.5])
            grid.setdefault("silence_phoneme", ["SP"])

        keys = list(grid)
        combos = list(itertools.product(*(grid[k] for k in keys)))
        print(f"\n[{mode}] {len(combos)} combinations")

        rows = []
        for combo in combos:
            params = dict(zip(keys, combo))
            if params.get("silence_threshold") is None:
                # The heuristic is off in this combination; keep the printed row
                # and the emitted config from implying otherwise.
                params.pop("min_silence_duration", None)
            out = score(
                entries, cfg, labels, mode, params, tolerances, args.no_use_offset,
                phoneme_types,
            )
            if out is None:
                print(f"  {params} -> unscorable")
                continue
            rows.append((out["per"], params, out))

        rows.sort(key=lambda r: r[0])
        f1_key = f"boundary_f1@{tolerances[-1]}ms"
        header = f"{'PER%':>7}  {'MAEms':>6}  {f1_key:>14}  params"
        print(header)
        print("-" * len(header))
        for per, params, out in rows[: args.top]:
            desc = " ".join(
                f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in params.items() if v is not None
            )
            print(f"{per:7.2f}  {out['boundary_mae_ms']:6.1f}  "
                  f"{out['boundary_f1'][tolerances[-1]] * 100:13.1f}%  {desc}")

        if rows:
            best_per, best_params, best_out = rows[0]
            print(f"\nBest [{mode}]: PER {best_per:.2f}%  "
                  f"MAE {best_out['boundary_mae_ms']:.1f} ms")
            baseline = next(
                (p for _, p, _ in rows
                 if p.get("self_loop_penalty", p.get("viterbi_bias"))
                 == cfg.get("postprocess", {}).get(
                     "forced_alignment_args", {}
                 ).get("self_loop_penalty", cfg.get("postprocess", {}).get("viterbi_bias"))),
                None,
            )
            if baseline is not None and baseline != best_params:
                base_row = next(r for r in rows if r[1] == baseline)
                delta = base_row[0] - best_per
                print(f"  vs config default {baseline}: "
                      f"PER {base_row[0]:.2f}% -> {best_per:.2f}% ({delta:+.2f})")
            results[mode] = {"params": best_params, "metrics": best_out}

    if args.out and results:
        snippet = {"postprocess": {}}
        for mode, payload in results.items():
            by_class, flat = split_per_class(payload["params"])
            if mode == "forced":
                block = dict(flat)
                if "self_loop_penalty_by_class" in by_class:
                    block["self_loop_penalty_by_class"] = by_class[
                        "self_loop_penalty_by_class"
                    ]
                snippet["postprocess"]["forced_alignment_args"] = block
            else:
                silence = {
                    k: v for k, v in flat.items()
                    if k.startswith("silence_") or k == "silence_phoneme"
                }
                # A per-class map fully replaces the scalar (see
                # decode.viterbi_decode), so do not write both and leave a
                # silent precedence for the reader to guess at. The key is the
                # BONUS name: writing the old cost key here would produce a
                # snippet that infer.py refuses to load.
                if "penalty_by_class" in by_class:
                    snippet["postprocess"]["stickiness_bonus_by_class"] = by_class[
                        "penalty_by_class"
                    ]
                else:
                    snippet["postprocess"]["viterbi_bias"] = flat["viterbi_bias"]
                if silence.get("silence_threshold") is not None:
                    snippet["postprocess"]["hard_silence"] = silence
        with open(args.out, "w", encoding="utf-8") as f:
            yaml.dump(snippet, f, sort_keys=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
