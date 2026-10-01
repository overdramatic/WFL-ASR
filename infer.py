import math
import os

import click
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

from decode import (
    apply_hard_silence,
    constrained_decode,
    continuous_segments,
    forced_align_viterbi,
    load_phoneme_types,
    viterbi_decode,
)
from model import BIOPhonemeTagger
from utils import (
    canonical_to_lang,
    decode_bio_tags,
    enforce_min_duration,
    forced_align_bio,
    insert_silence_phones,
    load_langs,
    load_phoneme_list,
    load_phoneme_merge_map,
    load_phones_txt,
    merge_adjacent_segments,
    save_lab,
)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def collect_wavs(path):
    if os.path.isfile(path) and path.lower().endswith(".wav"):
        return [path]
    if os.path.isdir(path):
        return [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".wav")
        ]
    raise ValueError(f"--input must be a .wav file or a directory: {path}")


def find_matching_txt(wav_path):
    base, _ = os.path.splitext(wav_path)
    txt_path = base + ".txt"
    return txt_path if os.path.isfile(txt_path) else None



def plan_windows(total_len, sr, max_sec=28.0, overlap_sec=2.0, frame_duration=0.02):
    """The windows to run the tagger over: (start, end, i_from, i_to) each.

    Windows advance by `max_sec - overlap_sec` rather than by `max_sec`, so
    neighbours share `overlap_sec` of audio, and each window then keeps only its
    middle. A phoneme on a seam is thereby decoded with context on both sides,
    which is the entire point: without it, an edge phoneme is decoded from one
    side only, and the windows are never compared to each other, so the loss is
    invisible in the aggregate metrics.

    Returns i_from/i_to already resolved, because the invariant they have to
    satisfy is global -- the kept frames must tile the file exactly once -- and
    that cannot be checked one window at a time. Duplicated frames would decode
    into duplicated phonemes, and a dropped frame shifts every timing after it,
    so both are worth the bookkeeping.

    The first and last windows keep their edges, and each interior window hands
    over to the next exactly where it stops trimming, so the spans are
    contiguous. `overlap_sec` is clamped to `max_sec - 0.5` so a window can
    always spare its two edges and leave no audio uncovered.

    All of the span arithmetic is done in whole SAMPLES, not seconds. In floats
    the same boundary comes out as 0.20000000000000007, which divides to
    10.000000000000004 and rounds up to 11 -- a frame dropped at every seam.
    """
    if not max_sec > 0:
        raise ValueError("max_sec must be positive.")
    if overlap_sec < 0:
        raise ValueError("overlap_sec must not be negative.")
    overlap_sec = max(0.0, min(float(overlap_sec), float(max_sec) - 0.5))

    if total_len <= 0:
        return []

    chunk = int(max_sec * sr)
    step = int(max(1e-3, max_sec - overlap_sec) * sr)
    edge = int(round(overlap_sec / 2.0 * sr))
    frame_samples = max(1, int(sr * frame_duration))

    def ceil_frames(n_samples):
        """How many whole frames span n_samples, rounded up."""
        return -(-int(n_samples) // frame_samples)

    def floor_frames(n_samples):
        return int(n_samples) // frame_samples

    def frame_count(n_samples):
        return max(1, ceil_frames(n_samples))

    # A file that already fits needs no overlap at all. Without this, the
    # shorter step would still produce a second, wholly redundant window whose
    # frames the first one has already emitted.
    if total_len <= chunk:
        return [(0, total_len, 0, frame_count(total_len))]

    spans = []
    covered = 0        # samples already emitted, exact integer
    for start in range(0, total_len, step):
        end = min(start + chunk, total_len)
        n_frames = frame_count(end - start)

        # The head is not trimmed against the edge length: it is taken from
        # wherever the previous window actually stopped. Rounding the edge up
        # to a whole frame and the tail down to a whole frame leave the two
        # disagreeing, and the difference is a dropped frame at every seam.
        i_from = ceil_frames(covered - start)
        # The tail is trimmed so the strip the next window re-reads is a real
        # strip of audio with context on both sides. The last window keeps its
        # tail, which is what closes the file.
        i_to = n_frames if end >= total_len else min(
            n_frames, floor_frames((end - edge) - start)
        )
        if i_to <= i_from:
            # Too short to spare its edges, or wholly inside what the previous
            # window already emitted: the strip adds nothing.
            continue
        spans.append((start, end, i_from, i_to))
        covered = start + i_to * frame_samples
    return spans


def encode_audio(
    model,
    audio,
    sr,
    config,
    device,
    lang_id=None,
    no_use_offset=False,
    max_sec=28.0,
    overlap_sec=2.0,
):
    """Run the tagger over a whole file. Returns (logits, offsets, duration).

    Split out of process_audio so tools that score the DECODER (see
    tune_decode.py) can pay for the forward pass once and then sweep decode
    parameters without touching the model again.

    Long audio is encoded in overlapping windows (see plan_windows). The call
    signature keeps the six positional arguments tune_decode.py passes, with
    the new chunking options as keywords only.
    """
    original_duration = len(audio) / sr

    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    total_len = len(audio)
    frame_duration = config["data"].get("frame_duration", 0.02)

    plan = plan_windows(total_len, sr, max_sec, overlap_sec, frame_duration)

    if lang_id is not None:
        lang_tensor = torch.tensor([lang_id], dtype=torch.long).to(device)
    else:
        lang_tensor = torch.zeros(1, dtype=torch.long).to(device)

    accumulated_logits = []
    accumulated_offsets = []

    for start, end, i_from, i_to in plan:
        chunk = audio[start:end]

        if len(chunk) < 1600:
            pad_res = 1600 - len(chunk)
            chunk = np.pad(chunk, (0, pad_res), mode="constant")

        expected_frames = math.ceil(
            (end - start) / sr / frame_duration
        )
        # The training path gets its mel from FeatureCollator (see train.py).
        # There is no DataLoader here, so forward() cannot expect precomputed
        # features: it raises unless we hand it (B, n_mels, frames). The
        # non-whisper path still wants the raw waveform.
        if (
            getattr(model, "encoder_type", "") == "whisper"
            and getattr(model, "precompute_features", False)
        ):
            input_values = model.feature_extractor(
                np.asarray(chunk, dtype=np.float32), sampling_rate=16000,
                return_tensors="pt",
            )["input_features"].to(device)
        else:
            input_values = (
                torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
            )
        lengths = torch.tensor([expected_frames], device=device)

        with torch.no_grad():
            logits, offsets = model(input_values, lang_tensor, lengths=lengths)

        # forward() already truncates to `lengths`, so this is a no-op guard that
        # also makes the slice below well defined.
        chunk_logits = logits.squeeze(0)[:expected_frames]
        chunk_offsets = (
            offsets.squeeze(0)[:expected_frames] if offsets is not None else None
        )

        # Never read past what the model actually returned.
        i_to = min(i_to, int(chunk_logits.size(0)))
        if i_to <= i_from:
            continue

        accumulated_logits.append(chunk_logits[i_from:i_to].cpu())
        if chunk_offsets is not None:
            accumulated_offsets.append(chunk_offsets[i_from:i_to].cpu())

    if not accumulated_logits:
        # plan_windows returns nothing for empty audio, and torch.cat rejects an
        # empty list. Callers get a correctly-shaped zero-row tensor rather
        # than an exception, so a stray empty .wav is skipped instead of ending
        # the run. len(id2label) is the width the decoder expects.
        n_labels = len(getattr(model, "id2label", {}) or {})
        return (
            torch.zeros((0, n_labels), dtype=torch.float32),
            None,
            original_duration,
        )

    full_logits = torch.cat(accumulated_logits, dim=0)

    full_offsets = None
    if accumulated_offsets and not no_use_offset:
        full_offsets = torch.cat(accumulated_offsets, dim=0)

    return full_logits, full_offsets, original_duration


def process_audio(
    model,
    audio,
    sr,
    config,
    device,
    lang_id=None,
    merge_map=None,
    lang_name=None,
    phones=None,
    no_use_offset=False,
    decoder="constrained",
    viterbi_bias=5,
    self_loop_penalty=None,
    penalty_by_class=None,
    phoneme_types=None,
    pad_silence=True,
    silence_phoneme="SP",
    silence_threshold=0.005,
    min_silence_duration=0.5,
    min_phoneme_duration=0.0,
    max_sec=28.0,
    overlap_sec=2.0,
):
    if len(audio) == 0:
        return []

    full_logits, full_offsets, original_duration = encode_audio(
        model, audio, sr, config, device, lang_id, no_use_offset,
        max_sec=max_sec, overlap_sec=overlap_sec,
    )
    frame_duration = config["data"].get("frame_duration", 0.02)

    # Only one spelling of the stay-cost may reach the decoder: `bias` is a
    # bonus on the free path and `penalty` is a cost on the forced one, so
    # picking silently between them is how a sweep ends up reporting a gain that
    # is really a sign flip. The free decoder takes a BONUS (see
    # decode.viterbi_decode), so a penalty passed here is negated on the way in.
    free_decode_kwargs = (
        {"stickiness_bonus": -float(self_loop_penalty),
         "penalty_by_class": penalty_by_class}
        if self_loop_penalty is not None
        else {"viterbi_bias": 1 if viterbi_bias is None else viterbi_bias,
              "penalty_by_class": penalty_by_class}
    )
    free_decode_kwargs["phoneme_types"] = phoneme_types
    # The phoneme map is keyed per language: "j" is a fricative in pt and an
    # affricate in ja, so the duration prior depends on which one this file is.
    free_decode_kwargs["lang"] = lang_name

    def free_segments():
        """The unconstrained decode. Output on its own, and the evidence for
        where the silences are when a transcript is available."""
        tags = viterbi_decode(
            full_logits, model.id2label, **free_decode_kwargs
        )
        return continuous_segments(
            decode_bio_tags(tags, frame_duration, full_offsets), original_duration
        )

    if phones:
        if pad_silence:
            # Forced alignment spreads the transcript over every frame, pauses
            # included. Without silence tokens it has to stretch the speech
            # phones across the gaps, which degrades exactly the common
            # phonemes. The unconstrained decode, which has no such constraint,
            # is the evidence for where the pauses actually are.
            evidence = apply_hard_silence(
                free_segments(), audio, sr,
                threshold=silence_threshold,
                min_duration=min_silence_duration,
                silence_phoneme=silence_phoneme,
            )
            phones = insert_silence_phones(
                phones, evidence, silence_phoneme,
                min_silence=min_silence_duration,
            )
        if decoder == "constrained":
            pred_tags = forced_align_bio(full_logits, model.id2label, phones)
        elif decoder == "viterbi":
            forced_align_args = config["postprocess"].get("forced_alignment_args", {})
            pred_tags = forced_align_viterbi(
                full_logits,
                model.id2label,
                phones,
                self_loop_penalty=forced_align_args.get("self_loop_penalty", -4.6),
                forward_penalty=forced_align_args.get("forward_penalty", -0.6),
                skip_penalty=forced_align_args.get("skip_penalty", -2.3),
                self_loop_penalty_by_class=forced_align_args.get(
                    "self_loop_penalty_by_class"
                ),
                phoneme_types=phoneme_types,
                lang=lang_name,
            )
    else:
        if decoder == "constrained":
            pred_tags = constrained_decode(full_logits, model.id2label)
        elif decoder == "viterbi":
            pred_tags = viterbi_decode(
                full_logits, model.id2label, **free_decode_kwargs
            )

    segments = decode_bio_tags(
        pred_tags, frame_duration, full_offsets
    )

    all_segments = []
    for s, e, ph in segments:
        if merge_map and lang_name:
            ph = canonical_to_lang(ph, lang_name, merge_map)
        all_segments.append((s, e, ph))

    segments = continuous_segments(all_segments, original_duration)
    if min_phoneme_duration > 0:
        # Boundaries only, so the phone sequence is preserved -- safe on top of
        # forced alignment, where the sequence is fixed by the transcript.
        segments = enforce_min_duration(segments, min_phoneme_duration)
        segments = continuous_segments(segments, original_duration)
    return segments


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    default="infer_test",
    help="Path to a .wav file or folder containing .wav files",
)
@click.option(
    "--checkpoint",
    "-ckpt",
    default="checkpoints_no_env/model.ckpt",
    help="Path to WFL .ckpt file",
)
@click.option(
    "--config",
    "-c",
    default="checkpoints_no_env/config.yaml",
    help="Path to config file",
)
@click.option(
    "--lang-id",
    "-l",
    type=int,
    default=None,
    help="Language ID (int) used during training. Example: `-l 0`",
)
@click.option(
    "--no_use_offset",
    is_flag=True,
    help="Disable offset head refinement (offsets ON by default).",
)
# long silence stuff
@click.option(
    "--silence-phoneme",
    default=None,
    help="The phoneme label to use for hard-coded silence "
         "(default: postprocess.hard_silence.silence_phoneme, SP)",
)
@click.option(
    "--silence-threshold",
    default=None,
    type=float,
    help="Amplitude threshold (0.0-1.0) to consider as silence "
         "(default: postprocess.hard_silence.threshold, 0.005)",
)
@click.option(
    "--min-silence-duration",
    default=None,
    type=float,
    help="Minimum duration (seconds) required to trigger hard silence "
         "(default: postprocess.hard_silence.min_duration, 0.5)",
)
@click.option(
    "--decoder-type",
    "-d",
    default="viterbi",
    type=click.Choice(["constrained", "viterbi"]),
    help="Decoder type for no transcription inference [constrained|viterbi] "
         "(default: viterbi)",
)
@click.option(
    "--viterbi-bias",
    default=None,
    type=float,
    help="Bonus (>=1) per frame spent on the same phoneme, no-transcript mode. "
         "Same number as stickiness_bonus=log(bias); give one or the other. "
         "Ignored when postprocess.stickiness_bonus_by_class is set.",
)
@click.option(
    "--chunk-sec",
    default=None,
    type=float,
    help=f"Window length for long audio (default: postprocess.chunking.max_sec, 28.0)",
)
@click.option(
    "--overlap-sec",
    default=None,
    type=float,
    help="Overlap between consecutive windows. Each window keeps only its middle, "
         "so a phoneme on a seam is decoded with context on both sides. "
         "0 disables it (default: postprocess.chunking.overlap_sec, 2.0).",
)
@click.option(
    "--pad-silence/--no-pad-silence",
    default=None,
    help="With a transcript, insert silence tokens where the unconstrained decode "
         "heard a pause. Without them the forced aligner stretches speech phones "
         "across the gaps (default: on).",
)
@click.option(
    "--min-phoneme-duration",
    default=None,
    type=float,
    help="Seconds. Moves boundaries so no phoneme is shorter than this. Disabled "
         "at 0.0 (default: postprocess.min_phoneme_duration).",
)
@click.option(
    "--phoneme-map",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="phoneme_map.yaml giving each phoneme its type, so the per-type decode "
         "penalties apply (default: postprocess.phoneme_map, then "
         "phoneme_map.yaml next to the config).",
)
def main(
    input_path,
    checkpoint,
    config,
    lang_id,
    no_use_offset,
    silence_phoneme,
    silence_threshold,
    min_silence_duration,
    decoder_type,
    viterbi_bias,
    chunk_sec,
    overlap_sec,
    pad_silence,
    min_phoneme_duration,
    phoneme_map,
):
    cfg = load_config(config)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.mps.is_available()
        else "cpu"
    )
    print(f"Running on: {device}")

    save_dir = cfg["output"]["save_dir"]
    phonemes_path = os.path.join(save_dir, "phonemes.txt")

    if not os.path.exists(phonemes_path):
        print(f"Error: {phonemes_path} not found.")
        return

    labels = load_phoneme_list(phonemes_path)
    merge_map = load_phoneme_merge_map(os.path.join(save_dir, "phoneme_merge_map.json"))

    # Postprocess defaults live in the config so a checkpoint carries the
    # decode settings it was tuned with; the flags above override them.
    post = cfg.get("postprocess", {})
    chunking = post.get("chunking", {})
    hard_silence = post.get("hard_silence", {})

    # The phoneme inventory belongs to the corpus, not to the code: read it from
    # the user's map so the per-type penalties apply to whatever languages were
    # trained. A missing map is a warning, not an error -- the decode still runs,
    # it just falls back to one class for everything, which is what the per-type
    # penalties were for in the first place.
    if phoneme_map is None:
        candidates = [
            post.get("phoneme_map"),
            os.path.join(os.path.dirname(os.path.abspath(config)), "phoneme_map.yaml"),
            "phoneme_map.yaml",
        ]
        phoneme_map = next((c for c in candidates if c and os.path.isfile(c)), None)
    if phoneme_map is None:
        if penalty_by_class:
            print(
                ">>> No phoneme_map.yaml found; every phoneme falls back to the "
                "'sonorant' class. The per-type penalties will not differentiate."
            )
        phoneme_types = None
    else:
        phoneme_types = load_phoneme_types(phoneme_map)
        print(
            f">>> Phoneme map: {phoneme_map} "
            f"({len(phoneme_types)} symbols)"
        )

    # `stickiness_bonus_by_class` is a BONUS block (positive values) and
    # `self_loop_penalty_by_class` was the COST block that used to live here. The
    # old key is still accepted so an existing config does not crash, but a
    # negative value there is the fragmentation bug and has to be refused rather
    # than quietly applied: reading only the new key would silently drop every
    # per-class override, which is the same failure wearing a different name.
    penalty_by_class = post.get("stickiness_bonus_by_class")
    legacy_by_class = post.get("self_loop_penalty_by_class")
    if penalty_by_class and legacy_by_class:
        raise ValueError(
            "postprocess has both stickiness_bonus_by_class and "
            "self_loop_penalty_by_class. They are the same block with opposite "
            "signs; keep stickiness_bonus_by_class and delete the other."
        )
    if legacy_by_class:
        negatives = {
            k: v for k, v in legacy_by_class.items() if isinstance(v, (int, float))
            and v < 0
        }
        if negatives:
            raise ValueError(
                f"postprocess.self_loop_penalty_by_class has negative values "
                f"{negatives}. On the free decoder that means a COST for staying "
                "on a phoneme, which shatters the output into one-frame "
                "phonemes -- use stickiness_bonus_by_class with the values "
                "negated, e.g. "
                + ", ".join(f"{k}: {-v:g}" for k, v in sorted(negatives.items()))
            )
        penalty_by_class = legacy_by_class
        print(">>> postprocess.self_loop_penalty_by_class is deprecated; "
              "renaming it to stickiness_bonus_by_class.")
    if viterbi_bias is not None and penalty_by_class:
        print(">>> postprocess.stickiness_bonus_by_class is set; "
              "--viterbi-bias is ignored.")
        viterbi_bias = None
    if viterbi_bias is None and not penalty_by_class:
        viterbi_bias = post.get("viterbi_bias", 5)

    chunk_sec = chunk_sec if chunk_sec is not None else chunking.get("max_sec", 28.0)
    overlap_sec = (
        overlap_sec if overlap_sec is not None else chunking.get("overlap_sec", 2.0)
    )
    if pad_silence is None:
        pad_silence = hard_silence.get("pad_transcript", True)
    if min_phoneme_duration is None:
        min_phoneme_duration = post.get("min_phoneme_duration", 0.0)
    # A flag that was actually given wins over the config. This is why the
    # silence options above default to None rather than to their historical
    # values: with a real default there is no way to tell "--silence-threshold
    # 0.005" from "not passed", and the config would silently win either way.
    if silence_threshold is None:
        silence_threshold = hard_silence.get("threshold", 0.005)
    if min_silence_duration is None:
        min_silence_duration = hard_silence.get("min_duration", 0.5)
    if silence_phoneme is None:
        silence_phoneme = hard_silence.get("silence_phoneme", "SP")

    lang_name = None
    if lang_id is not None:
        lang_path = os.path.join(save_dir, "langs.txt")
        if os.path.exists(lang_path):
            lang2id = load_langs(lang_path)
            id2lang = {v: k for k, v in lang2id.items()}
            lang_name = id2lang.get(lang_id)
            print(f"Language: {lang_name} (ID: {lang_id})")

    print("Loading model...")
    model = BIOPhonemeTagger(cfg, labels).to(device)
    model.eval()

    # weights_only=False because I dont like the the 'untrusted-models' warning
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = (
        checkpoint_data["state_dict"]
        if "state_dict" in checkpoint_data
        else checkpoint_data
    )

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[6:]] = v

    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        return

    files = collect_wavs(input_path)
    print(f"Found {len(files)} files.")

    for wav_path in files:
        print(f"Processing: {wav_path}")

        # auto-detect .txt
        txt_path = find_matching_txt(wav_path)
        phones = None
        if txt_path:
            try:
                phones = load_phones_txt(txt_path)
                if phones:
                    print(
                        f"  Forced-align enabled (found: {os.path.basename(txt_path)})"
                    )
                else:
                    phones = None
            except Exception as e:
                print(f"  Warning: failed to read {txt_path}: {e}")
                phones = None

        try:
            audio, sr = sf.read(wav_path)
        except Exception as e:
            print(f"Error reading {wav_path}: {e}")
            continue

        if sr != cfg["data"]["sample_rate"]:
            audio_t = torch.tensor(audio, dtype=torch.float32)
            if audio_t.dim() > 1:
                audio_t = audio_t.mean(dim=1)
            audio = torchaudio.functional.resample(
                audio_t, sr, cfg["data"]["sample_rate"]
            ).numpy()
            sr = cfg["data"]["sample_rate"]

        segments = process_audio(
            model,
            audio,
            sr,
            cfg,
            device,
            lang_id=lang_id,
            merge_map=merge_map,
            lang_name=lang_name,
            phones=phones,
            no_use_offset=no_use_offset,
            decoder=decoder_type,
            viterbi_bias=viterbi_bias,
            penalty_by_class=penalty_by_class,
            phoneme_types=phoneme_types,
            pad_silence=pad_silence,
            silence_phoneme=silence_phoneme,
            silence_threshold=silence_threshold,
            min_silence_duration=min_silence_duration,
            min_phoneme_duration=min_phoneme_duration,
            max_sec=chunk_sec,
            overlap_sec=overlap_sec,
        )

        if cfg.get("postprocess", {}).get("merge_segments", "right") != "none":
            segments = merge_adjacent_segments(
                segments, cfg["postprocess"]["merge_segments"]
            )

        # Apply hard silence ONLY if we are NOT using forced alignment
        # Forced alignment already knows where silence is based on the text "SP" tag if its in the txt
        # adding heuristic silence on top of forced alignment usually breaks things so yea no
        if phones is None:
            segments = apply_hard_silence(
                segments,
                audio,
                sr,
                threshold=silence_threshold,
                min_duration=min_silence_duration,
                silence_phoneme=silence_phoneme,
            )

        segments = continuous_segments(segments, len(audio) / sr)
        out_path = wav_path.replace(".wav", ".lab")
        save_lab(out_path, segments)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
