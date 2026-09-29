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
    viterbi_decode,
)
from model import BIOPhonemeTagger
from utils import (
    canonical_to_lang,
    decode_bio_tags,
    forced_align_bio,
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



def encode_audio(
    model,
    audio,
    sr,
    config,
    device,
    lang_id=None,
    no_use_offset=False,
):
    """Run the tagger over a whole file. Returns (logits, offsets, duration).

    Split out of process_audio so tools that score the DECODER (see
    tune_decode.py) can pay for the forward pass once and then sweep decode
    parameters without touching the model again.
    """
    original_duration = len(audio) / sr

    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    total_len = len(audio)

    MAX_SEC = 28.0
    CHUNK_SIZE = int(MAX_SEC * sr)

    if lang_id is not None:
        lang_tensor = torch.tensor([lang_id], dtype=torch.long).to(device)
    else:
        lang_tensor = torch.zeros(1, dtype=torch.long).to(device)

    accumulated_logits = []
    accumulated_offsets = []

    for start in range(0, total_len, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_len)
        chunk = audio[start:end]

        if len(chunk) < 1600:
            if len(chunk) == 0:
                continue
            pad_res = 1600 - len(chunk)
            chunk = np.pad(chunk, (0, pad_res), mode="constant")

        expected_frames = math.ceil(
            (end - start) / sr / config["data"]["frame_duration"]
        )
        input_values = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([expected_frames], device=device)

        with torch.no_grad():
            logits, offsets = model(input_values, lang_tensor, lengths=lengths)

        accumulated_logits.append(logits.squeeze(0).cpu())
        if offsets is not None:
            accumulated_offsets.append(offsets.squeeze(0).cpu())

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
):
    if len(audio) == 0:
        return []

    full_logits, full_offsets, original_duration = encode_audio(
        model, audio, sr, config, device, lang_id, no_use_offset
    )

    if phones:
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
            )
    else:
        if decoder == "constrained":
            pred_tags = constrained_decode(full_logits, model.id2label)
        elif decoder == "viterbi":
            pred_tags = viterbi_decode(
                full_logits, model.id2label, viterbi_bias=viterbi_bias
            )

    segments = decode_bio_tags(
        pred_tags, config["data"]["frame_duration"], full_offsets
    )

    all_segments = []
    for s, e, ph in segments:
        if merge_map and lang_name:
            ph = canonical_to_lang(ph, lang_name, merge_map)
        all_segments.append((s, e, ph))

    return continuous_segments(all_segments, original_duration)


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
    default="SP",
    help="The phoneme label to use for hard-coded silence (default: SP)",
)
@click.option(
    "--silence-threshold",
    default=0.005,
    type=float,
    help="Amplitude threshold (0.0-1.0) to consider as silence",
)
@click.option(
    "--min-silence-duration",
    default=0.5,
    type=float,
    help="Minimum duration (seconds) required to trigger hard silence",
)
@click.option(
    "--decoder-type",
    "-d",
    default="viterbi",
    type=str,
    help="Decoder type for no transcription inference [constrained|viterbi] (default: viterbi)",
)
@click.option(
    "--viterbi-bias",
    default=5,
    type=float,
    help="Amount of bias (>=1) added to frames of the same phoneme for viterbi decoding. (default: 5)",
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
