import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

import json
import yaml
import torch
import random
import argparse
import numpy as np
import soundfile as sf
import torchaudio
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from model import BIOPhonemeTagger, FocalLoss
from utils import (
    PhoneConfusionMatrix,
    boundary_counts,
    collapse_repeated,
    decode_bio_tags,
    load_phoneme_list,
    phone_error_rate,
    plot_phone_confusion,
    visualize_prediction,
)
import pytorch_optimizer as optim
from decode import continuous_segments, viterbi_decode


class StatusLine:
    """One line of training state, rewritten in place instead of appended.

    Lightning's progress bar is what fills a Colab cell with thousands of lines:
    stdout there is not a terminal, so tqdm gives up on in-place redrawing and
    emits a fresh line on every refresh, and the bar also carries a ~15-line
    metrics table per refresh. Over a long run the cell becomes unusable and the
    number you actually want sits far off-screen.

    This keeps the display to a single physical line: rewrite it in place, and
    pad with spaces so a shrinking value cannot leave debris from the previous,
    longer one. Callers should only pass fixed-width fields, otherwise the line
    jitters as digits change.

    Outside a terminal (redirected to a file, `tee`) the carriage return is
    meaningless, so `plain` mode writes one line per update instead. That is
    also what `--plain` forces, for reading the log after the fact.
    """

    def __init__(self, enabled=True, plain=False, min_interval=0.2):
        self.enabled = enabled
        self.plain = plain or not sys.stdout.isatty()
        self.min_interval = min_interval
        self._width = 0
        self._open = False
        self._last = 0.0

    def update(self, text):
        """Redraw the line. Cheap to call every batch: it throttles itself."""
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last < self.min_interval:
            return
        self._last = now
        if self.plain:
            print(text, flush=True)
            return
        # The pad clears whatever the previous, possibly longer, line left.
        pad = max(0, self._width - len(text))
        sys.stdout.write("\r" + text + " " * pad)
        sys.stdout.flush()
        self._width = len(text)
        self._open = True

    def line(self, text=""):
        """Finish the live line and print a permanent one."""
        if not self.enabled:
            return
        self.clear()
        print(text, flush=True)

    def clear(self):
        if not self.enabled or not self._open:
            return
        if not self.plain:
            sys.stdout.write("\r" + " " * self._width + "\r")
            sys.stdout.flush()
        self._width = 0
        self._open = False


def collate_fn(batch):
    input_values, label_ids, wavs, segments_gt, wav_paths, lang_ids = zip(*batch)
    label_lengths = torch.tensor([len(x) for x in label_ids])
    padded_input = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=-100)
    return padded_input, padded_labels, wavs, segments_gt, wav_paths, torch.tensor(lang_ids, dtype=torch.long), label_lengths


class FeatureCollator:
    """Computes the Whisper mel inside the DataLoader worker.

    Measured on T4 (small.yaml, bs=4): the WhisperFeatureExtractor costs 77 ms
    per step -- 31% of the whole step -- and runs serially in the main process,
    inside forward(). Here the SAME computation (same object, bit-for-bit same
    output) runs in the worker, in parallel with the GPU, and the cost leaves
    the critical path.

    We deliberately do NOT cache the mel on disk: with augmentation.enable=true
    the waveform changes every epoch, so a pre-computed cache would be invalid.
    """

    def __init__(self, encoder_dir, enabled=True):
        self.encoder_dir = encoder_dir
        self.enabled = enabled
        self._fe = None

    def __getstate__(self):
        # the feature extractor is rebuilt inside the worker; not worth
        # shipping through pickle
        state = self.__dict__.copy()
        state["_fe"] = None
        return state

    def _get_fe(self):
        if self._fe is None:
            from transformers import WhisperFeatureExtractor
            self._fe = WhisperFeatureExtractor.from_pretrained(self.encoder_dir)
        return self._fe

    def __call__(self, batch):
        padded_input, padded_labels, wavs, segs, paths, lang_ids, label_lengths = collate_fn(batch)
        if not self.enabled:
            return padded_input, padded_labels, wavs, segs, paths, lang_ids, label_lengths
        feats = self._get_fe()(
            [np.asarray(w, dtype=np.float32) for w in wavs],
            sampling_rate=16000, return_tensors="pt",
        )["input_features"]
        return feats, padded_labels, wavs, segs, paths, lang_ids, label_lengths

class PhonemeDataset(Dataset):
    def __init__(self, dataset_path, label_list, max_seq_len=None, aug_cfg=None,
                 frame_duration=0.02):
        with open(dataset_path, "r") as f: self.samples = json.load(f)
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_seq_len = max_seq_len
        self.frame_duration = frame_duration
        self.aug_cfg = aug_cfg or {"enable": False}

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        wav, sr = sf.read(sample["wav_path"], dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), sr, 16000
            ).numpy()

        if self.max_seq_len:
            wav = wav[:self.max_seq_len]

        duration = len(wav) / 16000
        num_frames = math.ceil(len(wav) / (16000 * self.frame_duration))
        tags = sample["bio_tags"][:num_frames]
        segments = [
            (s, min(e, duration), ph)
            for s, e, ph in sample["phoneme_segments"]
            if 0 <= s < min(e, duration)
        ]

        if self.aug_cfg.get("enable", False) and random.random() < self.aug_cfg.get("prob", 0.5):
            wav *= random.uniform(*self.aug_cfg.get("volume_range", [0.9, 1.1]))
            if self.aug_cfg.get("noise_std", 0) > 0:
                wav += np.random.normal(0, self.aug_cfg["noise_std"], wav.shape)
            wav = np.clip(wav, -1.0, 1.0)

        label_ids = torch.tensor(
            [-100 if tag == "O" else self.label2id[tag] for tag in tags],
            dtype=torch.long,
        )
        wav_tensor = torch.tensor(wav, dtype=torch.float32)
        return (
            wav_tensor, label_ids, wav, segments,
            sample["wav_path"], sample["lang_id"],
        )

def resolve_encoder_dir(config):
    """Mesma resolucao de model.py: ./encoder local, senao o nome no HF."""
    encoder_dir = os.path.join(os.getcwd(), "encoder")
    if os.path.exists(encoder_dir) and os.listdir(encoder_dir):
        return encoder_dir
    return config["model"].get("whisper_model", "openai/whisper-base")


class WFLDataModule(pl.LightningDataModule):
    def __init__(self, config, label_list):
        super().__init__()
        self.config = config
        self.label_list = label_list
        self.save_dir = config["output"]["save_dir"]
        self.batch_size = config["training"]["batch_size"]
        self.num_workers = config["training"]["num_workers"]

    def setup(self, stage=None):
        dataset_path = os.path.join(self.save_dir, "dataset.json")
        max_seq_len = self.config["data"].get("max_seq_len")
        if self.config["model"].get("encoder_type", "whisper").lower() == "whisper":
            max_seq_len = min(max_seq_len or 480000, 480000)
            
        train_dataset = PhonemeDataset(
            dataset_path,
            self.label_list,
            max_seq_len=max_seq_len,
            frame_duration=self.config["data"].get("frame_duration", 0.02),
            aug_cfg=self.config.get("augmentation"),
        )
        val_dataset = PhonemeDataset(
            dataset_path,
            self.label_list,
            max_seq_len=max_seq_len,
            frame_duration=self.config["data"].get("frame_duration", 0.02),
            aug_cfg={"enable": False},
        )

        paths = [s["wav_path"] for s in train_dataset.samples]
        indices = {p: i for i, p in enumerate(paths)}
        val_count = self.config["data"]["num_val_files"]
        split_path = os.path.join(self.save_dir, "data_split.json")

        if len(indices) != len(paths):
            raise ValueError("Duplicate audio paths in dataset")
        if not 0 < val_count < len(paths):
            raise ValueError("Invalid num_val_files")

        if os.path.exists(split_path):
            with open(split_path, encoding="utf-8") as f:
                split = json.load(f)
        else:
            train, val = random_split(
                sorted(paths),
                [len(paths) - val_count, val_count],
                generator=torch.Generator().manual_seed(
                    self.config["data"].get("split_seed", 42)
                ),
            )
            split = {"train_paths": list(train), "val_paths": list(val)}
            with open(split_path, "w", encoding="utf-8") as f:
                json.dump(split, f, indent=2, ensure_ascii=False)

        saved = split["train_paths"] + split["val_paths"]
        if (
            len(saved) != len(paths)
            or set(saved) != set(paths)
            or len(split["val_paths"]) != val_count
        ):
            raise ValueError("Split mismatch. Restore the dataset/config or rename data_split.json to create a new split")

        self.train_ds = Subset(
            train_dataset, [indices[p] for p in split["train_paths"]]
        )
        self.val_ds = Subset(
            val_dataset, [indices[p] for p in split["val_paths"]]
        )

        if not any(
            train_dataset.samples[i]["phoneme_segments"] for i in self.val_ds.indices
        ):
            raise ValueError(
                "The validation split contains no labelled phoneme, so val/per "
                "cannot be computed and ModelCheckpoint has nothing to monitor. "
                "Fix data_split.json or data.num_val_files."
            )

        is_whisper = self.config["model"].get("encoder_type", "whisper").lower() == "whisper"
        use_cached_mel = is_whisper and self.config["model"].get("precompute_features", True)
        if use_cached_mel:
            self.collator = FeatureCollator(resolve_encoder_dir(self.config), enabled=True)
        else:
            self.collator = collate_fn
        if is_whisper:
            print(
                f">>> WhisperFeatureExtractor in DataLoader worker: "
                f"{'ON' if use_cached_mel else 'off'} "
                f"(num_workers={self.num_workers}). Gain requires num_workers > 0."
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.collator, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.collator, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=self.num_workers > 0)

class WFLModel(pl.LightningModule):
    def __init__(self, config, label_list):
        super().__init__()
        self.save_hyperparameters(ignore=['model']) 
        self.config = config
        self.label_list = label_list
        self.id2label = {i: l for i, l in enumerate(label_list)}
        
        self.model = BIOPhonemeTagger(config, label_list)
        
        if config.get("finetune", {}).get("freeze_backbone", False):
            print(">>> Fine-tuning mode: Freezing Conformer backbone.")
            for param in self.model.conformer.parameters():
                param.requires_grad = False
                
        self.criterion = FocalLoss(alpha=0.5, gamma=2.0, ignore_index=-100)
        self.offset_weight = config["model"].get("subframe_loss_weight", 5.0)
        self.frame_duration = config["data"].get("frame_duration", 0.02)

        # Decode exactly like inference does, so validation measures the
        # decoder that will actually run. viterbi_bias lives under postprocess
        # next to the other decode knobs; the old validation.viterbi_bias is
        # still honoured.
        self.viterbi_bias = config.get("postprocess", {}).get(
            "viterbi_bias", config.get("validation", {}).get("viterbi_bias", 5)
        )
        # Boundary accuracy in ms is the metric that matters for a labeller:
        # PER collapses repeated phonemes and throws all timing away.
        self.boundary_tolerances_ms = sorted(
            config.get("validation", {}).get("boundary_tolerance_ms", [20, 40])
        )
        self.primary_tolerance_ms = self.boundary_tolerances_ms[-1]

        # PER counts the phonemes the model gets wrong but not which ones, and
        # that is the only question left once the rate is high: a flat 15% can be
        # one phoneme collapsing onto its neighbour or fifteen phones sharing it.
        # The confusion matrix is accumulated over the same decoded sequences
        # PER is measured on, so the two always agree.
        cm_cfg = config.get("validation", {}).get("confusion_matrix", {}) or {}
        self.cm_enabled = cm_cfg.get("enabled", True)
        self.cm_top_k = int(cm_cfg.get("top_k", 25))
        self.cm_min_count = int(cm_cfg.get("min_count", 1))
        self.cm_normalize = cm_cfg.get("normalize", True)
        self.cm_console_top = int(cm_cfg.get("console_top", 3))
        self.cm_csv = cm_cfg.get("csv", True)
        self.cm_dir = config["output"]["save_dir"]
        # Rebuilt per validation pass in on_validation_epoch_start.
        self._confusion = PhoneConfusionMatrix()

        total_val = config["data"]["num_val_files"]
        self.num_vis_samples = min(total_val, 8)

        # One rewritten line instead of a scrolling bar. See StatusLine.
        console = config.get("training", {})
        self.status = StatusLine(
            enabled=console.get("live_console", True),
            plain=console.get("plain_console", False),
        )
        # Initialised here, not in on_train_epoch_start: the sanity-check pass
        # runs before any training step and reads these.
        self._loss_ema = None
        self._epoch_start = time.monotonic()
        self._run_start = self._epoch_start
        self._steps_per_epoch = 0
        self._val_batches = 0
        self._val_printed_epoch = None

    def forward(self, x, lang_ids, lengths):
        return self.model(x, lang_ids, lengths=lengths)

    def calculate_loss(self, logits, offsets, labels, segs_gt, lengths):
        cls_loss = self.criterion(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        )

        total_offset_loss = torch.tensor(0.0, device=self.device)
        if offsets is not None:
            target_map = torch.zeros_like(offsets)
            mask_map = torch.zeros_like(offsets)

            for b_idx in range(len(segs_gt)):
                for start_t, end_t, _ in segs_gt[b_idx]:
                    s_f = int(start_t / self.frame_duration)
                    e_f = int(end_t / self.frame_duration)

                    if s_f < lengths[b_idx]:
                        target_map[b_idx, s_f, 0] = start_t / self.frame_duration - s_f
                        mask_map[b_idx, s_f, 0] = 1.0
                    if e_f < lengths[b_idx]:
                        target_map[b_idx, e_f, 1] = end_t / self.frame_duration - e_f
                        mask_map[b_idx, e_f, 1] = 1.0

            diff = torch.abs(offsets - target_map) * mask_map
            total_offset_loss = (
                diff.sum() / (mask_map.sum() + 1e-8)
            ) * self.offset_weight

        total_loss = cls_loss + total_offset_loss
        return total_loss, cls_loss, total_offset_loss

    def training_step(self, batch, batch_idx):
        inputs, labels, wavs, segs_gt, _, langs, lengths = batch
        logits, offsets = self(inputs, langs, lengths)
        loss, cls_loss, off_loss = self.calculate_loss(
            logits, offsets, labels, segs_gt, lengths
        )

        # Safety net for mixed precision. fp16 can overflow (Inf/NaN) in
        # attention over 1500 frames; if that happens in the forward pass, the
        # whole step is wasted AND the weights can be poisoned by the NaN. We
        # skip and count instead of silently training with corrupted weights.
        if not torch.isfinite(loss):
            self._nonfinite_steps = getattr(self, "_nonfinite_steps", 0) + 1
            self.log("train/nonfinite_steps", self._nonfinite_steps, on_step=True,
                     prog_bar=(self._nonfinite_steps <= 5))
            if self._nonfinite_steps <= 5:
                self.status.line(
                    f"epoch {self.current_epoch + 1}/{self.trainer.max_epochs}"
                    f" [!] non-finite loss at step {batch_idx}"
                    f" (total {self._nonfinite_steps}) -- step skipped."
                    f" The encoder is in fp32; if it persists, try"
                    f" training.precision: 32."
                )
            return None

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))
        self.log("train/cls_loss", cls_loss, on_step=False, on_epoch=True, batch_size=inputs.size(0))
        self.log("train/off_loss", off_loss, on_step=False, on_epoch=True, batch_size=inputs.size(0))
        return loss

    def on_train_epoch_start(self):
        self._loss_ema = None
        self._epoch_start = time.monotonic()
        self._run_start = getattr(self, "_run_start", time.monotonic())
        try:
            self._steps_per_epoch = len(self.trainer.train_dataloader)
        except (TypeError, RuntimeError):
            self._steps_per_epoch = 0

    def on_train_batch_end(self, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            return  # non-finite step, already counted in training_step
        value = float(loss.detach())
        self._loss_ema = value if self._loss_ema is None else (
            0.9 * self._loss_ema + 0.1 * value
        )
        self.status.update(self._train_line(batch_idx, value))

    def _phase(self):
        # Lightning runs a validation pass before training starts. It is worth
        # seeing when it fails, but it is not an epoch and must not be labelled
        # as one.
        return "sanity check" if self.trainer.sanity_checking else (
            f"epoch {self.current_epoch + 1:>4}/{self.trainer.max_epochs:<4}"
        )

    def _train_line(self, batch_idx, value):
        total = self._steps_per_epoch or 1
        done = (batch_idx + 1) / total
        elapsed = time.monotonic() - self._epoch_start
        rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0.0
        remaining = (self.trainer.max_epochs - self.current_epoch - 1) * total
        eta = remaining / rate if rate > 0 else 0.0
        return (
            f"{self._phase()}"
            f" batch {batch_idx + 1:>5}/{total:<5}"
            f" loss {value:7.4f}"
            f" lr {self._lr():>10}"
            f"  {done * 100:5.1f}%  elapsed {int(elapsed):>5}s"
            f"  eta {int(eta) // 3600:d}:{int(eta) % 3600 // 60:02d}:"
            f"{int(eta) % 60:02d}"
        )

    def _lr(self):
        try:
            return f"{self.optimizers().param_groups[0]['lr']:.2e}"
        except (RuntimeError, AttributeError, IndexError):
            return "-"

    def on_train_epoch_end(self):
        # Lightning fires this AFTER the validation pass for the epoch, so when
        # validation ran, its line has to carry the training numbers too --
        # otherwise the two lines read out of order.
        if getattr(self, "_val_printed_epoch", None) == self.current_epoch:
            return
        metrics = self.trainer.callback_metrics
        if self._loss_ema is None:
            return
        self.status.line(
            f"{self._phase()} train  mean loss "
            f"{metrics.get('train/loss_epoch', float('nan')):.4f}"
            f"  cls {metrics.get('train/cls_loss', float('nan')):.4f}"
            f"  off {metrics.get('train/off_loss', float('nan')):.4f}"
            f"  (no validation this epoch)"
        )

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None:
            return
        total = self._val_batches or 1
        self.status.update(
            f"{self._phase()} validating {batch_idx + 1:>4}/{total:<4}"
            f" loss {float(loss.detach()):7.4f}"
        )

    def on_validation_epoch_start(self):
        self.val_vis_count = 0
        self._confusion = PhoneConfusionMatrix()
        try:
            self._val_batches = len(self.trainer.val_dataloaders[0])
        except (TypeError, IndexError, RuntimeError):
            self._val_batches = 0

    def validation_step(self, batch, batch_idx):
        inputs, labels, wavs, segs_gt, _, langs, lengths = batch
        logits, offsets = self(inputs, langs, lengths)
        loss, cls_loss, off_loss = self.calculate_loss(
            logits, offsets, labels, segs_gt, lengths
        )

        valid = labels != -100
        frame_count = int(valid.sum().item())
        total_frames = int(lengths.sum().item())
        if frame_count:
            acc = (logits.argmax(-1)[valid] == labels[valid]).float().mean() * 100
            self.log("val/acc", acc, on_step=False, on_epoch=True,
                     prog_bar=True, batch_size=frame_count)
        if total_frames:
            # Share of the validation audio with no label at all. The .lab files
            # are expected to cover every frame, so anything above zero means
            # the reported numbers come from a filtered subset.
            self.log(
                "val/unlabelled_pct",
                100.0 * (total_frames - frame_count) / total_frames,
                on_step=False, on_epoch=True, batch_size=inputs.size(0),
            )
        for name, value in (
            ("loss", loss), ("cls_loss", cls_loss), ("off_loss", off_loss)
        ):
            self.log(f"val/{name}", value, on_step=False, on_epoch=True,
                     prog_bar=name == "loss", batch_size=inputs.size(0))

        errors, phone_count = 0, 0
        files_scored, files_empty = 0, 0
        ref_bnd_n, hyp_bnd_n = 0, 0
        matched = {tol: 0 for tol in self.boundary_tolerances_ms}
        abs_err, abs_err_n = 0.0, 0

        for i, wav in enumerate(wavs):
            length = int(lengths[i].item())
            duration = len(wav) / 16000
            tags = viterbi_decode(
                logits[i, :length], self.id2label, viterbi_bias=self.viterbi_bias
            )
            pred_segments = continuous_segments(
                decode_bio_tags(
                    tags, self.frame_duration, offsets[i, :length].detach().cpu()
                ),
                duration,
            )
            # Normalise the reference the same way inference normalises its
            # output, otherwise the leading unlabelled frames show up as a
            # boundary error that no model can fix.
            gt_segments = continuous_segments(segs_gt[i], duration)

            reference = collapse_repeated(
                [ph for _, _, ph in gt_segments]
            )
            prediction = collapse_repeated(
                [ph for _, _, ph in pred_segments]
            )

            # Every file with reference phonemes counts. The previous gate
            # required a file with ZERO unlabelled frames, which silently
            # dropped any file whose tail was not annotated -- HTK .lab files
            # routinely end a few ms before the audio does.
            if reference:
                files_scored += 1
                errors_i, ref_n = phone_error_rate(reference, prediction)
                errors += errors_i
                phone_count += ref_n
                if self.cm_enabled:
                    # Only scored files, for the same reason PER skips the rest:
                    # a file with no lab phonemes has nothing to be confused
                    # with, and its predictions would inflate the insertions.
                    self._confusion.add(reference, prediction)
            else:
                files_empty += 1

            matches, err_sum, err_n, n_ref, n_hyp = boundary_counts(
                gt_segments, pred_segments, self.boundary_tolerances_ms
            )
            for tol_ms, count in matches.items():
                matched[tol_ms] += count
            abs_err += err_sum
            abs_err_n += err_n
            ref_bnd_n += n_ref
            hyp_bnd_n += n_hyp

            if self.val_vis_count < self.num_vis_samples:
                self._log_visualization(
                    wav, pred_segments, segs_gt[i], self.val_vis_count
                )
                self.val_vis_count += 1

        if files_scored + files_empty:
            self.log(
                "val/per_files_pct",
                100.0 * files_scored / (files_scored + files_empty),
                on_step=False, on_epoch=True, batch_size=inputs.size(0),
            )
        if phone_count:
            self.log("val/per", 100.0 * errors / phone_count,
                     on_step=False, on_epoch=True, prog_bar=True,
                     batch_size=phone_count)
        if abs_err_n:
            self.log("val/boundary_mae_ms", 1000.0 * abs_err / abs_err_n,
                     on_step=False, on_epoch=True, batch_size=abs_err_n)
        boundary_denom = ref_bnd_n + hyp_bnd_n
        if boundary_denom:
            for tol_ms in self.boundary_tolerances_ms:
                self.log(
                    f"val/boundary_f1@{tol_ms}ms",
                    2.0 * matched[tol_ms] / boundary_denom,
                    on_step=False, on_epoch=True,
                    prog_bar=(tol_ms == self.primary_tolerance_ms),
                    batch_size=boundary_denom,
                )
        return loss

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if not self.trainer.sanity_checking:
            # The sanity check runs as current_epoch 0, before any training. If
            # it set this, the first real epoch's train line would be suppressed.
            self._val_printed_epoch = self.current_epoch
        f1 = "  ".join(
            f"F1@{tol} {metrics.get(f'val/boundary_f1@{tol}ms', float('nan')) * 100:.1f}%"
            for tol in self.boundary_tolerances_ms
        )
        # One line per epoch. The previous version printed two, with a leading
        # blank line, which is what tore the progress bar apart.
        self.status.line(
            f"{self._phase()} VALID"
            f"  loss {metrics.get('val/loss', float('nan')):7.4f}"
            f"  acc {metrics.get('val/acc', float('nan')):6.2f}%"
            f"  PER {metrics.get('val/per', float('nan')):6.2f}%"
            f"  bMAE {metrics.get('val/boundary_mae_ms', float('nan')):5.1f}ms"
            f"  {f1}"
            f"  files {metrics.get('val/per_files_pct', float('nan')):5.1f}%"
            f"  unlab {metrics.get('val/unlabelled_pct', float('nan')):.2f}%"
            # train/loss_epoch is not reduced yet -- on_epoch metrics land after
            # this hook -- so the step EMA is the only current figure available.
            f"  | trn {self._loss_ema if self._loss_ema is not None else float('nan'):7.4f}"
            # Which phonemes, not just how many: the first pairs are the ones
            # worth fixing. The full matrix goes to TensorBoard.
            f"{self._confusion_note()}"
        )
        self._log_confusion()

    def _confusion_note(self):
        """Worst lab->decoded pairs as a suffix for the VALID line ('' if none).

        `s->SH 12` is a lab phoneme decoded as another one; `s-><del>` was never
        decoded and `<ins>->s` was decoded without the lab asking for it.
        """
        if not self.cm_enabled:
            return ""
        pairs = self._confusion.top_confusions(self.cm_console_top)
        if not pairs:
            return ""
        return "  conf " + " ".join(
            f"{ref}->{hyp} {n}" for n, ref, hyp, _ in pairs
        )

    def _log_confusion(self):
        """Confusion matrix to TensorBoard, plus raw counts on disk.

        Figure and text carry the history epoch by epoch; the CSV is overwritten
        each pass, so it always describes the model as of the last validation --
        the sanity check writes the untrained baseline first.
        """
        if not self.cm_enabled or not self._confusion.counts:
            return

        if self.cm_csv:
            try:
                self._confusion.save_csv(
                    os.path.join(self.cm_dir, "confusion_matrix.csv")
                )
            except OSError as exc:
                self.status.line(f"[!] could not write confusion_matrix.csv: {exc}")

        if not self.logger:
            return
        try:
            self.logger.experiment.add_text(
                "val/confusion_top", self._confusion.to_text(),
                global_step=self.global_step,
            )
            fig = plot_phone_confusion(
                self._confusion,
                top_k=self.cm_top_k,
                min_count=self.cm_min_count,
                normalize=self.cm_normalize,
            )
            try:
                self.logger.experiment.add_figure(
                    "val/confusion_matrix", fig, global_step=self.global_step,
                )
            finally:
                plt.close(fig)
        except Exception as exc:  # a plot must not take the epoch down with it
            self.status.line(f"[!] confusion matrix not logged: {exc}")

    def _log_visualization(self, wav, pred_segments, gt_segments, sample_idx=0):
        fig = visualize_prediction(wav, 16000, pred_segments, gt_segments)
        try:
            if self.logger:
                self.logger.experiment.add_figure(
                    f"val/prediction_{sample_idx}", fig,
                    global_step=self.global_step,
                )
        finally:
            plt.close(fig)

    def configure_optimizers(self):
        opt_name = self.config["training"].get("optimizer", "AdamW")
        lr = self.config["training"]["learning_rate"]
        decay = self.config["training"].get("weight_decay", 1e-4)
        
        try:
            opt_cls = getattr(optim, opt_name)
        except AttributeError:
            opt_cls = getattr(torch.optim, opt_name)
            
        optimizer = opt_cls(self.parameters(), lr=lr, weight_decay=decay)
        
        step_size = self.config["training"].get("lr_decay_every_n_epochs", 10)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=self.config["training"]["lr_decay_gamma"]
        )
        return [optimizer], [scheduler]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="checkpoints_micro/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to .ckpt file to resume training from")
    args = parser.parse_args()

    with open(args.config, "r") as f: 
        config = yaml.safe_load(f)
    
    pl.seed_everything(42)

    save_dir = config["output"]["save_dir"]
    phoneme_path = os.path.join(save_dir, "phonemes.txt")
    if not os.path.exists(phoneme_path):
        raise FileNotFoundError(f"Phoneme list not found at {phoneme_path}. Run preprocess.py first.")
        
    label_list = load_phoneme_list(phoneme_path)
    
    data_module = WFLDataModule(config, label_list)
    
    ft_cfg = config.get("finetune", {})
    if ft_cfg.get("enabled", False) and ft_cfg.get("checkpoint_path"):
        ckpt = ft_cfg["checkpoint_path"]
        print(f"Loading weights for fine-tuning from: {ckpt}")
        model = WFLModel.load_from_checkpoint(ckpt, config=config, label_list=label_list)
    else:
        model = WFLModel(config, label_list)

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="model-ep{epoch:02d}-{val/per:.2f}",
        auto_insert_metric_name=False,
        monitor="val/per",
        mode="min",
        save_top_k=config["training"]["max_checkpoints"],
        save_last=True,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    max_epochs = config["training"].get("max_epochs", 100)
    check_val_every_n_epoch = config["training"].get("check_val_every_n_epoch", 1)
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=pl.loggers.TensorBoardLogger(save_dir=config["training"]["log_dir"], name="lightning_logs"),
        accelerator="auto",
        devices=1,
        # Measured on T4: 1.26x speedup with 16-mixed. The frozen encoder stays
        # in fp32 inside forward() (see BIOPhonemeTagger.forward), so the gain
        # comes entirely from the trainable head, with no precision risk.
        precision=config["training"].get("precision", "16-mixed"),
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        # The built-in bar is replaced by StatusLine. Off a terminal (which is
        # what Colab gives you) tqdm stops redrawing in place and emits a full
        # line plus a ~15-line metrics table per refresh, so the cell fills up
        # with scrollback and the live numbers scroll out of view.
        enable_progress_bar=config["training"].get("progress_bar", False),
        enable_model_summary=False,
    )

    print(f"Starting Training for {max_epochs} epochs (Validation every {check_val_every_n_epoch} epochs)...")
    trainer.fit(model, data_module, ckpt_path=args.resume)

if __name__ == "__main__":
    main()
