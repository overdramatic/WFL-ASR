# WFL-ASR: Whisper/WavLM for Phoneme Labeling

**WFL-ASR** is a configurable deep learning model designed for automatic phoneme segmentation using frame-level BIO tagging. It supports both Whisper and WavLM as audio encoders, and is structured for flexible and efficient training on phoneme-aligned datasets.

---

## How It Works

This model performs **frame-level phoneme labeling** using the BIO tag format (`B-`, `I-`, `O`).

### 1. Label Preprocessing
- `.lab` files define phoneme segments using HTK format.
- Each segment is converted into BIO tags aligned to time frames based on `frame_duration` (hardcoded to 20ms for Whisper compatibility).
- Tags are stored along with the audio path in a training JSON.

### 2. Feature Extraction
- **Whisper** or **WavLM** encoders process the audio waveform into frame-wise feature vectors.
  - Whisper uses fixed 20ms frame stride.
  - WavLM offers flexible windowing via HuBERT-style encoding.

### 3. Neural Architecture
The encoded features go through a stack of optional, configurable layers:

- `BiLSTM` - sequential modeling (optional)
- `Conformer Blocks` - long + short-term feature modeling
- `Dilated Conv Stack` - local context enhancement (optional)

### 4. Classification
- A linear layer maps each time step to a BIO tag.

### 5. Inference and Postprocessing
- Predict BIO tags from audio.
- Optional smoothing (median filtering) and merging for better boundary clarity.
- Convert tags back to `.lab` segments.

---

## Features

- Whisper/WavLM encoder support
- Frame-level BIO tag training
- Configurable architecture (BiLSTM, Conformer, Conv)
- HTK-compatible `.lab` output format
- Optional waveform augmentation via the `augmentation` config section

---

## Augmentation Options

The `config.yaml` file now includes an optional `augmentation` section used during training. When enabled it randomly applies volume scaling and Gaussian noise:

```yaml
augmentation:
  enable: true
  noise_std: 0.005      # standard deviation of Gaussian noise
  prob: 0.5             # probability to augment a sample
  volume_range: [0.9, 1.1]  # random scaling of audio volume
```

Disable augmentation by setting `enable: false`.

---

## Validation Metrics

Every validation pass reports:

| metric | meaning |
| --- | --- |
| `val/loss`, `val/cls_loss`, `val/off_loss` | total, tag and boundary-regression losses |
| `val/acc` | frame-level tag accuracy |
| `val/per` | phone error rate over the collapsed phoneme sequences |
| `val/boundary_mae_ms` | mean boundary position error |
| `val/boundary_f1@{20,40}ms` | boundary F-score at each tolerance |
| `val/per_files_pct` | share of validation files that had reference phonemes |
| `val/unlabelled_pct` | share of the validation audio no `.lab` covers |

### Confusion Matrix

PER says *how many* phonemes are wrong; the confusion matrix says *which*. It is
built by aligning the decoded phoneme sequence against the original `.lab` with
the same Levenshtein alignment PER uses, so its off-diagonal total **is** the PER
numerator — the matrix explains the number instead of telling a different story.

- `s -> SH` a lab phoneme decoded as another one (substitution)
- `s -> <del>` a lab phoneme never decoded (deletion)
- `<ins> -> s` a phoneme decoded without the lab asking for it (insertion)

The worst pairs are appended to the `VALID` console line, and each pass logs a
figure (`val/confusion_matrix`), a text summary (`val/confusion_top`) and the raw
counts to `<save_dir>/confusion_matrix.csv`:

```yaml
validation:
  confusion_matrix:
    enabled: true     # false skips the matrix entirely
    top_k: 25         # worst lab->decoded pairs to plot, one bar each
    min_count: 1      # drop pairs below this count
    normalize: true   # label each bar with the % of that lab phoneme too
    console_top: 3    # worst pairs appended to the VALID line
    csv: true         # raw counts to <save_dir>/confusion_matrix.csv
```

The figure is a bar chart of the worst pairs — label on the left (`s->SH`,
`s-><del>`, `<ins>->s`), count along the axis below, worst on top, one colour per
error kind (keyed in the title, since a legend box would sit on top of a bar).
Ranking by count is deliberate: a phoneme wrong 400 times costs far more PER than
one wrong twice, so the rare-but-broken phone does not sit at the top of the
chart. The trailing `%` on each bar is the share of that lab phoneme, which is how
you tell the two apart. The CSV keeps every scored pair, not just the plotted
ones.

---

### Phoneme Merging
Phonemes can be merged across languages by defining `merged_phoneme_groups` in
`config.yaml`. Each group starts with a merge label such as `merged_1` (can be anything) followed
by language specific phonemes:

```yaml
training:
   # define phonemes group that has the same sound (like-phoneme) throughout the dataset across labeling systems
  merged_phoneme_groups:
    - ["merged_1", "en/ah", "ja/a"]
    - ["merged_2", "en/ih", "ja/i"]
    - ["custom_var", "en/AP", "ja/AP"]
    - ["CustomVar", "en/SP", "ja/SP"]
```

During preprocessing these phonemes are replaced with the merged label. For
TensorBoard visualisation and inference, the labels are mapped back to the original phoneme for
the sample's language
