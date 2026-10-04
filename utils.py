import csv
import os, json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Row/column labels for the two error kinds that have no counterpart phone:
# a lab phoneme the decoder never produced, and a phoneme the decoder invented.
DELETION = "<del>"
INSERTION = "<ins>"


def decode_bio_tags(tags, frame_duration=0.02, offsets=None):
    segments = []
    curr_ph, start_idx = None, None

    def finalize(end_idx):
        nonlocal curr_ph, start_idx
        s_offset = offsets[start_idx, 0] if offsets is not None else 0.0
        s_time = (start_idx + float(s_offset)) * frame_duration

        last_frame = end_idx - 1
        if last_frame < 0:
            last_frame = 0

        # get offset for the end of the last frame
        if offsets is not None:
            if last_frame < len(offsets):
                e_offset = offsets[last_frame, 1]
            else:
                e_offset = 1.0
        else:
            e_offset = 1.0

        e_time = (last_frame + float(e_offset)) * frame_duration
        segments.append((s_time, e_time, curr_ph))
        curr_ph, start_idx = None, None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            if curr_ph:
                finalize(i)
            curr_ph = tag[2:]
            start_idx = i
        elif tag == "O":
            if curr_ph:
                finalize(i)
        elif tag.startswith("I-"):
            ph = tag[2:]
            if ph != curr_ph:
                if curr_ph:
                    finalize(i)
                curr_ph = ph
                start_idx = i

    if curr_ph:
        finalize(len(tags))
    return segments


def save_lab(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for s, e, p in segments:
            f.write(f"{int(s*1e7)} {int(e*1e7)} {p}\n")


def load_phoneme_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def load_phones_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    text = text.replace("\n", " ").replace("\t", " ")
    return [tok for tok in text.split(" ") if tok.strip()]


def load_langs(path):
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            k, v = l.strip().split(",")
            d[k] = int(v)
    return d


def load_phoneme_merge_map(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def canonical_to_lang(ph, lang, m_map):
    return m_map.get(ph, {}).get(lang, ph)


def merge_adjacent_segments(segs, mode="right"):
    if not segs or mode == "none":
        return segs
    merged = [segs[0]]
    for s, e, p in segs[1:]:
        ls, le, lp = merged[-1]
        if p == lp:
            merged[-1] = (ls, e, lp)
        else:
            merged.append((s, e, p))
    return merged


def segment_boundaries(segments, tolerance=0.001):
    """Transition times of a segment list: every start, plus the final end.

    Adjacent segments share a time (end == next start), so those duplicates are
    collapsed; counting them twice would inflate every boundary score. 1 ms is
    far below the 20 ms frame grid, so nothing meaningful is lost.
    """
    if not segments:
        return []
    ordered = sorted(segments, key=lambda seg: seg[0])
    times = [float(s) for s, _, _ in ordered]
    times.append(float(ordered[-1][1]))
    unique = []
    for t in times:
        if not unique or t - unique[-1] > tolerance:
            unique.append(t)
    return unique


def match_boundaries(reference, hypothesis, tolerance):
    """Monotone 1-1 matching of two ascending boundary lists.

    Returns the matched (reference, hypothesis) pairs. Both lists must already
    be sorted -- a sweep with a moving pointer is optimal for sorted input and
    keeps the pairing monotonic, so two predictions can never claim the same
    reference boundary.
    """
    pairs = []
    h = 0
    for r in reference:
        while h < len(hypothesis) and hypothesis[h] < r - tolerance:
            h += 1
        if h < len(hypothesis) and abs(hypothesis[h] - r) <= tolerance:
            pairs.append((r, hypothesis[h]))
            h += 1
    return pairs


def collapse_repeated(phones):
    """Drop consecutive duplicates.

    A labeller that chops one AA into A|A must not be punished for it: the
    evaluation works on phoneme sequences, not on tag transitions.
    """
    collapsed = []
    for phone in phones:
        if not collapsed or phone != collapsed[-1]:
            collapsed.append(phone)
    return collapsed


def phone_error_rate(reference, prediction):
    """Levenshtein distance between two phone sequences and the reference length."""
    row = list(range(len(prediction) + 1))
    for i, ref in enumerate(reference, 1):
        next_row = [i]
        for j, pred in enumerate(prediction, 1):
            next_row.append(min(
                row[j] + 1,
                next_row[j - 1] + 1,
                row[j - 1] + (ref != pred),
            ))
        row = next_row
    return row[-1], len(reference)


def align_phones(reference, prediction):
    """Levenshtein alignment of two phone sequences, as (lab, decoded) pairs.

    Each pair is a substitution or a correct match. `None` in a slot means the
    phone has no counterpart on the other side: lab=None is an insertion (the
    decoder emitted a phoneme the .lab does not have) and decoded=None is a
    deletion (the lab phoneme was never decoded).

    The path is an optimal edit alignment -- the same distance
    phone_error_rate() measures -- so every error PER counts lands in exactly
    one pair, which is what makes the confusion matrix below add up.
    """
    n, m = len(reference), len(prediction)
    # The whole cost table, unlike phone_error_rate()'s rolling row: the
    # traceback needs cost[i-1][j-1] of every cell, not just the last one.
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        row, prev_row = cost[i], cost[i - 1]
        ref = reference[i - 1]
        for j in range(1, m + 1):
            row[j] = min(
                prev_row[j] + 1,        # lab phoneme left undecoded (deletion)
                row[j - 1] + 1,        # phoneme decoded out of thin air (insertion)
                prev_row[j - 1] + (ref != prediction[j - 1]),
            )

    # Substitution/match first among the optimal moves, so a lab phone that was
    # decoded as itself is never reported as a delete plus an insert next door.
    pairs = []
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i - 1][j - 1] + (
            reference[i - 1] != prediction[j - 1]
        ):
            pairs.append((reference[i - 1], prediction[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((reference[i - 1], None))
            i -= 1
        else:
            pairs.append((None, prediction[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


def _ranked_labels(totals):
    """Labels by descending count, ties by name, so the order is reproducible."""
    return [
        label
        for label, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    ]


class PhoneConfusionMatrix:
    """Counts of lab phoneme -> decoded phoneme pairs over a set of files.

    PER says how many phonemes are wrong, this says which. A row is a phoneme of
    the reference .lab and a column what the decoder put in its place: the
    diagonal is correct, everything off it is one of the three error kinds --
    a substitution, DELETION (lab phoneme missed) or INSERTION (phoneme
    invented). Insertions and deletions need a slot of their own because they
    have no phone to sit in.

    `errors` is the off-diagonal total, i.e. exactly the numerator of the PER
    built from the same files, so the matrix explains the PER rather than
    telling a different story.
    """

    def __init__(self):
        self.counts = {}

    def add(self, reference, prediction):
        """Accumulate one file. Both sequences must already be collapsed."""
        for ref, hyp in align_phones(reference, prediction):
            key = (
                INSERTION if ref is None else ref,
                DELETION if hyp is None else hyp,
            )
            self.counts[key] = self.counts.get(key, 0) + 1

    @property
    def errors(self):
        return sum(n for (ref, hyp), n in self.counts.items() if ref != hyp)

    @property
    def total(self):
        """Lab phonemes counted, matches included."""
        return sum(n for (ref, _), n in self.counts.items() if ref != INSERTION)

    def row_totals(self):
        totals = {}
        for (ref, _), n in self.counts.items():
            totals[ref] = totals.get(ref, 0) + n
        return totals

    def column_totals(self):
        totals = {}
        for (_, hyp), n in self.counts.items():
            totals[hyp] = totals.get(hyp, 0) + n
        return totals

    def rows(self):
        return _ranked_labels(self.row_totals())

    def columns(self):
        return _ranked_labels(self.column_totals())

    def matrix(self, rows=None, columns=None):
        """Counts as a (rows, columns) array, in the given label order."""
        rows = self.rows() if rows is None else list(rows)
        columns = self.columns() if columns is None else list(columns)
        row_at = {label: i for i, label in enumerate(rows)}
        col_at = {label: j for j, label in enumerate(columns)}
        out = np.zeros((len(rows), len(columns)), dtype=np.int64)
        for (ref, hyp), n in self.counts.items():
            i, j = row_at.get(ref), col_at.get(hyp)
            if i is not None and j is not None:
                out[i, j] += n
        return out

    def top_confusions(self, k=3, min_count=1):
        """Worst off-diagonal cells as (count, lab, decoded, % of that lab phone).

        Sorted by how often the mistake happens, which is not the same as by
        percentage: a rare phone confused 100% of the time is one count here.
        """
        totals = self.row_totals()
        ranked = sorted(
            (
                (n, ref, hyp, 100.0 * n / totals[ref])
                for (ref, hyp), n in self.counts.items()
                if ref != hyp and n >= min_count
            ),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        return ranked[:k] if k else ranked

    def to_text(self, k=10, min_count=1):
        ranked = self.top_confusions(k, min_count)
        lines = [
            f"{len(ranked)} lab->decoded confusions out of {self.errors} errors"
            f" / {self.total} lab phonemes",
            *(f"{ref} -> {hyp}: {n} ({pct:.0f}% of {ref})" for n, ref, hyp, pct in ranked),
        ]
        if not ranked:
            lines.append("none above the min_count threshold")
        return "\n".join(lines)

    def save_csv(self, path):
        """Full matrix as counts. Row 1 is the header: `lab\\decoded`."""
        rows, columns = self.rows(), self.columns()
        if not rows or not columns:
            return
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["lab\\decoded"] + columns)
            for label, values in zip(rows, self.matrix(rows, columns)):
                writer.writerow([label] + [int(v) for v in values])


def boundary_counts(ref_segments, hyp_segments, tolerances_ms):
    """Raw boundary counts for one file, ready to be summed over a set.

    PER collapses repeated phonemes and discards all timing, so on its own it
    cannot tell a model that places boundaries well from one that does not.
    These counts back the F-score and the mean error of the boundary positions,
    which is what actually decides the output .lab for a labeller.

    Returns (matches, abs_err, matched_n, n_ref, n_hyp) where `matches` maps
    each tolerance in ms to the number of matched boundaries. Callers
    micro-average by summing, then divide.
    """
    ref_bnd = segment_boundaries(ref_segments)
    hyp_bnd = segment_boundaries(hyp_segments)
    matches = {}
    abs_err, matched_n = 0.0, 0
    if tolerances_ms:
        for tol_ms in tolerances_ms:
            pairs = match_boundaries(ref_bnd, hyp_bnd, tol_ms / 1000.0)
            matches[tol_ms] = len(pairs)
            if tol_ms == max(tolerances_ms):
                abs_err = sum(abs(r - h) for r, h in pairs)
                matched_n = len(pairs)
    return matches, abs_err, matched_n, len(ref_bnd), len(hyp_bnd)


def _to_numpy_2d(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected (T, V) logits, got shape {x.shape}")
    return x.astype(np.float64, copy=False)


def _log_softmax(x):
    m = np.max(x, axis=1, keepdims=True)
    x2 = x - m
    lse = np.log(np.sum(np.exp(x2), axis=1, keepdims=True)) + m
    return x - lse


def forced_align_bio(
    logits,
    id2label,
    phones,
    *,
    allow_end_in_last_phone=True
):
    x = _to_numpy_2d(logits)
    T, V = x.shape
    logp = _log_softmax(x)

    # Build label2id
    if isinstance(id2label, dict):
        label2id = {lab: i for i, lab in id2label.items()}
    else:
        label2id = {lab: i for i, lab in enumerate(id2label)}

    o_id = label2id.get("O", None)
    if o_id is None:
        raise ValueError("Label set must contain 'O'")

    phones = phones or []
    N = len(phones)

    b_ids = []
    i_ids = []
    for p in phones:
        b = label2id.get(f"B-{p}")
        ii = label2id.get(f"I-{p}")
        if b is None or ii is None:
            raise ValueError(f"Phoneme '{p}' missing from label set (need B-{p} and I-{p}).")
        b_ids.append(b)
        i_ids.append(ii)

    def idx_O(k): return k
    def idx_B(k): return (N + 1) + k
    def idx_I(k): return (N + 1) + N + k

    S = (N + 1) + 2 * N

    emit = np.full((T, S), -np.inf, dtype=np.float64)
    for k in range(N + 1):
        emit[:, idx_O(k)] = logp[:, o_id]
    for k in range(N):
        emit[:, idx_B(k)] = logp[:, b_ids[k]]
        emit[:, idx_I(k)] = logp[:, i_ids[k]]

    trans_from = [[] for _ in range(S)]

    # O_k -> O_k, O_k -> B_k
    for k in range(N + 1):
        trans_from[idx_O(k)].append(idx_O(k))
        if k < N:
            trans_from[idx_O(k)].append(idx_B(k))

    # B_k -> I_k
    for k in range(N):
        trans_from[idx_B(k)].append(idx_I(k))

    # I_k -> I_k, I_k -> O_{k+1}, I_k -> B_{k+1}
    for k in range(N):
        trans_from[idx_I(k)].append(idx_I(k))
        trans_from[idx_I(k)].append(idx_O(k + 1))
        if k + 1 < N:
            trans_from[idx_I(k)].append(idx_B(k + 1))

    dp = np.full((T, S), -np.inf, dtype=np.float64)
    back = np.full((T, S), -1, dtype=np.int32)

    dp[0, idx_O(0)] = emit[0, idx_O(0)]

    for t in range(1, T):
        prev = dp[t - 1]
        cur = dp[t]
        for s_prev in range(S):
            ps = prev[s_prev]
            if ps == -np.inf:
                continue
            for s_next in trans_from[s_prev]:
                sc = ps + emit[t, s_next]
                if sc > cur[s_next]:
                    cur[s_next] = sc
                    back[t, s_next] = s_prev

    end_states = [idx_O(N)]
    if allow_end_in_last_phone and N > 0:
        end_states.append(idx_I(N - 1))
    end_state = max(end_states, key=lambda s: dp[T - 1, s])

    path = [end_state]
    for t in range(T - 1, 0, -1):
        p = back[t, path[-1]]
        if p < 0:
            p = idx_O(0)
        path.append(p)
    path.reverse()

    # States -> tags
    tags = []
    for s in path:
        if s <= N:
            tags.append("O")
        elif s < (N + 1) + N:
            k = s - (N + 1)
            tags.append(f"B-{phones[k]}")
        else:
            k = s - ((N + 1) + N)
            tags.append(f"I-{phones[k]}")
    return tags


# One colour per error kind, so a missed phone reads differently from a phone
# decoded as its neighbour without having to decode the label.
ERROR_KINDS = {
    "substitution": "#4C72B0",
    "deletion": "#DD8452",
    "insertion": "#C44E52",
}

# The colour key goes in the title rather than in a legend box: on a chart whose
# bars start at the left edge there is no corner a legend can sit in without
# covering a bar, and the worst bars are the ones worth looking at.
ERROR_KIND_HINTS = {
    "substitution": "blue = decoded as another phoneme",
    "deletion": "orange = <del>, never decoded",
    "insertion": "red = <ins>, decoded out of nothing",
}


def error_kind(lab, decoded):
    if lab == INSERTION:
        return "insertion"
    return "deletion" if decoded == DELETION else "substitution"


def plot_phone_confusion(confusion, *, top_k=25, min_count=1, normalize=True, title=None):
    """Bar chart of the worst lab -> decoded confusions, worst on top.

    One bar per confused pair, labelled on the left as `s->SH`, `s-><del>` or
    `<ins>->s`, with the count along the axis below. Ranking by bar length
    answers "what do I fix first" in one look: a phoneme wrong 400 times costs
    far more PER than one wrong twice, and a heatmap of percentages would rank
    the rare broken phoneme above it. With `normalize` each bar also carries the
    share of that lab phoneme, which is what separates "this phone is rare and
    messy" from "this phone is broken".

    Pairs below `min_count` are dropped and the `top_k` worst are kept; the full
    untruncated counts are in the CSV, not here.
    """
    ranked = confusion.top_confusions(top_k, min_count)

    fig, ax = plt.subplots(figsize=(10.0, 0.34 * len(ranked) + 1.8))
    if not ranked:
        ax.text(
            0.5, 0.5, "no confusions above the threshold", ha="center",
            va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        ax.set_title(_confusion_title(confusion, title), fontsize=10)
        return fig

    counts = [n for n, *_ in ranked]
    labels = [f"{ref}->{hyp}" for _, ref, hyp, _ in ranked]
    colours = [ERROR_KINDS[error_kind(ref, hyp)] for _, ref, hyp, _ in ranked]

    # barh draws the first row at the bottom; invert so the worst is on top.
    positions = np.arange(len(ranked))
    ax.barh(positions, counts, color=colours, height=0.68)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9, fontfamily="monospace")
    ax.invert_yaxis()

    longest = max(counts)
    ax.set_xlim(0, longest * 1.16)  # room for the value at the end of the bar
    ax.set_xlabel(
        "count" if not normalize
        else "count   --   trailing % is the share of that lab phoneme",
        fontsize=9,
    )
    ax.xaxis.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (n, _, _, pct) in enumerate(ranked):
        text = f"{n}  {pct:.0f}%" if normalize else str(n)
        ax.text(
            n + longest * 0.015, i, text, va="center", ha="left", fontsize=8,
            color="black",
        )

    kinds = {error_kind(ref, hyp) for _, ref, hyp, _ in ranked}
    heading = _confusion_title(confusion, title)
    if len(kinds) > 1:
        heading += "\n" + "   ".join(
            ERROR_KIND_HINTS[kind] for kind in ERROR_KINDS if kind in kinds
        )
    ax.set_title(heading, fontsize=10)
    fig.tight_layout()
    return fig


def _confusion_title(confusion, title=None):
    if title is not None:
        return title
    return (
        f"lab -> decoded: {confusion.errors} errors / {confusion.total} "
        f"lab phonemes"
    )


def visualize_prediction(wav, sr, pred, gt=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.linspace(0, len(wav) / sr, len(wav)), wav, color="lightblue", alpha=0.85, linewidth=1)

    ax.set_ylim(-1.3, 1.3)
    ax.set_xlim(0, len(wav) / sr)
    ax.set_yticks([])

    # pred
    y_pred = 0.8
    for s, e, p in pred:
        ax.axvline(s, color="red", alpha=0.6, linestyle="--", linewidth=1)
        if e - s > 0.02:
            ax.text((s + e) / 2, y_pred, p, color="red", ha="center", va="center", fontsize=12, fontweight="bold")

    # gt
    y_gt = -0.8
    if gt:
        for item in gt:
            s, e, p = item[:3]
            ax.axvline(s, color="green", alpha=0.6, linewidth=1)
            if e - s > 0.02:
                ax.text((s + e) / 2, y_gt, p, color="green", ha="center", va="center", fontsize=12, fontweight="bold")

    legend_elements = [
        Line2D([0], [0], color="red", marker="o", linestyle="none", label="Pred"),
        Line2D([0], [0], color="green", marker="o", linestyle="none", label="GT"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig
