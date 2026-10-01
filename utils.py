import os, json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


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


def phone_error_detail(reference, prediction):
    """Levenshtein WITH a traceback, so the errors can be attributed.

    `phone_error_rate` returns (errors, n_reference), which is all the PER needs.
    It cannot say WHICH phonemes fail or HOW -- a substitution, a deletion and an
    insertion all collapse into one integer. Recovering the alignment path is
    what separates "the model merged /ɾ/ into /l/" from "the model never emitted
    /ɾ/ at all", and those need different fixes.

    Returns:
        S, D, I    substitution / deletion / insertion counts
        confusion  {(ref, hyp): n}, substitutions only
        per_phone  {phone: {"n": occurrences, "S": .., "D": ..}}
    """
    n, m = len(reference), len(prediction)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + (reference[i - 1] != prediction[j - 1]),
            )

    subs = dels = ins = 0
    confusion = {}
    stats = {}
    i, j = n, m
    while i > 0 or j > 0:
        # Diagonal first: matching (or substituting) is always preferred over
        # deleting, so the traceback reports the cheapest explanation.
        if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + (
            reference[i - 1] != prediction[j - 1]
        ):
            ref = reference[i - 1]
            entry = stats.setdefault(ref, {"n": 0, "S": 0, "D": 0})
            entry["n"] += 1
            if ref != prediction[j - 1]:
                subs += 1
                entry["S"] += 1
                key = (ref, prediction[j - 1])
                confusion[key] = confusion.get(key, 0) + 1
            i, j = i - 1, j - 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            ref = reference[i - 1]
            entry = stats.setdefault(ref, {"n": 0, "S": 0, "D": 0})
            entry["n"] += 1
            dels += 1
            entry["D"] += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return {
        "S": subs, "D": dels, "I": ins,
        "confusion": confusion, "per_phone": stats,
    }


class PhoneErrorAccumulator:
    """Sums S/D/I, the confusion pairs and the per-phone table over many files.

    Per-phone rates measured on a handful of files are noise: a phoneme seen
    twice with one error reads 50%. `min_count` exists so the ranking means
    something -- below that threshold a phone is excluded instead of topping
    the list.
    """

    def __init__(self, min_count=2):
        self.min_count = min_count
        self.reset()

    def reset(self):
        self.S = self.D = self.I = 0
        self.n_ref = 0
        self.files_scored = 0
        self.files_empty = 0
        self.confusion = {}
        self.per_phone = {}

    def update(self, reference, prediction):
        """Fold one file into the totals. Returns this file's detail, or None.

        The per-file return is what lets the caller hand the same numbers to
        Lightning's `self.log`. Logging has to happen inside the step, not in
        `on_validation_epoch_end`: by then Lightning has already reduced the
        epoch, so a value logged there is not aggregated like the others. Doing
        the DP once and logging from both sides keeps `val/per` and the console
        line provably identical.
        """
        if not reference:
            self.files_empty += 1
            return None
        out = phone_error_detail(reference, prediction)
        self.files_scored += 1
        self.S += out["S"]
        self.D += out["D"]
        self.I += out["I"]
        for key, count in out["confusion"].items():
            self.confusion[key] = self.confusion.get(key, 0) + count
        for ph, v in out["per_phone"].items():
            slot = self.per_phone.setdefault(ph, {"n": 0, "S": 0, "D": 0})
            slot["n"] += v["n"]
            slot["S"] += v["S"]
            slot["D"] += v["D"]
        self.n_ref = sum(v["n"] for v in self.per_phone.values())
        return out

    def rates(self):
        """Error rates as percentages of the reference length."""
        total = max(self.n_ref, 1)
        errors = self.S + self.D + self.I
        return {
            "per": 100.0 * errors / total,
            "sub": 100.0 * self.S / total,
            "del": 100.0 * self.D / total,
            "ins": 100.0 * self.I / total,
        }

    def phone_error_rate_table(self):
        """{phone: error percent}, only for phones seen often enough to rank."""
        out = {}
        for ph, v in self.per_phone.items():
            if v["n"] < self.min_count:
                continue
            out[ph] = 100.0 * (v["S"] + v["D"]) / v["n"]
        return out

    def worst_phones(self, limit=10):
        table = self.phone_error_rate_table()
        return sorted(table.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]

    def top_confusions(self, limit=10):
        return sorted(self.confusion.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]

    def summary_line(self, limit=8):
        """One line with the S/D/I split -- the fastest read on WHAT broke."""
        if not self.n_ref:
            return "phone S/D/I: (no reference phonemes)"
        r = self.rates()
        return (
            f"phones S {r['sub']:.2f}%  D {r['del']:.2f}%  I {r['ins']:.2f}%"
            f"  (PER {r['per']:.2f}%, {self.n_ref} ref, {self.files_scored} files)"
        )

    def detail_line(self, limit=8):
        """Worst phones and the pairs that eat them, for the status console."""
        worst = self.worst_phones(limit)
        pairs = self.top_confusions(limit)
        phones = "  ".join(f"{ph} {er:.0f}%" for ph, er in worst) or "n/a"
        conf = "  ".join(f"{r}->{h} {c}" for (r, h), c in pairs) or "n/a"
        return f"worst phones: {phones}   |   top confusions: {conf}"


def plot_confusion_matrix(confusion, limit=15, title="Substituicoes (ref -> hip)"):
    """Horizontal bar chart of the most frequent substitution pairs.

    Counts span orders of magnitude (a common pair can be 200x a rare one), so
    the axis is log10 -- otherwise every pair but the top one is invisible.
    Returns None when there is nothing to draw, so the caller can skip the
    figure instead of logging an empty axes.
    """
    if not confusion:
        return None
    pairs = sorted(confusion.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    if not pairs:
        return None

    labels = [f"{ref} -> {hyp}" for (ref, hyp), _ in pairs]
    counts = np.array([count for _, count in pairs], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.45 * len(pairs)), 3.2))
    ax.barh(range(len(pairs)), np.log10(np.maximum(counts, 1.0)), color="#c0392b")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("log10(contagem)", fontsize=8)
    ax.set_title(
        f"{title} -- top {len(pairs)} de {int(counts.sum())} substituicoes",
        fontsize=9,
    )
    plt.tight_layout()
    return fig


def enforce_min_duration(segments, min_seconds, max_passes=8):
    """Stretches too-short segments by taking time from their neighbours.

    Only boundaries move: no phoneme is added, removed or reordered, which is
    what makes this safe for forced alignment (the sequence is fixed by the
    transcript) and acceptable for free decoding (nothing constrains it). A
    neighbour is never pushed below `min_seconds` either, so a dense stretch of
    short phones simply stays short rather than propagating the violation
    outwards forever.

    Returns the segments unchanged if there is no spare time to redistribute.
    """
    if not segments or min_seconds <= 0 or len(segments) < 2:
        return list(segments)

    segs = [[float(s), float(e), ph] for s, e, ph in segments]
    for _ in range(max_passes):
        changed = False
        for i in range(len(segs)):
            deficit = min_seconds - (segs[i][1] - segs[i][0])
            if deficit <= 1e-9:
                continue

            neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(segs)]
            spare = {
                j: max(0.0, (segs[j][1] - segs[j][0]) - min_seconds)
                for j in neighbours
            }
            available = sum(spare.values())
            if available <= 1e-9:
                continue

            take = min(deficit, available)
            for j in neighbours:
                if spare[j] <= 0.0:
                    continue
                share = take * (spare[j] / available)
                if j < i:      # the left neighbour gives up its tail
                    segs[i][0] -= share
                    segs[j][1] -= share
                else:          # the right neighbour gives up its head
                    segs[i][1] += share
                    segs[j][0] += share
            changed = True
        if not changed:
            break

    return [(s, e, ph) for s, e, ph in segs]


def insert_silence_phones(
    transcript, free_segments, silence_phoneme, min_silence=0.05
):
    """Puts the silence phoneme into the transcript wherever the free decode found a pause.

    Forced alignment spreads the transcript's phonemes over EVERY frame,
    pauses included. If the transcript carries no silence token, the aligner
    has to stretch vowels and consonants across the gaps, and the degradation
    lands on exactly the common phonemes -- the opposite of what a forced
    aligner is for.

    The transcript is preserved: no phoneme is replaced, dropped or reordered,
    only silence tokens are added at the gaps where the unconstrained decode
    heard a pause. With no pauses the transcript comes back unchanged, so
    turning this on is safe for files that are already continuous speech.

    The mapping from free-decode positions to transcript positions is by index,
    which is an approximation: the two sequences are not the same length. It
    degrades to "insert silence roughly here", which is still far better than
    ignoring the pauses entirely.
    """
    if not transcript or not free_segments or not silence_phoneme:
        return list(transcript)

    # Free-decode sequence, collapsed, with pauses too short to be real dropped.
    free_seq = []
    for start, end, ph in free_segments:
        if ph == silence_phoneme and (end - start) < min_silence:
            continue
        if not free_seq or free_seq[-1] != ph:
            free_seq.append(ph)

    # Free index -> transcript position. len(transcript) means "after the last
    # phoneme", i.e. trailing silence.
    insert_at = {
        min(i, len(transcript))
        for i, ph in enumerate(free_seq)
        if ph == silence_phoneme
    }
    if not insert_at:
        return list(transcript)

    padded = []
    for k, ph in enumerate(transcript):
        if k in insert_at and (not padded or padded[-1] != silence_phoneme):
            padded.append(silence_phoneme)
        padded.append(ph)
    if len(transcript) in insert_at and padded[-1] != silence_phoneme:
        padded.append(silence_phoneme)
    return padded


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
