"""Turn tagger logits into phoneme segments.

Everything here is pure decoding: it takes a (frames, labels) matrix and gives
back BIO tags or a segment list. No model, no config, no file access and no CLI
live in this module, so the decoder can be imported and tested on its own --
train.py scores it during validation, infer.py runs it in production and
tune_decode.py sweeps its parameters against a checkpoint.

The Viterbi implementations are vendored here rather than imported from a
library because the decode is part of what this project publishes: a change in
the decoder silently changes every .lab file, with no error to notice. See
infer.py's git history for the version that used librosa's private
librosa.sequence._viterbi.
"""

import numpy as np
import torch
from numba import bool, float64, int32, njit, uint16

def bio_inputs(logits, id2label):
    labels = [id2label[i] for i in range(len(id2label))]
    scores = logits.detach().double().cpu().numpy()

    if scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError("Expected logits shaped (frames, labels).")
    if not np.isfinite(scores).all():
        raise ValueError("Decoder received non-finite logits.")
    if any(
        tag != "O" and not (tag.startswith(("B-", "I-")) and len(tag) > 2)
        for tag in labels
    ):
        raise ValueError("Expected O and valid B-/I- labels.")

    keep = [i for i, tag in enumerate(labels) if tag != "O"]
    scores = scores[:, keep]
    labels = [labels[i] for i in keep]

    starts = np.array([tag.startswith("B-") for tag in labels], dtype=np.bool)
    if not starts.any():
        raise ValueError("Decoder requires at least one B- label.")

    allowed = np.array(
        [
            [
                nxt.startswith("B-")
                or (nxt.startswith("I-") and prev in (f"B-{nxt[2:]}", nxt))
                for nxt in labels
            ]
            for prev in labels
        ],
        dtype=np.bool,
    )

    return scores, labels, allowed, starts


def constrained_decode(logits, id2label):
    scores, labels, allowed, valid = bio_inputs(logits, id2label)
    preds = []
    for frame in scores:
        best = int(np.argmax(np.where(valid, frame, -np.inf)))
        preds.append(labels[best])
        valid = allowed[best]
    return preds


@njit(
    int32[:](
        float64[:, ::1],  # log_prob (T, S), C-contiguous on purpose
        int32[:],         # trans_ptr (S + 1,)
        int32[:],         # trans_idx (nnz,)
        float64[:],       # trans_val (nnz,)
        float64[:],       # log_p_init (S,)
    )
)
def _viterbi_path(log_prob, trans_ptr, trans_idx, trans_val, log_p_init):
    n_frames, n_states = log_prob.shape

    # Both tables are (frames, states), not (states, frames): the DP fills one
    # row per frame, so the writes stay sequential. Numba also miscompiles the
    # backtracking when the pointer table is stored the other way round.
    scores = np.full((n_frames, n_states), -np.inf)
    pointers = np.full((n_frames, n_states), -1, dtype=np.int32)

    for s in range(n_states):
        scores[0, s] = log_p_init[s] + log_prob[0, s]

    for t in range(1, n_frames):
        for s in range(n_states):
            best = -np.inf
            best_prev = -1
            for e in range(trans_ptr[s], trans_ptr[s + 1]):
                prev = trans_idx[e]
                cand = scores[t - 1, prev] + trans_val[e]
                if cand > best:
                    best = cand
                    best_prev = prev
            if best_prev < 0:
                continue
            scores[t, s] = best + log_prob[t, s]
            pointers[t, s] = best_prev

    state = 0
    best = -np.inf
    for s in range(n_states):
        if scores[n_frames - 1, s] > best:
            best = scores[n_frames - 1, s]
            state = s

    path = np.empty(n_frames, dtype=np.int32)
    path[n_frames - 1] = state
    for t in range(n_frames - 2, -1, -1):
        prev = pointers[t + 1, state]
        state = prev
        path[t] = state if state >= 0 else 0
    return path


def _transition_csr(transitions):
    """CSR view of a dense (source, target) log-transition matrix.

    The forward DP walks
        scores[t, s] = log_prob[t, s] + max_p (scores[t-1, p] + trans[p, s])
    so it needs the allowed SOURCES of every TARGET -- the transpose of the
    usual row-major CSR. Getting this backwards still returns a legal-looking
    path, just the wrong one, so the orientation is spelled out here.

    The BIO constraint matrix is very sparse -- from a state you may reach at
    most (#phones + 1) successors -- so the DP walks the compressed rows
    instead of all S predecessors. With ~160 tags and 1500 frames that is
    roughly 100x fewer inner iterations than the dense form.
    """
    by_target = np.isfinite(transitions).T
    trans_ptr = np.zeros(by_target.shape[0] + 1, dtype=np.int32)
    np.cumsum(by_target.sum(axis=1), out=trans_ptr[1:])
    # np.nonzero returns (row, col) pairs in row-major order; the rows of the
    # transposed mask are the TARGETS, so the second array is the source list.
    targets, sources = np.nonzero(by_target)
    return trans_ptr, sources.astype(np.int32), transitions[sources, targets]


def viterbi_decode(logits, id2label, viterbi_bias=1):
    if not np.isfinite(viterbi_bias) or viterbi_bias < 1:
        raise ValueError("viterbi_bias must be finite and at least 1.")

    scores, labels, allowed, starts = bio_inputs(logits, id2label)
    if scores.shape[0] == 0:
        return []

    log_probs = scores - np.logaddexp.reduce(scores, axis=1, keepdims=True)
    # The kernel is declared C-contiguous ([:, ::1]) so a bad layout raises
    # instead of silently returning a different path -- which is exactly what
    # happens with an 'A'-layout signature, because bio_inputs() slices columns.
    log_probs = np.ascontiguousarray(log_probs)

    transitions = np.where(allowed, 0.0, -np.inf)
    for j, tag in enumerate(labels):
        if tag.startswith("I-"):
            transitions[allowed[:, j], j] = np.log(viterbi_bias)

    initial = np.where(starts, -np.log(starts.sum()), -np.inf)
    trans_ptr, trans_idx, trans_val = _transition_csr(transitions)
    path = _viterbi_path(log_probs, trans_ptr, trans_idx, trans_val, initial)
    return [labels[int(i)] for i in path]


@njit(
    uint16[:](float64[:, :], uint16[:], bool[:], uint16[:], float64, float64, float64)
)
def _forced_align_viterbi(
    log_probs,
    target_seq,
    begin_mask,
    optional_mask,
    self_loop_penalty,
    forward_penalty,
    skip_penalty,
):
    K = len(target_seq)
    _, T = log_probs.shape
    scores = np.full((K, T), -np.inf)
    pointers = np.zeros((K, T), dtype=np.uint16)

    scores[0, 0] = log_probs[target_seq[0], 0]

    for t in range(1, T):
        for s in range(K):
            state_idx = target_seq[s]

            curr_states = np.zeros(3, dtype=np.uint16)
            curr_scores = np.zeros(3, dtype=np.float64)
            count = 0
            if begin_mask[s]:
                curr_states[count] = s
                curr_scores[count] = -np.inf
            else:
                curr_states[count] = s
                curr_scores[count] = scores[s, t - 1] + self_loop_penalty
            count += 1

            if s > 0:
                curr_states[count] = s - 1
                curr_scores[count] = scores[s - 1, t - 1] + forward_penalty
                count += 1

            if s > 1 and optional_mask[s - 1] > 0:
                curr_states[count] = s - optional_mask[s - 1] - 1
                curr_scores[count] = (
                    scores[s - optional_mask[s - 1] - 1, t - 1] + skip_penalty
                )
                count += 1

            curr_states = curr_states[:count]
            curr_scores = curr_scores[:count]
            best_state_idx = np.argmax(curr_scores)

            scores[s, t] = curr_scores[best_state_idx] + log_probs[state_idx, t]
            pointers[s, t] = curr_states[best_state_idx]

    path = np.zeros(T, dtype=np.uint16)
    path[-1] = (
        K - 2 if optional_mask[-1] > 0 and scores[-2, -1] > scores[-1, -1] else K - 1
    )

    for t in range(T - 2, -1, -1):
        path[t] = pointers[path[t + 1], t + 1]

    return path


def forced_align_viterbi(
    logits,
    id2label,
    phones,
    self_loop_penalty=-4.6,
    forward_penalty=-0.6,
    skip_penalty=-2.3,
):
    label2id = {label: id for id, label in id2label.items()}
    # turn to log probs
    log_probs = (
        torch.log_softmax(logits, dim=-1).detach().double().cpu().numpy().transpose()
    )

    # make target sequence
    target_seq = []
    begin_mask = []
    for phn in phones:
        target_seq.extend([f"B-{phn}", f"I-{phn}"])
        begin_mask.extend([True, False])
    target_seq_idx = np.array([label2id[phn] for phn in target_seq], dtype=np.uint16)
    begin_mask = np.array(begin_mask, dtype=np.bool)
    optional_mask = []
    for phn in target_seq:
        if phn.startswith("I-"):
            if len(optional_mask) > 0:
                optional_mask.append(optional_mask[-1] + 1)
            else:
                optional_mask.append(1)
        else:
            optional_mask.append(0)
    optional_mask = np.array(optional_mask, dtype=np.uint16)

    path = _forced_align_viterbi(
        log_probs,
        target_seq_idx,
        begin_mask,
        optional_mask,
        self_loop_penalty,
        forward_penalty,
        skip_penalty,
    )

    return [target_seq[p] for p in path]


def apply_hard_silence(segments, audio, sr, threshold, min_duration, silence_phoneme):
    if len(audio) == 0:
        return segments

    frame_length = int(sr * 0.01)
    if frame_length < 1:
        frame_length = 1

    pad_len = (frame_length - (len(audio) % frame_length)) % frame_length
    padded_audio = np.pad(np.abs(audio), (0, pad_len), mode="constant")

    frames = padded_audio.reshape(-1, frame_length)
    frame_max = np.max(frames, axis=1)

    is_silent_frame = frame_max < threshold

    silence_intervals = []
    in_silence = False
    start_frame = 0

    for i, silent in enumerate(is_silent_frame):
        if silent and not in_silence:
            in_silence = True
            start_frame = i
        elif not silent and in_silence:
            in_silence = False
            duration = (i - start_frame) * 0.01
            if duration >= min_duration:
                silence_intervals.append((start_frame * 0.01, i * 0.01))

    if in_silence:
        duration = (len(is_silent_frame) - start_frame) * 0.01
        if duration >= min_duration:
            silence_intervals.append((start_frame * 0.01, len(is_silent_frame) * 0.01))

    if not silence_intervals:
        return segments

    temp_segments = segments.copy()

    for sil_start, sil_end in silence_intervals:
        next_temp_segments = []
        for s_start, s_end, s_label in temp_segments:
            if s_end <= sil_start or s_start >= sil_end:
                next_temp_segments.append((s_start, s_end, s_label))
                continue

            if s_start < sil_start:
                next_temp_segments.append((s_start, sil_start, s_label))
            if s_end > sil_end:
                next_temp_segments.append((sil_end, s_end, s_label))

        temp_segments = next_temp_segments

    for s, e in silence_intervals:
        temp_segments.append((s, e, silence_phoneme))

    temp_segments.sort(key=lambda x: x[0])
    return temp_segments


def continuous_segments(segments, duration):
    if duration <= 0:
        return []
    starts = []
    for s, _, ph in sorted(segments, key=lambda seg: seg[0]):
        s = float(s)
        s = max(0.0, s)
        if s >= duration or (starts and s <= starts[-1][0]):
            continue
        starts.append((s, ph))

    if not starts:
        return []
    starts[0] = (0.0, starts[0][1])
    return [
        (s, starts[i + 1][0] if i + 1 < len(starts) else duration, ph)
        for i, (s, ph) in enumerate(starts)
    ]
