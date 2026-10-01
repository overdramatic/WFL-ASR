"""Turn tagger logits into phoneme segments.

Everything here is pure decoding: it takes a (frames, labels) matrix and gives
back BIO tags or a segment list. No model, no config and no CLI live in this
module, so the decoder can be imported and tested on its own -- train.py scores
it during validation, infer.py runs it in production and tune_decode.py sweeps
its parameters against a checkpoint.

The one file this module reads is the phoneme map, and that is deliberate: the
phoneme inventory belongs to the corpus, not to the code, because a model can
be trained on any language with any dictionary. See load_phoneme_types().

The Viterbi implementations are vendored here rather than imported from a
library because the decode is part of what this project publishes: a change in
the decoder silently changes every .lab file, with no error to notice. See
infer.py's git history for the version that used librosa's private
librosa.sequence._viterbi.
"""

import os

import numpy as np
import torch
import yaml
from numba import bool, float64, int32, njit, uint16

# Duration tiers, ordered by typical length: vowel > sonorant > fricative >
# stop. All the self-loop prior has to say is how long a phoneme is allowed to
# stay open, so these are duration classes and not phonetic ones -- which is why
# a language contributes a `type` (see phoneme_map.yaml) and only the code knows
# how each type maps onto a tier.
#
# Splitting the old three-way split in two is what buys the plosives back. A
# 15 ms burst and an 80 ms fricative are not the same length, and lumping both
# with the vowels is exactly what let a confident but wrong phoneme swallow its
# neighbour, which reads as a fusion and lands on the common phonemes.
PHONE_CLASSES = ("vowel", "sonorant", "fricative", "stop")

# phoneme_map.yaml's `type` vocabulary -> duration class. Adding a language
# means adding entries to that file, not to this table: this only records how
# long each KIND of sound tends to be, and that does not change with the
# language.
TYPE_CLASSES = {
    "vowel": "vowel",
    "semivowel": "sonorant",
    "trill": "sonorant",
    "liquid": "sonorant",
    "nasal": "sonorant",
    "rhotic": "sonorant",
    "fricative": "fricative",
    "affricate": "fricative",
    "aspirate": "fricative",
    "stop": "stop",
}

# A phoneme missing from the map lands in the middle tier on purpose: it is the
# value closest to both a long vowel and a short burst, so guessing wrong here
# costs less than guessing wrong at either end.
DEFAULT_TYPE = "sonorant"
DEFAULT_CLASS = "sonorant"

# "obstruent" was a single class before phoneme_map.yaml existed, and the
# configs in this repository still use it. Kept as an alias so those files keep
# working untouched; it expands to the two classes it was later split into.
CLASS_ALIASES = {"obstruent": ("fricative", "stop")}


class PhonemeTypes:
    """The {phoneme: type} inventory of a corpus, resolved per language.

    The map is the user's, not the code's: a model can be trained on any
    language with any dictionary, so the phoneme -> type assignment lives in a
    file next to the corpus instead of in a table here.

    Lookups are language-aware because the same string is not the same sound
    everywhere: "j" is a fricative in pt and an affricate in ja, "zh" is an
    affricate in zh and a fricative in ko, and reading either as the other
    gives a wrong duration prior for a phoneme the corpus does contain. A
    language is known at every call site -- preprocess.py puts each .lab under
    its language directory, train.py has lang_ids in the batch, infer.py has
    lang_name -- so the language is asked for rather than thrown away.

    Two fallbacks sit behind the language-specific entry, in this order:

    1. the COMMON set (SP, AP, cl, vf, exh), written without a prefix and
       declared to work in every language;
    2. the bare phoneme, matched against every language, which is what makes a
       merged label resolvable: `merged_phoneme_groups` rewrites pt/a to the
       canonical "aa", and "aa" is typed under en, so without this fallback a
       merged vowel would silently become unrecognised.

    Anything left over is DEFAULT_TYPE, the middle duration tier, so an unknown
    phoneme costs a compromise rather than a distortion.
    """

    def __init__(self, by_lang, common, bare):
        self._by_lang = by_lang  # {(lang, phoneme): type}
        self._common = common    # {phoneme: type}, no language prefix
        self._bare = bare        # {phoneme: type}, best type across languages

    def get(self, phoneme, lang=None):
        # COMMON first: those entries are declared to work in every language, so
        # a same-named entry under some language is a duplicate, not an
        # override. Letting the language entry win would make SP resolve to a
        # different type per corpus, which is the opposite of what COMMON is for.
        if phoneme in self._common:
            return self._common[phoneme]
        if lang is not None:
            found = self._by_lang.get((lang, phoneme))
            if found is not None:
                return found
        return self._bare.get(phoneme, DEFAULT_TYPE)

    def types(self):
        """Every type the file declares, so resolve_penalties covers them all."""
        return set(self._by_lang.values()) | set(self._common.values())

    def __len__(self):
        return len(self._by_lang) + len(self._common)


def load_phoneme_types(path, warn=print):
    """Read a phoneme_map.yaml into a PhonemeTypes.

    Raises FileNotFoundError if there is no map and ValueError if the map is
    malformed -- including a `type` outside TYPE_CLASSES. That one is a hard
    error on purpose: an unrecognised type means those phonemes silently fell
    back to the middle tier, which is the difference between the per-type
    penalties working and not working, and nothing downstream would say so. A
    typo ("afficate") is exactly the case where a warning gets ignored and the
    run finishes with numbers that look fine and are not.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No phoneme map at {path}")

    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}

    entries = doc.get("symbols") if isinstance(doc, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a top-level 'symbols:' list.")

    by_lang, common = {}, {}
    unknown = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: symbols[{i}] is not a mapping.")
        # Not str() on purpose. YAML 1.1 resolves a bare `on`, `off`, `yes`,
        # `no`, `y`-like and `n`-like scalars to booleans, so the Portuguese
        # vowel `on` [õ] arrives here as Python True. str(True) is "True",
        # which is a plausible-looking phoneme that appears in no corpus and
        # silently replaces the real one -- so refuse it and say which entry.
        raw_symbol = entry.get("symbol")
        raw_type = entry.get("type")
        if not isinstance(raw_symbol, str) or not isinstance(raw_type, str):
            offenders = []
            if not isinstance(raw_symbol, str):
                offenders.append(f"symbol={raw_symbol!r}")
            if not isinstance(raw_type, str):
                offenders.append(f"type={raw_type!r}")
            raise ValueError(
                f"{path}: symbols[{i}] has non-string "
                + " and ".join(offenders)
                + f" (entry: {entry!r}). YAML reads a bare 'on', 'off', "
                "'yes', 'no', 'true' or 'false' as a boolean, so a phoneme "
                "like pt `on` [õ] has to be quoted: {symbol: \"on\", ...}."
            )
        symbol = raw_symbol.strip()
        ph_type = raw_type.strip()
        if not symbol or not ph_type:
            raise ValueError(f"{path}: symbols[{i}] needs 'symbol' and 'type'.")
        if ph_type not in TYPE_CLASSES:
            unknown.setdefault(ph_type, []).append(symbol)
            continue

        if "/" in symbol:
            lang, phoneme = symbol.split("/", 1)
            by_lang.setdefault((lang, phoneme), ph_type)
        else:
            common.setdefault(symbol, ph_type)

    if unknown:
        # Every offending entry is listed, not just the first: fixing one typo
        # at a time across a 350-line file is not a reasonable workflow.
        detail = "; ".join(
            f"{t!r} on {', '.join(symbols[:6])}"
            + (f" (+{len(symbols) - 6} more)" if len(symbols) > 6 else "")
            for t, symbols in sorted(unknown.items())
        )
        raise ValueError(
            f"{path}: unknown phoneme type(s) not in decode.TYPE_CLASSES: "
            f"{detail}. Expected one of: {', '.join(sorted(TYPE_CLASSES))}."
        )

    # The bare fallback resolves a merged or otherwise language-less label. Where
    # languages disagree, the most frequent type wins and ties go to the entry
    # read first, so the result depends only on the file.
    counts = {}
    for (_, phoneme), ph_type in by_lang.items():
        counts.setdefault(phoneme, {}).setdefault(ph_type, []).append(phoneme)
    bare = {
        phoneme: max(seen, key=lambda t: (len(seen[t]), t))
        for phoneme, seen in counts.items()
    }
    # COMMON overrides the language-less table, matching the lookup order in
    # get(): a language-free label for SP must read as SP, not as whatever one
    # language happened to spell it.
    bare.update(common)

    if not by_lang and not common:
        warn(f"[phoneme_map] {path}: 'symbols:' is empty; every phoneme falls "
             f"back to {DEFAULT_TYPE!r}.")

    return PhonemeTypes(by_lang, common, bare)


def phone_type(phoneme, phoneme_types=None, lang=None):
    """The phoneme's type for `lang`; DEFAULT_TYPE when there is no map."""
    if phoneme_types is None:
        return DEFAULT_TYPE
    return phoneme_types.get(phoneme, lang)


def phone_class(phoneme, phoneme_types=None, lang=None):
    """The duration class of a phoneme. Unknown phonemes land in the middle."""
    return TYPE_CLASSES.get(
        phone_type(phoneme, phoneme_types, lang), DEFAULT_CLASS
    )


def resolve_penalties(by_class, default, phoneme_types=None):
    """Flatten a per-class or per-type map into one number per type.

    For a phoneme of type T in class C the lookup is by_class[T], then
    by_class[C], then the scalar default. A config may therefore name either
    granularity and get what it expects: `stop` is both a type and a class and
    resolves to the same value either way, `sonorant` is only a class and covers
    every type in it, and `obstruent` still resolves as the alias for the two
    classes it was split into.

    Returns {} for an empty input, which callers read as "no per-class override,
    use the scalar".
    """
    if not by_class:
        return {}

    # A class-level key stands for every type inside it, so expand the aliases
    # and the class names once and leave the lookup below a plain dict get.
    expanded = {}
    for key, value in by_class.items():
        for target in CLASS_ALIASES.get(key, (key,)):
            expanded.setdefault(target, float(value))

    used = set(TYPE_CLASSES) | {DEFAULT_TYPE}
    if phoneme_types is not None:
        used |= phoneme_types.types()
    return {
        ph_type: expanded.get(
            ph_type,
            expanded.get(TYPE_CLASSES.get(ph_type, DEFAULT_CLASS), default),
        )
        for ph_type in sorted(used)
    }



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


def viterbi_decode(
    logits,
    id2label,
    viterbi_bias=None,
    stickiness_bonus=None,
    penalty_by_class=None,
    phoneme_types=None,
    lang=None,
):
    """Viterbi over the BIO tags, with no transcript to constrain it.

    `stickiness_bonus` (>= 0) is added to the transition INTO an `I-` tag, so a
    larger value makes the decoder stay on the current phoneme longer. It is a
    BONUS and the name says so. `viterbi_bias` (>= 1) is the older spelling of
    the same number, log(bias) per frame. Passing both is an error rather than a
    silent precedence rule.

    WHY BONUS HERE AND PENALTY IN THE FORCED ALIGNER
    ------------------------------------------------
    The two paths look symmetric and are not, and getting this wrong shatters
    every phoneme into one-frame fragments:

    - Free decode: leaving the current phoneme is FREE (the transition to any
      `B-` costs zero). So the `I-` value alone decides how long a phoneme
      lasts, and a NEGATIVE value pays the decoder to leave. That is a
      fragmentation prior wearing a penalty's name.
    - Forced aligner: the DP is already committed to phoneme `s` and only
      chooses between staying (`self_loop_penalty`) and advancing
      (`forward_penalty`). Both are costs there, so negative is right, and a
      more negative `self_loop_penalty` closes a phoneme sooner.

    Unifying the sign across the two is what this function used to claim, and it
    silently inverted the free decode: `viterbi_bias: 5` became a penalty of
    -1.61 on staying, so the decoder preferred to leave after every single frame.
    The two numbers are NOT comparable in magnitude for the same reason -- see
    forced_align_viterbi. Keep them separate and keep the names honest.

    `penalty_by_class` overrides the scalar per phonetic type or duration class
    (see resolve_penalties). This is the knob that matters for common phonemes:
    one global value has to stretch a 120 ms vowel and a 15 ms plosive burst at
    the same time, and ends up good at neither.

    `phoneme_types` is the map from load_phoneme_types() and `lang` the name of
    the language being decoded; the map is keyed per language because the same
    string is not the same sound everywhere. Without them every phoneme gets the
    same class and the override collapses to the scalar.
    """
    if viterbi_bias is not None and stickiness_bonus is not None:
        raise ValueError(
            "Give viterbi_bias or stickiness_bonus, not both: they are the "
            "same quantity spelled two ways."
        )

    if stickiness_bonus is None:
        bias = 1 if viterbi_bias is None else viterbi_bias
        if not np.isfinite(bias) or bias < 1:
            raise ValueError("viterbi_bias must be finite and at least 1.")
        bonus = float(np.log(bias))
    else:
        if not np.isfinite(stickiness_bonus):
            raise ValueError("stickiness_bonus must be finite.")
        bonus = float(stickiness_bonus)
        if bonus < 0:
            # Raised, not absorbed. A negative bonus is the fragmentation prior
            # described above, and it does not look wrong in a config or in
            # training curves -- it just quietly produces one phoneme per frame.
            raise ValueError(
                f"stickiness_bonus must be >= 0, got {bonus:g}. This decoder "
                "rewards staying on a phoneme, so a negative value makes it "
                "leave after every frame and shatters the output. If you are "
                "porting a self_loop_penalty from the forced aligner, negate it: "
                f"{bonus:g} -> {-bonus:g}, or use viterbi_bias={np.exp(-bonus):.4g}."
            )

    scores, labels, allowed, starts = bio_inputs(logits, id2label)
    if scores.shape[0] == 0:
        return []

    log_probs = scores - np.logaddexp.reduce(scores, axis=1, keepdims=True)
    # The kernel is declared C-contiguous ([:, ::1]) so a bad layout raises
    # instead of silently returning a different path -- which is exactly what
    # happens with an 'A'-layout signature, because bio_inputs() slices columns.
    log_probs = np.ascontiguousarray(log_probs)

    transitions = np.where(allowed, 0.0, -np.inf)
    by_type = resolve_penalties(penalty_by_class, bonus, phoneme_types)
    # The per-class map gets the same sign check as the scalar, because it is a
    # far easier way in: `vowel_penalty=-1.79` looks like the old cost
    # convention, and a sweep grid accepts any float without complaint. Without
    # this, one negative class shatters exactly that class's phonemes while the
    # rest decode fine, which reads as a data problem rather than a sign.
    if penalty_by_class:
        bad = {k: v for k, v in penalty_by_class.items() if float(v) < 0}
        if bad:
            raise ValueError(
                f"penalty_by_class has negative values {bad}. On this decoder "
                "those are a COST for staying on a phoneme, which fragments "
                "every phoneme in that class into one-frame segments. Negate "
                "them (they are bonuses): "
                + ", ".join(f"{k}: {-float(v):g}" for k, v in sorted(bad.items()))
            )
    for j, tag in enumerate(labels):
        if tag.startswith("I-"):
            transitions[allowed[:, j], j] = by_type.get(
                phone_type(tag[2:], phoneme_types, lang), bonus
            )

    initial = np.where(starts, -np.log(starts.sum()), -np.inf)
    trans_ptr, trans_idx, trans_val = _transition_csr(transitions)
    path = _viterbi_path(log_probs, trans_ptr, trans_idx, trans_val, initial)
    return [labels[int(i)] for i in path]


@njit(
    uint16[:](
        float64[:, :],  # log_probs (K, T)
        uint16[:],      # target_seq
        bool[:],        # begin_mask
        uint16[:],      # optional_mask
        float64[:],     # self_loop_pen (K,) -- per target state
        float64,        # forward_penalty
        float64,        # skip_penalty
    )
)
def _forced_align_viterbi(
    log_probs,
    target_seq,
    begin_mask,
    optional_mask,
    self_loop_pen,
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
                curr_scores[count] = scores[s, t - 1] + self_loop_pen[s]
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
    self_loop_penalty_by_class=None,
    phoneme_types=None,
    lang=None,
):
    """Align a KNOWN phone sequence to the frames.

    Here `self_loop_penalty` really is a penalty: a non-positive cost per frame
    spent on the current phone. The sign is the opposite of the free decoder's
    `stickiness_bonus`, on purpose -- see the long note in viterbi_decode for why
    the two paths are not symmetric.

    `self_loop_penalty_by_class` overrides it per phonetic type or duration class
    (see resolve_penalties), which matters here more than in the free path: the
    sequence is fixed, so the ONLY freedom the DP has is where it puts the
    boundaries, and a scalar that suits a 120 ms vowel will happily eat the
    15 ms burst of a plosive.

    Note the magnitude is not comparable with the free decoder's. -4.6 nats per
    frame is a much stronger preference than the free path's log(5) = 1.6,
    because the forced path knows the phone order and only has to place
    boundaries, while the free path also has to decide what was said. They were
    tuned separately; sweep them on the same validation split (tune_decode.py).
    """
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

    # Per-state self-loop cost. B- states are unreachable from themselves (the
    # kernel gives them -inf), so their entry is never read; it is set to 0.0
    # rather than to a penalty that would look like it does something.
    by_type = resolve_penalties(
        self_loop_penalty_by_class, self_loop_penalty, phoneme_types
    )
    self_loop_pen = np.array(
        [
            by_type.get(
                phone_type(tag[2:], phoneme_types, lang), self_loop_penalty
            )
            if tag.startswith("I-") else 0.0
            for tag in target_seq
        ],
        dtype=np.float64,
    )

    path = _forced_align_viterbi(
        log_probs,
        target_seq_idx,
        begin_mask,
        optional_mask,
        self_loop_pen,
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
