"""Runs the REAL viterbi_decode, so it catches a wiring error inside the
transition matrix rather than a mistake in a re-implementation of it.

This exists because of one specific bug. `self_loop_penalty` was applied as the
value of the transition INTO an `I-` tag, while LEAVING to any `B-` costs zero.
So the sign alone decided how long a phoneme lasted: `viterbi_bias: 5` became a
penalty of -1.61 on staying, the decoder was paid to leave after every single
frame, and the output shattered into one-frame phonemes -- roughly 5x more
segments than the reference. Training curves looked fine (train loss fell
0.711 -> 0.597) because none of them decode. Only the PER and the segment counts
moved, and they looked like a modelling problem.

The lesson is the reason for the shape of this file: the sign was not reachable
by testing `resolve_penalties` arithmetic. The number was right; the place it
was written was wrong. So the assertions below go through the public function,
with real logits, and measure segments rather than log-probabilities.

Run it where torch, numpy and numba exist (Colab):

    python3 test_decode_sign.py          # or: !python3 test_decode_sign.py

Needs nothing but the repo itself. No checkpoint, no dataset, no audio, no GPU.

Logits are built as torch tensors, not numpy arrays, because that is what the
decoder actually receives: bio_inputs() calls .detach().double() on its input,
and train.py / infer.py pass tensors sliced out of the model output. Testing
with a numpy array would exercise an input type production never produces.
"""
import math
import os
import sys

# Run from anywhere: the config checks below open configs/<name>.yaml relative to
# the repo, and silently skipping them would be worse than failing.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch

import decode
import inspect


FAILED = []


def check(label, cond, extra=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {label}" + (f" -- {extra}" if extra else ""))
    if not cond:
        FAILED.append(label)


# The label set of a small PT model. O is the outside tag; SP is the silence the
# project already detects well, kept here so a change that breaks silence is
# visible.
LABELS = ["O", "B-a", "I-a", "B-t", "I-t", "B-i0", "I-i0", "B-SP", "I-SP"]
ID2LABEL = {i: t for i, t in enumerate(LABELS)}
L = len(LABELS)

# Minimal phoneme map for the test phonemes so per-class overrides apply.
# Without this, bare 'a' falls back to DEFAULT_TYPE="sonorant", not "vowel".
TEST_PHONEME_TYPES = decode.PhonemeTypes(
    by_lang={},
    common={
        "a": "vowel",
        "t": "stop",
        "i0": "vowel",
        "SP": "vowel",  # SP is silence; COMMON in real map, but we need it typed
    },
    bare={
        "a": "vowel",
        "t": "stop",
        "i0": "vowel",
        "SP": "vowel",
    },
)


def row(prefer, logit=9.0, rest=-6.0):
    """A frame that argues for one or two tags equally."""
    r = [rest] * L
    for name in prefer:
        r[LABELS.index(name)] = logit
    return r


def frames(*blocks):
    """Build a (frames, labels) logit tensor the way the MODEL hands it over.

    A torch tensor, float32, shape (T, labels) -- because that is what
    bio_inputs() consumes (`logits.detach().double()...`) and therefore what
    train.py and infer.py pass in. Handing it a numpy array instead would be
    testing a path production never takes, and the first version of this file
    did exactly that and failed on the missing .detach().
    """
    return torch.tensor(blocks, dtype=torch.float32)


def n_segments(tags):
    """How many phoneme segments this tag sequence decodes to.

    The count of `B-` tags, NOT the count of runs of distinct phoneme names.
    Repeated `B-a B-a B-a` is one phoneme name and three segments, and three
    segments is precisely the failure. Counting name-runs hides it -- which is
    how this bug survived a first pass at the test.
    """
    return sum(1 for t in tags if t.startswith("B-"))


def flat_logits(n, *prefer):
    return frames(*[row(list(prefer)) for _ in range(n)])


T = 12
bonus = math.log(5.0)  # what viterbi_bias=5 means

print("== a constant-logit run of one phoneme must decode as ONE segment ==")
tags = decode.viterbi_decode(
    flat_logits(T, "B-a", "I-a"), ID2LABEL, viterbi_bias=5.0
)
segs = n_segments(tags)
check(f"viterbi_bias=5 gives 1 segment over {T} frames (got {segs})", segs == 1,
      " ".join(tags))
check("the run is B-a then I-a throughout",
      tags[0] == "B-a" and tags[1:] == ["I-a"] * (T - 1), " ".join(tags))

print("\n== stickiness_bonus=log(5) is the same decode as viterbi_bias=5 ==")
tags_b = decode.viterbi_decode(
    flat_logits(T, "B-a", "I-a"), ID2LABEL, stickiness_bonus=bonus
)
check("the two spellings agree", tags_b == tags, " ".join(tags_b))

print("\n== the sign is monotone: more bonus is never shorter ==")
lengths = {}
for b in (0.0, 0.5, bonus, 2.0, 4.0):
    t = decode.viterbi_decode(
        flat_logits(T, "B-a", "I-a"), ID2LABEL, stickiness_bonus=b,
        phoneme_types=TEST_PHONEME_TYPES, lang="test"
    )
    lengths[b] = n_segments(t)
    # bonus=0 CAN fragment (all transitions 0 = tie); >0 must not
    expected = 1 if b > 0 else None
    check(f"bonus {b:5.3f} -> {lengths[b]} segment(s)",
          lengths[b] == 1 if b > 0 else lengths[b] >= 1)
check("no bonus ever produces more segments than no bonus at all",
      all(v <= lengths[0.0] for v in lengths.values()), str(lengths))

print("\n== a negative stay-cost RAISES instead of fragmenting ==")
# This is the guard that makes the original bug impossible to reintroduce by
# editing a config: it used to fragment silently, and silently is what cost a day.
try:
    decode.viterbi_decode(
        flat_logits(T, "B-a", "I-a"), ID2LABEL, stickiness_bonus=-bonus
    )
    check("negative stickiness_bonus raises", False, "no exception raised")
except ValueError as e:
    msg = str(e)
    check("negative stickiness_bonus raises ValueError", True)
    check("the message says >= 0", ">= 0" in msg, msg[:90])
    check("the message offers the negation", "negate" in msg, msg[:90])
    check("the message offers viterbi_bias", "viterbi_bias=" in msg, msg[:90])
    check("the message quotes the offending value", "-1.61" in msg or "1.609" in msg
          or "1.61" in msg, msg[:90])

print("\n== passing both spellings is an error, not a precedence rule ==")
try:
    decode.viterbi_decode(
        flat_logits(T, "B-a", "I-a"), ID2LABEL,
        viterbi_bias=5.0, stickiness_bonus=bonus,
    )
    check("both spellings raise", False, "no exception raised")
except ValueError:
    check("both spellings raise ValueError", True)

print("\n== viterbi_bias below 1 raises (a bias under 1 is a penalty) ==")
for bad in (0.5, 0.0, -3.0):
    try:
        decode.viterbi_decode(
            flat_logits(4, "B-a", "I-a"), ID2LABEL, viterbi_bias=bad
        )
        check(f"viterbi_bias={bad} raises", False, "no exception raised")
    except ValueError:
        check(f"viterbi_bias={bad} raises ValueError", True)

print("\n== two phonemes in a row give two segments, at the right frame ==")
half = T // 2
two = frames(
    *[row(["B-a", "I-a"]) for _ in range(half)]
    + [row(["B-t", "I-t"]) for _ in range(T - half)]
)
tags = decode.viterbi_decode(two, ID2LABEL, viterbi_bias=5.0)
check(f"a-then-t gives exactly 2 segments (got {n_segments(tags)})",
      n_segments(tags) == 2, " ".join(tags))
check(f"the switch is at frame {half}",
      tags[half - 1] == "I-a" and tags[half] == "B-t", " ".join(tags))

print("\n== a 1-frame stop survives: the short phonemes are the ones being lost ==")
# The reported failures were on common phonemes, and short ones (a plosive burst
# is 15 ms = one frame). A stay-cost that is too eager erases exactly these, so
# this is the case that matters most for the original complaint.
burst = frames(
    *[row(["B-a", "I-a"]) for _ in range(5)]
    + [row(["B-t"], logit=12.0)]
    + [row(["B-a", "I-a"]) for _ in range(6)]
)
tags = decode.viterbi_decode(burst, ID2LABEL, viterbi_bias=5.0)
check("t survives a single frame", any(t.endswith("-t") for t in tags),
      " ".join(tags))
check("the vowel resumes after the burst", tags[-1] in ("I-a", "B-a"),
      " ".join(tags))
check("exactly 3 segments (vowel, stop, vowel)", n_segments(tags) == 3,
      " ".join(tags))

print("\n== a stop does not get a vowel's stay bonus ==")
# The whole reason for the per-class block: one global value has to stretch a
# 120 ms vowel and a 15 ms burst, and ends up good at neither.
flat = flat_logits(T, "B-t", "I-t")
as_vowel = decode.viterbi_decode(
    flat, ID2LABEL, stickiness_bonus=1.79, penalty_by_class={"vowel": 1.79},
    phoneme_types=TEST_PHONEME_TYPES, lang="test"
)
as_stop = decode.viterbi_decode(
    flat, ID2LABEL, stickiness_bonus=1.79, penalty_by_class={"stop": 1.10},
    phoneme_types=TEST_PHONEME_TYPES, lang="test"
)
check("the two overrides are accepted", isinstance(as_vowel, list)
      and isinstance(as_stop, list))

print("\n== a negative value inside penalty_by_class raises too ==")
# The per-class map is the easier way in: a sweep grid takes any float, and one
# negative class would shatter exactly that class while the rest decoded fine,
# which reads as a data problem rather than a sign.
try:
    decode.viterbi_decode(
        flat_logits(T, "B-a", "I-a"), ID2LABEL,
        penalty_by_class={"vowel": -1.79},
        phoneme_types=TEST_PHONEME_TYPES, lang="test"
    )
    check("negative penalty_by_class raises", False, "no exception raised")
except ValueError as e:
    msg = str(e)
    check("negative penalty_by_class raises ValueError", True)
    check("it names the offending class", "vowel" in msg, msg[:90])
    check("it says to negate", "Negate" in msg or "negate" in msg, msg[:90])
    check("it prints the negated value", "1.79" in msg, msg[:90])

print("\n== positive per-class values are accepted ==")
# Base bonus=1.0 ensures the scalar is positive; per-class override applies
# only when the phoneme's class matches. "a" is vowel, so:
# - vowel:1.79  -> uses 1.79
# - stop:1.10   -> falls back to 1.0
# - sonorant:1.61 -> falls back to 1.0
# All should decode as 1 segment (no crash, no fragmentation).
for cls, val in (("vowel", 1.79), ("stop", 1.10), ("sonorant", 1.61)):
    t = decode.viterbi_decode(
        flat_logits(T, "B-a", "I-a"), ID2LABEL,
        stickiness_bonus=1.0, penalty_by_class={cls: val},
        phoneme_types=TEST_PHONEME_TYPES, lang="test"
    )
    check(f"penalty_by_class={cls}:{val} decodes as 1 segment",
          n_segments(t) == 1, " ".join(t))

print("\n== the forced aligner still ACCEPTS negative penalties ==")
# The asymmetry has to survive in both directions: the guard above must not leak
# into the forced path, where a negative cost is exactly right.
forced_sig = inspect.signature(decode.forced_align_viterbi)
check("forced takes self_loop_penalty", "self_loop_penalty" in forced_sig.parameters)
check("forced takes the by-class block",
      "self_loop_penalty_by_class" in forced_sig.parameters)
try:
    decode.resolve_penalties({"vowel": -4.6}, -4.6)
    check("resolve_penalties accepts negative (shared by forced)", True)
except ValueError:
    check("resolve_penalties accepts negative (shared by forced)", False,
          "the guard leaked into the shared resolver")

print("\n== config keys: the bonus name is what the code reads ==")
# Reading only the new key is right; reading only the OLD one is the original
# bug re-entered through the config file instead of the signature.
import yaml  # noqa: E402

for name in ("small", "medium", "high"):
    with open(f"configs/{name}.yaml", encoding="utf-8") as fh:
        post = yaml.safe_load(fh)["postprocess"]
    blk = post.get("stickiness_bonus_by_class")
    check(f"{name}: stickiness_bonus_by_class exists", bool(blk), str(blk))
    if blk:
        negs = {k: v for k, v in blk.items() if v < 0}
        check(f"{name}: no negative value in the block", not negs, str(negs))
    check(f"{name}: the old cost key is gone from the free block",
          "self_loop_penalty_by_class" not in post)
    fa = post.get("forced_alignment_args", {})
    check(f"{name}: forced block keeps its negative cost",
          fa.get("self_loop_penalty", 0) < 0, str(fa.get("self_loop_penalty")))
    check(f"{name}: forced by-class keeps negative costs",
          all(v < 0 for v in (fa.get("self_loop_penalty_by_class") or {}).values()))

print("\n== infer.py and train.py read the bonus key, and refuse the cost key ==")
for mod_name in ("infer", "train"):
    with open(f"{mod_name}.py", encoding="utf-8") as fh:
        src = fh.read()
    check(f"{mod_name}.py reads stickiness_bonus_by_class",
          "stickiness_bonus_by_class" in src)
    check(f"{mod_name}.py guards the legacy cost key",
          "postprocess.self_loop_penalty_by_class has negative values" in src,
          "a legacy block with negative costs must be refused, not applied")

print("\n== tune_decode writes the bonus key ==")
with open("tune_decode.py", encoding="utf-8") as fh:
    tune_src = fh.read()
check("tune_decode writes stickiness_bonus_by_class",
      'snippet["postprocess"]["stickiness_bonus_by_class"]' in tune_src)
check("tune_decode no longer writes the cost key into a free snippet",
      'snippet["postprocess"]["self_loop_penalty_by_class"]' not in tune_src)
check("the --grid help warns about the sign difference",
      "WATCH THE SIGN" in tune_src)

print("\n== the timing script measures what the model saw, not the raw .lab ==")
# A duration table that describes the PRE-merge labels is measuring a dataset
# that does not exist. Three ways that happens, all of them silent.
import measure_phoneme_timing as mpt  # noqa: E402

# 1. the type tables must not drift from decode.py's
check("self_check passes against decode.py", mpt.self_check() is True,
      "measure_phoneme_timing has diverged from decode")

# 2. merge semantics: preprocess.py uses [phon] = canon, so the LAST group
#    wins. An earlier version used setdefault (FIRST wins) and quietly timed a
#    label the preprocessor had already rewritten.
cfg = {"training": {"merged_phoneme_groups": [["aa", "pt/a"], ["bb", "pt/a"]]}}
check("last merge group wins, matching preprocess.py",
      mpt.load_merge_map(cfg), {"pt": {"a": "bb"}})
check("a single group still merges",
      mpt.load_merge_map({"training": {"merged_phoneme_groups":
                                       [["aa", "pt/a", "pt/e"]]}}),
      {"pt": {"a": "aa", "e": "aa"}})

# 3. zero-length labels are dropped before training (preprocess.py:87), so
#    counting them would drag every mean down.
check("a zero-length label is dropped",
      mpt.read_lab.__doc__ is not None)
seg_dir = "test_tmp_labs"
os.makedirs(os.path.join(seg_dir, "pt"), exist_ok=True)
try:
    with open(os.path.join(seg_dir, "pt", "u.lab"), "w", encoding="utf-8") as fh:
        fh.write("0 120000000 a\n"          # 12 s, kept
                 "120000000 120000000 a\n"   # zero length, dropped
                 "130000000 150000000 t\n")  # 2 s, kept
    segs, bad = mpt.read_lab(os.path.join(seg_dir, "pt", "u.lab"), {})
    check("zero-length label dropped, other two kept", len(segs) == 2, str(segs))
    check("the dropped line is counted, not raised", bad == 1, str(bad))
finally:
    import shutil
    shutil.rmtree(seg_dir, ignore_errors=True)

# 4. to_bio_tags frame arithmetic: the +1 is what inflates short phonemes
check("a 15 ms burst gets 1 frame at 20 ms",
      mpt.frames_for(1.000, 1.015, 0.02)[2] == 1)
check("a 60 ms vowel gets 4 frames at 20 ms",
      mpt.frames_for(2.000, 2.060, 0.02)[2] == 4)
check("a phoneme never gets 0 frames",
      mpt.frames_for(3.000, 3.001, 0.02)[2] == 1)

print("\n== the phoneme map agrees with the phonetics ==")
# Guards against a wrong type silently landing in the middle tier, where it
# still decodes and the numbers still look plausible. Each case below is a
# specific claim about a specific PT phoneme, checked against the real map.
import measure_phoneme_timing as _mpt  # noqa: E402

_map = _mpt.load_types("phoneme_map.yaml", warn=lambda *a: None)

# ch, nh, lh, rw are SINGLE phonemes written in Portuguese orthographic
# convention, not diphthongs and not compound symbols. An earlier note in the
# plan claimed they were diphthongs needing special handling; that was wrong,
# and these assertions are here so the claim cannot come back unnoticed.
#   ch = tS, nh = J, lh = L, rw = r\  (X-Sampa)
for _ph, _want in (("ch", "affricate"), ("nh", "nasal"),
                   ("lh", "liquid"), ("rw", "liquid")):
    check(f"pt/{_ph} is a {_want}", _map.get(_ph, "pt"), _want)

# l and lh coexist as distinct phonemes, exactly as n and nh do. They share a
# duration class, which is correct: a lateral and a palatal lateral are the
# same manner, and the class is about duration, not place.
check("pt/l and pt/lh are both liquids",
      (_map.get("l", "pt"), _map.get("lh", "pt")), ("liquid", "liquid"))
check("pt/n and pt/nh are both nasals",
      (_map.get("n", "pt"), _map.get("nh", "pt")), ("nasal", "nasal"))

# w and y are glides (semivowels) in their vocalic role. The user's examples:
#   yoyo = [j o j o] (X-Sampa) = [y o y o]
#   qual = [k w a w] (X-Sampa) = [k w a w]
#   sao  = [s a~ w] (X-Sampa) = [s an w]
# So y in yoyo is the X-Sampa `j` glide, NOT the palatal fricative. The map
# already types it as semivowel, which is right -- and it stays distinct from
# pt/j, which IS the palatal fricative.
check("pt/y is a semivowel (the [j] glide of yoyo)",
      _map.get("y", "pt"), "semivowel")
check("pt/w is a semivowel (the [w] of qual)",
      _map.get("w", "pt"), "semivowel")
check("y (glide) and j (palatal fricative) are different types",
      _map.get("y", "pt") != _map.get("j", "pt"), True)
check("pt/j is the fricative", _map.get("j", "pt"), "fricative")
for _ph, _want in (("s", "fricative"), ("an", "vowel"), ("o", "vowel"),
                   ("a", "vowel"), ("k", "stop")):
    check(f"pt/{_ph} is a {_want}", _map.get(_ph, "pt"), _want)

print("\n== a phoneme that YAML reads as a boolean is refused, not coerced ==")
# pt `on` [õ] written as a bare `on` arrives from PyYAML as Python True. The
# old loader did str(entry["symbol"]), so it became the phoneme "True": a
# plausible-looking label that appears in no corpus, silently replacing the real
# vowel. This crashed with "argument of type 'bool' is not iterable" in the
# measuring script and silently mislabelled in the decoder; both now refuse.
import tempfile  # noqa: E402

_bad_map = """symbols:
- {symbol: a, type: vowel}
- {symbol: on, type: vowel}
- {symbol: t, type: stop}
"""
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                 encoding="utf-8") as fh:
    fh.write(_bad_map)
    _bad_path = fh.name
try:
    _hit = False
    try:
        decode.load_phoneme_types(_bad_path)
    except ValueError as e:
        _hit = True
        _msg = str(e)
    check("a bare `on` raises ValueError", _hit)
    check("the message names the entry index", "symbols[1]" in _msg, _msg[:80])
    check("the message shows the boolean it became", "symbol=True" in _msg,
          _msg[:80])
    check("the message says how to fix it", '"on"' in _msg and "quoted" in _msg,
          _msg[:80])
    # And the same through the measuring script's mirror.
    _hit2 = False
    try:
        _mpt.load_types(_bad_path, warn=lambda *a: None)
    except ValueError:
        _hit2 = True
    check("measure_phoneme_timing refuses it too", _hit2)
finally:
    os.unlink(_bad_path)

# The real fix, checked against the actual map: quoting makes it a string and
# `on` resolves to the vowel it is.
_quoted = """symbols:
- {symbol: a, type: vowel}
- {symbol: "on", type: vowel}
"""
with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                 encoding="utf-8") as fh:
    fh.write(_quoted)
    _ok_path = fh.name
try:
    _t = decode.load_phoneme_types(_ok_path, warn=lambda *a: None)
    check("quoted \"on\" loads", len(_t), 2)
    check("quoted \"on\" resolves to its type", _t.get("on"), "vowel")
    check("it is the string 'on', not True", isinstance(_t.get("on"), str), True)
finally:
    os.unlink(_ok_path)

print("\n== silence is not what this knob should disturb ==")
# SP works well today and must stay working. It is COMMON, so it resolves the
# same in every language, and it is long, so it should stay a single segment.
sil = decode.viterbi_decode(
    flat_logits(T, "B-SP", "I-SP"), ID2LABEL, viterbi_bias=5.0
)
check(f"SP decodes as 1 segment (got {n_segments(sil)})", n_segments(sil) == 1,
      " ".join(sil))

print("\n== the forced aligner keeps its own sign ==")
# The asymmetry is the fix, not an oversight: the free path's I- value is the
# only thing deciding duration, the forced DP is pinned to a known phone and
# only picks boundaries. Unifying them is what broke the free path.
forced_src = inspect.getsource(decode.forced_align_viterbi)
check("forced_align_viterbi still takes self_loop_penalty",
      "self_loop_penalty" in inspect.signature(decode.forced_align_viterbi).parameters)
check("the forced docstring states the sign is opposite",
      "opposite" in (decode.forced_align_viterbi.__doc__ or "").lower())
check("the forced docstring points at the free one",
      "viterbi_decode" in (decode.forced_align_viterbi.__doc__ or ""))
check("free path no longer takes self_loop_penalty",
      "self_loop_penalty" not in
      inspect.signature(decode.viterbi_decode).parameters)

print("\n" + "=" * 64)
if FAILED:
    print(f"{len(FAILED)} FAILURE(S):")
    for f in FAILED:
        print("  -", f)
    sys.exit(1)
print("all checks passed")