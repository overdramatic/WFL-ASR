#!/usr/bin/env python3
"""Measure the REAL duration of every phoneme, per type, from the .lab files.

WHY THIS EXISTS
---------------
Every duration prior in decode.py -- `stickiness_bonus_by_class`, the
`min_phoneme_duration` floor -- started life as a guess. `vowel: 1.79` and
`stop: 1.10` were reasoned from the idea that vowels are long and plosive bursts
are short, and that is all they were: reasoned. A guess is fine as a starting
point for a sweep and terrible as a starting point for the conclusion, because
the two activities want different things and the guess cannot tell them apart.

The labels are the ground truth for exactly the quantity the decoder needs.
`pt/a` is not "about 120 ms because it is a vowel"; it is a distribution with a
measured median, a spread, and a tail. Reading it costs one pass over the
dataset and removes the guess from the loop.

WHAT IT MEASURES, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------
This reads the .lab files, NOT the decoded output. A duration prior describes
how long the phoneme IS; measuring what the decoder produces instead would fit
the prior to its own errors, which is how a wrong prior becomes self-confirming.

Three things preprocess.py does to the labels are reproduced here, because
measuring the pre-processed phonemes while the model trained on the
post-processed ones would quietly compare two different datasets:

  1. `merged_phoneme_groups` is applied (preprocess.py:85). A merged label is
     what the model saw, so a merged label is what gets timed.
  2. Zero-length and out-of-audio labels are dropped (preprocess.py:87).
  3. Times are converted from HTK ticks by /1e7 (preprocess.py:86).

One thing is measured that preprocess.py does not do, and it is the whole
point of having this script: `to_bio_tags` turns a phoneme into
`e_idx - s_idx + 1` frames by integer division, and that count is NOT the real
duration. For a 15 ms burst at 20 ms per frame it can be 1, and for a 60 ms
fricative it can be 3 or 4 depending on where it happens to fall relative to the
frame grid. Both the true duration and the frame count the model actually sees
are reported, because they disagree, and that disagreement is a lead -- see the
SHORT-PHONEME REPORT at the end.

USAGE
-----
    python3 measure_phoneme_timing.py \
        --data-dir training_dataset \
        --phoneme-map phoneme_map.yaml \
        --config configs/small.yaml

With --out it also writes the per-phoneme table as CSV, so the numbers can be
diffed between runs instead of re-read from a screen.

With --suggest it prints a `stickiness_bonus_by_class` block ready to paste.
Treat that block as a much better-informed INITIAL value, not as an answer: the
mapping from duration to bonus is not derived here, it is fitted by sweeping
(tune_decode.py), because the right value also depends on how confident the
model is, which this script does not and cannot see.
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

try:
    import yaml
except ImportError:
    print("This script needs pyyaml: pip install pyyaml", file=sys.stderr)
    raise

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# decode.py imports numpy and numba, which a measuring script has no use for:
# reading timestamps is arithmetic, not signal processing. Requiring a GPU-less
# numpy just to print a table would mean this script is only runnable in the
# exact environment it is meant to be an escape from, so the type resolution is
# mirrored here and checked against decode.py's tables whenever those import.
# The mirror is three constants and one lookup order; the check at the bottom of
# this file is what keeps it honest.
DEFAULT_TYPE = "sonorant"
DEFAULT_CLASS = "sonorant"
PHONE_CLASSES = ("vowel", "sonorant", "fricative", "stop")
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
CLASS_ALIASES = {"obstruent": ("fricative", "stop")}
VALID_TYPES = set(TYPE_CLASSES)


class _Types:
    """Same resolution order as decode.PhonemeTypes.get, without the imports.

    1. COMMON first, so SP cannot change meaning per corpus.
    2. (lang, phoneme), because the same string is not the same sound
       everywhere ("j" is a fricative in pt and an affricate in ja).
    3. bare, by majority across languages, so a merged label still resolves.
    4. DEFAULT_TYPE, the middle duration tier.
    """

    def __init__(self, by_lang, common, bare):
        self._by_lang = by_lang
        self._common = common
        self._bare = bare

    def get(self, phoneme, lang=None):
        if phoneme in self._common:
            return self._common[phoneme]
        if lang is not None:
            found = self._by_lang.get((lang, phoneme))
            if found is not None:
                return found
        return self._bare.get(phoneme, DEFAULT_TYPE)

    def types(self):
        return set(self._by_lang.values()) | set(self._common.values())

    def __len__(self):
        return len(self._by_lang) + len(self._common)


def load_types(path, warn=print):
    """Read phoneme_map.yaml. Raises the same way decode.load_phoneme_types does.

    A `type` outside VALID_TYPES is an error, not a warning: it means those
    phonemes silently fell back to the middle tier, and this table is exactly
    where that would become visible and be ignored.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No phoneme map at {path}")
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    entries = doc.get("symbols")
    if not entries:
        raise ValueError(f"{path}: no 'symbols:' list")

    by_lang, common, bare = {}, {}, {}
    bad = []
    for i, e in enumerate(entries):
        sym, typ = e.get("symbol"), e.get("type")
        # Not str(): YAML 1.1 turns a bare `on`/`off`/`yes`/`no`/`true`/`false`
        # into booleans, so pt `on` [õ] arrives as True. str(True) == "True",
        # a phoneme in no corpus, which would quietly skew every measurement.
        if not isinstance(sym, str) or not isinstance(typ, str):
            raise ValueError(
                f"{path}: symbols[{i}] has a non-string value: {e!r}. "
                "YAML reads a bare 'on', 'off', 'yes', 'no', 'true' or "
                "'false' as a boolean, so a phoneme like pt `on` [õ] has to "
                'be quoted: {symbol: "on", ...}.'
            )
        if not sym:
            raise ValueError(f"{path}: entry without 'symbol': {e!r}")
        if not typ:
            raise ValueError(f"{path}: {sym} has no 'type'")
        if typ not in VALID_TYPES:
            bad.append(f"{sym}={typ}")
            continue
        if "/" in sym:
            lang, phon = sym.split("/", 1)
            by_lang[(lang, phon)] = typ
            bare.setdefault(phon, typ)
        else:
            common[sym] = typ
            bare.setdefault(sym, typ)
    if bad:
        raise ValueError(
            f"{path}: unknown type in {len(bad)} entries: {', '.join(bad)}\n"
            f"Expected one of: {', '.join(sorted(VALID_TYPES))}"
        )
    warn(f"Phoneme map: {path} ({len(by_lang) + len(common)} symbols)")
    return _Types(by_lang, common, bare)


def phone_type(phoneme, types, lang=None):
    return types.get(phoneme, lang) if types is not None else DEFAULT_TYPE


def phone_class(phoneme, types, lang=None):
    return TYPE_CLASSES.get(phone_type(phoneme, types, lang), DEFAULT_CLASS)


def self_check():
    """Fail loudly if this mirror has drifted from decode.py's tables.

    Only runs when decode is importable. A silent divergence between the
    measuring script and the decoder would make the measurements quietly
    describe the wrong priors, which is the exact failure mode this script
    exists to remove.
    """
    try:
        import decode
    except Exception:
        return None
    for name, mine in (
        ("TYPE_CLASSES", TYPE_CLASSES), ("CLASS_ALIASES", CLASS_ALIASES),
        ("DEFAULT_TYPE", DEFAULT_TYPE), ("DEFAULT_CLASS", DEFAULT_CLASS),
        ("PHONE_CLASSES", PHONE_CLASSES),
    ):
        theirs = getattr(decode, name, None)
        if theirs != mine:
            raise AssertionError(
                f"measure_phoneme_timing.{name} has drifted from decode.{name}: "
                f"{mine!r} != {theirs!r}"
            )
    return True


def load_merge_map(config):
    """Reproduce preprocess.py:48-54 exactly.

    `merge.setdefault(lang, {})[phon] = canon` -- LAST group wins, not first.
    That asymmetry against the setdefault on the outer dict is easy to
    misremember, and getting it backwards times a phoneme the model never
    trained on: if two groups claim pt/a, the preprocessor writes the second
    one's canonical form and this script has to agree, or the table describes a
    label that does not exist in the dataset.

    (Checked, not assumed: an earlier version of this used setdefault on the
    inner dict, which keeps the FIRST group and silently disagreed.)
    """
    merge = {}
    for group in (config or {}).get("training", {}).get(
        "merged_phoneme_groups", []
    ):
        canon = group[0]
        for item in group[1:]:
            if "/" in item:
                lang, phon = item.split("/", 1)
                merge.setdefault(lang, {})[phon] = canon
    return merge


def read_lab(path, merge_for_lang):
    """Yield (start_s, end_s, phoneme) the way preprocess.py would see them.

    Malformed lines are counted and reported, not raised on: a dataset with one
    bad line should still yield a duration table for the other 40 hours of audio,
    and the count of skipped lines is itself worth seeing.
    """
    segs, skipped = [], 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 3:
                skipped += 1
                continue
            start, end = float(parts[0]) / 1e7, float(parts[1]) / 1e7
            phon = parts[2]
            phon = merge_for_lang.get(phon, phon)
            if start == end or start >= end:
                skipped += 1
                continue
            segs.append((start, end, phon))
    return segs, skipped


def frames_for(start, end, frame_dur, num_frames=None):
    """Reproduce to_bio_tags' integer arithmetic (preprocess.py:26-27).

    The `+ 1` on the frame count is not a rounding detail, it is the bug's
    hiding place: a phoneme that is shorter than one frame still gets a B- tag,
    and a phoneme that straddles a frame boundary gets rounded UP to the next
    frame. Reporting this count next to the true duration is what makes the
    difference visible.
    """
    s_idx = int(start / frame_dur)
    e_idx = int(end / frame_dur)
    if num_frames is not None:
        e_idx = min(e_idx, num_frames - 1)
    return s_idx, e_idx, max(1, e_idx - s_idx + 1)


def percentile(sorted_vals, q):
    """Linear-interpolated percentile. numpy is not guaranteed to be here."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def stats(durations):
    s = sorted(durations)
    n = len(s)
    return {
        "n": n,
        "mean_ms": 1000.0 * sum(s) / n,
        "median_ms": 1000.0 * percentile(s, 0.5),
        "p10_ms": 1000.0 * percentile(s, 0.10),
        "p90_ms": 1000.0 * percentile(s, 0.90),
        "min_ms": 1000.0 * s[0],
        "max_ms": 1000.0 * s[-1],
    }


def main():
    ap = argparse.ArgumentParser(
        description="Measure real phoneme durations from .lab files."
    )
    ap.add_argument("--data-dir", required=True,
                    help="folder with <lang>/*.lab (see data.data_dir)")
    ap.add_argument("--phoneme-map", default="phoneme_map.yaml")
    ap.add_argument("--config", default=None,
                    help="config yaml, for merged_phoneme_groups and frame_duration")
    ap.add_argument("--frame-duration", type=float, default=None)
    ap.add_argument("--min-count", type=int, default=1,
                    help="hide phonemes rarer than this from the table "
                         "(default 1 = show everything)")
    ap.add_argument("--suggest", action="store_true",
                    help="print a stickiness_bonus_by_class block")
    ap.add_argument("--out", default=None, help="write the table as CSV")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write every stat as JSON")
    args = ap.parse_args()

    config = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    frame_dur = args.frame_duration
    if frame_dur is None:
        frame_dur = (config.get("data") or {}).get("frame_duration", 0.02)
    merge_map = load_merge_map(config)

    try:
        types = load_types(args.phoneme_map)
    except FileNotFoundError:
        print(f"Error: no phoneme map at {args.phoneme_map}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as e:
        # A malformed map is a config error, not a crash. It stays a hard error
        # rather than a warning -- a bad `type` means those phonemes fell back
        # to the middle tier, and a table that silently mislabels them is worse
        # than no table -- but it does not need a traceback to say so.
        print(f"Error in {args.phoneme_map}:\n  {e}", file=sys.stderr)
        return 2
    if self_check():
        print("Type tables match decode.py.")
    print(f"Frame duration: {frame_dur * 1000:.0f} ms")
    if any(merge_map.values()):
        print(f"Merged groups: "
              f"{ {k: dict(v) for k, v in merge_map.items()} }")

    lang_dirs = sorted(
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    )
    if not lang_dirs:
        print(f"Error: no language subfolders in {args.data_dir}",
              file=sys.stderr)
        return 2

    # (lang, phoneme) -> [durations in seconds]
    by_lang_phone = defaultdict(list)
    # (lang, phoneme) -> [frame counts the model saw]
    by_lang_phone_frames = defaultdict(list)
    # phoneme -> [(lang, seconds, frames)]
    per_occurrence = defaultdict(list)
    unknown = defaultdict(int)
    total_files = total_bad_lines = total_skipped = 0
    total_phones = 0

    for lang in lang_dirs:
        merge_for_lang = merge_map.get(lang, {})
        labs = sorted(glob.glob(os.path.join(args.data_dir, lang, "*.lab")))
        for lab in labs:
            segs, bad = read_lab(lab, merge_for_lang)
            total_files += 1
            total_bad_lines += bad
            for start, end, phon in segs:
                _, _, nframes = frames_for(start, end, frame_dur)
                dur = end - start
                key = (lang, phon)
                by_lang_phone[key].append(dur)
                by_lang_phone_frames[key].append(nframes)
                per_occurrence[phon].append((lang, dur, nframes))
                total_phones += 1
                if phon not in types._bare and (
                    lang, phon
                ) not in types._by_lang and types.get(phon, lang) == "sonorant":
                    # Not conclusive (sonorant is also a real type), so it is
                    # counted rather than raised. The table prints which ones.
                    unknown[phon] += 1

    if total_phones == 0:
        print("No phonemes read. Check --data-dir.", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ tables
    print(f"\n{'-' * 78}")
    print(f"PER PHONEME  ({total_files} files, {total_phones} occurrences, "
          f"{total_bad_lines} bad lines, {total_skipped} zero-length dropped)")
    print(f"{'-' * 78}")
    header = (f"{'lang/phone':<14}{'type':<11}{'class':<10}"
              f"{'n':>6}{'median':>9}{'mean':>9}{'p10':>8}{'p90':>9}{'frames':>8}")
    print(header)
    print(f"{'-' * len(header)}")

    rows = []
    lang_phone_keys = sorted(by_lang_phone.items())
    for (lang, phon), durs in lang_phone_keys:
        st = stats(durs)
        fr = by_lang_phone_frames[(lang, phon)]
        ph_type = phone_type(phon, types, lang)
        ph_class = phone_class(phon, types, lang)
        rows.append({
            "lang": lang, "phoneme": phon, "type": ph_type, "class": ph_class,
            **st,
            "median_frames": percentile(sorted(fr), 0.5),
            "mean_frames": sum(fr) / len(fr),
            "min_frames": min(fr), "max_frames": max(fr),
        })
        if st["n"] < args.min_count:
            continue
        name = f"{lang}/{phon}" if lang else phon
        print(f"{name:<14}{ph_type:<11}{ph_class:<10}"
              f"{st['n']:>6}{st['median_ms']:>9.0f}{st['mean_ms']:>9.0f}"
              f"{st['p10_ms']:>8.0f}{st['p90_ms']:>9.0f}"
              f"{percentile(sorted(fr), 0.5):>8.1f}")

    # ----------------------------------------------------------------- by type
    print(f"\n{'-' * 78}")
    print("BY TYPE  (all languages pooled; a type is one sound, so pooling is right)")
    print(f"{'-' * 78}")
    print(f"{'type':<13}{'class':<11}{'n':>7}{'median':>9}{'mean':>9}"
          f"{'p10':>8}{'p90':>9}")
    by_type = defaultdict(list)
    for (lang, phon), durs in by_lang_phone.items():
        by_type[phone_type(phon, types, lang)].extend(durs)
    for ph_type in sorted(by_type, key=lambda t: -stats(by_type[t])["median_ms"]):
        durs = by_type[ph_type]
        st = stats(durs)
        print(f"{ph_type:<13}{TYPE_CLASSES.get(ph_type, '?'):<11}"
              f"{st['n']:>7}{st['median_ms']:>9.0f}{st['mean_ms']:>9.0f}"
              f"{st['p10_ms']:>8.0f}{st['p90_ms']:>9.0f}")

    # ---------------------------------------------------------------- by class
    print(f"\n{'-' * 78}")
    print("BY DURATION CLASS  (this is what a prior is actually keyed on)")
    print(f"{'-' * 78}")
    print(f"{'class':<13}{'n':>7}{'median':>9}{'mean':>9}{'p10':>8}{'p90':>9}")
    by_class = defaultdict(list)
    for (lang, phon), durs in by_lang_phone.items():
        by_class[phone_class(phon, types, lang)].extend(durs)
    class_stats = {}
    for ph_class in sorted(by_class, key=lambda c: -stats(by_class[c])["median_ms"]):
        durs = by_class[ph_class]
        st = stats(durs)
        class_stats[ph_class] = st
        print(f"{ph_class:<13}{st['n']:>7}{st['median_ms']:>9.0f}"
              f"{st['mean_ms']:>9.0f}{st['p10_ms']:>8.0f}{st['p90_ms']:>9.0f}")

    # ---------------------------------------------------------- silence check
    # SP is COMMON and works today. It is reported so that a run where silence
    # moved is visible, and it is deliberately NOT used to set a prior: a long
    # silence and a long vowel are not the same kind of long, and stretching a
    # prior to accommodate SP is how SP ended up sharing a knob with the vowels.
    # Per language, because the same key can carry different lengths per corpus
    # and averaging them would hide exactly that.
    sp_by_lang = {
        lang: durs for (lang, ph), durs in by_lang_phone.items() if ph == "SP"
    }
    for lang in sorted(sp_by_lang):
        st = stats(sp_by_lang[lang])
        print(f"\nSP in {lang} (silence, for reference only, NOT a prior): "
              f"n={st['n']} median={st['median_ms']:.0f} ms "
              f"p10={st['p10_ms']:.0f} p90={st['p90_ms']:.0f} ms")
    if not sp_by_lang:
        print("\nSP: not present in any .lab. If the corpus marks silence some "
              "other\nway, note it: silence handling keys off the literal "
              "'SP' string.")

    # ------------------------------------------------------------ suggest block
    if args.suggest and class_stats:
        print(f"\n{'-' * 78}")
        print("DURATION ORDER, measured  (this is the useful output)")
        print(f"{'-' * 78}")
        ordered = sorted(class_stats.items(), key=lambda kv: -kv[1]["median_ms"])
        for ph_class, st in ordered:
            ratio = st["median_ms"] / (frame_dur * 1000)
            print(f"  {ph_class:<11} median {st['median_ms']:>6.0f} ms "
                  f"= {ratio:>4.1f} frames   p10..p90 "
                  f"{st['p10_ms']:.0f}..{st['p90_ms']:.0f} ms")
        print()
        print("What this does and does not license:")
        print()
        print("  DOES: tells you which class is genuinely longer, by measurement")
        print("        rather than by assumption. If the order disagrees with what")
        print("        the config assumes, that config is wrong.")
        print()
        print("  DOES NOT: justify a bonus value. The BONUS is a per-frame")
        print("        stickiness competing against the model's own logits, and")
        print("        this script never sees those logits. The right bonus for a")
        print("        class is fitted by sweeping it (tune_decode.py), and the")
        print("        duration only tells you the ORDER to sweep in.")
        print()
        print("  So: read the order above, confirm it matches")
        print("  postprocess.stickiness_bonus_by_class, and sweep the values.")
        print("  An automated bonus from a single corpus would overwrite a tuned")
        print("  value with one measured run's worth of evidence, which is the")
        print("  same mistake as the reasoned defaults, in the other direction.")

        shipped = ((config.get("postprocess") or {}).get(
            "stickiness_bonus_by_class") or {})
        if shipped:
            present = [c for c in PHONE_CLASSES
                       if c in shipped and c in class_stats]
            if not present:
                print()
                print("The config's block names none of the measured classes "
                      f"({', '.join(PHONE_CLASSES)}), so there is nothing to "
                      "compare.")
            else:
                # Rank by MEASURED duration, and tie-break on the measured
                # median rather than on the tuple, so two classes with nearly
                # equal durations are not ordered by dictionary position. A rank
                # that flipped because of a tie would be reported as a real
                # disagreement with the config, which is worse than useless.
                order_index = {c: i + 1 for i, (c, _) in enumerate(ordered)}

                # Same tie-break on the config side: sort by bonus descending,
                # then by measured median descending, then by class name so the
                # result is deterministic across runs.
                cfg_order = sorted(
                    present,
                    key=lambda c: (-shipped.get(c, 0),
                                   -class_stats[c]["median_ms"], c),
                )
                cfg_rank = {c: i + 1 for i, c in enumerate(cfg_order)}

                print()
                print("Against the config you are about to sweep:")
                print(f"  {'class':<11}{'bonus':>8}{'median ms':>11}"
                      f"{'dur rank':>10}{'cfg rank':>10}  verdict")
                verdict_count = 0
                for ph_class in present:
                    d_rank = order_index.get(ph_class, "?")
                    c_rank = cfg_rank.get(ph_class, "?")
                    ok = d_rank == c_rank
                    verdict_count += (not ok)
                    print(f"  {ph_class:<11}{shipped[ph_class]:>8.2f}"
                          f"{class_stats[ph_class]['median_ms']:>11.0f}"
                          f"{str(d_rank):>10}{str(c_rank):>10}"
                          f"  {'ok' if ok else 'differs'}")
                if verdict_count:
                    print()
                    print(f"  {verdict_count} class(es) where the config's bonus")
                    print("  order differs from the measured duration order:")
                    for ph_class in present:
                        if order_index.get(ph_class) != cfg_rank.get(ph_class):
                            print(f"    - {ph_class}")
                    print("  Two readings, and the data cannot choose between")
                    print("  them: the config encodes a belief the data does not")
                    print("  support, OR these classes genuinely want a prior")
                    print("  opposite to their duration. A plosive burst is the")
                    print("  case where that is plausible -- it is short but it is")
                    print("  also unambiguous, so the decoder may need to be")
                    print("  pushed to commit to it rather than stretched.")
                    print("  Sweep before assuming the model is at fault:")
                    print("    --grid vowel_penalty=1.5,2,2.5 "
                          "sonorant_penalty=1.2,1.6,2")
                else:
                    print()
                    print("  The config's bonus order matches the measured")
                    print("  duration order. The values themselves are still")
                    print("  unmeasured guesses, but the ORDER -- which is the")
                    print("  part a guess is most likely to get wrong -- is right.")
        else:
            print()
            print("No postprocess.stickiness_bonus_by_class in the config, so")
            print("there is nothing to compare against. That is the pre-Group-1")
            print("state: one scalar for every phoneme, which cannot stretch a")
            print("120 ms vowel and a 15 ms burst at the same time.")

    # ------------------------------------------------------- short-phoneme report
    # The suspected cause of the worst confusions (ae, ax, i0, u0) is that these
    # are the SHORTEST PT vowels, and that to_bio_tags may be eating them. This
    # section is the direct test: it counts how many occurrences round to a
    # single frame, and how many the integer arithmetic inflates.
    print(f"\n{'-' * 78}")
    print("SHORT-PHONEME REPORT  (is to_bio_tags eating the short ones?)")
    print(f"{'-' * 78}")
    print("to_bio_tags computes s_idx = int(start/d) and e_idx = int(end/d), then")
    print(f"writes B- at s_idx and I- through e_idx. With d = {frame_dur * 1000:.0f} ms:")
    print("  * a phoneme shorter than one frame still gets a B- tag, so it")
    print("    SURVIVES, but occupies a whole frame and is inflated, not erased;")
    print("  * two phonemes landing on the SAME s_idx means the second overwrites")
    print("    the first's B- tag -- that one IS erased, silently.")
    print()
    # Erosion by tag overwrite needs the per-file order of consecutive phonemes,
    # which is deliberately not retained here: counting it properly belongs with
    # the to_bio_tags test (B2), not in a duration table. What IS countable
    # without the order is the rounding damage, which is the same underlying
    # integer arithmetic.
    single_frame = 0
    inflation_ms = []
    for phon, occ in per_occurrence.items():
        for _lang, dur, nframes in occ:
            if nframes == 1:
                single_frame += 1
            # How much longer the model is told the phoneme lasted than it did.
            # Always >= 0, because of the +1, so the MEDIAN is the readable
            # number: a boolean "was it inflated" is 100% and tells nothing.
            inflation_ms.append(max(0.0, nframes * frame_dur - dur) * 1000.0)
    inflation_ms.sort()
    print(f"rounding inflation, median:          "
          f"{percentile(inflation_ms, 0.5):.0f} ms per phoneme")
    print(f"rounding inflation, p90:             "
          f"{percentile(inflation_ms, 0.9):.0f} ms per phoneme")
    print(f"  (the +1 in `e_idx - s_idx + 1` makes this always positive; for a")
    print(f"   15 ms plosive burst the median inflation is a large fraction of the")
    print(f"   phoneme itself, which is what makes short phonemes hard to place)")
    print(f"occurrences landing on 1 frame:      {single_frame} "
          f"({100.0 * single_frame / total_phones:.1f}% of all)")
    print(f"(erosion by tag overwrite is not counted here; it needs per-file order)")

    if unknown:
        print(f"\nPhonemes not explicitly in the map (fell back to sonorant): "
              f"{sorted(unknown)}")
        print("Add them to phoneme_map.yaml if they are common -- a missing entry")
        print("means a duration prior that was never applied to them.")

    # ------------------------------------------------------------------- output
    if args.out:
        cols = ["lang", "phoneme", "type", "class", "n", "median_ms", "mean_ms",
                "p10_ms", "p90_ms", "min_ms", "max_ms", "median_frames",
                "mean_frames", "min_frames", "max_frames"]
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in sorted(rows, key=lambda r: -r["median_ms"]):
                w.writerow({k: r[k] for k in cols})
        print(f"\nWrote {args.out}")

    if args.json_out:
        payload = {
            "frame_duration": frame_dur,
            "n_files": total_files,
            "n_occurrences": total_phones,
            "per_phoneme": rows,
            "by_type": {t: stats(v) for t, v in by_type.items()},
            "by_class": {c: stats(v) for c, v in by_class.items()},
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())