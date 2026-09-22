"""
instrumentation_similarity.py
=============================

Fast, precise, deterministic, dependency-free similarity scoring between two
MIDI instrumentations (lists of GM program numbers: 0-127 melodic, 128 = drums).

Public API
----------
instrumentation_similarity(src, trg, *, strict=False, detail=False) -> float
"""

from numbers import Integral
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple, Union

__all__ = ["instrumentation_similarity"]

# --------------------------------------------------------------------------- #
# Tuning constants                                                            #
# --------------------------------------------------------------------------- #

DRUM_PROGRAM = 128            # GM "drum kit" pseudo-program (channel 10)
MAX_MELODIC_PROGRAM = 127     # highest melodic program number

# --- timbre affinity matrix ---
_SUB_BASE = 0.95              # affinity of the #1 ranked "gentle substitute"
_SUB_STEP = 0.10              # affinity decay per rank in a substitute list
_SUB_FLOOR = 0.55             # floor for very deep substitute ranks
_FAMILY_AFFINITY = 0.30       # affinity for two instruments of the same GM family

# --- melodic score blending ---
_W_EXACT_BONUS = 0.25         # strength of the exact-overlap (Jaccard) bonus

# --- final score blending ---
_W_MELODIC = 0.90             # weight of the melodic component
_W_DRUM = 0.10                # weight of drum-channel agreement

_EMPTY_VS_EMPTY = 0.00        # score when neither input carries usable data


# --------------------------------------------------------------------------- #
# Knowledge base: ranked "gentle substitutes" (closest timbre first).         #
# --------------------------------------------------------------------------- #
_SUBSTITUTES = {

    # --- Pianos ---
    0: [1, 2, 4],   # Acoustic Grand Piano -> Bright Acoustic, Electric Grand, E.Piano 1
    1: [0, 2, 4],   # Bright Acoustic Piano -> Acoustic Grand, Electric Grand, E.Piano 1
    2: [4, 5, 0],   # Electric Grand Piano -> E.Piano 1, E.Piano 2, Acoustic Grand
    3: [0, 1, 4],   # Honky-Tonk Piano -> Acoustic Grand, Bright Acoustic, E.Piano 1
    4: [5, 2, 0],   # Electric Piano 1 -> E.Piano 2, Electric Grand, Acoustic Grand
    5: [4, 2, 0],   # Electric Piano 2 -> E.Piano 1, Electric Grand, Acoustic Grand
    6: [7, 0],      # Harpsichord -> Clavinet, Acoustic Grand
    7: [6, 0],      # Clavinet -> Harpsichord, Acoustic Grand

    # --- Chromatic Percussion ---
    8: [10, 14, 9],   # Celesta -> Music Box, Tubular Bells, Glockenspiel
    9: [10, 14, 8],   # Glockenspiel -> Music Box, Tubular Bells, Celesta
    10: [8, 9, 14],   # Music Box -> Celesta, Glockenspiel, Tubular Bells
    11: [12, 13, 9],  # Vibraphone -> Marimba, Xylophone, Glockenspiel
    12: [11, 13],     # Marimba -> Vibraphone, Xylophone
    13: [12, 11],     # Xylophone -> Marimba, Vibraphone
    14: [8, 9, 15],   # Tubular Bells -> Celesta, Glockenspiel, Dulcimer
    15: [24, 14, 8],  # Dulcimer -> Acoustic Guitar (Nylon), Tubular Bells, Celesta

    # --- Organs ---
    16: [17, 20, 19], # Drawbar Organ -> Percussive Organ, Reed Organ, Church Organ
    17: [16, 20, 19], # Percussive Organ -> Drawbar Organ, Reed Organ, Church Organ
    18: [16, 17, 19], # Rock Organ -> Drawbar, Percussive, Church
    19: [20, 16, 17], # Church Organ -> Reed Organ, Drawbar Organ, Percussive Organ
    20: [19, 21, 16], # Reed Organ -> Church Organ, Accordion, Drawbar Organ
    21: [23, 22, 20], # Accordion -> Tango Accordion, Harmonica, Reed Organ
    22: [21, 23, 85], # Harmonica -> Accordion, Tango Accordion, Synth Voice
    23: [21, 22, 20], # Tango Accordion -> Accordion, Harmonica, Reed Organ

    # --- Guitars ---
    24: [25, 26, 15], # Acoustic Guitar Nylon -> Acoustic Steel, Jazz Guitar, Dulcimer
    25: [24, 26, 31], # Acoustic Guitar Steel -> Nylon, Jazz Guitar, Harmonics
    26: [27, 24, 25], # Electric Guitar Jazz -> Clean Electric, Nylon, Steel
    27: [26, 28, 31], # Electric Guitar Clean -> Jazz Electric, Muted, Harmonics
    28: [27, 26, 31], # Electric Guitar Muted -> Clean, Jazz, Harmonics
    29: [30, 118],    # Overdriven Guitar -> Distortion Guitar, Synth Drum
    30: [29, 118],    # Distortion Guitar -> Overdriven Guitar, Synth Drum
    31: [27, 24, 25], # Guitar Harmonics -> Clean Electric, Nylon, Steel

    # --- Basses ---
    32: [35, 33, 34], # Acoustic Bass -> Fretless, Finger, Pick
    33: [34, 35, 32], # Electric Bass Finger -> Pick, Fretless, Acoustic
    34: [33, 35, 32], # Electric Bass Pick -> Finger, Fretless, Acoustic
    35: [32, 33, 34], # Fretless Bass -> Acoustic, Finger, Pick
    36: [37, 38, 39], # Slap Bass 1 -> Slap Bass 2, Synth Bass 1, Synth Bass 2
    37: [36, 38, 39], # Slap Bass 2 -> Slap Bass 1, Synth Bass 1, Synth Bass 2
    38: [39, 36, 37], # Synth Bass 1 -> Synth Bass 2, Slap 1, Slap 2
    39: [38, 36, 37], # Synth Bass 2 -> Synth Bass 1, Slap 1, Slap 2

    # --- Strings ---
    40: [41, 42, 48], # Violin -> Viola, Cello, String Ensemble 1
    41: [40, 42, 48], # Viola -> Violin, Cello, String Ensemble 1
    42: [41, 40, 48], # Cello -> Viola, Violin, String Ensemble 1
    43: [42, 32, 48], # Contrabass -> Cello, Acoustic Bass, String Ensemble 1
    44: [48, 49, 40], # Tremolo Strings -> Ensemble 1, Ensemble 2, Violin
    45: [48, 32, 12], # Pizzicato Strings -> Ensemble, Acoustic Bass, Marimba
    46: [14, 8, 107], # Orchestral Harp -> Tubular Bells, Celesta, Koto

    # --- Ensemble ---
    48: [49, 50, 40], # String Ensemble 1 -> Ensemble 2, Synth Strings 1, Violin
    49: [48, 50, 40], # String Ensemble 2 -> Ensemble 1, Synth Strings 1, Violin
    50: [51, 48, 49], # Synth Strings 1 -> Synth Strings 2, String Ensembles
    51: [50, 48, 49], # Synth Strings 2 -> Synth Strings 1, String Ensembles
    52: [53, 54, 91], # Choir Aahs -> Voice Oohs, Synth Voice, Pad 4 (Choir)
    53: [52, 54, 85], # Voice Oohs -> Choir Aahs, Synth Voice, Lead 6 (Voice)
    54: [85, 52, 53], # Synth Voice -> Lead 6 (Voice), Choir Aahs, Voice Oohs

    # --- Brass ---
    56: [59, 57, 61], # Trumpet -> Muted Trumpet, Trombone, Brass Section
    57: [58, 56, 61], # Trombone -> Tuba, Trumpet, Brass Section
    58: [57, 56, 61], # Tuba -> Trombone, Trumpet, Brass Section
    59: [56, 57, 61], # Muted Trumpet -> Trumpet, Trombone, Brass Section
    60: [61, 62, 58], # French Horn -> Brass Section, Synth Brass 1, Tuba
    61: [60, 62, 56], # Brass Section -> French Horn, Synth Brass 1, Trumpet
    62: [63, 61, 60], # Synth Brass 1 -> Synth Brass 2, Brass Section, French Horn
    63: [62, 61, 60], # Synth Brass 2 -> Synth Brass 1, Brass Section, French Horn

    # --- Reed ---
    64: [65, 66, 67], # Soprano Sax -> Alto, Tenor, Baritone
    65: [64, 66, 67], # Alto Sax -> Soprano, Tenor, Baritone
    66: [65, 64, 67], # Tenor Sax -> Alto, Soprano, Baritone
    67: [66, 65, 68], # Baritone Sax -> Tenor, Alto, Oboe
    68: [69, 71, 70], # Oboe -> English Horn, Clarinet, Bassoon
    69: [68, 70, 71], # English Horn -> Oboe, Bassoon, Clarinet
    70: [71, 69, 68], # Bassoon -> Clarinet, English Horn, Oboe
    71: [68, 73, 69], # Clarinet -> Oboe, Flute, English Horn

    # --- Pipe ---
    72: [73, 74, 76], # Piccolo -> Flute, Recorder, Blown Bottle
    73: [72, 74, 76], # Flute -> Piccolo, Recorder, Blown Bottle
    74: [75, 73, 72], # Recorder -> Pan Flute, Flute, Piccolo
    75: [74, 73, 76], # Pan Flute -> Recorder, Flute, Blown Bottle
    76: [77, 75, 74], # Blown Bottle -> Shakuhachi, Pan Flute, Recorder
    77: [76, 75, 73], # Shakuhachi -> Blown Bottle, Pan Flute, Flute
    78: [79, 73, 74], # Whistle -> Ocarina, Flute, Recorder
    79: [78, 74, 73], # Ocarina -> Whistle, Recorder, Flute

    # --- Synth Lead ---
    80: [81, 86, 82], # Square -> Sawtooth, Fifths, Calliope
    81: [80, 84, 82], # Sawtooth -> Square, Charang, Calliope
    82: [83, 81, 84], # Calliope -> Chiff, Sawtooth, Charang
    83: [82, 80, 88], # Chiff -> Calliope, Square, New Age Pad
    84: [81, 85, 80], # Charang -> Sawtooth, Voice Lead, Square
    85: [54, 53, 81], # Voice -> Synth Voice, Voice Oohs, Sawtooth
    86: [80, 81, 82], # Fifths -> Square, Sawtooth, Calliope
    87: [81, 38, 39], # Bass+lead -> Sawtooth, Synth Bass 1, Synth Bass 2

    # --- Synth Pad ---
    88: [89, 90, 91], # New Age -> Warm, Polysynth, Choir
    89: [88, 90, 92], # Warm -> New Age, Polysynth, Bowed
    90: [88, 89, 62], # Polysynth -> New Age, Warm, Synth Brass 1
    91: [52, 54, 89], # Choir -> Choir Aahs, Synth Voice, Warm
    92: [89, 88, 90], # Bowed -> Warm, New Age, Polysynth
    93: [90, 98, 99], # Metallic -> Polysynth, Crystal, Atmosphere
    94: [89, 88, 91], # Halo -> Warm, New Age, Choir
    95: [89, 88, 94], # Sweep -> Warm, New Age, Halo

    # --- Synth FX ---
    96: [97, 98, 88],  # Rain -> Soundtrack, Crystal, New Age
    97: [96, 98, 88],  # Soundtrack -> Rain, Crystal, New Age
    98: [99, 100, 88], # Crystal -> Atmosphere, Brightness, New Age
    99: [100, 98, 88], # Atmosphere -> Brightness, Crystal, New Age
    100: [99, 98, 88], # Brightness -> Atmosphere, Crystal, New Age
    101: [98, 99, 102],# Goblins -> Crystal, Atmosphere, Echoes
    102: [101, 103, 99],# Echoes -> Goblins, Sci-Fi, Atmosphere
    103: [102, 101, 99],# Sci-Fi -> Echoes, Goblins, Atmosphere

    # --- Ethnic ---
    104: [105, 106, 24], # Sitar -> Banjo, Shamisen, Acoustic Guitar Nylon
    105: [104, 24, 25],  # Banjo -> Sitar, Acoustic Guitar Nylon, Steel
    106: [107, 104, 24], # Shamisen -> Koto, Sitar, Acoustic Guitar Nylon
    107: [106, 104, 24], # Koto -> Shamisen, Sitar, Acoustic Guitar Nylon
    108: [12, 11, 13],   # Kalimba -> Marimba, Vibraphone, Xylophone
    109: [111, 68, 69],  # Bagpipe -> Shanai, Oboe, English Horn
    110: [40, 48, 41],   # Fiddle -> Violin, String Ensemble, Viola
    111: [68, 69, 109],  # Shanai -> Oboe, English Horn, Bagpipe

    # --- Percussive ---
    112: [9, 14, 113],   # Tinkle Bell -> Glockenspiel, Tubular Bells, Agogo
    113: [112, 114, 9],  # Agogo -> Tinkle Bell, Steel Drums, Glockenspiel
    114: [113, 112, 9],  # Steel Drums -> Agogo, Tinkle Bell, Glockenspiel
    115: [116, 117, 118],# Woodblock -> Taiko Drum, Melodic Tom, Synth Drum
    116: [117, 118, 115],# Taiko Drum -> Melodic Tom, Synth Drum, Woodblock
    117: [118, 116, 115],# Melodic Tom -> Synth Drum, Taiko Drum, Woodblock
    118: [117, 116, 115] # Synth Drum -> Melodic Tom, Taiko Drum, Woodblock
}

# Small supplement for instruments the table above leaves uncovered
# (119+ "Sound Effects" are intentionally left to the family fallback).
_EXTRA_SUBSTITUTES = {
    47: [116, 117, 118],   # Timpani -> Taiko, Melodic Tom, Synth Drum
    55: [62, 63, 48],      # Orchestra Hit -> Synth Brass 1/2, String Ensemble 1
}


# --------------------------------------------------------------------------- #
# Precomputed 128x128 timbre affinity matrix (built lazily, exactly once).    #
# --------------------------------------------------------------------------- #
_AFFINITY_MATRIX: Optional[Tuple[Tuple[float, ...], ...]] = None


def _build_affinity_matrix() -> Tuple[Tuple[float, ...], ...]:
    """
    Build the melodic affinity matrix M where M[a][b] in [0.0, 1.0] expresses
    how interchangeable instruments a and b are:

        1.00  identical instrument
        0.95  rank-1 "gentle substitute"
        0.85  rank-2 "gentle substitute"
        0.75  rank-3 "gentle substitute"
        0.30  same General MIDI family (8-instrument block), no direct link
        0.00  unrelated

    The directed substitution table is symmetrized with a max() so that a
    link in *either* direction counts (e.g. Honky-Tonk lists Grand Piano as a
    gentle substitute even though Grand Piano's own list omits Honky-Tonk).
    Program 128 (drums) is intentionally absent: it is handled logically.
    """
    n = MAX_MELODIC_PROGRAM + 1                       # 128
    directed = [[0.0] * n for _ in range(n)]

    for program, subs in {**_SUBSTITUTES, **_EXTRA_SUBSTITUTES}.items():
        if not 0 <= program < n:
            continue
        for rank, other in enumerate(subs):
            if not 0 <= other < n:
                continue                              # guard against bad data
            weight = max(_SUB_BASE - _SUB_STEP * rank, _SUB_FLOOR)
            if weight > directed[program][other]:
                directed[program][other] = weight

    family_of = [p // 8 for p in range(n)]            # GM families: blocks of 8
    matrix = []
    for a in range(n):
        row_a = directed[a]
        cells = [1.0] * n                             # diagonal = 1.0
        for b in range(n):
            if b == a:
                continue
            best = row_a[b]
            back = directed[b][a]
            if back > best:
                best = back
            if best == 0.0 and family_of[a] == family_of[b]:
                best = _FAMILY_AFFINITY
            cells[b] = best
        matrix.append(tuple(cells))
    return tuple(matrix)


def _affinity_matrix() -> Tuple[Tuple[float, ...], ...]:
    """Lazy singleton accessor (benign race: rebuilds identical matrix)."""
    global _AFFINITY_MATRIX
    if _AFFINITY_MATRIX is None:
        _AFFINITY_MATRIX = _build_affinity_matrix()
    return _AFFINITY_MATRIX


# --------------------------------------------------------------------------- #
# Input normalization                                                         #
# --------------------------------------------------------------------------- #
def _normalize_instruments(
    programs: Iterable[Any], name: str, strict: bool
) -> Tuple[bool, FrozenSet[int]]:
    """
    Reduce an input to (has_drums, melodic_frozenset).

    * duplicates / ordering are destroyed on purpose (set semantics)
    * program 128 is peeled off into the drum flag
    * bools are rejected explicitly (True/False are not instrument programs)
    * non-integers and out-of-range values are silently dropped, or raise
      ValueError in strict mode; numpy integers are accepted (numbers.Integral)
    """
    if programs is None:
        raise TypeError(
            f"'{name}' must be an iterable of MIDI instrument numbers, not None"
        )
    try:
        items = iter(programs)
    except TypeError:
        raise TypeError(
            f"'{name}' must be an iterable of MIDI instrument numbers, "
            f"got {type(programs).__name__}"
        ) from None

    melodic: set = set()
    has_drums = False
    for value in items:
        if isinstance(value, bool) or not isinstance(value, Integral):
            if strict:
                raise ValueError(
                    f"{name} contains a non-integer instrument value: {value!r}"
                )
            continue
        program = int(value)
        if program == DRUM_PROGRAM:
            has_drums = True
        elif 0 <= program <= MAX_MELODIC_PROGRAM:
            melodic.add(program)
        elif strict:
            raise ValueError(
                f"{name} contains an out-of-range instrument value: {value!r} "
                f"(expected 0..{DRUM_PROGRAM})"
            )
        # lenient mode: junk is silently ignored
    return has_drums, frozenset(melodic)


# --------------------------------------------------------------------------- #
# Melodic soft affinity                                                       #
# --------------------------------------------------------------------------- #
def _melodic_soft_affinity(mel_a: FrozenSet[int], mel_b: FrozenSet[int]) -> float:
    """
    Bidirectional best-match affinity between two melodic sets:
    every instrument in one set is matched to its single closest partner in
    the other set; the forward and backward means are averaged, which makes
    the score symmetric and robust to different set sizes.
    """
    matrix = _affinity_matrix()
    fwd_total = 0.0
    for a in mel_a:
        row = matrix[a]
        fwd_total += max(map(row.__getitem__, mel_b))   # C-speed inner max
    bwd_total = 0.0
    for b in mel_b:
        row = matrix[b]
        bwd_total += max(map(row.__getitem__, mel_a))
    return 0.5 * (fwd_total / len(mel_a) + bwd_total / len(mel_b))


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def instrumentation_similarity(
    src: Iterable[Any],
    trg: Iterable[Any],
    *,
    strict: bool = False,
    detail: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Compute a similarity score in [0.0, 1.0] between two MIDI instrumentations.

    Parameters
    ----------
    src, trg : iterable of int
        GM program numbers, 0-127 (melodic) and/or 128 (drum track).
        Duplicates and ordering are ignored (inputs are reduced to sets).
        Non-integer / out-of-range junk is silently dropped unless strict=True.
    strict : bool, keyword-only
        If True, raise ValueError on any invalid element instead of ignoring it.
    detail : bool, keyword-only
        If True, return (score, diagnostics-dict) instead of just the score.

    Scoring model
    -------------
    1. Each input is split into a melodic set (0-127) and a drum flag (128).
    2. Melodic similarity = soft affinity + exact-overlap bonus:
       * soft affinity: via a precomputed 128x128 timbre affinity matrix
         (exact = 1.0, ranked gentle substitutes = 0.95/0.85/0.75,
         same GM family = 0.30, unrelated = 0.0), matched bidirectionally;
       * exact-overlap bonus: Jaccard * 0.25 * (1 - soft) -- exact matches
         lift the score but can never drag it down.
    3. Drum component: excluded if neither list has drums; 1.0 if both do;
       0.0 if only one does.
    4. Final score = 0.9 * melodic + 0.1 * drum when the drum component
       applies.  Hence incompatible melodic content can never be rescued by
       a shared drum track (capped at 0.1), while identical melodic content
       scores 0.9-1.0.

    Edge cases handled
    ------------------
    * duplicates / ordering      -> set semantics
    * empty vs empty             -> 0.0 (no usable data; tune _EMPTY_VS_EMPTY)
    * empty vs non-empty         -> 0.0
    * drums-only vs drums-only   -> 1.0 ; drums-only vs empty -> 0.0
    * one-sided drums            -> small mismatch penalty, never a bonus
    * junk elements (str, bool, float, out-of-range) -> dropped / strict-raise
    * None / non-iterables       -> TypeError
    * fully deterministic        -> no randomness, same inputs -> same score

    Examples
    --------
    >>> instrumentation_similarity([0, 40, 128], [0, 40, 128])   # identical
    1.0
    >>> instrumentation_similarity([40], [41])                   # Violin vs Viola
    0.95
    >>> instrumentation_similarity([0, 40, 128], [65, 80, 128])  # incompatible, shared drums
    0.1
    >>> instrumentation_similarity([0, 40], [0, 40, 128])        # drum-channel mismatch
    0.9
    """
    drum_src, mel_src = _normalize_instruments(src, "src", strict)
    drum_trg, mel_trg = _normalize_instruments(trg, "trg", strict)

    # ---------------- melodic component ----------------
    soft: Optional[float] = None
    jaccard: Optional[float] = None
    if not mel_src and not mel_trg:
        melodic: Optional[float] = None            # nothing melodic on either side
    elif not mel_src or not mel_trg:
        melodic = 0.0                              # melodic vs drums-only/empty
        soft = jaccard = 0.0
    else:
        intersection = len(mel_src & mel_trg)
        union = len(mel_src | mel_trg)
        jaccard = intersection / union
        soft = _melodic_soft_affinity(mel_src, mel_trg)
        # Jaccard acts as a fading exact-overlap bonus: it vanishes as the
        # soft score approaches 1.0 and can never lower the result.
        melodic = min(1.0, soft + _W_EXACT_BONUS * jaccard * (1.0 - soft))

    # ---------------- drum component ----------------
    if not drum_src and not drum_trg:
        drum: Optional[float] = None               # drums irrelevant on both sides
    else:
        drum = 1.0 if (drum_src and drum_trg) else 0.0

    # ---------------- combine ----------------
    if melodic is None and drum is None:           # both inputs carry no data
        score: float = _EMPTY_VS_EMPTY
    elif melodic is None:                          # drums-only vs drums-only
        score = drum
    elif drum is None:                             # neither side uses drums
        score = melodic
    else:                                          # drums matter on >= 1 side
        score = _W_MELODIC * melodic + _W_DRUM * drum

    score = min(1.0, max(0.0, score))              # defensive clamp

    if detail:
        return score, {
            "score": score,
            "melodic_score": melodic,
            "soft_affinity": soft,
            "jaccard": jaccard,
            "drum_score": drum,
            "src_melodic": sorted(mel_src),
            "trg_melodic": sorted(mel_trg),
            "src_has_drums": drum_src,
            "trg_has_drums": drum_trg,
        }
    return score


# --------------------------------------------------------------------------- #
# Self-test / demo                                                            #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from math import isclose

    cases = [
        ([0, 40, 128], [0, 40, 128], 1.0,    "identical (with drums)"),
        ([40, 0], [0, 40, 40], 1.0,          "order & duplicates ignored"),
        ([40], [41], 0.95,                   "Violin <-> Viola (rank-1 substitute)"),
        ([33], [34], 0.95,                   "Finger bass <-> Pick bass"),
        ([6], [2], 0.30,                     "same GM family fallback"),
        ([40], [80], 0.0,                    "unrelated families"),
        ([0, 40, 128], [65, 80, 128], 0.1,   "incompatible + shared drums -> LOW"),
        ([0, 40], [0, 40, 128], 0.9,         "identical, drum-channel mismatch"),
        ([0, 40, 72], [0, 40], 31 / 36,      "partial overlap (extra Piccolo in src)"),
        ([0], [0, 40], 0.78125,              "one-sided extra instrument"),
        ([128], [128], 1.0,                  "drums-only on both sides"),
        ([], [], 0.0,                        "no data at all"),
        (["x", 999, 40.5, 40], [40], 1.0,    "lenient: junk silently dropped"),
    ]

    width = max(len(note) for _, _, _, note in cases)
    all_ok = True
    for s, t, expected, note in cases:
        got = instrumentation_similarity(s, t)
        ok = isclose(got, expected, rel_tol=0.0, abs_tol=1e-9)
        all_ok = all_ok and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {note:<{width}}  "
              f"expected={expected:<10.6g} got={got:.6g}")

    print()
    score, info = instrumentation_similarity([0, 40, 128], [65, 80, 128], detail=True)
    print("detail example:", score, "->", info)

    try:
        instrumentation_similarity([40, "guitar"], [40], strict=True)
    except ValueError as exc:
        print("strict mode correctly raised:", exc)

    print("\nAll checks passed." if all_ok else "\nSome checks FAILED.")