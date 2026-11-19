"""BLEU and chrF, implemented from their definitions (Section 11 of the paper).

Both follow the default settings of sacreBLEU, the standard tool in MT research,
so that scores are comparable with published work:

  BLEU  = BP * exp( sum_{n=1..4} (1/4) * log p_n ),   BP = min(1, exp(1 - r/c))
  chrF  = (1 + b^2) * chrP * chrR / (b^2 * chrP + chrR),  b = 2, character n-grams 1..6

Command line:
    python metrics.py hypotheses.txt references.txt
(one sentence per line, the same number of lines in both files)
"""
import math
import re
import sys
import unicodedata
from collections import Counter

# ---------------------------------------------------------------------------
# Tokenisation (the "13a" tokenizer used by sacreBLEU and the WMT campaigns)
# ---------------------------------------------------------------------------
_13A_RULES = [
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),  # split punctuation
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),               # period/comma not after a digit
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),               # period/comma not before a digit
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),                    # dash after a digit
]


def tokenize_13a(line):
    line = line.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    if "&" in line:
        line = (line.replace("&quot;", '"').replace("&amp;", "&")
                    .replace("&lt;", "<").replace("&gt;", ">"))
    line = f" {line} "
    for pattern, repl in _13A_RULES:
        line = pattern.sub(repl, line)
    # Also split non-English punctuation (Arabic comma and question mark,
    # Chinese full stops, quotation marks, etc.), as sacreBLEU's "intl"
    # tokenizer does. Without this, "حالك؟" would never match "حالك".
    line = "".join(f" {ch} " if ord(ch) > 127 and unicodedata.category(ch)[0] in "PS"
                   else ch for ch in line)
    return line.split()


def ngrams(items, n):
    """Counter of all n-grams (as tuples or strings) of length n."""
    return Counter(tuple(items[i:i + n]) for i in range(len(items) - n + 1))


# ---------------------------------------------------------------------------
# BLEU (Papineni et al., 2002)
# ---------------------------------------------------------------------------
def corpus_bleu(hypotheses, references, max_order=4):
    """Corpus-level BLEU with one reference per sentence.

    Matches and totals are summed over the whole corpus before the precisions
    are computed, as in the original definition. Zero precisions are smoothed
    with the 'exp' method (sacreBLEU's default) so that one missing 4-gram
    match does not force the score to 0.
    """
    if len(hypotheses) != len(references):
        raise ValueError("Need exactly one reference per hypothesis.")
    matches = [0] * max_order
    totals = [0] * max_order
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        h, r = tokenize_13a(hyp), tokenize_13a(ref)
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_order + 1):
            h_ngrams, r_ngrams = ngrams(h, n), ngrams(r, n)
            # modified (clipped) precision: an n-gram counts at most as often
            # as it appears in the reference
            matches[n - 1] += sum(min(c, r_ngrams[g]) for g, c in h_ngrams.items())
            totals[n - 1] += max(len(h) - n + 1, 0)

    if hyp_len == 0:
        return dict(score=0.0, precisions=[0.0] * max_order, bp=0.0,
                    hyp_len=0, ref_len=ref_len)

    precisions, smooth = [], 1.0
    for m, t in zip(matches, totals):
        if t == 0:
            precisions.append(0.0)
        elif m == 0:                       # 'exp' smoothing
            smooth *= 2
            precisions.append(100.0 / (smooth * t))
        else:
            precisions.append(100.0 * m / t)

    # "effective order": only use n-gram orders that exist in the output, so a
    # short sentence (e.g. three words, which has no 4-grams) is not scored 0.
    # This is what sacreBLEU does for sentence-level BLEU; for longer texts all
    # four orders exist and the result is the standard corpus BLEU.
    eff_order = sum(1 for t in totals if t > 0)
    used = precisions[:eff_order]
    # no word in common with the reference at all: BLEU is 0 (as in sacreBLEU)
    if eff_order == 0 or matches[0] == 0 or min(used) <= 0:
        log_mean = -math.inf
    else:
        log_mean = sum(math.log(p / 100.0) for p in used) / eff_order

    # brevity penalty: punishes output shorter than the reference
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)
    score = 0.0 if log_mean == -math.inf else 100.0 * bp * math.exp(log_mean)
    return dict(score=score, precisions=precisions, bp=bp,
                hyp_len=hyp_len, ref_len=ref_len)


# ---------------------------------------------------------------------------
# chrF (Popovic, 2015)
# ---------------------------------------------------------------------------
def corpus_chrf(hypotheses, references, char_order=6, beta=2.0):
    """Corpus-level chrF on character n-grams of order 1..char_order.

    Whitespace is removed before counting, statistics are summed over the
    corpus, precision and recall are averaged over the n-gram orders, and the
    two are combined in an F-score that weights recall beta times as much.
    """
    if len(hypotheses) != len(references):
        raise ValueError("Need exactly one reference per hypothesis.")
    hyp_counts = [0] * char_order
    ref_counts = [0] * char_order
    match_counts = [0] * char_order

    for hyp, ref in zip(hypotheses, references):
        h = "".join(hyp.split())
        r = "".join(ref.split())
        for n in range(1, char_order + 1):
            h_ngrams, r_ngrams = ngrams(h, n), ngrams(r, n)
            hyp_counts[n - 1] += sum(h_ngrams.values())
            ref_counts[n - 1] += sum(r_ngrams.values())
            match_counts[n - 1] += sum((h_ngrams & r_ngrams).values())

    eps = 1e-16
    avg_p = avg_r = 0.0
    effective_order = 0
    for n_hyp, n_ref, n_match in zip(hyp_counts, ref_counts, match_counts):
        avg_p += n_match / n_hyp if n_hyp > 0 else eps
        avg_r += n_match / n_ref if n_ref > 0 else eps
        if n_hyp > 0 and n_ref > 0:
            effective_order += 1

    if effective_order == 0:
        return dict(score=0.0, precision=0.0, recall=0.0)
    chr_p, chr_r = avg_p / effective_order, avg_r / effective_order
    b2 = beta ** 2
    if chr_p + chr_r == 0:
        return dict(score=0.0, precision=0.0, recall=0.0)
    score = 100.0 * (1 + b2) * chr_p * chr_r / (b2 * chr_p + chr_r)
    return dict(score=score, precision=100.0 * chr_p, recall=100.0 * chr_r)


def score(hypotheses, references):
    """BLEU and chrF together, rounded for display."""
    bleu = corpus_bleu(hypotheses, references)
    chrf = corpus_chrf(hypotheses, references)
    return {
        "bleu": round(bleu["score"], 2),
        "bleu_precisions": [round(p, 1) for p in bleu["precisions"]],
        "brevity_penalty": round(bleu["bp"], 3),
        "chrf": round(chrf["score"], 2),
        "chrf_precision": round(chrf["precision"], 1),
        "chrf_recall": round(chrf["recall"], 1),
        "settings": "BLEU: 13a tokens + Unicode punctuation, up to 4-grams, exp smoothing | chrF: char 6-grams, beta=2",
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python metrics.py hypotheses.txt references.txt")
    with open(sys.argv[1], encoding="utf-8") as f:
        hyps = [line.rstrip("\n") for line in f]
    with open(sys.argv[2], encoding="utf-8") as f:
        refs = [line.rstrip("\n") for line in f]
    result = score(hyps, refs)
    print(f"BLEU = {result['bleu']:.2f}  "
          f"(precisions {'/'.join(map(str, result['bleu_precisions']))}, "
          f"BP = {result['brevity_penalty']})")
    print(f"chrF = {result['chrf']:.2f}  "
          f"(precision {result['chrf_precision']}, recall {result['chrf_recall']})")
