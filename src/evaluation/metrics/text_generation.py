# SPDX-FileCopyrightText: 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Text generation metrics (BLEU, ROUGE-L) for captioning / free-form generation tasks."""

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute sentence-level BLEU between a reference and hypothesis.

    Uses NLTK's smoothing method 1 so that short hypotheses with zero
    n-gram overlap don't collapse to 0.0.
    """
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F-measure between a reference and hypothesis."""
    if not reference or not hypothesis:
        return 0.0
    scores = _rouge.score(reference, hypothesis)
    return scores["rougeL"].fmeasure
