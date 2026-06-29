"""Text similarity metrics for VLM annotation scoring.

Implements BLEU, ROUGE-L, and CIDEr metrics in pure Python (no nltk dependency)
for comparing VLM-generated scene descriptions between clean reference and
distorted images.  Optionally supports Sentence-BERT and BERTScore for
richer semantic similarity.

References:
- BLEU: Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine
  Translation", ACL 2002.  Sentence-level BLEU with smoothing method 7.
- ROUGE-L: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries",
  ACL 2004.  LCS-based F-measure.
- CIDEr: Vedantam et al., "CIDEr: Consensus-based Image Description
  Evaluation", CVPR 2015.  TF-IDF weighted n-gram cosine similarity.
- Sentence-BERT: Reimers & Gurevych, "Sentence-BERT", EMNLP 2019.
- BERTScore: Zhang et al., "BERTScore", ICLR 2020.
"""

import logging
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_log = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False
    SentenceTransformer = None

try:
    from bert_score import BERTScorer

    _BS_AVAILABLE = True
except ImportError:
    _BS_AVAILABLE = False
    BERTScorer = None


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _lcs(seq1: Sequence[str], seq2: Sequence[str]) -> int:
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_bleu(
    reference: str,
    hypothesis: str,
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """Compute sentence-level BLEU score.

    Uses smoothed precision (method 7 from Chen & Cherry, 2014) and brevity
    penalty.  Returns a float in [0, 1].

    Args:
        reference: Reference text.
        hypothesis: Candidate/hypothesis text.
        max_n: Maximum n-gram order (BLEU-4 by default).
        smooth: Apply smoothing to avoid zero precisions.
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not hyp_tokens:
        return 0.0

    r = len(ref_tokens)
    c = len(hyp_tokens)

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        ref_ngrams = _ngrams(ref_tokens, n)
        if not ref_ngrams:
            precisions.append(0.0)
            continue

        hyp_ngrams = _ngrams(hyp_tokens, n)
        total = sum(hyp_ngrams.values())  # type: ignore[arg-type]

        clipped = 0
        for ngram, hyp_count in hyp_ngrams.items():
            ref_count = ref_ngrams.get(ngram, 0)
            clipped += min(hyp_count, ref_count)

        if total == 0:
            p = 0.0
        else:
            p = clipped / total
            if smooth and p == 0.0:
                p = 1.0 / (total + 1)  # Smoothing method 7

        precisions.append(p)

    # Smooth zero precisions to avoid log(0) in geometric mean
    precisions = [max(p, 1e-10) for p in precisions]
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    bp = 1.0 if c > r else math.exp(1.0 - r / max(c, 1))

    return min(1.0, bp * geo_mean)


def compute_rouge_l(
    reference: str,
    hypothesis: str,
    beta: float = 1.0,
) -> float:
    """Compute ROUGE-L F-score based on longest common subsequence.

    Args:
        reference: Reference text.
        hypothesis: Candidate/hypothesis text.
        beta: Parameter controlling precision-recall tradeoff
              (beta > 1 favours recall, beta < 1 favours precision).

    Returns:
        ROUGE-L F-score in [0, 1].
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not hyp_tokens:
        return 0.0

    lcs_len = _lcs(ref_tokens, hyp_tokens)
    r = len(ref_tokens)
    c = len(hyp_tokens)

    if r == 0 or c == 0:
        return 0.0

    recall = lcs_len / r
    precision = lcs_len / c

    beta_sq = beta * beta
    if recall + precision == 0:
        return 0.0

    return (1.0 + beta_sq) * recall * precision / (recall + beta_sq * precision)


def compute_tf_idf_vectors(
    documents: List[str],
    max_n: int = 4,
) -> Tuple[Dict[str, float], List[Dict[str, float]], List[str]]:
    """Compute TF-IDF vectors for a collection of documents.

    Args:
        documents: List of text documents.
        max_n: Maximum n-gram order.

    Returns:
        Tuple of (idf_dict, tf_idf_vectors, all_ngrams_in_order).
    """
    N = len(documents)
    if N == 0:
        return {}, [], []

    all_ngram_sets: List[Counter] = []
    all_ngrams: set = set()
    for doc in documents:
        tokens = _tokenize(doc)
        doc_ngrams = Counter()
        for n in range(1, max_n + 1):
            doc_ngrams.update(_ngrams(tokens, n))
        all_ngram_sets.append(doc_ngrams)
        all_ngrams.update(doc_ngrams.keys())

    ngram_list = sorted(all_ngrams, key=str)

    df = Counter()
    for doc_ngrams in all_ngram_sets:
        for ngram in set(doc_ngrams.keys()):
            df[ngram] += 1

    idf: Dict[str, float] = {}
    for ngram in ngram_list:
        idf[str(ngram)] = math.log((N + 1) / (df.get(ngram, 0) + 1)) + 1.0

    tf_idf_vectors: List[Dict[str, float]] = []
    for doc_ngrams in all_ngram_sets:
        total = sum(doc_ngrams.values())
        vec = {}
        for ngram in ngram_list:
            ng_key = str(ngram)
            if total > 0:
                vec[ng_key] = (doc_ngrams.get(ngram, 0) / total) * idf[ng_key]
            else:
                vec[ng_key] = 0.0
        tf_idf_vectors.append(vec)

    return idf, tf_idf_vectors, [str(ng) for ng in ngram_list]


def _cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    keys = set(vec1.keys()) | set(vec2.keys())
    dot = sum(vec1.get(k, 0.0) * vec2.get(k, 0.0) for k in keys)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def compute_cider(
    references: List[str],
    hypothesis: str,
    corpus: Optional[List[str]] = None,
    max_n: int = 4,
) -> float:
    """Compute CIDEr score for a single hypothesis against references.

    CIDEr measures consensus by computing TF-IDF weighted cosine similarity
    between the hypothesis and reference n-gram vectors.  Returns a score
    in [0, ~10], where higher is better.

    Args:
        references: List of reference texts.
        hypothesis: Candidate/hypothesis text.
        corpus: Optional full corpus for TF-IDF computation.
                If None, uses references as the corpus.
        max_n: Maximum n-gram order.

    Returns:
        CIDEr score (averaged over references).
    """
    if not references or not hypothesis:
        return 0.0

    all_docs = list(references) + [hypothesis]
    if corpus:
        all_docs = list(corpus) + [hypothesis]

    _, tf_idf_vectors, _ = compute_tf_idf_vectors(all_docs, max_n)

    if len(tf_idf_vectors) < 2:
        return 0.0

    hyp_vec = tf_idf_vectors[-1]
    ref_vecs = tf_idf_vectors[: len(references)] if not corpus else tf_idf_vectors[:-1]

    if not ref_vecs:
        return 0.0

    scores = [_cosine_similarity(hyp_vec, rv) for rv in ref_vecs]
    return float(np.mean(scores) * 10.0)


def compute_cognitive_score(
    ref_texts: List[str],
    dist_texts: List[str],
    weights: Tuple[float, float, float] = (1.0, 1.0, 0.1),
    cider_corpus: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute the cognitive quality score via description comparison.

    For each prompt (pair of reference/distorted descriptions), computes
    BLEU, ROUGE-L, and CIDEr, then averages across prompts and combines
    with the proposal's weighting scheme (1:1:0.1).

    Args:
        ref_texts: List of reference descriptions (one per prompt).
        dist_texts: List of distorted descriptions (one per prompt).
        weights: (bleu_weight, rouge_weight, cider_weight). Default (1, 1, 0.1).
        cider_corpus: Optional full corpus of reference descriptions for CIDEr
                      TF-IDF; if None, uses ref_texts.

    Returns:
        Dict with keys: bleu, rouge_l, cider, cognitive_score.
    """
    if len(ref_texts) != len(dist_texts):
        raise ValueError(
            f"Reference and distorted text lists must have same length: "
            f"{len(ref_texts)} vs {len(dist_texts)}"
        )

    if not ref_texts:
        return {"bleu": 0.0, "rouge_l": 0.0, "cider": 0.0, "cognitive_score": 0.0}

    bleu_total = 0.0
    rouge_total = 0.0
    cider_total = 0.0
    per_prompt: List[dict] = []

    for ref, dist in zip(ref_texts, dist_texts):
        b = compute_bleu(ref, dist)
        r = compute_rouge_l(ref, dist)
        c = compute_cider([ref], dist, corpus=cider_corpus)
        bleu_total += b
        rouge_total += r
        cider_total += c
        per_prompt.append(
            {
                "bleu": round(b, 6),
                "rouge_l": round(r, 6),
                "cider": round(c, 6),
            }
        )

    n = len(ref_texts)
    avg_bleu = bleu_total / n
    avg_rouge = rouge_total / n
    avg_cider = cider_total / n

    w_bleu, w_rouge, w_cider = weights
    cognitive = w_bleu * avg_bleu + w_rouge * avg_rouge + w_cider * avg_cider
    # Normalize by sum of weights so scale is [0, ~1]
    cognitive = cognitive / (w_bleu + w_rouge + w_cider)

    return {
        "bleu": round(avg_bleu, 6),
        "rouge_l": round(avg_rouge, 6),
        "cider": round(avg_cider, 6),
        "cognitive_score": round(cognitive, 6),
        "per_prompt": per_prompt,
    }


_ST_CACHE: Optional["SentenceTransformer"] = None
_BS_CACHE: Optional["BERTScorer"] = None


def compute_similarity(
    reference: str,
    hypothesis: str,
    cider_corpus: Optional[List[str]] = None,
) -> dict:
    """Compute all text similarity metrics between two descriptions.

    Computes BLEU, ROUGE-L, CIDEr (always available), plus optionally
    Sentence-BERT cosine similarity and BERTScore F1 if the respective
    packages are installed.

    Args:
        reference: Reference text (clean image description).
        hypothesis: Hypothesis text (distorted image description).
        cider_corpus: Optional corpus for CIDEr TF-IDF computation.

    Returns:
        Dict with keys: bleu, rouge_l, cider, sentence_bert (if available),
        bertscore_f1 (if available).
    """
    result: dict = {
        "bleu": round(compute_bleu(reference, hypothesis), 6),
        "rouge_l": round(compute_rouge_l(reference, hypothesis), 6),
        "cider": round(compute_cider([reference], hypothesis, corpus=cider_corpus), 6),
    }

    global _ST_CACHE
    if _ST_AVAILABLE:
        try:
            if _ST_CACHE is None:
                _ST_CACHE = SentenceTransformer("all-MiniLM-L6-v2")
            emb_ref = _ST_CACHE.encode(reference, convert_to_tensor=True)
            emb_hyp = _ST_CACHE.encode(hypothesis, convert_to_tensor=True)
            cos_sim = float(
                (emb_ref @ emb_hyp.T).cpu().item()
                / (emb_ref.norm().cpu().item() * emb_hyp.norm().cpu().item() + 1e-10)
            )
            result["sentence_bert"] = round(cos_sim, 6)
        except Exception:
            _log.debug("Sentence-BERT similarity computation failed", exc_info=True)

    global _BS_CACHE
    if _BS_AVAILABLE:
        try:
            if _BS_CACHE is None:
                _BS_CACHE = BERTScorer(lang="en", rescale_with_baseline=True)
            P, R, F1 = _BS_CACHE.score([hypothesis], [reference])
            result["bertscore_f1"] = round(float(F1[0]), 6)
        except Exception:
            _log.debug("BERTScore computation failed", exc_info=True)

    return result


def derive_cognitive_score(
    similarity: dict,
    method: str = "weighted",
    weights: Tuple[float, float, float] = (1.0, 1.0, 0.1),
) -> float:
    """Derive a single cognitive quality score from similarity metrics.

    Args:
        similarity: Dict from ``compute_similarity()``.
        method: Aggregation strategy:
            - ``"weighted"``: weighted mean of bleu/rouge_l/cider (default).
            - ``"bertscore"``: use bertscore_f1 directly (requires bert-score).
            - ``"bleu_only"``, ``"rouge_l_only"``, ``"cider_only"``: single metric.
        weights: (bleu, rouge_l, cider) for weighted method.

    Returns:
        Float cognitive score, normalized to approximately [0, 1].
    """
    if method == "bertscore":
        bs = similarity.get("bertscore_f1", None)
        if bs is not None:
            return round(bs, 6)
        _log.warning("bertscore_f1 not available, falling back to weighted")
        method = "weighted"

    if method == "weighted":
        w_bleu, w_rouge, w_cider = weights
        weight_sum = w_bleu + w_rouge + w_cider
        score = (
            w_bleu * similarity.get("bleu", 0.0)
            + w_rouge * similarity.get("rouge_l", 0.0)
            + w_cider * similarity.get("cider", 0.0)
        ) / weight_sum
        return round(score, 6)

    single_key = {
        "bleu_only": "bleu",
        "rouge_l_only": "rouge_l",
        "cider_only": "cider",
    }.get(method)

    if single_key and single_key in similarity:
        return round(similarity[single_key], 6)

    raise ValueError(f"Unknown cognitive score method: {method}")
