"""Pure-Python, per-query retrieval metrics and paired percentile bootstrap."""

from __future__ import annotations

import math
import random

from .data import relevant_doc_ids
from .types import Question


def _alpha_dcg(ranking: list[str], groups: list[set[str]], alpha: float, limit: int) -> float:
    """Accumulate binary subtopic novelty, ignoring repeated IDs at their original ranks."""
    seen_docs: set[str] = set()
    covered = [0] * len(groups)
    total = 0.0
    for rank, doc_id in enumerate(ranking[:limit], 1):
        if doc_id in seen_docs:
            continue
        seen_docs.add(doc_id)
        gain = 0.0
        for index, group in enumerate(groups):
            if doc_id in group:
                gain += (1 - alpha) ** covered[index]
                covered[index] += 1
        total += gain / math.log2(rank + 1)
    return total


def _alpha_ideal(groups: list[set[str]], alpha: float, limit: int) -> list[str]:
    """Use ndeval's greedy ideal ranking, breaking gain ties by descending document ID."""
    remaining = set().union(*groups) if groups else set()
    covered = [0] * len(groups)
    ranking = []
    while remaining and len(ranking) < limit:
        def gain(doc_id: str) -> float:
            """Compute the marginal novelty before adding a document."""
            return sum((1 - alpha) ** covered[i] for i, group in enumerate(groups) if doc_id in group)

        best = max(remaining, key=lambda doc_id: (gain(doc_id), doc_id))
        ranking.append(best)
        remaining.remove(best)
        for index, group in enumerate(groups):
            if best in group:
                covered[index] += 1
    return ranking


def evaluate_query(
    ranked_ids: list[str], query: Question | dict, ks: tuple[int, ...] = (5, 10, 20, 50), alpha: float = 0.5,
) -> dict[str, float]:
    """Evaluate retrieval independently of generators, answers and online providers.

    Args:
        ranked_ids: Ordered corpus IDs. Duplicates occupy slots but add no relevance or novelty.
        query: Offline Question or dictionary with nested nugget relevance lists/maps.
        ks: Positive integer cutoffs. Short runs are padded with nonrelevant slots for precision.
        alpha: Novelty decay parameter in [0,1]; FreshStack uses 0.5.

    Returns:
        Recall against the union of binary-positive qrels; official FreshStack coverage against
        all nuggets, including unjudged ones; binary precision/nDCG; greedy alpha-nDCG; full-run MRR.
        Empty or wholly unjudged queries return zero for all metrics, never NaN. The alpha ideal
        follows ndeval.c's greedy/descending-ID convention; no executable parity is claimed here.
        Cutoffs above 20 extend the same formula beyond the official evaluator's supported depth.

    Raises:
        ValueError: Cutoffs or alpha are invalid.
    """
    if any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in ks):
        raise ValueError("metric cutoffs must be positive integers")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    nuggets = query.nuggets if isinstance(query, Question) else query.get("nuggets", [])
    groups = [relevant_doc_ids(nugget) for nugget in nuggets]
    relevant = set().union(*groups) if groups else set()
    ideal = _alpha_ideal(groups, alpha, max(ks, default=0))
    results = {"mrr": next((1 / rank for rank, doc_id in enumerate(ranked_ids, 1) if doc_id in relevant), 0.0)}
    for k in ks:
        seen: set[str] = set()
        dcg = 0.0
        for rank, doc_id in enumerate(ranked_ids[:k], 1):
            if doc_id in relevant and doc_id not in seen:
                dcg += 1 / math.log2(rank + 1)
            seen.add(doc_id)
        found = seen & relevant
        idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
        alpha_idcg = _alpha_dcg(ideal, groups, alpha, k)
        results.update({
            f"recall@{k}": len(found) / len(relevant) if relevant else 0.0,
            f"coverage@{k}": sum(bool(seen & group) for group in groups) / len(groups) if groups else 0.0,
            f"precision@{k}": len(found) / k,
            f"ndcg@{k}": dcg / idcg if idcg else 0.0,
            f"alpha_ndcg@{k}": _alpha_dcg(ranked_ids, groups, alpha, k) / alpha_idcg if alpha_idcg else 0.0,
        })
    return results


def _percentile(values: list[float], probability: float) -> float:
    """Interpolate a percentile of sorted bootstrap means."""
    position = (len(values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def bootstrap_paired(
    a: list[float], b: list[float], seed: int = 20261008, samples: int = 2000,
) -> dict:
    """Estimate mean b-a and a paired percentile-bootstrap 95% confidence interval.

    Args:
        a: Baseline per-query metric values in a fixed query order.
        b: Candidate metric values in the same query order.
        seed: Local resampling RNG seed.
        samples: Positive number of resampled paired means.

    Returns:
        mean_delta, ci_low, ci_high, n, seed and samples. Percentiles use linear interpolation.

    Raises:
        ValueError: Inputs are empty, unequal, nonfinite or samples is not a positive integer.
    """
    if not a or len(a) != len(b):
        raise ValueError("paired inputs must be nonempty and equal length")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if any(not math.isfinite(value) for value in a + b):
        raise ValueError("paired inputs must contain finite values")
    differences = [candidate - baseline for baseline, candidate in zip(a, b)]
    if any(not math.isfinite(value) for value in differences):
        raise ValueError("paired differences must be finite")
    n = len(differences)
    rng = random.Random(seed)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    return {"mean_delta": sum(differences) / n, "ci_low": _percentile(means, 0.025),
            "ci_high": _percentile(means, 0.975), "n": n, "seed": seed, "samples": samples}
