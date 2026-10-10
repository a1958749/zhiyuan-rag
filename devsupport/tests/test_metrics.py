"""Independent retrieval metric expectations, all artificial."""

import math

import pytest

from devsupport.metrics import bootstrap_paired, evaluate_query
from devsupport.types import Question


def query(*groups: list[str]) -> Question:
    """Construct an artificial offline query from relevance groups."""
    return Question("artificial", "title", "body", "offline answer", [
        {"relevant_corpus_ids": group} for group in groups
    ], {})


def test_recall_is_fraction_of_all_relevant_documents_not_hit_rate() -> None:
    result = evaluate_query(["a", "x"], query(["a", "b"], ["c"]), ks=(1, 2, 5))
    assert result["recall@1"] == pytest.approx(1 / 3)
    assert result["coverage@1"] == 0.5
    assert result["precision@2"] == 0.5
    assert result["precision@5"] == 0.2
    assert result["mrr"] == 1
    assert result["ndcg@2"] == pytest.approx(1 / (1 + 1 / math.log2(3)))


def test_overlapping_relevance_counts_doc_once_and_nuggets_independently() -> None:
    result = evaluate_query(["a"], query(["a", "b"], ["a", "c"]), ks=(1,))
    assert result["recall@1"] == pytest.approx(1 / 3)
    assert result["coverage@1"] == 1
    assert result["alpha_ndcg@1"] == 1


def test_duplicate_result_occupies_rank_but_adds_no_gain() -> None:
    result = evaluate_query(["a", "a", "c"], query(["a", "b"], ["c"]), ks=(2, 3))
    assert result["recall@2"] == pytest.approx(1 / 3)
    assert result["precision@2"] == 0.5
    assert result["coverage@2"] == 0.5
    assert result["recall@3"] == pytest.approx(2 / 3)
    assert evaluate_query(["x", "x", "a"], query(["a"]), ks=(3,))["mrr"] == pytest.approx(1 / 3)


def test_alpha_novelty_rewards_new_nugget_over_repeat() -> None:
    q = query(["a", "b"], ["c"])
    repeated = evaluate_query(["a", "b"], q, ks=(2,))["alpha_ndcg@2"]
    diverse = evaluate_query(["a", "c"], q, ks=(2,))["alpha_ndcg@2"]
    assert repeated == pytest.approx((1 + 0.5 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert diverse == 1
    assert evaluate_query(["a", "b"], q, ks=(2,), alpha=0)["alpha_ndcg@2"] == 1


def test_alpha_ideal_tie_uses_descending_doc_id_as_ndeval_does() -> None:
    # First all have two subtopics: z wins; its successor covers one fresh subtopic.
    q = query(["z", "a"], ["z", "b"], ["a", "b"])
    result = evaluate_query(["z", "b", "a"], q, ks=(3,))
    assert result["alpha_ndcg@3"] == 1


def test_binary_qrels_ignore_grade_magnitude_and_nonpositive_grades() -> None:
    q = {"nuggets": [{"relevant_corpus_ids": {"a": 3, "b": 0, "c": -1}},
                     {"relevant_corpus_ids": ["d"]}]}
    assert evaluate_query(["b", "a"], q, ks=(2,))["recall@2"] == 0.5
    assert evaluate_query(["b", "a"], q, ks=(2,))["mrr"] == 0.5


def test_empty_nuggets_remain_in_official_coverage_denominator_and_unjudged_returns_zero() -> None:
    assert evaluate_query(["a"], query([], ["a"]), ks=(1,))["coverage@1"] == 0.5
    result = evaluate_query(["anything"], query([]), ks=(1, 5))
    assert all(value == 0 for value in result.values())
    assert all(value == 0 for value in evaluate_query([], query(["a"])).values())


@pytest.mark.parametrize("kwargs", [{"ks": (0,)}, {"ks": (-1,)}, {"alpha": 1.1}, {"alpha": -0.1}])
def test_invalid_metric_parameters_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        evaluate_query([], query(["a"]), **kwargs)


def test_paired_bootstrap_reports_b_minus_a_and_preserves_pairing() -> None:
    a = [0, 100, -50]
    b = [1, 101, -49]
    result = bootstrap_paired(a, b, samples=100)
    assert result["mean_delta"] == 1
    assert result["ci_low"] == result["ci_high"] == 1
    assert result["n"] == 3
    assert result == bootstrap_paired(a, b, samples=100)
    assert bootstrap_paired([0, 0, 0], [-1, 0, 1], samples=100) == bootstrap_paired(
        [0, 0, 0], [-1, 0, 1], samples=100,
    )


@pytest.mark.parametrize("a,b,kwargs", [([], [], {}), ([1], [], {}), ([1], [2], {"samples": 0}),
                                       ([float("nan")], [1], {})])
def test_invalid_bootstrap_inputs_raise(a: list[float], b: list[float], kwargs: dict) -> None:
    with pytest.raises(ValueError):
        bootstrap_paired(a, b, **kwargs)
