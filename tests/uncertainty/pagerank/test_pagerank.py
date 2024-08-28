import pytest

from cs50_assignments.uncertainty.pagerank.pagerank import (
    iterate_pagerank,
    sample_pagerank,
    transition_model,
)


def test_transition_model_returns_all_keys():
    corpus = {"page1": set(), "page2": set(), "page3": set()}

    assert set(corpus.keys()) == set(transition_model(corpus, "page1", 0.5).keys())


@pytest.mark.parametrize(
    "corpus, page, damping_factor, expected_result",
    [
        (
            {
                "1.html": {"2.html", "3.html"},
                "2.html": {"3.html"},
                "3.html": {"2.html"},
            },
            "1.html",
            0.85,
            {"1.html": 0.05, "2.html": 0.475, "3.html": 0.475},
        ),
        (
            {
                "1.html": {"3.html", "4.html"},
                "2.html": {"3.html"},
                "3.html": {"2.html"},
                "4.html": {"2.html"},
            },
            "1.html",
            0.85,
            {"1.html": 0.0375, "2.html": 0.0375, "3.html": 0.4625, "4.html": 0.4625},
        ),
        (
            {
                "1.html": {"3.html", "4.html"},
                "2.html": {"3.html"},
                "3.html": {"2.html"},
                "4.html": {},
            },
            "4.html",
            0.85,
            {"1.html": 0.25, "2.html": 0.25, "3.html": 0.25, "4.html": 0.25},
        ),
    ],
)
def test_transition_model_returns_correct_probabilities(
    corpus, page, damping_factor, expected_result
):
    result = transition_model(corpus, page, damping_factor)
    for key in expected_result:
        assert result[key] == pytest.approx(expected_result[key], rel=1e-6)


def test_sample_pagerank_probabilities_sum_to_1():
    corpus = {
        "1.html": {"2.html", "3.html"},
        "2.html": {"3.html"},
        "3.html": {"2.html"},
    }
    damping_factor = 0.85
    n = 100
    result = sample_pagerank(corpus, damping_factor, n)

    assert set(corpus.keys()) == set(result.keys())
    assert sum(result.values()) == pytest.approx(1, rel=1e-6)


def test_iterate_pagerank_probabilities_sum_to_1():
    corpus = {
        "1.html": {"2.html", "3.html"},
        "2.html": {"3.html"},
        "3.html": {"2.html"},
    }
    damping_factor = 0.85
    result = iterate_pagerank(corpus, damping_factor)

    assert set(corpus.keys()) == set(result.keys())
    assert sum(result.values()) == pytest.approx(1, rel=1e-6)
