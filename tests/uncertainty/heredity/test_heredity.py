import pytest

from cs50_assignments.uncertainty.heredity.heredity import (
    joint_probability,
    normalize,
    update,
)


def test_joint_probability_returns_correct_probabilities():
    people = {
        "Harry": {"name": "Harry", "mother": "Lily", "father": "James", "trait": None},
        "James": {"name": "James", "mother": None, "father": None, "trait": True},
        "Lily": {"name": "Lily", "mother": None, "father": None, "trait": False},
    }

    one_gene = {"Harry"}
    two_genes = {"James"}
    have_trait = {"James"}

    expected = 0.9504 * 0.0065 * 0.431288
    assert joint_probability(people, one_gene, two_genes, have_trait) == pytest.approx(
        expected, rel=1e-6
    )


def test_update_returns_correct_probabilities():
    people = {"Harry", "James", "Lily"}
    probabilities = {
        person: {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
        for person in people
    }

    one_gene = {"Harry"}
    two_genes = {"James"}
    have_trait = {"James"}
    p = 0.9504 * 0.0065 * 0.431288

    update(probabilities, one_gene, two_genes, have_trait, p)
    assert probabilities["Harry"]["gene"][1] == p
    assert probabilities["Harry"]["trait"][False] == p
    assert probabilities["James"]["gene"][2] == p
    assert probabilities["James"]["trait"][True] == p
    assert probabilities["Lily"]["gene"][0] == p
    assert probabilities["Lily"]["trait"][False] == p


def test_normalize_returns_correct_probabilities():
    probabilities = {
        "Harry": {
            "gene": {2: 0.75, 1: 0.55, 0: 0.33},
            "trait": {True: 0.9, False: 0.8},
        },
        "James": {"gene": {2: 0.3, 1: 0, 0: 0.1}, "trait": {True: 0.1, False: 0.2}},
    }

    normalize(probabilities)

    assert probabilities["Harry"]["gene"][2] == pytest.approx(0.75 / 1.63, rel=1e-6)
    assert probabilities["Harry"]["gene"][1] == pytest.approx(0.55 / 1.63, rel=1e-6)
    assert probabilities["Harry"]["gene"][0] == pytest.approx(0.33 / 1.63, rel=1e-6)

    assert probabilities["Harry"]["trait"][True] == pytest.approx(0.9 / 1.7, rel=1e-6)
    assert probabilities["Harry"]["trait"][False] == pytest.approx(0.8 / 1.7, rel=1e-6)

    assert probabilities["James"]["gene"][2] == pytest.approx(0.3 / 0.4, rel=1e-6)
    assert probabilities["James"]["gene"][1] == pytest.approx(0, rel=1e-6)
    assert probabilities["James"]["gene"][0] == pytest.approx(0.1 / 0.4, rel=1e-6)

    assert probabilities["James"]["trait"][True] == pytest.approx(0.1 / 0.3, rel=1e-6)
    assert probabilities["James"]["trait"][False] == pytest.approx(0.2 / 0.3, rel=1e-6)
