from pathlib import Path

from cs50_assignments.learning.shopping.shopping import load_data


def test_load_data():
    # We simply test the happy path here
    dir = Path(__file__).parent / "test_data/valid.csv"
    evidence, labels = load_data(dir)
    assert len(evidence) == 2
    assert len(labels) == 2
    assert evidence[0] == [
        12,
        2.5,
        3,
        4.5,
        5,
        6.5,
        7.5,
        8.5,
        9.5,
        0.5,
        0,
        1,
        2,
        3,
        4,
        1,
        0,
    ]
    assert labels[0] == 0
    assert evidence[1] == [
        13,
        3.5,
        4,
        4.8,
        9,
        6.9,
        8,
        9.3,
        10.8,
        1.5,
        6,
        2,
        3,
        4,
        5,
        0,
        1,
    ]
    assert labels[1] == 1
