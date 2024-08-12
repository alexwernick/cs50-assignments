from cs50_assignments.search.degrees.degrees import shortest_path


def test_return_none_if_no_possible_path():
    assert shortest_path("102", "158") is None  # test data Kevin Bacon & Tom Hanks


def test_return_shortest_path_if_possible():
    assert shortest_path("102", "129") == [
        ("104257", "129")
    ]  # test data Kevin Bacon & Tom Cruise.
    # There are two possible paths but the smallest is length 1
