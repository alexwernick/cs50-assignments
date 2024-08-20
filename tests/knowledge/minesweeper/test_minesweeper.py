from unittest.mock import patch

import pytest

from cs50_assignments.knowledge.minesweeper.minesweeper import MinesweeperAI, Sentence


def test_known_mines_when_count_matches_number_of_cells():
    cells = {(1, 2), (2, 4)}
    count = 2
    sentence = Sentence(cells, count)
    assert sentence.known_mines() == cells


def test_known_mines_when_count_less_than_number_of_cells():
    cells = {(1, 2), (2, 4)}
    count = 1
    sentence = Sentence(cells, count)
    assert sentence.known_mines() == set()  # returns empty set


@pytest.mark.parametrize(
    "count",
    [-1, 3],
)
def test_known_mines_when_count_invalid(count):
    cells = {(1, 2), (2, 4)}
    sentence = Sentence(cells, count)
    with pytest.raises(Exception):
        sentence.known_mines()


def test_known_safes_when_count_0():
    cells = {(1, 2), (2, 4)}
    count = 0
    sentence = Sentence(cells, count)
    assert sentence.known_safes() == cells


def test_known_safes_when_count_greater_than_zero():
    cells = {(1, 2), (2, 4)}
    count = 1
    sentence = Sentence(cells, count)
    assert sentence.known_safes() == set()  # returns empty set


@pytest.mark.parametrize(
    "count",
    [-1, 3],
)
def test_known_safes_when_count_invalid(count):
    cells = {(1, 2), (2, 4)}
    sentence = Sentence(cells, count)
    with pytest.raises(Exception):
        sentence.known_safes()


@pytest.mark.parametrize(
    "cell, cells, count, expected_cells, expected_count",
    [
        ((1, 2), {(1, 2), (2, 4)}, 1, {(2, 4)}, 0),
        ((1, 3), {(1, 2), (2, 4)}, 1, {(1, 2), (2, 4)}, 1),
    ],
)
def test_mark_mine(cell, cells, count, expected_cells, expected_count):
    cells = {(1, 2), (2, 4)}
    sentence = Sentence(cells, count)
    sentence.mark_mine(cell)
    assert sentence.count == expected_count
    assert sentence.cells == expected_cells


@pytest.mark.parametrize(
    "cell, cells, count, expected_cells, expected_count",
    [
        ((1, 2), {(1, 2), (2, 4)}, 1, {(2, 4)}, 1),
        ((1, 3), {(1, 2), (2, 4)}, 1, {(1, 2), (2, 4)}, 1),
    ],
)
def test_mark_safe(cell, cells, count, expected_cells, expected_count):
    cells = {(1, 2), (2, 4)}
    sentence = Sentence(cells, count)
    sentence.mark_safe(cell)
    assert sentence.count == expected_count
    assert sentence.cells == expected_cells


def test_add_knowledge_updates_moves_made():
    ai = MinesweeperAI()
    cell = (1, 2)
    count = 1
    ai.add_knowledge(cell, count)
    assert cell in ai.moves_made


def test_add_knowledge_marks_as_safe():
    ai = MinesweeperAI()
    cell = (1, 2)
    count = 1
    with patch.object(MinesweeperAI, "mark_safe") as mock_mark_safe:
        ai.add_knowledge(cell, count)
        mock_mark_safe.assert_called_once()


def test_add_knowledge_appends_sentence():
    ai = MinesweeperAI()
    cell = (0, 1)
    count = 1
    ai.moves_made.add((0, 0))  # add 0,0 here to make check it is excluded from sentence
    expected_sentence = Sentence({(1, 0), (1, 1), (1, 2), (0, 2)}, count)
    ai.add_knowledge(cell, count)
    expected_sentence in ai.knowledge


def test_add_knowledge_marks_any_known_safe():
    ai = MinesweeperAI()
    cell = (0, 0)
    count = 0
    implied_safe_set = {(0, 1), (1, 0), (1, 1)}
    ai.add_knowledge(cell, count)
    assert implied_safe_set.issubset(ai.safes)


def test_add_knowledge_marks_any_known_mines():
    ai = MinesweeperAI()
    cell = (0, 0)
    count = 3
    implied_safe_set = {(0, 1), (1, 0), (1, 1)}
    ai.add_knowledge(cell, count)
    assert implied_safe_set.issubset(ai.mines)


def test_add_knowledge_infers_new_sentences():
    ai = MinesweeperAI()
    sentence1 = Sentence({(1, 0), (1, 1), (1, 2), (2, 0), (2, 2)}, 2)
    sentence2 = Sentence({(1, 0), (1, 1), (1, 2)}, 1)
    inferred_sentence = Sentence({(2, 0), (2, 2)}, 1)

    cell = (7, 7)  # choosing a cell that won't affect the test case
    count = 0
    ai.knowledge.append(sentence1)
    ai.knowledge.append(sentence2)
    ai.add_knowledge(cell, count)
    assert inferred_sentence in ai.knowledge


def test_make_safe_move():
    safes = {(0, 1), (0, 0)}
    moves_made = {(0, 0)}
    possible_safe_moves = {(0, 1)}
    ai = MinesweeperAI()
    ai.safes = safes
    ai.moves_made = moves_made
    safe_move = ai.make_safe_move()

    if possible_safe_moves == set():
        assert safe_move is None
    else:
        assert safe_move in possible_safe_moves
