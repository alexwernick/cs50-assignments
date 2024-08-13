from unittest.mock import patch

import pytest

from cs50_assignments.search.tictactoe.tictactoe import (
    actions,
    minimax,
    player,
    result,
    terminal,
    utility,
    validate,
    winner,
)


@pytest.mark.parametrize(
    "board, expected",
    [
        ([[None, None, None], [None, None, None], [None, None, None]], "X"),
        ([["X", None, None], [None, None, None], [None, None, None]], "O"),
        ([["X", None, None], [None, "O", None], [None, None, None]], "X"),
        ([["X", "O", "X"], ["X", "O", "O"], ["O", "X", "X"]], None),
    ],
)
def test_player(board, expected):
    with patch("cs50_assignments.search.tictactoe.tictactoe.validate") as mock_validate:
        assert player(board) == expected
        mock_validate.assert_called_once()


@pytest.mark.parametrize(
    "board, expected",
    [
        (
            [[None, None, None], [None, None, None], [None, None, None]],
            {
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
            },
        ),
        (
            [["X", None, None], [None, None, None], [None, None, None]],
            {
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
            },
        ),
        (
            [["X", None, None], [None, "O", None], [None, None, None]],
            {
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
            },
        ),
        ([["X", "O", "X"], ["X", "O", "O"], ["O", "X", "X"]], set()),
    ],
)
def test_actions(board, expected):
    with patch("cs50_assignments.search.tictactoe.tictactoe.validate") as mock_validate:
        assert actions(board) == expected
        mock_validate.assert_called_once()


@pytest.mark.parametrize(
    "board, action, expected",
    [
        (
            [[None, None, None], [None, None, None], [None, None, None]],
            (0, 0),
            [["X", None, None], [None, None, None], [None, None, None]],
        ),
        (
            [["X", None, None], [None, None, None], [None, None, None]],
            (1, 1),
            [["X", None, None], [None, "O", None], [None, None, None]],
        ),
        (
            [["X", None, None], [None, "O", None], [None, None, None]],
            (1, 0),
            [["X", None, None], ["X", "O", None], [None, None, None]],
        ),
        ([["X", None, None], [None, "O", None], [None, None, None]], (1, 1), Exception),
    ],
)
def test_result(board, action, expected):
    with patch("cs50_assignments.search.tictactoe.tictactoe.validate") as mock_validate:
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                result(board, action)
        else:
            assert result(board, action) == expected
            assert (
                expected != board
            )  # making sure that we have not modified the board that was input
        mock_validate.assert_called()


@pytest.mark.parametrize(
    "board, expected",
    [
        ([[None, None, None], [None, None, None], [None, None, None]], None),
        ([["X", None, None], [None, None, None], [None, None, None]], None),
        ([["X", None, None], [None, "O", None], [None, None, None]], None),
        ([["X", "O", "X"], ["X", "O", "O"], ["O", "X", "X"]], None),
        ([["X", "O", "X"], ["X", "O", "O"], ["X", "X", "O"]], "X"),
        ([["X", "X", None], ["O", "O", "O"], ["O", "X", "X"]], "O"),
    ],
)
def test_winner(board, expected):
    with patch("cs50_assignments.search.tictactoe.tictactoe.validate") as mock_validate:
        assert winner(board) == expected
        mock_validate.assert_called()


@pytest.mark.parametrize(
    "board, winner_response, expected",
    [
        ([[None, None, None], [None, None, None], [None, None, None]], "X", True),
        ([[None, None, None], [None, None, None], [None, None, None]], "O", True),
        ([[None, None, None], [None, None, None], [None, None, None]], None, False),
        ([["X", None, None], [None, "O", None], [None, None, None]], None, False),
        ([["X", "O", "X"], ["X", "O", "O"], ["O", "X", "X"]], None, True),
    ],
)
def test_terminal(board, winner_response, expected):
    with patch(
        "cs50_assignments.search.tictactoe.tictactoe.winner",
        return_value=winner_response,
    ) as mock_func:
        assert terminal(board) == expected
        mock_func.assert_called()


@pytest.mark.parametrize(
    "winner_response, expected",
    [
        ("X", 1),
        ("O", -1),
        (None, 0),
    ],
)
def test_utility(winner_response, expected):
    with patch(
        "cs50_assignments.search.tictactoe.tictactoe.winner",
        return_value=winner_response,
    ) as mock_winner, patch(
        "cs50_assignments.search.tictactoe.tictactoe.terminal", return_value=True
    ) as mock_terminal, patch(
        "cs50_assignments.search.tictactoe.tictactoe.validate", return_value=True
    ):
        assert utility(None) == expected
        mock_winner.assert_called_once()
        mock_terminal.assert_called_once()
        mock_terminal.assert_called()


@pytest.mark.parametrize(
    "board, expected",
    [
        ([["X", None, None], [None, None, None], [None, None, None]], [(1, 1)]),
        ([["X", None, None], [None, "O", None], [None, None, None]], [(0, 1), (1, 0)]),
        ([["X", "O", "X"], ["X", "O", "O"], ["O", "X", "X"]], [None]),
        ([["X", "O", "X"], ["X", "O", "O"], [None, "X", "O"]], [(2, 0)]),
    ],
)
def test_minimax(board, expected):
    assert minimax(board) in expected


@pytest.mark.parametrize(
    "board, expected",
    [
        ([[None, None, None], [None, None, None], [None, None, None]], False),
        ([["X", None, None], [None, None, None], [None, None, None]], False),
        ([["X", None, None], [None, "O", None], [None, None, None]], False),
        ([["X", "O", "X"], ["X", "O", "O"], ["O", "X", "X"]], False),
        ([["X", "X", "X"], ["X", "X", "O"], ["O", "X", None]], True),
        (
            [["X", None, "something-invalid"], [None, None, None], [None, None, None]],
            True,
        ),
        (
            [
                [None, None, None],
                [None, None, None],
                [None, None, None],
                [None, None, None],
            ],
            True,
        ),
        ([[None, None, None], [None, None, None]], True),
    ],
)
def test_validate(board, expected):
    exeption_thrown = False
    try:
        validate(board)
    except Exception:
        exeption_thrown = True

    assert exeption_thrown == expected
