"""
Tic Tac Toe Player
"""

import copy

LETTER_X = "X"
LETTER_O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    validate(board)
    number_of_x = get_number_of_elements(board, "X")
    number_of_o = get_number_of_elements(board, "O")

    # We know this is true as we know we have a valid board
    if number_of_x == 5 and number_of_o == 4:
        return None

    if number_of_x > number_of_o:
        return "O"

    return "X"


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    validate(board)

    # as we have a vlaid board we can simply get the co-ordinates of the EMPTY cells
    return {
        (i, j)
        for i, row in enumerate(board)
        for j, element in enumerate(row)
        if element is EMPTY
    }


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    validate(board)
    i = action[0]
    j = action[1]

    if i < 0 or i > 2 or j < 0 or j > 2:
        raise Exception("Action is not valid. Indicies must be between 0 and 2")

    if board[i][j] is not EMPTY:
        raise Exception(
            f"Can not make an action on postition ({i}, {j}) as it is not empty"
        )

    copied_board = copy.deepcopy(board)
    copied_board[i][j] = player(board)
    return copied_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    validate(board)

    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not None:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] is not None:
            return board[0][i]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        return board[0][2]

    return None


def terminal(board):
    validate(board)
    if winner(board) is not None:
        return True

    return get_number_of_elements(board, EMPTY) == 0


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    validate(board)
    if terminal(board) is False:
        raise Exception(
            "Can not get the utility of a board that is not in a terminal state"
        )

    win = winner(board)

    if win == LETTER_X:
        return 1
    elif win == LETTER_O:
        return -1
    elif win is None:
        return 0
    else:
        raise Exception(f"Can not handle winner value of {win}")


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    validate(board)

    possible_actions = actions(board)

    if terminal(board):
        return None

    current_player = player(board)
    max_min = max if current_player == LETTER_X else min

    current_best_value = None
    current_best_action = None
    for possible_action in possible_actions:
        action_value = minimax_value_with_pruning(
            result(board, possible_action), current_best_value
        )

        if current_best_value is None or action_value == max_min(
            action_value, current_best_value
        ):
            current_best_value = action_value
            current_best_action = possible_action

    return current_best_action


def minimax_value_with_pruning(board, current_best):
    """
    Returns the optimal value of a given board recursively
    """
    validate(board)

    if terminal(board):
        return utility(board)

    possible_actions = actions(board)
    current_player = player(board)
    max_min = max
    max_min = max if current_player == LETTER_X else min

    action_best = None
    for possible_action in possible_actions:
        board_after_action = result(board, possible_action)
        action_value = minimax_value_with_pruning(board_after_action, action_best)

        if (
            current_best is not None
            and action_value == max_min(action_value, current_best)
        ) and action_value != current_best:
            return action_value  # we can bomb out here to improve efficiency

        action_best = (
            action_value if action_best is None else max_min(action_best, action_value)
        )

    return action_best


def validate(board):
    valid_elements = {LETTER_X, LETTER_O, EMPTY}
    if len(board) != 3 or all(len(row) == 3 for row in board) is not True:
        raise Exception("Board is not a 3X3 matrix")

    if all(element in valid_elements for row in board for element in row) is False:
        raise Exception("Board contains invalid elements")

    number_of_x = get_number_of_elements(board, "X")
    number_of_o = get_number_of_elements(board, "O")

    if number_of_o > number_of_x or number_of_x > number_of_o + 1:
        raise Exception("Board contains invalid number of Xs and Os")


def get_number_of_elements(board, char):
    return sum(element == char for row in board for element in row)
