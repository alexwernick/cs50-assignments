from cs50_assignments.learning.nim.nim import NimAI


def test_get_q_value():
    ai = NimAI()
    ai.q[(1, 1, 2, 2), (1, 1)] = 1
    assert ai.get_q_value((1, 1, 4, 4), (1, 1)) == 0
    assert ai.get_q_value((1, 1, 2, 2), (1, 1)) == 1


def test_update_q_value():
    ai = NimAI(0.5, 0.1)

    ai.update_q_value((1, 1, 2, 2), (1, 1), 1, 1, 0)
    assert ai.q[(1, 1, 2, 2), (1, 1)] == 1.0

    ai.update_q_value((1, 1, 2, 2), (1, 1), 1.5, 1, 1)
    assert ai.q[(1, 1, 2, 2), (1, 1)] == 1.75

    ai.update_q_value((1, 1, 2, 2), (1, 1), 1.75, -1, -1)
    assert ai.q[(1, 1, 2, 2), (1, 1)] == -0.125


def test_best_future_reward():
    ai = NimAI(0.5, 0.1)

    ai.q[(1, 1, 2, 2), (1, 1)] = 1
    ai.q[(1, 1, 2, 2), (3, 2)] = 2
    ai.q[(1, 1, 2, 2), (2, 2)] = 3

    assert ai.best_future_reward((1, 1, 2, 2)) == 3

    ai.q[(1, 1, 2, 2), (2, 2)] = -1
    assert ai.best_future_reward((1, 1, 2, 2)) == 2

    assert ai.best_future_reward((1, 1, 2, 1)) == 0


def test_choose_action():
    ai = NimAI(0.5, 0.1)
    ai.q[(1, 1, 2, 2), (1, 1)] = 1
    ai.q[(1, 1, 2, 2), (3, 2)] = 2
    ai.q[(1, 1, 2, 2), (2, 2)] = 3

    assert ai.choose_action((1, 1, 2, 2), False) == (2, 2)
