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
