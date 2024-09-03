from cs50_assignments.optimization.crossword.crossword import Variable


def test_enforce_node_consistency(crossword_creator):
    crossword_creator.enforce_node_consistency()

    for variable, words in crossword_creator.domains.items():
        for word in words:
            assert len(word) == variable.length


def test_revise(crossword_creator):
    crossword_creator.enforce_node_consistency()

    x = get_variable(crossword_creator.domains, 0, 1, Variable.ACROSS)
    y = get_variable(crossword_creator.domains, 0, 1, Variable.DOWN)
    assert crossword_creator.revise(x, y)
    assert crossword_creator.domains[x] == {"TWO", "TEN", "SIX"}

    x = get_variable(crossword_creator.domains, 0, 1, Variable.DOWN)
    y = get_variable(crossword_creator.domains, 0, 1, Variable.ACROSS)
    assert crossword_creator.revise(x, y)
    assert crossword_creator.domains[x] == {"THREE", "SEVEN"}

    x = get_variable(crossword_creator.domains, 0, 1, Variable.DOWN)
    y = get_variable(crossword_creator.domains, 4, 1, Variable.ACROSS)
    assert crossword_creator.revise(x, y)
    assert crossword_creator.domains[x] == {"SEVEN"}

    x = get_variable(crossword_creator.domains, 0, 1, Variable.ACROSS)
    y = get_variable(crossword_creator.domains, 0, 1, Variable.DOWN)
    assert crossword_creator.revise(x, y)
    assert crossword_creator.domains[x] == {"SIX"}

    x = get_variable(crossword_creator.domains, 4, 1, Variable.ACROSS)
    y = get_variable(crossword_creator.domains, 0, 1, Variable.DOWN)
    assert crossword_creator.revise(x, y)
    assert crossword_creator.domains[x] == {"NINE"}

    x = get_variable(crossword_creator.domains, 1, 4, Variable.DOWN)
    y = get_variable(crossword_creator.domains, 4, 1, Variable.ACROSS)
    assert crossword_creator.revise(x, y)
    assert crossword_creator.domains[x] == {"FIVE", "NINE"}


def test_ac3(crossword_creator):
    crossword_creator.enforce_node_consistency()
    crossword_creator.ac3()

    assert crossword_creator.domains[
        get_variable(crossword_creator.domains, 0, 1, Variable.DOWN)
    ] == {"SEVEN"}
    assert crossword_creator.domains[
        get_variable(crossword_creator.domains, 0, 1, Variable.ACROSS)
    ] == {"SIX"}
    assert crossword_creator.domains[
        get_variable(crossword_creator.domains, 4, 1, Variable.ACROSS)
    ] == {"NINE"}
    assert crossword_creator.domains[
        get_variable(crossword_creator.domains, 1, 4, Variable.DOWN)
    ] == {"FIVE", "NINE"}


def get_variable(domains, i, j, direction):
    for variable in domains:
        if variable.i == i and variable.j == j and variable.direction == direction:
            return variable
    return None
