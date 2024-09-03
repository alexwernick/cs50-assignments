from pathlib import Path

import pytest

from cs50_assignments.optimization.crossword.crossword import Crossword
from cs50_assignments.optimization.crossword.generate import CrosswordCreator


@pytest.fixture
def crossword_creator():
    structure_file = Path(__file__).parent / "test_data/structure0.txt"
    words_file = Path(__file__).parent / "test_data/words0.txt"
    return CrosswordCreator(Crossword(structure_file, words_file))
