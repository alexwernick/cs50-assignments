import copy
import sys
from pathlib import Path

from cs50_assignments.optimization.crossword.crossword import Crossword, Variable


class CrosswordCreator:
    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy() for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont

        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size, self.crossword.height * cell_size),
            "black",
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                rect = [
                    (j * cell_size + cell_border, i * cell_size + cell_border),
                    (
                        (j + 1) * cell_size - cell_border,
                        (i + 1) * cell_size - cell_border,
                    ),
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (
                                rect[0][0] + ((interior_size - w) / 2),
                                rect[0][1] + ((interior_size - h) / 2) - 10,
                            ),
                            letters[i][j],
                            fill="black",
                            font=font,
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack({key: None for key in self.domains})

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for variable, words in self.domains.items():
            valid_words = {word for word in words if len(word) == variable.length}
            self.domains[variable] = valid_words

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        overlap = self.crossword.overlaps[x, y]
        if overlap is None:
            return False

        x_words = self.domains[x]
        y_words = self.domains[y]
        valid_words = {
            word
            for word in x_words
            if self._contains_word_with_letter(y_words, overlap[1], word[overlap[0]])
        }
        did_revise = len(valid_words) < len(self.domains[x])
        self.domains[x] = valid_words
        return did_revise

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            arcs = set()
            for variable in self.domains:
                for neighbor in self.crossword.neighbors(variable):
                    arcs.add((variable, neighbor))

        while len(arcs) > 0:
            arc = arcs.pop()
            if self.revise(*arc):
                if len(self.domains[arc[0]]) == 0:
                    return False
                for neighbor in self.crossword.neighbors(arc[0]):
                    arcs.add((arc[0], neighbor))

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for word in assignment.values():
            if word is None:
                return False

        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        # values not distinct
        non_none_values = [item for item in assignment.values() if item is not None]
        if len(non_none_values) != len(set(non_none_values)):
            return False

        for variable, word in assignment.items():
            if word is None:
                continue

            if len(word) != variable.length:
                return False

            for neighbor in self.crossword.neighbors(variable):
                if assignment[neighbor] is None:
                    continue

                overlap = self.crossword.overlaps[variable, neighbor]

                if word[overlap[0]] != assignment[neighbor][overlap[1]]:
                    return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        heuristic_values = []
        for value in self.domains[var]:
            words_ruled_out = 0
            for neighbor in self.crossword.neighbors(var):
                if assignment[neighbor] is not None:
                    continue

                overlap = self.crossword.overlaps[var, neighbor]
                for neighbor_value in self.domains[neighbor]:
                    if value[overlap[0]] != neighbor_value[overlap[1]]:
                        words_ruled_out += 1
            heuristic_values.append((value, words_ruled_out))

        sorted_values = sorted(heuristic_values, key=lambda heuristic: heuristic[1])
        return [value[0] for value in sorted_values]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_variables = [
            variable for variable in assignment if assignment[variable] is None
        ]
        variable_domain_sizes = [
            (variable, len(self.domains[variable])) for variable in unassigned_variables
        ]
        sorted_variable_domain_sizes = sorted(
            variable_domain_sizes, key=lambda num: num[1]
        )
        smallest_variable_domain_sizes = [
            variable_domain
            for variable_domain in sorted_variable_domain_sizes
            if variable_domain[1] == sorted_variable_domain_sizes[0][1]
        ]

        if len(smallest_variable_domain_sizes) == 1:
            return smallest_variable_domain_sizes[0][0]

        variable_degrees = [
            (variable[0], len(self.crossword.neighbors(variable[0])))
            for variable in smallest_variable_domain_sizes
        ]
        sorted_variable_degrees = sorted(
            variable_degrees, key=lambda x: x[1], reverse=True
        )
        return sorted_variable_degrees[0][0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        variable = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(variable, assignment):
            # make assignment
            deep_copy = copy.deepcopy(assignment)
            assignment[variable] = value
            if self.consistent(assignment):
                arcs = set()
                for neighbor in self.crossword.neighbors(variable):
                    arcs.add((neighbor, variable))

                if self.ac3(arcs) and self.backtrack(assignment):
                    return assignment

            assignment = deep_copy
        return None

    def _contains_word_with_letter(_, words, index, letter):
        for word in words:
            if len(word) > index and word[index] == letter:
                return True
        return False


def main():
    # Set defaults
    structure = Path(__file__).parent / "data/structure1.txt"
    words = Path(__file__).parent / "data/words1.txt"
    output = None

    # Parse command-line arguments
    if len(sys.argv) in [3, 4]:
        structure = Path(__file__).parent / sys.argv[1]
        words = Path(__file__).parent / sys.argv[2]
        output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
