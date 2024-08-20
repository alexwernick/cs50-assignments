import random


class Minesweeper:
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):
        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence:
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count > len(self.cells) or self.count < 0:
            raise Exception(
                f"The count of the sentence is invalid count: {self.count}, "
                f"length cells: {len(self.cells)}"
            )

        if len(self.cells) == self.count:
            return self.cells

        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count > len(self.cells) or self.count < 0:
            raise Exception(
                f"The count of the sentence is invalid count: {self.count}, "
                f"length cells: {len(self.cells)}"
            )

        if self.count == 0:
            return self.cells

        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell not in self.cells:
            return

        self.count -= 1
        self.cells.remove(cell)

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell not in self.cells:
            return

        self.cells.remove(cell)


class MinesweeperAI:
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):
        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # valdidate

        self.moves_made.add(cell)
        self.mark_safe(cell)

        neighboring_cells = self._get_neighboring_cells(cell)
        neighboring_cells_without_known_mines = neighboring_cells - self.mines
        count -= len(neighboring_cells) - len(neighboring_cells_without_known_mines)
        neighboring_cells_without_known_mines_and_safes = (
            neighboring_cells_without_known_mines - self.safes
        )
        sentence = Sentence(neighboring_cells_without_known_mines_and_safes, count)
        self.knowledge.append(sentence)

        keep_recursing = True

        while keep_recursing:
            keep_recursing = (
                self._recursively_update_known_mines_and_safes()
                or self._recursively_infer_new_sentences_from_knowledge()
            )

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        safe_moves = self.safes - self.moves_made

        if len(safe_moves) == 0:
            return None

        return next(iter(safe_moves))  # arbitrary safe move

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        grid_set = {(i, j) for i in range(self.height) for j in range(self.width)}
        random_moves = (grid_set - self.moves_made) - self.mines

        if len(random_moves) == 0:
            return None

        return next(iter(random_moves))  # arbitrary safe move

    def _get_neighboring_cells(self, cell):
        """
        Returns neighboring cells, only considering cells
        who's state is not yet determined
        """
        cells = set()

        for i in range(max(cell[0] - 1, 0), min(cell[0] + 2, self.height)):
            for j in range(max(cell[1] - 1, 0), min(cell[1] + 2, self.width)):
                neighbor_cell = (i, j)
                if neighbor_cell not in self.moves_made and neighbor_cell != cell:
                    cells.add(neighbor_cell)

        return cells

    def _recursively_update_known_mines_and_safes(self):
        """
        Recursively goes through knowledge and marks
        safes and mines accordingly
        """
        found_new_information = False
        while True:
            infered_safes = set()
            infered_mines = set()
            for sentence in self.knowledge:
                infered_safes.update(sentence.known_safes() - self.safes)
                infered_mines.update(sentence.known_mines() - self.mines)

            if len(infered_safes) == 0 and len(infered_mines) == 0:
                return found_new_information

            found_new_information = True

            for inferred_safe in infered_safes:
                self.mark_safe(inferred_safe)

            for inferred_mine in infered_mines:
                self.mark_mine(inferred_mine)

    def _recursively_infer_new_sentences_from_knowledge(self):
        """
        Recursively goes through knowledge and
        generates new knowledge
        """
        found_new_information = False

        while True:
            infered_sentences = []
            for sentence1 in self.knowledge:
                for sentence2 in self.knowledge:
                    if sentence1 == sentence2:
                        continue

                    inferred_sentence = self._infer_new_sentences_from_two_sentences(
                        sentence1, sentence2
                    )
                    if inferred_sentence is not None:
                        infered_sentences.append(inferred_sentence)
                        continue

                    inferred_sentence = self._infer_new_sentences_from_two_sentences(
                        sentence2, sentence1
                    )
                    if inferred_sentence is not None:
                        infered_sentences.append(inferred_sentence)
                        continue

            if len(infered_sentences) == 0:
                return found_new_information

            found_new_information = True
            for inferred_sentence in infered_sentences:
                self.knowledge.append(inferred_sentence)

    def _infer_new_sentences_from_two_sentences(self, sentence1, sentence2):
        if sentence1.cells.issubset(sentence2.cells):
            inferred_sentence = Sentence(
                sentence2.cells - sentence1.cells, sentence2.count - sentence1.count
            )
            if inferred_sentence not in self.knowledge:
                return inferred_sentence
        return None