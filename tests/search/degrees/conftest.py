import csv
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def patch_movie_data(monkeypatch):
    names, people, movies = load_data(Path(__file__).parent / "test_data")
    monkeypatch.setattr("cs50_assignments.search.degrees.degrees.names", names)
    monkeypatch.setattr("cs50_assignments.search.degrees.degrees.people", people)
    monkeypatch.setattr("cs50_assignments.search.degrees.degrees.movies", movies)


def load_data(directory):
    """
    Load data from CSV files into memory.
    """
    names = {}
    people = {}
    movies = {}

    # Load people
    with open(f"{directory}/people.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            people[row["id"]] = {
                "name": row["name"],
                "birth": row["birth"],
                "movies": set(),
            }
            if row["name"].lower() not in names:
                names[row["name"].lower()] = {row["id"]}
            else:
                names[row["name"].lower()].add(row["id"])

    # Load movies
    with open(f"{directory}/movies.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[row["id"]] = {
                "title": row["title"],
                "year": row["year"],
                "stars": set(),
            }

    # Load stars
    with open(f"{directory}/stars.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                people[row["person_id"]]["movies"].add(row["movie_id"])
                movies[row["movie_id"]]["stars"].add(row["person_id"])
            except KeyError:
                pass

    return names, people, movies
