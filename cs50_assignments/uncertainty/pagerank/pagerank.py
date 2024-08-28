import os
import random
import re
import sys
from pathlib import Path

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")

    dir = Path(__file__).parent / sys.argv[1]
    corpus = crawl(dir)
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")  # noqa: E231
    ranks = iterate_pagerank(corpus, DAMPING)
    print("PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")  # noqa: E231


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # will code without validating here to keep things simple but
    # in a production program we should code more defensively

    linked_pages = corpus[page]
    if len(linked_pages) == 0:
        return dict.fromkeys(corpus.keys(), 1 / len(corpus))

    distribution = dict.fromkeys(corpus.keys(), (1 - damping_factor) / len(corpus))

    for linked_page in linked_pages:
        distribution[linked_page] += damping_factor / len(linked_pages)

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    estimated_rank = dict.fromkeys(corpus.keys(), 0)
    current_page = random.choice(list(corpus.keys()))
    estimated_rank[current_page] += 1

    for _ in range(n - 1):
        distribution = transition_model(corpus, current_page, damping_factor)
        choices = list(distribution.keys())
        probabilities = list(distribution.values())
        selected_page = random.choices(choices, weights=probabilities, k=1)[0]
        estimated_rank[selected_page] += 1
        current_page = selected_page

    # normalise by dividing by n
    return {key: value / n for key, value in estimated_rank.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # make correction if there is a page with no links
    # we need to make page link to all pages including itself
    for linking_page, linking_page_pages in corpus.items():
        if len(linking_page_pages) == 0:
            corpus[linking_page] = corpus.keys()

    tolerance = 0.001
    n = len(corpus)
    p_of_random_selection = (1 - damping_factor) / n
    estimated_distibution = dict.fromkeys(corpus.keys(), 1 / len(corpus))
    tolerance_condition_satisfied = False
    while tolerance_condition_satisfied is False:
        new_estimated_distibution = {}
        tolerance_condition_satisfied = True
        for key in estimated_distibution:
            sum_of_pages_that_link = 0
            for linking_page, linking_page_pages in corpus.items():
                if key in linking_page_pages:
                    sum_of_pages_that_link += estimated_distibution[linking_page] / len(
                        linking_page_pages
                    )

            new_rank_estimate = (
                p_of_random_selection + damping_factor * sum_of_pages_that_link
            )
            value_change_within_tolerance = (
                abs(new_rank_estimate - estimated_distibution[key]) < tolerance
            )
            tolerance_condition_satisfied = (
                tolerance_condition_satisfied and value_change_within_tolerance
            )
            new_estimated_distibution[key] = new_rank_estimate

        estimated_distibution = new_estimated_distibution

    return estimated_distibution


if __name__ == "__main__":
    main()
