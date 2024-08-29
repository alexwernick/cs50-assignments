import csv
import itertools
import sys
from pathlib import Path

PROBS = {
    # Unconditional probabilities for having gene
    "gene": {2: 0.01, 1: 0.03, 0: 0.96},
    "trait": {
        # Probability of trait given two copies of gene
        2: {True: 0.65, False: 0.35},
        # Probability of trait given one copy of gene
        1: {True: 0.56, False: 0.44},
        # Probability of trait given no gene
        0: {True: 0.01, False: 0.99},
    },
    # Mutation probability
    "mutation": 0.01,
}


def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")

    dir = Path(__file__).parent / sys.argv[1]
    people = load_data(dir)

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):
        # Check if current set of people violates known information
        fails_evidence = any(
            (
                people[person]["trait"] is not None
                and people[person]["trait"] != (person in have_trait)
            )
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")  # noqa: E231
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")  # noqa: E231
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")  # noqa: E231


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (
                    True
                    if row["trait"] == "1"
                    else False
                    if row["trait"] == "0"
                    else None
                ),
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s)
        for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    # will code without validating here to keep things simple but
    # in a production program we should code more defensively
    joint_probability = 1.0

    for person in people:
        number_of_genes = _resolve_number_of_genes(person, one_gene, two_genes)
        is_trait = person in have_trait
        # we only consider in this code the case of both parents
        # we do not consider where one parent is known
        mum = people[person]["mother"]
        dad = people[person]["father"]
        has_parents = mum is not None and dad is not None
        prob_number_of_genes = None

        if has_parents:
            prob_number_of_genes = _prob_number_of_genes_given_parents(
                dad, mum, one_gene, two_genes, number_of_genes
            )
        else:
            prob_number_of_genes = PROBS["gene"][number_of_genes]

        joint_probability *= (
            prob_number_of_genes * PROBS["trait"][number_of_genes][is_trait]
        )

    return joint_probability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        number_of_genes = _resolve_number_of_genes(person, one_gene, two_genes)
        is_trait = person in have_trait
        probabilities[person]["gene"][number_of_genes] += p
        probabilities[person]["trait"][is_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        sum_of_gene_ps = (
            probabilities[person]["gene"][0]
            + probabilities[person]["gene"][1]
            + probabilities[person]["gene"][2]
        )
        sum_of_trait_ps = (
            probabilities[person]["trait"][True] + probabilities[person]["trait"][False]
        )

        probabilities[person]["gene"][0] = (
            probabilities[person]["gene"][0] / sum_of_gene_ps
        )
        probabilities[person]["gene"][1] = (
            probabilities[person]["gene"][1] / sum_of_gene_ps
        )
        probabilities[person]["gene"][2] = (
            probabilities[person]["gene"][2] / sum_of_gene_ps
        )

        probabilities[person]["trait"][True] = (
            probabilities[person]["trait"][True] / sum_of_trait_ps
        )
        probabilities[person]["trait"][False] = (
            probabilities[person]["trait"][False] / sum_of_trait_ps
        )


def _p_of_gene_given_parent_num(num_genes_in_parent):
    if num_genes_in_parent == 2:
        return 1 - PROBS["mutation"]
    elif num_genes_in_parent == 1:
        return 0.5 * (1 - PROBS["mutation"]) + 0.5 * PROBS["mutation"]
    else:
        return PROBS["mutation"]


def _resolve_number_of_genes(person, one_gene, two_genes):
    is_one = person in one_gene
    is_two = person in two_genes
    return 1 if is_one else (2 if is_two else 0)


def _prob_number_of_genes_given_parents(dad, mum, one_gene, two_genes, number_of_genes):
    number_of_genes_in_dad = _resolve_number_of_genes(dad, one_gene, two_genes)
    p_gene_from_dad = _p_of_gene_given_parent_num(number_of_genes_in_dad)
    number_of_genes_in_mum = _resolve_number_of_genes(mum, one_gene, two_genes)
    p_gene_from_mum = _p_of_gene_given_parent_num(number_of_genes_in_mum)

    if number_of_genes == 2:
        return p_gene_from_dad * p_gene_from_mum
    elif number_of_genes == 1:
        return (
            p_gene_from_dad * (1 - p_gene_from_mum)
            + (1 - p_gene_from_dad) * p_gene_from_mum
        )
    else:
        return (1 - p_gene_from_mum) * (1 - p_gene_from_dad)


if __name__ == "__main__":
    main()
