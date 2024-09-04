import csv
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(Path(__file__).parent / sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")  # noqa: E231
    print(f"True Negative Rate: {100 * specificity:.2f}%")  # noqa: E231


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []

    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row

        for row in reader:
            # Convert values to appropriate types
            administrative = int(row[0])
            administrative_duration = float(row[1])
            informational = int(row[2])
            informational_duration = float(row[3])
            product_related = int(row[4])
            product_related_duration = float(row[5])
            bounce_rates = float(row[6])
            exit_rates = float(row[7])
            page_values = float(row[8])
            special_day = float(row[9])
            month = convert_month(row[10])
            operating_systems = int(row[11])
            browser = int(row[12])
            region = int(row[13])
            traffic_type = int(row[14])
            visitor_type = convert_visitor_type(row[15])
            weekend = convert_weekend(row[16])
            label = convert_label(row[17])

            # Create evidence list
            evidence_row = [
                administrative,
                administrative_duration,
                informational,
                informational_duration,
                product_related,
                product_related_duration,
                bounce_rates,
                exit_rates,
                page_values,
                special_day,
                month,
                operating_systems,
                browser,
                region,
                traffic_type,
                visitor_type,
                weekend,
            ]
            evidence.append(evidence_row)

            # Add label to labels list
            labels.append(label)

    return evidence, labels


def convert_month(month):
    """
    Convert month string to corresponding index.

    January -> 0
    February -> 1
    March -> 2
    ...
    December -> 11
    """
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "June",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    return months.index(month)


def convert_visitor_type(visitor_type):
    """
    Convert visitor type string to corresponding integer.

    Returning Visitor -> 1
    Other -> 0
    """
    if visitor_type == "Returning_Visitor":
        return 1
    else:
        return 0


def convert_weekend(weekend):
    """
    Convert weekend string to corresponding integer.

    TRUE -> 1
    FALSE -> 0
    """
    if weekend == "TRUE":
        return 1
    else:
        return 0


def convert_label(label):
    """
    Convert label string to corresponding integer.

    TRUE -> 1
    FALSE -> 0
    """
    if label == "TRUE":
        return 1
    else:
        return 0


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    true_positives = sum(
        1
        for label, prediction in zip(labels, predictions)
        if label == 1 and prediction == 1
    )
    true_negatives = sum(
        1
        for label, prediction in zip(labels, predictions)
        if label == 0 and prediction == 0
    )
    positive_labels = sum(1 for label in labels if label == 1)
    negative_labels = sum(1 for label in labels if label == 0)

    sensitivity = true_positives / positive_labels
    specificity = true_negatives / negative_labels

    return sensitivity, specificity


if __name__ == "__main__":
    main()
