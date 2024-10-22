
# Shopping Predictor

This project is a machine learning application that predicts whether a user will make a purchase on an online shopping website based on their browsing behavior. The application uses a k-nearest neighbors classifier.

## Files

- `shopping.py`: The main script that loads data, trains the model, and evaluates its performance.

## Prerequisites

- Python 3.x
- pandas
- scikit-learn

You can install the required packages using pip:
pip install pandas scikit-learn


## Functions

### `main()`

The main function that orchestrates the loading of data, training of the model, and evaluation of the results.

### `load_data(filename)`

Loads shopping data from a CSV file and converts it into a list of evidence lists and a list of labels.

- `filename`: The path to the CSV file.
- Returns: A tuple `(evidence, labels)`.

### `train_model(evidence, labels)`

Trains a k-nearest neighbors model (k=1) on the provided evidence and labels.

- `evidence`: A list of evidence lists.
- `labels`: A list of labels.
- Returns: A trained k-nearest neighbors model.

### `evaluate(labels, predictions)`

Evaluates the performance of the model by calculating its sensitivity and specificity.

- `labels`: A list of actual labels.
- `predictions`: A list of predicted labels.
- Returns: A tuple `(sensitivity, specificity)`.

## Data Format

The CSV file contains the following columns:

- `Administrative`: An integer.
- `Administrative_Duration`: A floating point number.
- `Informational`: An integer.
- `Informational_Duration`: A floating point number.
- `ProductRelated`: An integer.
- `ProductRelated_Duration`: A floating point number.
- `BounceRates`: A floating point number.
- `ExitRates`: A floating point number.
- `PageValues`: A floating point number.
- `SpecialDay`: A floating point number.
- `Month`: A string representing the month (e.g., "Jan", "Feb").
- `OperatingSystems`: An integer.
- `Browser`: An integer.
- `Region`: An integer.
- `TrafficType`: An integer.
- `VisitorType`: A string ("Returning_Visitor", "New_Visitor", "Other").
- `Weekend`: A boolean.
- `Revenue`: A boolean indicating whether the user made a purchase.

## License

This project is licensed under the MIT License.
