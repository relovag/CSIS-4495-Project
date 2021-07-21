from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

metrics = {
    "category": {
        "Accuracy": [accuracy_score, {}],
        "F1 Score": [f1_score, {"average": "weighted"}],
        "Precision": [precision_score, {"average": "weighted"}],
        "Recall": [recall_score, {"average": "weighted"}],
    },
    "number": {
        "R2": [r2_score, {}],
        "Mean Squared Error": [mean_squared_error, {}],
        "Mean Absolute Error": [mean_absolute_error, {}],
    },
}
