from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

metrics = {
    "Accuracy": [accuracy_score, {}],
    "F1 Score": [f1_score, {"average": "weighted"}],
    "Precision": [precision_score, {"average": "weighted"}],
    "Recall": [recall_score, {"average": "weighted"}],
}
