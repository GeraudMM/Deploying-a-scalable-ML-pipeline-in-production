# Script to train machine learning model.
from ml.data import process_data
import ml.model as utils
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import yaml
import pickle

# Add the necessary imports for the starter code.

# Add code to load in the data.

with open('params.yaml') as f:
        params = yaml.safe_load(f)
input_data = pd.read_csv(params['input_data_path'])

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(input_data, test_size=params['train_model']['test_size'])

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", encoder=encoder, lb=lb, training=False
    )
# Train and save a model.
model = utils.train_model(X_train, y_train)

preds = utils.inference(model, X_test)
model_config = {
    "model": model,
    "encoder": encoder,
    "lb": lb,
    "cat_features": cat_features
}

with open(params['model_path'], 'wb') as file:
    pickle.dump(model_config, file)

slice_data_metrics = utils.compute_metric_on_slice_of_data(model, input_data, cat_features, label = "salary", encoder=encoder, lb=lb)
# Save to output text
slice_data_metrics.to_csv(params['metrics_folder'] + "metrics_on_slice.csv")


precision, recall, fbeta = utils.compute_model_metrics(y_test, preds)

model_metric = {
    "precision": precision,
    "recall": recall,
    "fbeta": fbeta
}
print(f"model metrics: {model_metric}")
with open(params['metrics_folder'] + "model_metric.pkl", 'wb') as file:
    pickle.dump(model_metric, file)
