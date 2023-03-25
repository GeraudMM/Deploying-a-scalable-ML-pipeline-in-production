# Script to train machine learning model.
from ml.data import process_data
import ml.model as utils
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import yaml

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
        test, categorical_features=cat_features, label=None, encoder=encoder, lb=lb, training=False
    )
# Train and save a model.
model = ml.model.train_model(X_train, y_train)

    model_config = {
        "model": model,
        "encoder": encoder,
        "lb": lb,
        "cat_features": cat_features
    }

    with open(params['model_path'], 'wb') as file:
        pickle.dump(model_config, file)