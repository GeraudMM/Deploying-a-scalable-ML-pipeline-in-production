# Script to train machine learning model.
from ml.data import process_data
import ml.model as utils
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import yaml
import pickle


with open('params.yaml') as f:
    params = yaml.safe_load(f)


logging.basicConfig(
    filename=params['logfile_path'],
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

logging.info(f"Loading data: {params['input_data_path']}")
input_data = pd.read_csv(params['input_data_path'])

logging.info("Loading done")

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

logging.info("Processing the data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", encoder=encoder, lb=lb, training=False
)
# Train and save a model.
logging.info("Train the model and predict")
model = utils.train_model(X_train, y_train)

preds = utils.inference(model, X_test)
model_config = {
    "model": model,
    "encoder": encoder,
    "lb": lb,
    "cat_features": cat_features
}

logging.info(f"save the model to {params['model_path']}")
with open(params['model_path'], 'wb') as file:
    pickle.dump(model_config, file)

slice_data_metrics = utils.compute_metric_on_slice_of_data(
    model, input_data, cat_features, label="salary", encoder=encoder, lb=lb)
# Save to output text
slice_data_metrics.to_csv(params['metrics_folder'] + "metrics_on_slice.csv")


precision, recall, fbeta = utils.compute_model_metrics(y_test, preds)

logging.info(f"Precision: {precision:.2f}; Recall: {recall:.2f}; Fbeta: {fbeta:.2f}")
model_metric = {
    "precision": precision,
    "recall": recall,
    "fbeta": fbeta
}

with open(params['metrics_folder'] + "model_metric.pkl", 'wb') as file:
    pickle.dump(model_metric, file)

logging.info("Training pipeline finished !")
