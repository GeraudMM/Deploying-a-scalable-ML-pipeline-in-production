from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
import pandas as pd
from ml.data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(random_state=31416)
    model.fit(X_train, y_train)

    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """

    return model.predict(X)

def compute_metric_on_slice_of_data(model, input_data, categorical_features=[], label = None, encoder=None, lb=None):
    """ compute metric on slice of the data
    
    Args:
        model: Trained machine learning model.
        X (pd.DataFrame): Dataframe containing the features and label. Columns in `categorical_features`
        categorical_features (list[str]): List containing the names of the categorical features (default=[])
        label (str): Name of the label column in `X`. If None, then an empty array will be returned for y (default=None)
        encoder (sklearn.preprocessing._encoders.OneHotEncoder) : Trained sklearn OneHotEncoder, only used if 
            training=False.
        lb (sklearn.preprocessing._label.LabelBinarizer) : Trained sklearn LabelBinarizer, only used if training=False.
        
    Returns:
        metrics_df (pd.DataFrame): Different metrics for the slice of data.
    """

    metrics = []

    for feature in categorical_features:
        unique_values = input_data[feature].unique()
        for value in unique_values:
            X, y, encoder, lb = process_data(input_data.loc[input_data[feature]==value], categorical_features=categorical_features, label=label, encoder=encoder, lb=lb, training=False)
            preds = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            metrics.append((feature, value, precision, recall, fbeta))
    
    metrics_df = pd.DataFrame(metrics, columns=['feature', 'value', 'precision', 'recall', 'fbeta'])
    return metrics_df
