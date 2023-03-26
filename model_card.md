# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model used for this release is a classical random forest classifier.

## Intended Use

The goal of this model is to predict the income based on the census data. 
More precisely we predict wether the income is over 50k per year or not.

## Training Data

All the data comes from the [UCI website](https://archive.ics.uci.edu/ml/datasets/census+income).
"Extraction was done by Barry Becker from the 1994 Census database."

## Evaluation Data

Evaluation data is a part of the data from the UCI website. for this release, it was 20% of the whole dataset. However this parameter can  be modified in the param.yaml file at the root of this directory.

## Metrics
The current metric used to observe the results are the precision, recall and fbeta scores.
for this release the results are the following:

- precison: 0.72

- recall: 0.61

- fbeta: 0.66

As we can see there is still room for improvement.

Those metric are stored in the metrics/model_metrics.pkl file.

## Ethical Considerations

Relation founds in that dataset may not be representative of the real life ones. Thus, in no case, this model should be a source of inspiration to search for implication between the different inputs and the salary.

## Caveats and Recommendations

This dataset is quite old and not well balanced between classes. If we would like to have a model with a better representation of the relation between census data and salary, we should ensure the dataset is made ethicaly and well balanced.