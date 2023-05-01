# Experiments on classifying toxic Wikipedia comments

This is an implementation of [jigsaw-toxic-comment-classification-challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data).

## Running

You must have [poetry](https://python-poetry.org/) installed. The following example runs an experiment and prints the results.

`poetry run python3 src/run-experiment.py ../dataset/train.csv ../dataset/test_w_labels.csv ../models/A/ preprocessors.simple models.multinomial_NB`
