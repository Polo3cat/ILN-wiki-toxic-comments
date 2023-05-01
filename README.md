# Experiments on classifying toxic Wikipedia comments

This is an implementation of [jigsaw-toxic-comment-classification-challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data).

## Dependencies

You must have [poetry](https://python-poetry.org/) installed. Dependencies can be installed with
```shell
$ poetry install
```

## Running

### Generate labelled test
Test corpus and its labels are split into two files. A single file for easier ingestion has to be generated. The following example shows how.
```shell
$ poetry run python3 src/utils/consolidate-test-labels.py ../dataset/test.csv ../dataset/test_labels.csv ../dataset/test_w_labels.csv
```

### Run an experiment
The following example shows how to run an experiment and prints the results. This uses the `simple` preprocessor and the `multinomial_NB` model.

```shell
$ poetry run python3 src/run-experiment.py ../dataset/train.csv ../dataset/test_w_labels.csv ../models/A/ preprocessors.simple models.multinomial_NB
[nltk_data] Downloading package punkt to /home/pablo/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package stopwords to /home/pablo/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
2023-05-01 19:30:11,384 INFO Beginning experiment
Fitting models: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 18.38it/s]
2023-05-01 19:30:13,318 INFO Model mean accuracy: {'toxic': 0.9193629059989371, 'severe_toxic': 0.9858232517427866, 'obscene': 0.9510769326956141, 'threat': 0.9938260026884241, 'insult': 0.948857419738035, 'identity_hate': 0.9843071055675389}
```
