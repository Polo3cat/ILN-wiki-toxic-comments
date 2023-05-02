import numpy as np

from tqdm import tqdm

from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier


def train(document_term_matrix, target, target_names):
	multioutput_mnnb = MultiOutputClassifier(MultinomialNB())
	multioutput_mnnb.fit(document_term_matrix, target)
	return multioutput_mnnb


def test(model, features, labels):
	return model.score(features, np.asarray(labels))
