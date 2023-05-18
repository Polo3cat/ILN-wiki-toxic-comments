import numpy as np

from tqdm import tqdm

from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score


def train(document_term_matrix, target, target_names):
	multioutput_mnnb = MultiOutputClassifier(MultinomialNB())
	multioutput_mnnb.fit(document_term_matrix, target)
	return multioutput_mnnb


def test(model, features, labels):
	y_true = np.asarray(labels)
	acc = model.score(features, y_true)
	prob = np.transpose([y_pred[:,1] for y_pred in model.predict_proba(features)])
	auc = roc_auc_score(y_true, prob, average=None)
	auc = sum(auc)/len(auc)
	return {"Accuracy": acc, "ROC AUC": auc}

