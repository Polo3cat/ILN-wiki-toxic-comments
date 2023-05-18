from tqdm import tqdm

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score


def train(document_term_matrix, target, target_names):
	multinominal_NBs = {t_name: MultinomialNB() for t_name in target_names}
	for i,mnnb in tqdm(enumerate(multinominal_NBs.values()), total=len(multinominal_NBs), desc='Fitting models'):
		mnnb.fit(document_term_matrix, [x[i] for x in target])
	return multinominal_NBs


def test(models, features, labels):
	accuracies = {}
	aucs = {}
	for i,(m_name,model) in enumerate(models.items()):
		y_true = [l[i] for l in labels]
		accuracies[m_name] = model.score(features,y_true)
		prob = model.predict_proba(features)[:,1]
		aucs[m_name] = roc_auc_score(y_true, prob)
	return {
		"Accuracy": accuracies,
		"MeanAcc": sum(accuracies.values())/len(models),
		"AUC": aucs,
		"MeanAUC": sum(aucs.values())/len(models)}
