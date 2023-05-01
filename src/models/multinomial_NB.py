from tqdm import tqdm

from sklearn.naive_bayes import MultinomialNB


def train(document_term_matrix, target, target_names):
	multinominal_NBs = {t_name: MultinomialNB() for t_name in target_names}
	for i,mnnb in tqdm(enumerate(multinominal_NBs.values()), total=len(multinominal_NBs), desc='Fitting models'):
		mnnb.fit(document_term_matrix, [x[i] for x in target])
	return multinominal_NBs


def test(models, features, labels):
	accuracies = {}
	for i,(m_name,model) in enumerate(models.items()):
		accuracies[m_name] = model.score(features,[l[i] for l in labels])
	return accuracies
