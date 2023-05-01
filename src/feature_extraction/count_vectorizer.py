import logging

from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer

from utils import cache


def extract(train_mat, preprocess, model_dir):
	preprocess_path = f"{model_dir}/preprocess_hash"
	train_mat_hash_path = f"{model_dir}/train_mat_hash"
	vectorizer_path = f"{model_dir}/vectorizer"
	doc_term_mat_path = f"{model_dir}/doc_term_mat"

	vectorizer = cache.load(vectorizer_path)
	doc_term_mat = cache.load(doc_term_mat_path)

	# Explicitly evaluate all conditions because
	# they have side effects
	relearn = not all(
		[
			cache.check_update_hashof(preprocess, preprocess_path),
			cache.check_update_hashof(train_mat, train_mat_hash_path),
			vectorizer is not None,
			doc_term_mat is not None
		]
	)
	if relearn:
		logging.info('Relearning count vectorizer feature extractor')
		logging.info('Re-extracting train features')
		vectorizer = CountVectorizer(
			input='content',
			decode_error='ignore',
			tokenizer=preprocess)

		doc_term_mat = vectorizer.fit_transform(tqdm((x[1] for x in train_mat), total=len(train_mat), desc='Fitting count vectorizer'))

		cache.store(vectorizer, vectorizer_path)
		cache.store(doc_term_mat, doc_term_mat_path)
	
	return doc_term_mat, vectorizer


def extract_with(extractor, test_matrix, model_dir):
	extractor_hash_path = f"{model_dir}/extractor_hash"
	test_mat_hash_path = f"{model_dir}/test_mat_hash"
	test_features_path = f"{model_dir}/test_features"

	test_features = cache.load(test_features_path)
	relearn = not all(
		[
			cache.check_update_hashof(extractor, extractor_hash_path),
			cache.check_update_hashof(test_matrix, test_mat_hash_path),
			test_features is not None
		]
	)
	if relearn:
		logging.info('Re-extracting test features')
		test_features = extractor.transform(tqdm((x[1] for x in test_matrix), total=len(test_matrix), desc='Extracting features from test'))
		cache.store(test_features, test_features_path)

	return test_features
