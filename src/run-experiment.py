import argparse
import csv
import importlib
import logging
import pprint

from tqdm import tqdm

from feature_extraction import count_vectorizer

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('train-file', help='training csv file containing text and associated labels')
	parser.add_argument('test-file', help='testing csv file containing text and associated labels')
	parser.add_argument('model-store-dir', help='directory where models will be dumped and loaded')
	parser.add_argument('preprocessor', help='preprocessor module used to preprocess text')
	parser.add_argument('model', help='model module used for training and testing')

	args = vars(parser.parse_args())

	preprocessor = importlib.import_module(args['preprocessor'])
	model_module = importlib.import_module(args['model'])

	logging.info(f'Beginning experiment')

	with open(args['train-file'],newline='') as f_train:
		# Throw header away, we already know it
		f_train.readline()
		# Load into memory, by building a list. This allows len to be measured.
		# Fields in the train csv
		# "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
		train_matrix = list(csv.reader(f_train, dialect=csv.unix_dialect))
	features, extractor = count_vectorizer.extract(train_matrix, preprocessor.sent_to_words, args['model-store-dir'])
	target = [[int(_x) for _x in x[2:]] for x in train_matrix]
	target_names = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
	model = model_module.train(features,target,target_names)

	with open(args['test-file'],newline='') as f_test:
		f_test.readline()
		test_matrix = list(csv.reader(f_test, dialect=csv.unix_dialect))
	test_features = count_vectorizer.extract_with(extractor, test_matrix, args['model-store-dir'])
	test_labels = [[int(_x) for _x in x[2:]] for x in test_matrix]
	result = model_module.test(model, test_features, test_labels)
	logging.info(f'Model results: {pprint.pformat(result)}')
