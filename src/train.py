import argparse
import csv
import string

from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from joblib import dump, load


# nltk dependencies
nltk.download('punkt')
nltk.download('stopwords')

def sent_to_words(sent):
	return [w for w in word_tokenize(sent.lower())
		if w not in stopwords.words('english')\
		and w not in string.punctuation\
		and all(c in string.printable for c in w)\
		and len(w) > 1]


def train(train_reader, model_dir):
	vectorizer_path = f"{args['model-dir']}/vectorizer"
	doc_term_mat_path = f"{args['model-dir']}/doc_term_mat"
	try:
		vectorizer = load(vectorizer_path)
		doc_term_mat = load(doc_term_mat_path)
	except:
		vectorizer = CountVectorizer(
			input='content',
			decode_error='ignore',
			tokenizer=sent_to_words)
		doc_term_mat = vectorizer.fit_transform(tqdm((x[1] for x in train_reader), total=len(train_reader), desc='Fitting vectorizer'))
		dump(vectorizer, vectorizer_path)
		dump(doc_term_mat, doc_term_mat_path)

	multinominal_NB = [MultinomialNB()]*6
	for i,gnb in tqdm(enumerate(multinominal_NB), total=len(multinominal_NB), desc='Fitting models'):
		gnb.fit(doc_term_mat, [x[i+2] for x in train_reader])
	return multinominal_NB, vectorizer


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('train-file')
	parser.add_argument('model-dir')

	args = vars(parser.parse_args())

	with open(args['train-file']) as f_train:
		# Load into memory, by building a list. This allows len to be measured.
		train_reader = list(csv.reader(f_train, dialect=csv.unix_dialect))
		# Fields in the train csv
		# "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
		model_desc = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
		models, vectorizer = train(train_reader, args['model-dir'])
		for model,desc in zip(models,model_desc):
			dump(model, f"{args['model-dir']}/{desc}")