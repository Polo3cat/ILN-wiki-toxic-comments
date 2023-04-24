import argparse
import csv
import string
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool
from pprint import pprint
import sys

from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def sent_to_words(sent):
	return [w for w in word_tokenize(sent.lower()) 
		if w not in stopwords.words('english')\
		and w not in string.punctuation\
		and all(c in string.printable for c in w)]


def _train(p,t):
	sys.stdout.flush()
	# toxic severe_toxic obscene threat insult identity_hate
	tag_bag = [defaultdict(int)]*6
	for _,sent,*tags in tqdm(t, desc=f'Process {p} training', position=p):
		for bag,tag in zip(tag_bag,tags):
			if tag == '1':
				words = sent_to_words(sent)
				for w in words:
					bag[w] += 1
	return tag_bag


def err(exc):
	print(exc)
	sys.stdout.flush()


def merge(list_of_lists):
	ret = list_of_lists[0]
	for ls in list_of_lists[1:]:
		for d1,d2 in zip(ls,ret):
			for k,v in d1.items():
				d2[k] += v
	return ret


def train(train_reader):
	PROCESSES = 8
	chunk = len(train_reader)//PROCESSES
	rem = len(train_reader)%PROCESSES
	print(f'{len(train_reader)} {chunk=} {rem=}')
	train_reader =  train_reader
	ch = [(i,train_reader[i*chunk:i*chunk+chunk+(rem if i+1==PROCESSES else 0)])
		 for i in range(PROCESSES)]
	with Pool(PROCESSES) as pool:
		# toxic severe_toxic obscene threat insult identity_hate
		tag_bags = [pool.apply_async(_train, c, error_callback=err) for c in ch]
		tag_bags = merge([x.get() for x in tag_bags])
	pprint(tag_bags)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('train-file')
	parser.add_argument('test-file')
	parser.add_argument('test-labels-file')

	args = vars(parser.parse_args())

	with open(args['train-file']) as f_train,\
		 open(args['test-file']) as f_test,\
		 open(args['test-labels-file']) as f_test_labels:
		# Load into memory, by building a list. This allows len to be measured.
		train_reader = list(csv.reader(f_train, dialect=csv.unix_dialect))
		# Fields in the train csv
		# "id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"
		train(train_reader)
