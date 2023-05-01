import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('test-file', help='csv file containing text')
parser.add_argument('test-labels', help='file with rows of test ids and labels')
parser.add_argument('consolidation-file', help='file with rows of test ids and labels')

args = vars(parser.parse_args())

with open(args['test-file'],newline='') as f_test, open(args['test-labels'],newline='') as f_labels:
	f_test.readline()
	f_labels.readline()
	test = {r[0]: r[1] for r in csv.reader(f_test, dialect=csv.unix_dialect)}
	labels = {r[0]: r[1:] for r in csv.reader(f_labels, dialect=csv.unix_dialect)}

with open(args['consolidation-file'],'w',newline='') as f:
	w = csv.writer(f, dialect=csv.unix_dialect,quoting=csv.QUOTE_NONNUMERIC)
	w.writerow(["id","comment_text","toxic","severe_toxic","obscene","threat","insult","identity_hate"])
	for k,v in labels.items():
		if v[0] != '-1':
			w.writerow([k,test[k],*[int(x) for x in v]])
