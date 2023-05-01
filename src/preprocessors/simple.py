import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk dependencies
nltk.download('punkt')
nltk.download('stopwords')


def sent_to_words(sent):
	return [w for w in word_tokenize(sent.lower())
		if w not in stopwords.words('english')\
		and w not in string.punctuation\
		and all(c in string.printable for c in w)\
		and len(w) > 1]