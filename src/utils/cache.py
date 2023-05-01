import hashlib
import pickle

from joblib import dump
from joblib import load as _load


def check_update_hashof(obj, cache_file):
	md5sum = hashlib.md5(pickle.dumps(obj)).hexdigest()
	with open(cache_file,'a+') as f:
		f.seek(0)
		if f.read() != md5sum:
			f.write(md5sum)
			return False
	return True


def load(cache_file):
	try:
		return _load(cache_file)
	except Exception:
		return None


def store(obj, cache_file):
	dump(obj, cache_file)
