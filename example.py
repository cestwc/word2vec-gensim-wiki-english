import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

import gensim.downloader as api
from gensim.models import Word2Vec

wiki = api.load("wiki-english-20171001")

def f(a):
    # The function I want to apply to everything
    return ''.join(x.lower() for s in a['section_texts'] for x in s+' ' if x.isalpha() or x == ' ').strip().split()

class IterableWrapper:
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = None
    def __iter__(self):
        # This is not ideal as it doesn't allow two different iterators at the same time
        self.iterator = iter(self.iterable)
        return self
    def __next__(self):
        return f(next(self.iterator))
	
dataset = IterableWrapper(wiki)
model = Word2Vec(dataset, size=128, window=15, iter=50, sg=1, min_count=150)
model.save("word2vec-wiki.model")
model.wv.save_word2vec_format('word2vec-wiki.txt')
print(model.wv.most_similar('cat'))
