# word2vec-gensim-wiki-english
Train your own word2vec embeddings using a wiki english dataset

You may want to have pretrained word2vec vectors, and this [repository](https://github.com/RaRe-Technologies/gensim-data) may just be a good idea for you. However, what makes it tricky is that there isn't pretrained vectors using the wiki-english dataset. What makes it even more tricky is that the given usage code, though works for ```text8``` dataset, cannot train vectors on the ```wiki-english-20171001``` dataset.

We have tested it several times, and the most probable reason is that the data structure of ```wiki-english-20171001``` is slightly different from the rest. It contains many sections, rather than just tokenized sentences.

To get it work, we refer to the ```IterableWrapper``` provided by this [post](https://www.reddit.com/r/learnprogramming/comments/d980aa/how_to_wrap_an_iterable_in_python/), and apply it on the wiki-english dataset.

## Usage
To see how fast your progress is, you'd better configure you logging like this
```python
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
```
Then we load the dataset as introduced in this [repository](https://github.com/RaRe-Technologies/gensim-data)
```python
import gensim.downloader as api
from gensim.models import Word2Vec

wiki = api.load("wiki-english-20171001")
```
The key idea is here, you wrap the dataset in a way that a model from [gensim](https://pypi.org/project/gensim/) can handle
```python
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
```
Finally, you train your vectors as usual and test how good they are
```python
dataset = IterableWrapper(wiki)
model = Word2Vec(dataset, size=128, window=15, iter=50, sg=1, min_count=150)
model.save("word2vec-wiki.model")
model.wv.save_word2vec_format('word2vec-wiki.txt')
print(model.wv.most_similar('cat'))
```
