from overrides import overrides
import pickle
import spacy
import sys
import math
import numpy as np
from numpy import linalg as la
import json
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.metrics.distance import *
from nltk.translate.bleu_score import *

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

logger = get_logger(__name__)
# nlp = spacy.load('en_core_web_lg', parser=False)
nlp = spacy.load('en_core_web_sm', parser=False, ner=False)

@register('rankingemb_model')
class RankingEmbModel(Component):
    def __init__(self, **kwargs):
        # pass
        self.glove_model = kwargs['embedder']
        pass

        # with open('/tmp/phones.pickle', 'rb') as handle:
        #     print('Data set is loading')
        #     data = pickle.load(handle)
        #     print('Data set loaded')

        # data_nlped = [nlp(item['Title']) for item in data]
        # print('everything is nlped')

        # self.mean_transform(data_nlped)
        # print('everything is transformed')

        
#    @overrides
#    def load(self):
#        pass

#    @overrides
#    def save(self):
#        pass

    @overrides
    def __call__(self, x, x_emb):

                
        # print(x)
        # print(x_emb)

        # y = []
        # for idx, a in enumerate(x):
        #     print(idx)
        #     print(a)
        #     print(x_emb[idx])
        #     print("--------------------")
        #     # for b in a:
        #         # print(b)
        #         # print(b.pos_)
        # y.append(1)
        return x

    # def train(self, data):
    #     print(str(data))
    #     pass

    # def fit(self, *args, **kwargs):
    #     super().__init__(**kwargs)  # self.opt initialized in here

    #     print(self.opt.pop('category'))
    #     # self.tokenizer = self.opt.pop('tokenizer')
    #     pass

    def mean_transform(self, X, debug = False):
        out = []
        for words in X:
            row = []
            for w in filter_nlp(words):
                if debug:
                    print(w)
                    print(w.lemma_)
                    print(w.vector)
                    print(len(w.vector))
        
                row.append(self.glove_model([[w.lemma_]]))
            if len(row)!=0:
                row_mean = np.mean(row, axis=0)
            else:
                row_mean = np.zeros(50)
            out.append(row_mean)
        return np.array(out)

    def shutdown(self):
        pass

    def reset(self):
        pass  

    # def fit(self, x, y, *args, **kwargs):
    #     pass

    # def load(self, *args, **kwargs):
    #     pass
    #     # logger.info('SquadVocabEmbedder: loading saved {}s vocab from {}'.format(self.level, self.load_path))
    #     # self.emb_dim, self.emb_mat, self.token2idx_dict = pickle.load(self.load_path.open('rb'))
    #     # self.loaded = True

    # def save(self, *args, **kwargs):
    #     pass
    #     # logger.info('SquadVocabEmbedder: saving {}s vocab to {}'.format(self.level, self.save_path))
    #     # self.save_path.parent.mkdir(parents=True, exist_ok=True)
    #     # pickle.dump((self.emb_dim, self.emb_mat, self.token2idx_dict), self.save_path.open('wb'))
    

def filter_nlp(tokens):
    res = []
    for word in tokens:
# WRB WDT WP
        if word.tag_ not in ['MD', 'SP', 'DT', 'PRP', 'TO']: #VBP VBZ
            res.append(word)
    return res

def filwords(words):
    tokens = nlp(words)
    return [w.lemma_ for w in filter_nlp(tokens)]
