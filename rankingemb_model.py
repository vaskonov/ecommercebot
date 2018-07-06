from overrides import overrides
import pickle
import spacy
from spacy.matcher import Matcher
import sys
import re
import math
import numpy as np
from numpy import linalg as la
import json
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.metrics.distance import *
from nltk.translate.bleu_score import *
from scipy.spatial.distance import cosine, euclidean

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download, jsonify_data
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

log = get_logger(__name__)
# nlp = spacy.load('en_core_web_lg', parser=False)
nlp = spacy.load('en', parser=False)

@register('rankingemb_model')

class RankingEmbModel(Component):
    def __init__(self, **kwargs):
        self.glove_model = kwargs['embedder']
        self.dim = int(kwargs['dim'])

        with open('/tmp/phones.pickle', 'rb') as handle:
            log.debug('Data set is loading')
            self.data = pickle.load(handle)
            log.debug('Data set loaded')

        self.data_nlped = [nlp(item['Title']) for item in self.data]
        log.debug('Data is nlped')

        self.feat_nlped = [nlp(item['Feature']) if 'Feature' in item else nlp('') for item in self.data]
        log.debug('Feature is nlped')

        self.data_mean = self.mean_transform(self.data_nlped)
        log.debug('everything is transformed')

    @overrides
    def __call__(self, x):

        log.debug(str(x))
        text = " ".join(x[0]).replace("$", " dollars ")

        # if type(x) == str:
            # text = x
        # elif type(x[0]) == str:
            # text = x[0]
        # elif type(x[0][0]) == str:
            # text = x[0][0]

        log.debug(text)

        # start = start[0]
        # stop = stop[0]
        doc = nlp(text)
        doc_fil = doc
        #doc_fil = [w for w in doc if w.tag_ in ['NNP', 'NN', 'JJ', 'PROPN']]
        
        doc, money_res = find_money(doc)
        print(doc)
        print(filter_nlp(doc))
        
        text_mean_emb = self.mean_transform([doc_fil])[0]
        results_mean = [cosine(text_mean_emb, emb) if np.sum(emb)!=0 else math.inf for emb in self.data_mean]
        results_blue = [bleu_string_distance(lemmas(feat), lemmas(filter_nlp(doc))) for feat in self.feat_nlped]
        results_title = [bleu_string_distance(lemmas(title), lemmas(filter_nlp(doc))) for title in self.data_nlped]

        scores = np.mean([results_mean, results_blue, results_title], axis=0).tolist()
        results_args = np.argsort(scores).tolist()

        if 'num1' in money_res:
            log.debug('results before money '+str(len(results_args)))
            results_args = [idx for idx in results_args if price(self.data[idx])>=money_res['num1'] and price(self.data[idx])<=money_res['num2']]
            log.debug('results after money '+str(len(results_args)))
            
        # fetch_data = [self.data[idx] for idx in results_args[start:stop+1]]
        ret = {
            'results_args': results_args,
            'scores': scores
        }
        return json.dumps(ret)
        
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
            for w in filter_nlp_emb(words):
                if debug:
                    print(w)
                    print(w.lemma_)
                    print(w.vector)
                    print(len(w.vector))
        
                row.append(self.glove_model([[w.lemma_]]))
            if len(row)!=0:
                row_mean = np.mean(row, axis=0)
            else:
                row_mean = np.zeros(self.dim)
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
    
def price(item):
    if 'ListPrice' in item:
        return float(item['ListPrice'].split('$')[1].replace(",",""))
    else:
        return math.inf

def find_money(doc):
    below = lambda text: bool(re.compile(r'below|cheap').match(text))
    BELOW = nlp.vocab.add_flag(below)

    above = lambda text: bool(re.compile(r'above|start').match(text))
    ABOVE = nlp.vocab.add_flag(above)

    matcher = Matcher(nlp.vocab)
    matcher.add('below', None, [{BELOW: True},{'LOWER':'than', 'OP':'?'},{'LOWER':'from', 'OP':'?'}, 
                        {'ORTH':'$', 'OP':'?'}, {'ENT_TYPE': 'MONEY', 'LIKE_NUM':True}])
    matcher.add('above', None, [{ABOVE: True},{'LOWER':'than', 'OP':'?'},{'LOWER':'from', 'OP':'?'}, 
                        {'ORTH':'$', 'OP':'?'}, {'ENT_TYPE': 'MONEY', 'LIKE_NUM':True}])

    #  {'ENT_TYPE': 'MONEY', 'LIKE_NUM':True}
    matches = matcher(doc)

    result = {}
    doc1 = list(doc)

    negated = False
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id] 
        span = doc[start:end]
        for child in doc[start].children:
            if child.dep_ == 'neg':
                negated = True

        print(match_id, string_id, start, end, span.text, negated)
        num_token = [token for token in span if token.like_num == True]
        if len(num_token)!=1:
            print("Error", str(num_token))

        if (string_id == 'below' and negated == False) or (string_id == 'above' and negated == True):
            result['num1'] = 0
            result['num2'] = float(num_token[0].text)

        if (string_id == 'above' and negated == False) or (string_id == 'below' and negated == True):
            result['num1'] = float(num_token[0].text)
            result['num2'] = 1000000

        # result['op'] = string_id
        # result['num'] = doc[end]
        
        log.debug('find_money '+str(result))
        del doc1[start:end+1]     

    return doc1, result


def filter_nlp_emb(doc):
    return [w for w in doc if w.tag_ in ['NNP', 'NN', 'JJ', 'PROPN']]    

def filter_nlp(tokens):
    res = []
    for word in tokens:
# WRB WDT WP
        if word.tag_ not in ['MD', 'SP', 'DT', 'PRP', 'TO'] and word.is_stop == False and word.is_punct == False: #VBP VBZ
            res.append(word)
    return res

def lemmas(doc):
    return [w.lemma_ for w in doc]

def filwords(words):
    tokens = nlp(words)
    return [w.lemma_ for w in filter_nlp(tokens)]

def bleu_string_distance(q_list,a_list):
  # 0 - no overlab (bad)
  # 1 - exact match (good)
    smooth = SmoothingFunction()
    return 1-sentence_bleu([q_list], a_list, weights=(0.3,0.7), auto_reweigh=False, emulate_multibleu=False, smoothing_function=smooth.method1)

# print(bleu_string_distance(['i', 'love','nature'], ['i','love'],))
