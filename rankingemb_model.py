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

        self.title_nlped = [nlp(item['Title']) for item in self.data]
        log.debug('Title is nlped')

        self.feat_nlped = [nlp(item['Title']+'.'+item['Feature']) if 'Feature' in item else nlp(item['Title']) for item in self.data]
        log.debug('Feature is nlped')

        self.title_mean = self.mean_transform(self.title_nlped)
        log.debug('everything is transformed')

        with open('../faq.json', 'r') as f:
            self.faq_js = json.load(f)

        with open('../intents.json', 'r') as f:
            self.intents = json.load(f)

    @overrides
    def __call__(self, x):

        log.debug(str(x))
      #  text = " ".join(x[0]).replace("$", " dollars ")
        text = " ".join(x[0])

        log.debug(text)

        doc = nlp(text)
        # doc_fil = doc
        #doc_fil = [w for w in doc if w.tag_ in ['NNP', 'NN', 'JJ', 'PROPN']]
        
        doc, money_res = find_money(doc)
        print('unfiltered_nlp:', doc)
        print('filter_nlp:', filter_nlp(doc))
        
        if 'num1' not in money_res:
            results = classify_faq(doc)

            results_arg = np.argsort(results)
            print('Results for faq "%s"', str(results_arg))

            if results[results_arg[0]] < 0.2:

                print('FAQ question is detected with answer', self.faq_js.values()[results_arg[0]])

                return json.dumps({
                    'intent': 'faq',
                    'text': self.faq_js.values()[results_arg[0]]
                    })
            else:

                doc = filter_nlp(doc)
                print('Going to intent classification, filtered:', doc)

                nns = [w.text for w in doc if w.tag_ in ['NNP', 'NN', 'JJ', 'PROPN']]
                sal_phrase = [w.text for w in doc if w.tag_ not in ['NNP', 'NN', 'JJ', 'PROPN']]
                
                print('salient phrase for intent', sal_phrase)
                print('nns for catalogue', nns)

                scores = sorted(classify_intent(sal_phrase), key=lambda x: x[1])
                print('intent classification:', str(scores))

                if scores[0][1] < 0.2 and scores[0][0] == 'payment':
                    return json.dumps({
                        'intent': 'payment'
                        })

                else:
                    return rank_items(nns, money_res):
        else:
            return rank_items(nns, money_res):


            
                

        
        # text_mean_emb = self.mean_transform([doc_fil])[0]
        # results_mean = [cosine(text_mean_emb, emb) if np.sum(emb)!=0 else math.inf for emb in self.title_mean]
    #     results_blue_feat = [bleu_string_distance(lemmas(feat), lemmas(filter_nlp(doc)), (0.3, 0.7)) for feat in self.feat_nlped]
    #     print("features calculated")

    #     results_blue_title = [bleu_string_distance(lemmas(title), lemmas(filter_nlp_title(doc)), (1,)) for title in self.title_nlped]
    #     print("blue calculated")

    #     scores = np.mean([results_blue_feat, results_blue_title], axis=0).tolist()
    #     results_args = np.argsort(scores).tolist()

    #     if 'num1' in money_res:
    #         log.debug('results before money '+str(len(results_args)))
    #         results_args = [idx for idx in results_args if price(self.data[idx])>=money_res['num1'] and price(self.data[idx])<=money_res['num2']]
    #         log.debug('results after money '+str(len(results_args)))
            
    #     # fetch_data = [self.data[idx] for idx in results_args[start:stop+1]]
    #     ret = {
    #         'results_args': results_args,
    #         'scores': scores
    #     }
    #     return json.dumps(ret)
        
    # # def train(self, data):
    #     print(str(data))
    #     pass

    # def fit(self, *args, **kwargs):
    #     super().__init__(**kwargs)  # self.opt initialized in here

    #     print(self.opt.pop('category'))
    #     # self.tokenizer = self.opt.pop('tokenizer')
    #     pass

    def rank_items(doc, money_res):
        results_blue_feat = [bleu_string_distance(lemmas(feat), lemmas(filter_nlp(doc)), (0.3, 0.7)) for feat in self.feat_nlped]
        print("features calculated")

        results_blue_title = [bleu_string_distance(lemmas(title), lemmas(filter_nlp_title(doc)), (1,)) for title in self.title_nlped]
        print("blue calculated")

        scores = np.mean([results_blue_feat, results_blue_title], axis=0).tolist()
        results_args = np.argsort(scores).tolist()

        if 'num1' in money_res:
            log.debug('results before money '+str(len(results_args)))
            results_args = [idx for idx in results_args if price(self.data[idx])>=money_res['num1'] and price(self.data[idx])<=money_res['num2']]
            log.debug('results after money '+str(len(results_args)))
            
        # fetch_data = [self.data[idx] for idx in results_args[start:stop+1]]
        ret = {
            'intent': 'catalog',
            'results_args': results_args,
            'scores': scores
        }
        return json.dumps(ret)
        

    def classify_faq(query_nlped):
        ques = list(self.faq_js.keys())
        ans = list(self.faq_js.values())
    
        items_meaned = [mean_transform([nlp(text)])[0] for text in ques]
        query_meaned = mean_transform([query_nlped])[0]
        results = [cosine(query_meaned, emb) if np.sum(emb)!=0 else math.inf for emb in items_meaned]

        return results

    def classify_intent(query_nlped):

        query_meaned = mean_transform([query_nlped])[0]
        intents_scores = []
        for intent, sens in self.intents.items():
            for sen in sens:
                sen_meaned = mean_transform([nlp(sen)])[0]

                if np.sum(sen_nlped)!=0:
                    score = cosine(query_meaned, sen_meaned)
                else:
                    score = math.inf
            
                intents_scores.append((intent, score))

        return [score for score in intents_scores if score[1]>=0]


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


def filter_nlp_title(doc):
    return [w for w in doc if w.tag_ in ['NNP', 'NN', 'PROPN'] and not w.like_num]

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

def bleu_string_distance(q_list,a_list,weights):
  # 0 - no overlab (bad)
  # 1 - exact match (good)
    smooth = SmoothingFunction()
    return 1-sentence_bleu([q_list], a_list, weights, auto_reweigh=False, emulate_multibleu=False, smoothing_function=smooth.method1)

# print(bleu_string_distance(['i', 'love','nature'], ['i','love'],))
