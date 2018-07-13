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

        with open('./faq.json', 'r') as f:
            self.faq_js = json.load(f)

        with open('./intents.json', 'r') as f:
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
        log.debug('doc before money:' + str(doc))
        log.debug(str([w.tag_  for w in doc]))
        for ent in doc.ents:
            log.debug(ent.text, ent.start_char, ent.end_char, ent.label_)

        doc, money_res = find_money(doc)
        
        print('doc after money:', doc)
        # print('filter_nlp:', filter_nlp(doc))
        
        if 'num1' not in money_res:
            results = self.classify_faq(doc)

            results_arg = np.argsort(results)
            print('Results for faq: ', str(results_arg), str(results))

            if results[results_arg[0]] < 0.2:

                print('FAQ question is detected with answer', list(self.faq_js.values())[results_arg[0]])

                return json.dumps({
                    'intent': 'faq',
                    'text': list(self.faq_js.values())[results_arg[0]]
                    })
            else:

                # doc = filter_nlp(doc)
                print('Going to intent classification, filtered:', doc)

                nns = [w for w in doc if w.tag_ in ['NNP', 'NN', 'JJ', 'PROPN', 'CD']]
                sal_phrase = [w for w in doc if w.tag_ not in ['NNP', 'NN', 'JJ', 'PROPN', 'CD']]

                print('salient phrase for intent', sal_phrase)
                print('nns for catalogue', nns)

                if len(sal_phrase) == 0:
                    print("salient is empty, go to rank")
                    return self.rank_items(nns, money_res)

                scores = sorted(self.classify_intent(sal_phrase), key=lambda x: x[1])
                print('intent classification:', str(scores))

                if scores[0][1] < 0.2 and scores[0][0] == 'payment':

                    print('payment detected')
                    return json.dumps({
                        'intent': 'payment'
                        })

                else:
                    return self.rank_items(nns, money_res)
        else:
            nns = [w for w in doc if w.tag_ in ['NNP', 'NN', 'JJ', 'PROPN']]
            return self.rank_items(nns, money_res)

    def rank_items(self, doc, money_res):
        results_blue_title = [bleu_string_distance(lemmas(title), lemmas(filter_nlp_title(doc)), (1,)) for title in self.title_nlped]
        print("blue calculated")

        results_blue_feat = [bleu_string_distance(lemmas(feat), lemmas(filter_nlp(doc)), (0.3, 0.7)) if results_blue_title[idx]<1 else 1 for idx, feat in enumerate(self.feat_nlped)]
        print("features calculated")

        scores = np.mean([results_blue_feat, results_blue_title], axis=0).tolist()
        raw_scores = [(score, len(self.data[idx]['Title'])) for idx, score in enumerate(scores)]
        raw_scores_ar = np.array(raw_scores, dtype=[('x', 'float_'), ('y', 'int_')])
        results_args = np.argsort(raw_scores_ar, order=('x','y')).tolist()

        # results_args = np.argsort(scores).tolist()

        print('minimal score:', np.min(scores))
        print([raw_scores[idx] for idx in results_args[:10]])
        print('10th score:', scores[results_args[10]])
        print('20th score:', scores[results_args[20]])
        print('30th score:', scores[results_args[30]])

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

        print(self.data[results_args[0]])
        return json.dumps(ret)
        

    def classify_faq(self, query_nlped):
        ques = list(self.faq_js.keys())
        ans = list(self.faq_js.values())
    
        items_meaned = [self.mean_transform([nlp(text)])[0] for text in ques]
        query_meaned = self.mean_transform([query_nlped])[0]
        results = [cosine(query_meaned, emb) if np.sum(emb)!=0 else math.inf for emb in items_meaned]

        return results

    def classify_intent(self, query_nlped):

        query_meaned = self.mean_transform([query_nlped])[0]
        intents_scores = []
        for intent, sens in self.intents.items():
            for sen in sens:
                sen_meaned = self.mean_transform([nlp(sen)])[0]

                if np.sum(sen_meaned)!=0:
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
        
                vec = self.glove_model([[w.lemma_]])
                
                if np.sum(vec)==0:
                    print('mean_transform:', w.lemma_, ':was not found')

                row.append(vec)
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
        if word.tag_ not in ['MD', 'SP', 'DT', 'TO']:# and word.is_stop == False and word.is_punct == False: #VBP VBZ
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