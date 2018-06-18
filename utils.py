import os
import re
import math
import numpy as np
from numpy import linalg as la
import spacy
import json
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.metrics.distance import *
from nltk.translate.bleu_score import *

def preprocess(sen):
  sen_prep = sen.lower()
  sen_prep = re.sub('(^|\s)ha(\s|$)', '', sen_prep)  
  sen_prep = re.sub('(^|\s)alex(\s|$)', '', sen_prep)  
  sen_prep = re.sub('(^|\s)helloalex(\s|$)', '', sen_prep)  
  sen_prep = re.sub('(^|\s)fb(\s|$)', ' facebook ', sen_prep)  
  sen_prep = re.sub('\?', '', sen_prep)  
  sen_prep = re.sub('\'s', ' is ', sen_prep)  
  sen_prep = re.sub('\'ve', ' have ', sen_prep)  
      
  return sen_prep

#  X_prep = []
#
#  for index, x in enumerate(X):
#    X_prep.append(x.lower().replace('helloalex', 'microsoft').replace('alex','microsoft'))
#  return X_prep

def load_data(folder):
  data = []
  all_files = []

  if os.path.isdir(folder):
    for root, dirs, files in os.walk(folder):
      files = [folder+file for file in files]
      all_files += files
  else:
     all_files.append(folder)

  print("Files to load:", all_files)
  for file in sorted(all_files):
    elem = {}
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
      for line in f:
        if line[:4] == "ITEM":
          if len(elem.keys())>0:
            data.append(elem)
          elem = {}
        else:
          ro = line.strip().split("=")
          if len(ro) == 2:
            if ro[0] in elem:
              elem[ro[0]] += " " + ro[1]
            else:
              elem[ro[0]] = ro[1]
  return data

#def filwords(words):
#  res = []
#  words_parsed = nlp(words)
#  for word in words_parsed:
#    if word.pos_ in ['NOUN', 'VERB', 'ADJ'] and word.tag_ not in ['WP', 'VBZ', 'VBP', 'MD']:
#                 if word.lemma_  not in ['be', 'do']:
#      res.append(word.lemma_)
#             else:
#                 res.append('<NOSTR>')
#  return res

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

def lemma(sen):
  lemma_out = []
  words = nlp(sen)
  for word in words:
    lemma_out.append(word.lemma_)

  return ' '.join(lemma_out)

class MeanEmbeddingVectorizer(object):
  def __init__(self, word2vec):
    self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
    self.dim = len(word2vec.itervalues().next())

  def fit(self, X, y):
    return self

  def transform(self, X):
    return np.array([
            np.mean([self.word2vec[w] for w in filwords(words) if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
  def __init__(self):
    #self.word2vec = word2vec
    self.word2weight = None
    # if a text is empty we should return a vector of zeros
    # with the same dimensionality as all the other vectors
    #self.dim = len(word2vec.itervalues().next())

#     def fit(self, X, y):
#         return self
    
#     def transform(self, X):
#         return np.array([
#             np.mean([self.word2vec[w] for w in filwords(words) if w in self.word2vec]
#                     or [np.zeros(self.dim)]
#                     , axis=0)
#              for words in X        
#         ])
  def fit(self, X):
    tfidf = TfidfVectorizer(analyzer=lambda x: x)
    tfidf.fit(X)
    # if a word was never seen - it must be at least as infrequent
    # as any of the known words - so the default idf is the max of 
    # known idf's
    max_idf = max(tfidf.idf_)
    self.word2weight = defaultdict(
      lambda: max_idf,
      [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

    return self

  def transform(self, X):
    out = []
    for words in X:
      row = []
      for w in filter_nlp(words):
        row.append(w.vector*self.word2weight[w])
      if len(row)!=0:
        row_mean = np.mean(row, axis=0)
      else:
        row_mean = np.zeros(384)
      out.append(row_mean)
    return np.array(out)

class MeanEmbeddingVectorizerSpacy(object):
  def fit(self, X):
    return self
    
  def transform(self, X, debug = False):
    out = []
    for words in X:
      row = []
      for w in filter_nlp(words):
        if debug:
          print(w)
          print(w.lemma_)
          print(w.vector)
          print(len(w.vector))

        row.append(w.vector)
      if len(row)!=0:
        row_mean = np.mean(row, axis=0)
      else:
        row_mean = np.zeros(384)
      out.append(row_mean)
    return np.array(out)

smooth1 = SmoothingFunction()
def bleu_string_distance(q_str,a_str, smooth = smooth1.method1):

  if type(q_str) != 'str':
    q_str = q_str.text

  if type(a_str) != 'str':
    a_str = a_str.text

  q_list = q_str.lower().split(' ')
  a_list = a_str.lower().split(' ')

  glob = []

  for q_word in q_list:
    r = []
    for a_word in a_list:
      hyp_list = list(q_word)
      ref_list = list(a_word)
      # r.append(float(edit_distance(q_word,a_word))/max(len(q_word), len(a_word)))
      r.append(float(sentence_bleu([hyp_list], ref_list, weights=(0.1, 0.15, 0.2, 0.25, 0.3), auto_reweigh=False, emulate_multibleu=False, smoothing_function=smooth)))

    glob.append(max(r))
  return sum(glob)/float(len(glob))

from scipy.spatial.distance import *

class MeanEmbeddingVectorizerSpacyTuple:
  def __init__(self, dist):
    self.dist = dist

  def fit(self, X, y):
    return self

  def transform(self, X):
    return [[self.dist(wtuple[0].vector, wtuple[1].vector)] for wtuple in X]

class BleuDistance():
  def __init__(self, smooth):
    self.smooth = smooth

  def fit(self, X, y):
    return self

  def transform(self, X):
    return [[bleu_string_distance(wtuple[0], wtuple[1], self.smooth)] for wtuple in X]
