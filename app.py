import logging
from logging.handlers import RotatingFileHandler
from flask import jsonify
from flask import Flask, render_template, request
from utils import *
from scipy.spatial.distance import cosine, euclidean
import numpy as np
import json
import pickle
import spacy
from spacy.tokens import Doc

#nlp = spacy.load('en', parser=False)
#nlp = spacy.load('en_vectors_web_lg', parser=False)
nlp = spacy.load('en_core_web_md', parser=False)
#nlp.vocab.vectors.from_disk('./glove')

# nlp.vocab.load_vectors('glove.6B.300d.txt')
# nlp.vocab.load_vectors_from_bin_loc("./100_bow10.bin")

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

@app.route('/classify', methods=["POST"])
def classify():
  js = request.get_json(force=True)
  app.logger.info('Request: %s' % json.dumps(js, indent=4))

  response = {}
  text_or = js['text']
  # text = preprocess(text_or)
  text = text_or

  print('text:', text)
  text_emb = emb.transform([nlp(text)], debug = True)[0]

# scores = [np.dot(text_emb, x_emb) for x_emb in X_emb]
# predicted_idx = np.argmax(scores)

  # y_predicted = []
  # y_score = []
  
  results_cos = []
  for idx, x_emb in enumerate(data_emb):

    if np.sum(x_emb) == 0:
      print('SKIP')
      continue
  
    # if y[idx] not in results_cos:
      # results_cos[y[idx]] = []
    
    results_cos.append([data_or[idx], cosine(text_emb, x_emb)])
   
  results_cos = sorted(results_cos,key=lambda x: x[1])
  
  # for key, value in results_cos.iteritems():
  #   results_min_cos[key] = np.nanmin(value) 

  # results_l = [(key, value) for key, value in results_min_cos.iteritems()]
  # results_l = sorted(results_l,key=lambda x: x[1])
    
  # score = results_l[0][1]
  # label = results_l[0][0]
  # syn = dataset[label][0]

  # if score > 0.19:
  #   syn = 'Out of domain'
  #   response['closest'] = label
  #   label = '-1'
  # else:
  #   if label == '26':
  #     c_scores = []
  #     for c_idx, comp in enumerate(work_with):
  #       c_scores.append(bleu_string_distance(comp, text))
  #     label += '_'+str(np.argmax(c_scores))
  #     syn +=' '+ work_with[np.argmax(c_scores)]
  
  # response['text'] = text_or
  # response['class'] = label
  # response['score'] = 1.0-score
  # response['paraphrase'] = syn
  
  # app.logger.info('Response: %s' % json.dumps(response, indent=4))

  return jsonify(results_cos[:5])

@app.route('/test')
def test():
  return "server is up"

if __name__=='__main__':

  data_or = load_data("./Amazon-E-commerce-Data-set/Data-sets/amazondata_Home_32865 668.txt")
  print("data loaded")

  with open("processed.pickle", "rb") as handle:
    doc_bytes, vocab_bytes = pickle.load(handle)
    print('pickle was loaded')

  nlp.vocab.from_bytes(vocab_bytes)
  docs = [Doc(nlp.vocab).from_bytes(b) for b in doc_bytes]
  print(len(docs))

  # docs_text = [doc.text for doc in docs]

  emb = MeanEmbeddingVectorizerSpacy()
  # emb = TfidfEmbeddingVectorizer()

  emb.fit(docs)
  data_emb = emb.transform(docs)
  print('tfidfed')

  handler = RotatingFileHandler('./foo.log', maxBytes=100000, backupCount=1)
  handler.setLevel(logging.INFO)
  app.logger.addHandler(handler)
  app.run(host='0.0.0.0', port=3030)
