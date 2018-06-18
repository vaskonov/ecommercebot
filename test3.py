import pickle
import spacy
from spacy.tokens import Doc
from utils import *

#nlp = spacy.load('en_vectors_web_lg', parser=False)
nlp = spacy.load('en_core_web_md', parser=False)

# data = load_data("./Amazon-E-commerce-Data-set/Data-sets/amazondata_Phones_1984 40.txt")
data = load_data("./Amazon-E-commerce-Data-set/Data-sets/amazondata_Home_32865 668.txt")
sens = [x['Title'] if 'Title' in x else '' for x in data]

docs = nlp.pipe(sens)

print('nlped')

doc_bytes = [doc.to_bytes() for doc in docs]
# doc_bytes = [doc.to_bytes(tensor=False, user_data=False) for doc in docs]
vocab_bytes = nlp.vocab.to_bytes()
  
with open("processed.pickle","wb") as handle:
	pickle.dump((doc_bytes, vocab_bytes), handle)

print('dumped')

# with open("processed.pickle", "rb") as handle:
# 	doc_bytes, vocab_bytes = pickle.load(handle)
# 	print('pickle was loaded')

# nlp.vocab.from_bytes(vocab_bytes)
# docs = [Doc(nlp.vocab).from_bytes(b) for b in doc_bytes]
# print(len(docs))

# for to in docs[2]:
# 	print(to.lemma_)
