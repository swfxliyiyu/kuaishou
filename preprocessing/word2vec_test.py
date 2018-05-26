from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np
import json
import time


W2V_PATH = '../../model/word2vec/word2vec.model'

def sentence2vec(sentence):
    model = Word2Vec.load(W2V_PATH)
    words_vecters = []
    for word in sentence:
        try:
            words_vecters.append(model.wv[word])
        except KeyError:
            print(word, "not in vocabulary")
            continue
    sentence_vec = np.mean(words_vecters, axis=0)
    return sentence_vec

if __name__ == '__main__':
    
    sentence2vec(s)
