# -*- coding: utf-8 -*-
'''
 @Time    : 18-6-8 下午3:08
 @Author  : sunhongru
 @Email   : sunhongru@sensetime.com
 @File    : text_lda.py
 @Software: PyCharm
'''

import pandas as pd
import numpy as np
import collections
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def text_lda(max_iter, n_topics):
    print('text lda topic {}'.format(n_topics))
    df_train_text = pd.read_csv('../../data/train/train_text.txt', header=None, names=['photo_id', 'text'], sep='\t')
    df_test_text = pd.read_csv('../../data/test/test_text.txt', header=None, names=['photo_id', 'text'], sep='\t')
    df_text = pd.concat([df_train_text, df_test_text], axis=0, ignore_index=True)
    print('num of photo_id', len(df_text))
    df_text = df_text.drop(index=df_text[df_text['text'] == '0'].index).reset_index()
    print('num of photo_id with text', len(df_text))

    # count sentence
    df_sentences = df_text['text'].apply(lambda t: t.split(','))
    df_slen = pd.DataFrame(df_sentences.apply(lambda t: len(t)))
    print('sentence len:', df_slen.describe())

    # count word
    word_box = []
    for sentence in df_sentences.values:
        word_box.extend(sentence)
    word_counter = collections.Counter(word_box)

    df_word = pd.DataFrame(list(zip(word_counter.keys(), word_counter.values())), columns=['word', 'count'])
    print('word count:', df_word.describe())
    # print(df_word.head())
    # print(df_text['text'].tolist())

    #lda
    corpus = df_text['text'].tolist()
    cntVector = CountVectorizer(max_df=0.95, min_df=2,
                                # max_features=100000
                                )
    cntTf = cntVector.fit_transform(corpus)
    print(len(cntVector.get_feature_names()))

    lda = LatentDirichletAllocation(n_topics=n_topics,
                                    max_iter=max_iter,
                                    learning_method='batch',
                                    evaluate_every=5,
                                    n_jobs=10,
                                    verbose=1)
    docres = lda.fit_transform(cntTf)
    print(docres.shape)
    # print(docres)
    print(lda.perplexity(cntTf))
    # print(lda.components_)
    column_names = ['topic_' + str(i) for i in range(1, n_topics+1)]
    df_text_lda = pd.DataFrame(docres, columns=column_names)
    df_text_lda['photo_id'] = df_text['photo_id']
    df_text_lda.to_pickle('../../data/text_lda_{}.pkl'.format(n_topics))
    print('doc size:', docres.shape)
    print(df_text.shape)
    print(df_text_lda)


if __name__ == '__main__':

    text_lda(max_iter=1500, n_topics=10)
    print('text lda finish')
