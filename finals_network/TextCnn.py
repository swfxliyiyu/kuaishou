import random

import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import pandas as pd
import numpy as np


class TextCnn(object):
    def __init__(self, input, words_embedding, latent_dim, embed_size, is_train, sequence_length, drop_keep_prob,
                 filter_sizes=(2, 3, 4), num_filters=32):
        self.input = input
        self.words_embedding = words_embedding
        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_filters_total = self.num_filters * len(filter_sizes)
        self.embed_size = embed_size
        self.initializer = tf.glorot_uniform_initializer()
        self.sequence_length = sequence_length
        self.dropout_keep_prob = drop_keep_prob

        self.instantiate_weights()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"):  # embedding matrix
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.latent_dim],
                                                initializer=self.initializer)  # [embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",
                                                shape=[self.latent_dim])  # [label_size] #ADD 2017.06.09

    def get_bag_of_words_emb(self):
        self.embedded_words = tf.nn.embedding_lookup(self.words_embedding,
                                                     self.input)  # [None,sentence_length,embed_size]
        self.pooled = tf.reduce_max(self.embedded_words, axis=1)  # [None, embed_size
        self.act = tf.nn.softsign(self.pooled)
        return self.act

    def get_text_cnn_emb(self):
        self.embedded_words = tf.nn.embedding_lookup(self.words_embedding,
                                                     self.input)  # [None,sentence_length,embed_size]
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words,
                                                           -1)  # [None,sentence_length,embed_size,1). expand
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                # ====>a.create filter
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters],
                                         initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # ====>c. apply nolinearity
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b),
                               "relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID',
                                        name="pool")  # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled_outputs.append(pooled)
        self.h_pool = tf.concat(pooled_outputs,
                                3)  # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        self.h_pool_flat = tf.reshape(self.h_pool, [-1,
                                                    self.num_filters_total])  # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)

        # 4.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]
        self.h_drop = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)
        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop,
                               self.W_projection) + self.b_projection  # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
            logits = tf.nn.relu(logits)
        return logits
