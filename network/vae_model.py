from __future__ import division
from __future__ import print_function
import os.path

import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

class VAE(object):
    def __init__(self, input_dim, hidden_encoder_dim, hidden_decoder_dim, latent_dim, lam, lr):
        self.input_dim = input_dim
        self.hidden_encoder_dim = hidden_encoder_dim
        self.hidden_decoder_dim = hidden_decoder_dim
        self.latent_dim = latent_dim
        self.lam = lam

        self.x = tf.placeholder("float", shape=[None, input_dim])
        self.l2_loss = tf.constant(0.0)

        # add op for merging summary
        # self.summary_op = tf.summary.merge_all()

        self.n_steps = 100

        mu_encoder, logvar_encoder = self.vae_encoder()
        self.z = self.sampler(mu_encoder, logvar_encoder)
        x_hat = self.vae_decoder()
        self.loss, regularized_loss = self.vae_loss(x_hat, logvar_encoder, mu_encoder)
        self.loss_summ = tf.summary.scalar("lowerbound", self.loss)
        self.train_step = tf.train.AdamOptimizer(lr).minimize(regularized_loss)

        # with tf.Session() as self.sess:
        self.sess = tf.Session()
        self.summary_writer = tf.summary.FileWriter('experiment', graph=self.sess.graph)
        if os.path.isfile("save128_mean_pow/model.ckpt.meta"):
            print("Restoring saved parameters")
            #saver = tf.train.import_meta_graph('save128_sum/model.ckpt.meta')
            saver=tf.train.Saver()
            saver.restore(self.sess, tf.train.latest_checkpoint('save128_mean_pow/'))
        else:
            print("Initializing parameters")
            self.sess.run(tf.global_variables_initializer())

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0., shape=shape)
        return tf.Variable(initial)

    def vae_encoder(self,):
        W_encoder_input_hidden = self.weight_variable([self.input_dim, self.hidden_encoder_dim])
        b_encoder_input_hidden = self.bias_variable([self.hidden_encoder_dim])
        self.l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

        # Hidden layer encoder
        hidden_encoder = tf.nn.relu(tf.matmul(self.x, W_encoder_input_hidden) + b_encoder_input_hidden)

        W_encoder_hidden_mu = self.weight_variable([self.hidden_encoder_dim, self.latent_dim])
        b_encoder_hidden_mu = self.bias_variable([self.latent_dim])
        self.l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

        # Mu encoder
        mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

        W_encoder_hidden_logvar = self.weight_variable([self.hidden_encoder_dim, self.latent_dim])
        b_encoder_hidden_logvar = self.bias_variable([self.latent_dim])
        self.l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

        # Sigma encoder
        logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar
        return mu_encoder, logvar_encoder

    def sampler(self, mu_encoder, logvar_encoder):
        # Sample epsilon
        epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')
        # Sample latent variable
        std_encoder = tf.exp(0.5 * logvar_encoder)
        z = mu_encoder + tf.multiply(std_encoder, epsilon)
        return z

    def vae_decoder(self,):
        W_decoder_z_hidden = self.weight_variable([self.latent_dim, self.hidden_decoder_dim])
        b_decoder_z_hidden = self.bias_variable([self.hidden_decoder_dim])
        self.l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

        # Hidden layer decoder
        hidden_decoder = tf.nn.relu(tf.matmul(self.z, W_decoder_z_hidden) + b_decoder_z_hidden)

        W_decoder_hidden_reconstruction = self.weight_variable([self.hidden_decoder_dim, self.input_dim])
        b_decoder_hidden_reconstruction = self.bias_variable([self.input_dim])
        self.l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)
        x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction
        return x_hat

    def vae_loss(self, x_hat, logvar_encoder, mu_encoder):
        KLD = -0.0001 * tf.reduce_mean(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder),
                                   reduction_indices=1)
        #BCE = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=self.x), reduction_indices=1)

        BCE = tf.reduce_mean(tf.pow(self.x - x_hat, 2), reduction_indices = 1)
        loss = tf.reduce_mean(BCE + KLD)
        regularized_loss = loss + self.lam * self.l2_loss
        return loss, regularized_loss

    def sklearn_shuffle(self, data, random_state):
        if isinstance(data, list):
            l = len(data)
            res = shuffle(data, random_state=random_state)
            return res[:l], res[-1]
        else:
            return shuffle(data, random_state=random_state)

    def train_on_batch(self, train):
        feed_dict = {self.x: np.array(train['visual'].tolist())}
        _, cur_loss = self.sess.run([self.train_step, self.loss], feed_dict=feed_dict)
        return _, cur_loss

    def evaluate(self, test):
        feed_dict = {self.x: np.array(test['visual'].tolist())}
        mid_layer = self.sess.run([self.z], feed_dict=feed_dict)
        test_loss = self.sess.run([self.loss], feed_dict=feed_dict)
        print(test_loss)
        return mid_layer[0]

    def fit(self, train, test, batch_size, min_display):
        best_loss = np.inf
        # n_samples = train.shape[0]
       # iters = (n_samples - 1) // batch_size + 1
       # print('n_samples, iters:', n_samples, iters)
        start_time = time.time()

        if train is not None:
            n_samples = train.shape[0]
            iters = (n_samples - 1) // batch_size + 1
            print('n_samples, iters:', n_samples, iters)
            for step in range(1, self.n_steps):
                train = self.sklearn_shuffle(train, 2018)
                for iter in range(iters):
                    batch_x = train.iloc[iter * (batch_size): (iter + 1) * batch_size, :]
                    _, cur_loss = self.train_on_batch(batch_x)
                    if iter % min_display == 0:
                        total_time = time.time() - start_time
                        #if test is None:
                        print("Epoch {0: 2d} Step {1: 4d}: tr_loss {2: 0.6f} tr_time {3: 0.1f}".format(step, iter,
                                                                                                           cur_loss,
                                                                                                           total_time))
                        # else:
                        #     _, val_loss = evaluate(test, train_step, loss)
                        #     print(
                        #         "Epoch {0: 2d} Step {1: 4d}: tr_loss {2: 0.6f} va_loss {3: 0.6f} tr_time {4: 0.1f}".format(
                        #             step, iter, cur_loss, val_loss, total_time))
                        if cur_loss < best_loss:
                            best_loss = cur_loss
                            save_path = tf.train.Saver().save(self.sess, "save128_mean_pow/model.ckpt")
                            # self.summary_writer.add_summary(summary_str, step)
        if test is not None:
            print('test-start!')
            print(test)
            test = test.rename(columns={'key': 'pid'})
            n_samples = test.shape[0]
            iters = n_samples // batch_size
            print(n_samples, iters)
            mid_layer = []
            for i in range(iters):
                batch_x = test.iloc[i * (batch_size): (i + 1) * batch_size, :]
                mid_layer_x = self.evaluate(batch_x)
                print(i, mid_layer_x, mid_layer_x.shape[1])
                mid_layer.extend(mid_layer_x)
            print(test.shape[0], len(mid_layer))
            test['mid_layer'] = mid_layer
            test[['pid', 'mid_layer']].to_pickle('../data/visual_feature/vae128_visual_feature_mean_pow.pkl')
