import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
import pandas as pd
import numpy as np


class VAE(object):
    def __init__(self, input_dim, hidden_encoder_dim, latent_dim, lam=0.001, kld_loss=0.001):
        self.input_dim = input_dim
        self.hidden_encoder_dim = hidden_encoder_dim
        self.latent_dim = latent_dim
        self.lam = lam
        self.kld_loss = kld_loss
        self.l2_loss = tf.constant(0.0)

    def weight_variable(self, shape, name):
        weight = tf.get_variable(shape=shape, dtype=tf.float32, initializer=tf.glorot_uniform_initializer(), name=name)
        return weight

    def bias_variable(self, shape, name):
        bias = tf.get_variable(shape=shape, initializer=tf.zeros_initializer(),
                               dtype=tf.float32, name=name)
        return bias

    def vae_encoder(self, x):
        W_encoder_input_hidden = self.weight_variable([self.input_dim, self.hidden_encoder_dim], 'hidden_weight')
        b_encoder_input_hidden = self.bias_variable([self.hidden_encoder_dim], 'hidden_bias')
        self.l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

        # Hidden layer encoder
        hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)

        W_encoder_hidden_mu = self.weight_variable([self.hidden_encoder_dim, self.latent_dim], 'hidden_mu_weight')
        b_encoder_hidden_mu = self.bias_variable([self.latent_dim], 'hidden_mu_bias')
        self.l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

        # Mu encoder
        mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

        W_encoder_hidden_logvar = self.weight_variable([self.hidden_encoder_dim, self.latent_dim],
                                                       'hidden_logvar_weight')
        b_encoder_hidden_logvar = self.bias_variable([self.latent_dim], 'hidden_logvar_bias')
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

    def get_vae_embbeding(self, x):
        mu_encoder, log_encoder = self.vae_encoder(x)
        z = self.sampler(mu_encoder, log_encoder)
        KLD = -tf.reduce_mean(1 + log_encoder - tf.pow(mu_encoder, 2) - tf.exp(log_encoder),
                                              reduction_indices=1)
        loss = self.lam * self.l2_loss + self.kld_loss * tf.reduce_mean(KLD)
        return z, loss
