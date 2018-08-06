# coding=utf-8
from __future__ import print_function, division
import pandas as pd
import tensorflow as tf
import time
import numpy as np

from VAE_Encoder import VAE
from utils import get_sample_num, new_variable_initializer, sklearn_shuffle
import os
from tensorflow.python.ops import random_ops


class Model(object):
    def __init__(self, num_user, num_recent_item, num_words, one_hots_dims, dim_k, att_dim_k, reg,
                 att_reg, lr, prefix,
                 dim_num_feat,
                 user_emb_feat,
                 dim_lda,
                 seed=1024,
                 use_deep=True, deep_dims=(256, 128, 64), dim_hidden_out=(64, 32, 16), checkpoint_path=None):
        self.att_reg = None
        self.seed = seed
        if checkpoint_path and checkpoint_path.count('/') < 2:
            raise ValueError('checkpoint_path must be dir/model_name format')
        self.checkpoint_path = checkpoint_path
        self.train_flag = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))

        with self.graph.as_default():
            tf.set_random_seed(self.seed)  # 设置随机种子
            self.global_step = tf.Variable(
                0, dtype=tf.int32, trainable=False, name='global_step')
            self.sample_weight = tf.placeholder(
                tf.float32, shape=[None, ], name='sample_weight')
        self.test_datas = {}
        self.val_datas = {}

        self.num_user = num_user
        self.num_history_item = num_recent_item
        self.num_words = num_words
        self.dim_num_feat = dim_num_feat
        self.dim_k = dim_k
        self.att_dim_k = att_dim_k
        self.reg = reg
        self.att_reg = att_reg
        self.lr = lr
        self.prefix = prefix
        self.one_hots_dims = one_hots_dims
        self.use_deep = use_deep
        self.deep_dims = deep_dims
        self.dim_hidden_out = dim_hidden_out
        self.user_emb_feat = user_emb_feat
        self.dim_lda = dim_lda
        self.num_faces_cols = 31
        self.dim_usr_cf_emb = 96  # 用户协同过滤embedding维度
        self.num_cross_layer = 5

        self._build_graph()

    def _get_optimizer_loss(self, ):
        return self.loss

    def _build_graph(self):
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            self._create_placeholders()
            self._create_weights()
            self._forward_pass()
            self._create_loss()

    def _create_placeholders(self, ):

        self.labels = tf.placeholder(shape=[None], dtype=tf.float32, name='labels')  # [batch_size]

        self.user_indices = tf.placeholder(shape=[None], dtype=tf.float32, name='user_indices')  # [batch_size]

        self.item_words_indices_a = tf.placeholder(shape=[None, 2], dtype=tf.float32,
                                                   name='item_words_indices_a')  # [num_indices, dim]

        self.item_words_values_a = tf.placeholder(shape=[None], dtype=tf.float32,
                                                  name='item_words_values_a')  # [num_indices]

        self.visual_emb_feat = tf.placeholder(shape=[None, 2048], dtype=tf.float32,
                                              name='visual_features')

        self.words_lda = tf.placeholder(shape=[None, self.dim_lda], dtype=tf.float32,
                                        name='words_lda')

        # 其余物品的one-hot信息
        self.one_hots_a = tf.placeholder(shape=[None, len(self.one_hots_dims)],
                                         dtype=tf.float32, name='one_hots_a')  # [batch_size, num_one_hots]

        self.batch_size = tf.placeholder(shape=[], dtype=tf.int32, name='batch_size')  # num of batch size

        self.train_phase = tf.placeholder(shape=[], dtype=tf.bool, name='train_phase')

        # embedding的dropout
        self.dropout_emb = tf.placeholder(shape=[], dtype=tf.float32, name='dropout_emb')

        # deep特征
        if self.use_deep:
            self.num_features = tf.placeholder(shape=[None, self.dim_num_feat], dtype=tf.float32,
                                               name='num_features')  # [batch_size, num_features_dims]
            self.face_num = tf.placeholder(shape=[None, self.num_faces_cols], dtype=tf.float32,
                                           name='face_num_features')  # [batch_size, num_faces_dims]

            self.dropout_deep = tf.placeholder(shape=[], dtype=tf.float32, name='dropout_deep')

    def _create_weights(self):

        # Embedding
        self.Wu_Emb = tf.get_variable(shape=[self.num_user, self.dim_k], initializer=tf.glorot_uniform_initializer(),
                                      dtype=tf.float32, name='user_embedding')

        self.W_usr_feat_emb = tf.get_variable(shape=[self.dim_usr_cf_emb, self.dim_k],
                                              initializer=tf.glorot_uniform_initializer(),
                                              dtype=tf.float32, name='user_feat_embedding')

        self.Wwords_Emb = tf.get_variable(shape=[self.num_words, self.dim_k],
                                          initializer=tf.glorot_uniform_initializer(),
                                          dtype=tf.float32, name='words_embedding')

        self.W_LDA_emb = tf.get_variable(shape=[self.dim_lda, self.dim_k],
                                         initializer=tf.glorot_uniform_initializer(),
                                         dtype=tf.float32, name='LDA_embedding')

        self.W_one_hots = []
        for i in range(len(self.one_hots_dims)):
            W_temp = tf.get_variable(shape=[self.one_hots_dims[i], self.dim_k],
                                     initializer=tf.glorot_uniform_initializer(),
                                     dtype=tf.float32, name='one_hot_{}'.format(i))
            self.W_one_hots.append(W_temp)

        self.W_Ctx = tf.get_variable(shape=[self.dim_num_feat, self.dim_k],
                                     initializer=tf.glorot_uniform_initializer(),
                                     dtype=tf.float32, name='context_embedding')

        # Item one-hot features attention
        self.Wu_oh_Att = tf.get_variable(shape=[self.dim_k, self.att_dim_k],
                                         initializer=tf.glorot_uniform_initializer(),
                                         dtype=tf.float32, name='user_attention')

        self.Wctx_Att = tf.get_variable(shape=[self.dim_k, self.att_dim_k],
                                        initializer=tf.glorot_uniform_initializer(),
                                        dtype=tf.float32, name='context_attention')

        self.Woh_Att = []
        for i in range(len(self.one_hots_dims)):
            W_temp = tf.get_variable(shape=[self.dim_k, self.att_dim_k],
                                     initializer=tf.glorot_uniform_initializer(),
                                     dtype=tf.float32, name='oh_attention_{}'.format(i))
            self.Woh_Att.append(W_temp)

        self.WW_Att = tf.get_variable(shape=[self.dim_k, self.att_dim_k],
                                      initializer=tf.glorot_uniform_initializer(),
                                      dtype=tf.float32, name='words_attention')

        self.W_LDA_Att = tf.get_variable(shape=[self.dim_k, self.att_dim_k],
                                         initializer=tf.glorot_uniform_initializer(),
                                         dtype=tf.float32, name='LDA_attention')

        self.b_oh_Att = tf.get_variable(shape=[self.att_dim_k], initializer=tf.zeros_initializer(),
                                        dtype=tf.float32, name='bias_attention')
        self.w_oh_Att = tf.get_variable(shape=[self.att_dim_k, 1], initializer=tf.glorot_uniform_initializer(),
                                        dtype=tf.float32, name='weight_attention')
        self.c_oh_Att = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(),
                                        dtype=tf.float32, name='bias2_attention')

        self.bias = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(), dtype=tf.float32, name='bias')

        self.bias_u = tf.get_variable(shape=[self.num_user, 1], initializer=tf.zeros_initializer(), dtype=tf.float32,
                                      name='bias_u')

        # deep 参数
        if self.use_deep:

            # self.Wu_deep_emb = tf.get_variable(shape=[self.num_user, self.dim_k],
            #                                    initializer=tf.glorot_uniform_initializer(),
            #                                    dtype=tf.float32, name='user_deep_emb')
            # self.Wwords_deep_emb = tf.get_variable(shape=[self.num_words, self.dim_k],
            #                                        initializer=tf.glorot_uniform_initializer(),
            #                                        dtype=tf.float32, name='words_deep_emb')
            self.W_deep_one_hots = []
            for i in range(len(self.one_hots_dims)):
                W_temp = tf.get_variable(shape=[self.one_hots_dims[i], self.dim_k],
                                         initializer=tf.glorot_uniform_initializer(),
                                         dtype=tf.float32, name='deep_one_hot_{}'.format(i))
                self.W_deep_one_hots.append(W_temp)

    def _batch_norm_layer(self, x, train_phase, scope_bn):
        with tf.variable_scope(scope_bn):
            beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
            # axises = np.arange(len(x.shape) - 1)
            batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(train_phase, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def _highway_layer(self, x, dim, name):
        W_t = tf.get_variable(shape=[dim, dim], initializer=tf.glorot_uniform_initializer(), dtype=tf.float32,
                              name=name + '_W_t')
        b_t = tf.get_variable(shape=[dim], initializer=tf.glorot_uniform_initializer(), dtype=tf.float32,
                              name=name + '_b_t')
        t = tf.nn.sigmoid(tf.matmul(x, W_t) + b_t)
        W = tf.get_variable(shape=[dim, dim], initializer=tf.glorot_uniform_initializer(), dtype=tf.float32,
                            name=name + '_W')
        b = tf.get_variable(shape=[dim], initializer=tf.glorot_uniform_initializer(), dtype=tf.float32,
                            name=name + '_b')
        h_x = tf.nn.relu(tf.matmul(x, W) + b)
        z = t * h_x + (1 - t) * x
        return z

    def _cross_layer(self, x, name):
        # x [batch_size, dim_cross]

        b = tf.get_variable(shape=[self.dim_cross], initializer=tf.glorot_uniform_initializer(), dtype=tf.float32,
                            name=name + '_b')
        x_mul_w = tf.layers.dense(inputs=x,
                                  kernel_initializer=tf.glorot_uniform_initializer(),
                                  units=1, activation=tf.nn.relu,)  # [batch_size, 1]
        output = self.cross_x0 * x_mul_w  # [batch_size, dim_cross]
        output = output + b + x
        return output

    def _forward_pass(self, ):
        # 用户的向量表示
        with tf.name_scope('user_express'):
            # 用户隐向量
            self.Usr_Emb = tf.nn.embedding_lookup(self.Wu_Emb,
                                                  tf.cast(self.user_indices, tf.int32))  # [batch_size, dim_k]
            self.Usr_Feat = tf.nn.embedding_lookup(self.user_emb_feat,
                                                   tf.cast(self.user_indices, tf.int32))  # [batch_size, dim_cf_emb]

            self.bias_usr_feat_emb = tf.get_variable(shape=[self.dim_k], initializer=tf.zeros_initializer(),
                                                     dtype=tf.float32,
                                                     name='bias_usr_feat_emb')

            self.Usr_Feat_Emb = tf.matmul(self.Usr_Feat,
                                          self.W_usr_feat_emb) + self.bias_usr_feat_emb  # [batch_size, dim_k]
            self.Usr_Feat_Emb = tf.nn.relu(self.Usr_Feat_Emb)
            # self.Usr_Feat_Emb = tf.layers.dropout(self.Usr_Feat_Emb, self.dropout_emb)
            # self.Usr_Feat_Emb = self._batch_norm_layer(self.Usr_Feat_Emb, self.train_phase, 'user_emb_bn')

            self.Usr_Expr_a = self.Usr_Emb + self.Usr_Feat_Emb  # [batch_size, dim_k]

        # 环境的向量表示
        with tf.name_scope('context_express'):
            self.bias_ctx_emb = tf.get_variable(shape=[self.dim_k], initializer=tf.zeros_initializer(),
                                                dtype=tf.float32,
                                                name='bias_ctx_emb')
            self.Ctx_Emb = tf.matmul(self.num_features, self.W_Ctx) + self.bias_ctx_emb  # [batch_size, dim_k]
            self.Ctx_Emb = self._batch_norm_layer(self.Ctx_Emb, self.train_phase, 'ctx_bn')
            self.Ctx_Emb = tf.nn.relu(self.Ctx_Emb)

        # 物品的向量表示
        with tf.name_scope('item_express'):
            self.I_Wds_a = tf.SparseTensor(indices=tf.cast(self.item_words_indices_a, dtype=np.int64),
                                           values=self.item_words_values_a,
                                           dense_shape=[tf.cast(self.batch_size, dtype=np.int64), self.num_words])
            self.att_u_a = tf.matmul(self.Usr_Expr_a, self.Wu_oh_Att)  # [batch_size, dim_att]
            self.att_ctx = tf.matmul(self.Ctx_Emb, self.Wctx_Att)
            self.att_oh = []
            self.bias_wds_emb = tf.get_variable(shape=[self.dim_k], initializer=tf.zeros_initializer(),
                                                dtype=tf.float32,
                                                name='bias_wds_emb')
            self.I_Wds_Emb_a = tf.sparse_tensor_dense_matmul(self.I_Wds_a,
                                                             self.Wwords_Emb) + self.bias_wds_emb  # [batch_size, dim_k]
            self.I_Wds_Emb_a = tf.nn.relu(self.I_Wds_Emb_a)

            self.att_I_Wds = tf.matmul(self.I_Wds_Emb_a, self.WW_Att)  # 词的attention
            self.att_I_Wds = tf.nn.relu(self.att_u_a + self.att_ctx + self.att_I_Wds + self.b_oh_Att)
            self.att_I_Wds = tf.matmul(self.att_I_Wds, self.w_oh_Att) + self.c_oh_Att
            self.att_oh.append(self.att_I_Wds)

            vae_encoder = VAE(input_dim=2048, hidden_encoder_dim=1024, latent_dim=96, lam=0.001, kld_loss=0.001)
            self.I_visual_Emb, self.vae_loss = vae_encoder.get_vae_embbeding(self.visual_emb_feat)
            # TODO
            self.I_visual_Emb = self._batch_norm_layer(self.I_visual_Emb, self.train_phase, 'vis_bn')
            self.bias_lda_emb = tf.get_variable(shape=[self.dim_k], initializer=tf.zeros_initializer(),
                                                dtype=tf.float32,
                                                name='bias_lda_emb')
            self.I_LDA_Emb = tf.matmul(self.words_lda, self.W_LDA_emb) + self.bias_lda_emb  # [batch_size, dim_k]
            self.I_LDA_Emb = tf.nn.relu(self.I_LDA_Emb)
            self.att_I_LDA = tf.matmul(self.I_LDA_Emb, self.W_LDA_Att)  # LDA的attention
            self.att_I_LDA = tf.nn.relu(self.att_u_a + self.att_ctx + self.att_I_LDA + self.b_oh_Att)
            self.att_I_LDA = tf.matmul(self.att_I_LDA, self.w_oh_Att) + self.c_oh_Att
            self.att_oh.append(self.att_I_LDA)

            self.I_One_hot_a = []
            for i in range(len(self.one_hots_dims)):
                I_Emb_temp_a = tf.nn.embedding_lookup(self.W_one_hots[i],
                                                      tf.cast(self.one_hots_a[:, i],
                                                              tf.int32))  # [batch_size, dim_k]
                att_oh_temp = tf.matmul(I_Emb_temp_a, self.Woh_Att[i])  # [batch_size, att_dim_k]
                att_temp = tf.nn.relu(
                    self.att_u_a + self.att_ctx + att_oh_temp + self.b_oh_Att)  # [batch_size, att_dim_k]
                att_temp = tf.matmul(att_temp, self.w_oh_Att) + self.c_oh_Att  # [batch_size, 1]
                self.att_oh.append(att_temp)
                self.I_One_hot_a.append(I_Emb_temp_a)
            self.att_oh = tf.nn.softmax(tf.concat(self.att_oh, axis=1))  # [batch_size, oh_dim] 第一列是词attention
            self.I_Wds_Emb_a = self.I_Wds_Emb_a * self.att_oh[:, 0:1]
            self.I_LDA_Emb = self.I_LDA_Emb * self.att_oh[:, 1:2]
            for i in range(2, len(self.one_hots_dims) + 2):
                self.I_One_hot_a[i - 2] = self.I_One_hot_a[i - 2] * self.att_oh[:, i:i + 1]

        with tf.name_scope('cross'):
            self.cross_inputs = [self.Usr_Emb, self.Usr_Feat_Emb, self.num_features, self.I_visual_Emb,
                                 self.I_Wds_Emb_a,
                                 self.I_LDA_Emb] + self.I_One_hot_a
            self.cross_x0 = tf.concat(self.cross_inputs, axis=1)  # [batch_size, self.dim_cross]
            self.cross_x0 = self._batch_norm_layer(self.cross_x0, self.train_phase, 'cross_bn')
            self.dim_cross = self.cross_x0.get_shape().as_list()[1]
            # self.cross_x0 = tf.reshape(self.cross_x0, [-1, self.dim_cross, 1])
            self.cross_xl = self.cross_x0
            for l in range(self.num_cross_layer):
                self.cross_xl = self._cross_layer(self.cross_xl, 'cross_{}'.format(l))
                self.cross_xl = tf.layers.dropout(inputs=self.cross_xl, rate=self.dropout_emb)
            self.cross_output = self.cross_xl
            # self.cross_output = tf.reshape(self.cross_xl, [-1, self.dim_cross])
        with tf.name_scope('deep'):
            if self.use_deep:
                self.I_one_hot_deep = []
                for i in range(len(self.one_hots_dims)):
                    I_Emb_temp_a = tf.nn.embedding_lookup(self.W_deep_one_hots[i], tf.cast(self.one_hots_a[:, i],
                                                                                           tf.int32))  # [batch_size, dim_k]
                    self.I_one_hot_deep.append(I_Emb_temp_a)

                #
                self.deep_input = tf.concat(
                    [self.num_features, self.visual_emb_feat] + self.I_one_hot_deep, axis=1)
                # 输入加入batch_norm
                for i, deep_dim in enumerate(self.deep_dims):
                    if i == 0:
                        self.deep_input = tf.layers.dense(inputs=self.deep_input,
                                                          kernel_initializer=tf.glorot_uniform_initializer(),
                                                          units=deep_dim, activation=tf.nn.relu, )
                        self.deep_input = self._batch_norm_layer(self.deep_input, self.train_phase,
                                                                 'deep_bn_{}'.format(i))
                    else:
                        self.deep_input = self._highway_layer(self.deep_input, deep_dim, 'deep_highway_{}'.format(i))
                    # 加入dropout
                    self.deep_input = tf.layers.dropout(inputs=self.deep_input, rate=self.dropout_deep)
                self.deep_output = tf.layers.dense(self.deep_input, 1, activation=None)
                self.deep_output = tf.reshape(self.deep_output, [-1])  # [batch_size]

        with tf.name_scope('output'):

            if self.use_deep:
                self.concated = tf.concat([self.cross_output, self.deep_input], axis=1)
            else:
                self.concated = self.cross_output

            self.hidden = self.concated
            for i, dim in enumerate(self.dim_hidden_out):
                if i == 0:
                    self.hidden = tf.layers.dense(inputs=self.hidden,
                                                  kernel_initializer=tf.glorot_uniform_initializer(),
                                                  units=dim, activation=tf.nn.relu)
                else:
                    self.hidden = self._highway_layer(self.hidden, dim, 'out_highway_{}'.format(i))
                self.hidden = tf.layers.dropout(inputs=self.hidden, rate=self.dropout_deep)
            self.bu = tf.nn.embedding_lookup(self.bias_u, tf.cast(self.user_indices, dtype=tf.int32))
            self.y_ui_a = tf.layers.dense(self.hidden, 1, activation=None) + self.bu
            self.y_ui_a = tf.reshape(tf.nn.sigmoid(self.y_ui_a), [-1])

    def _create_loss(self):

        self.biases = [self.bias, self.bu,
                       self.c_oh_Att, self.b_oh_Att,
                       self.bias_usr_feat_emb, self.bias_ctx_emb, self.bias_wds_emb, self.bias_lda_emb,
                       ]

        self.params = [self.Usr_Emb, self.Wwords_Emb, self.W_Ctx, self.W_usr_feat_emb,
                       self.W_LDA_emb, ] + self.I_One_hot_a + self.biases

        self.att_params = [self.WW_Att, self.Wu_oh_Att, self.Wctx_Att, self.W_LDA_Att,
                           self.w_oh_Att,
                           ] + self.Woh_Att

        self.loss = tf.keras.losses.categorical_crossentropy(self.labels, self.y_ui_a)
        # 正则项
        for param in self.params:
            self.loss = tf.add(self.loss, self.reg * tf.nn.l2_loss(param))
        for param in self.att_params:
            self.loss = tf.add(self.loss, self.att_reg * tf.nn.l2_loss(param))
            # self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        self.loss += self.vae_loss

    def _create_metrics(self, metric):
        """
        返回每个样本上的评分
        """
        return [self._get_optimizer_loss()]

    def _create_optimizer(self, optimizer='sgd'):
        """

        :param optimizer: str of optimizer or predefined optimizer in tensorflow
        :return: optimizer object
        """

        optimizer_dict = {'sgd': tf.train.GradientDescentOptimizer(self.lr),
                          'adam': tf.train.AdamOptimizer(self.lr),
                          'adagrad': tf.train.AdagradOptimizer(self.lr),
                          'rmsprop': tf.train.RMSPropOptimizer(self.lr),
                          'moment': tf.train.MomentumOptimizer(self.lr, 0.9),
                          'ftrl': tf.train.FtrlOptimizer(self.lr)
                          }
        if isinstance(optimizer, str):
            if optimizer in optimizer_dict.keys():
                return optimizer_dict[optimizer]
            else:
                raise ValueError('invalid optimizer name')
        elif isinstance(optimizer, tf.train.Optimizer):
            return optimizer
        else:
            raise ValueError('invalid parm for optimizer')

    def save_model(self, save_path):
        self.saver.save(self.sess, save_path + '.ckpt')

    def load_model(self, meta_graph_path, ckpt_dir=None, ckpt_path=None):
        """
        :meta_graph_path .meta文件路径
        :ckpt_dir 最新的检查点所在目录
        :ckpt_path 指定检查点
        """
        if ckpt_dir is None and ckpt_path is None:
            raise ValueError('Must specify ckpt_dir or ckpt_path')

        # restore_saver = tf.train.import_meta_graph(meta_graph_path, )

        if ckpt_path is None:
            ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
            print(ckpt_path)
        # restore_saver.restore(self.sess, ckpt_path)
        self.saver.restore(self.sess, ckpt_path)

    def compile(self, optimizer='sgd', metrics=None,
                only_init_new=False, ):
        """
        compile the model with optimizer and loss function
        :param optimizer:str or predefined optimizer in tensorflow
        ['sgd','adam','adagrad','rmsprop','moment','ftrl']
        :param loss: str  not used
        :param metrics: str ['logloss','mse','mean_squared_error','logloss_with_logits']
        :param loss_weights:
        :param sample_weight_mode:
        :param only_init_new bool
        :return:
        """
        # TODO: 添加loss
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # 根据指定的优化器和损失函数初始化
            # self.metric_list = self._create_metrics(metrics)  # 创建度量列表
            # for the use of BN,tf.get_collection get default Graph
            update_ops = self.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):  # for the use of BN
                self.op = self._create_optimizer(optimizer)
                self.optimizer = self.op.minimize(
                    loss=self._get_optimizer_loss(), global_step=self.global_step)  # 创建优化器
                # 执行初始化操作
            self.saver = tf.train.Saver()  # saver要定义在所有变量定义结束之后，且在计算图中
            if only_init_new is False:
                print("init all variables")
                init_op = tf.global_variables_initializer()
            else:
                print("init new variables")
                init_op = new_variable_initializer(
                    self.sess)  # 如果更换了优化器，需要重新初始化一些变量
            self.sess.run(init_op)

    def _item_words_indices_and_values(self, input_data):
        def func(input):
            for ix, words in enumerate(input):
                for word in words:
                    yield [ix, word]

        indices = list(func(input_data['words']))
        indices = np.asarray(indices, np.uint16)
        values = np.ones([indices.shape[0]], dtype=np.int8)
        if len(indices) == 0:
            return np.zeros([1, 2], np.int8), np.zeros([1], np.int8)
        return indices, values

    def fit(self, input_data, batch_size=1024, epochs=50, validation_data=None, shuffle=True, initial_epoch=0,
            min_display=50, max_iter=-1, drop_out_deep=0.5, drop_out_emb=0.6, save_path=None, test_data=None, ):

        n_samples = get_sample_num(input_data)
        iters = (n_samples - 1) // batch_size + 1
        self.tr_loss_list = []
        self.val_loss_list = []
        self.drop_out_deep_on_train = drop_out_deep
        self.drop_out_emb_on_train = drop_out_emb
        print(iters, "steps per epoch")
        print(batch_size, "samples per step")
        start_time = time.time()
        stop_flag = False
        self.best_loss = np.inf
        self.best_ckpt = None

        for i in range(epochs):
            if i < initial_epoch:
                continue
            if shuffle:
                input_data = sklearn_shuffle(input_data, random_state=np.random.randint(2018))
            for j in range(iters):
                print("Epoch {0: 2d} Step {1: 4d}...".format(i, j))
                if isinstance(input_data, list):
                    batch_x = [
                        item[j * batch_size:(j + 1) * batch_size] for item in input_data]
                elif isinstance(input_data, pd.DataFrame):
                    batch_x = input_data.iloc[j * batch_size:(j + 1) * batch_size, :]
                else:
                    batch_x = input_data[j * batch_size:(j + 1) * batch_size]
                loss = self.train_on_batch(
                    batch_x)
                # TODO 显示频率
                if j / iters < .5:
                    display = int(min_display * 5)
                elif j / iters < .85:
                    display = int(min_display * 2)
                else:
                    display = int(min_display / 2)
                if j % display == 0 or j == iters - 1:
                    tr_loss = loss
                    self.tr_loss_list.append(tr_loss)
                    total_time = time.time() - start_time
                    if validation_data is None:
                        print("Epoch {0: 2d} Step {1: 4d}: tr_loss {2: 0.6f} tr_time {3: 0.1f}".format(i, j, tr_loss,
                                                                                                       total_time))
                    else:
                        val_loss = self.evaluate(validation_data)
                        self.val_loss_list.append(val_loss)
                        print(
                            "Epoch {0: 2d} Step {1: 4d}: tr_loss {2: 0.6f} va_loss {3: 0.6f} tr_time {4: 0.1f}".format(
                                i, j, tr_loss, val_loss, total_time))
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            if test_data is not None:
                                self.preds = self.pred_prob(test_data)
                            if i > 1:
                                try:
                                    self.save_model(self.checkpoint_path + '.best')
                                except Exception as e:
                                    print(e)
                # self.save_model(self.checkpoint_path)
                if (i * iters) + j == max_iter:
                    stop_flag = True
                    break
            try:
                self._save_preds(test_data, self.preds, save_path)
            except Exception as e:
                print(e)
            if stop_flag:
                break

    def train_on_batch(self, input_data):  # fit a batch
        user_ids = input_data['user_indices'].values

        labels = input_data['click'].values
        onehots_1 = input_data['face_cols_01'].tolist()
        # onehots_2 = input_data[[col for col in input_data if '_01' in col]].values
        onehots = np.concatenate([onehots_1], axis=1)
        item_words_indices, item_words_values = self._item_words_indices_and_values(input_data)
        visual_emb_feat = input_data['visual'].tolist()
        words_lda = input_data['topics'].tolist()
        # num_features = np.asarray(input_data['context'].tolist())
        num_features = input_data[ctx_cols].values
        face_cols_num = input_data['face_cols_num'].tolist()
        feed_dict_ = {
            self.user_indices: user_ids,
            self.visual_emb_feat: visual_emb_feat,
            self.item_words_indices_a: item_words_indices,
            self.item_words_values_a: item_words_values,
            self.words_lda: words_lda,
            self.labels: labels,
            self.one_hots_a: onehots,
            self.batch_size: user_ids.shape[0],
            self.num_features: num_features,
            self.face_num: face_cols_num,
            self.dropout_deep: self.drop_out_deep_on_train,
            self.dropout_emb: self.drop_out_emb_on_train,
            self.train_phase: True,
        }
        # batch_size = self.sess.run([self.batch_size], feed_dict=feed_dict_)
        y, loss, _ = self.sess.run([self.y_ui_a, self.loss, self.optimizer], feed_dict=feed_dict_)
        return loss

    def train_on_split_batch(self, data, split=1):
        loss_sum = 0
        for input_data in np.array_split(data, split):
            user_ids = input_data['user_indices'].values

            labels = input_data['click'].values
            onehots_1 = np.asarray(input_data['face_cols_01'].tolist())
            # onehots_2 = input_data[[col for col in input_data if '_01' in col]].values
            onehots = np.concatenate([onehots_1], axis=1)
            item_words_indices, item_words_values = self._item_words_indices_and_values(input_data)
            visual_emb_feat = input_data['visual'].tolist()
            words_lda = np.asarray(input_data['topics'].tolist())
            # num_features = input_data[[col for col in input_data if '_N' in col]].values
            num_features = np.asarray(input_data['context'].tolist())
            face_cols_num = np.asarray(input_data['face_cols_num'].tolist())
            feed_dict_ = {
                self.user_indices: user_ids,
                self.visual_emb_feat: visual_emb_feat,
                self.item_words_indices_a: item_words_indices,
                self.item_words_values_a: item_words_values,
                self.words_lda: words_lda,
                self.labels: labels,
                self.one_hots_a: onehots,
                self.batch_size: user_ids.shape[0],
                self.num_features: num_features,
                self.face_num: face_cols_num,
                self.dropout_deep: self.drop_out_deep_on_train,
                self.dropout_emb: self.drop_out_emb_on_train,
                self.train_phase: True,
            }
            # batch_size = self.sess.run([self.batch_size], feed_dict=feed_dict_)
            y, loss, _ = self.sess.run([self.y_ui_a, self.loss, self.optimizer], feed_dict=feed_dict_)
            loss_sum += loss
        return loss_sum

    def test_on_batch(self, test_data):
        """
        evaluate sum of batch loss
        """
        pass

    def scoreAUC(self, labels, probs):
        i_sorted = sorted(range(len(probs)), key=lambda i: probs[i],
                          reverse=True)
        auc_temp = 0.0
        TP = 0.0
        TP_pre = 0.0
        FP = 0.0
        FP_pre = 0.0
        P = 0
        N = 0
        last_prob = probs[i_sorted[0]] + 1.0
        for i in range(len(probs)):
            if last_prob != probs[i_sorted[i]]:
                auc_temp += (TP + TP_pre) * (FP - FP_pre) / 2.0
                TP_pre = TP
                FP_pre = FP
                last_prob = probs[i_sorted[i]]
            if labels[i_sorted[i]] == 1:
                TP = TP + 1
            else:
                FP = FP + 1
        auc_temp += (TP + TP_pre) * (FP - FP_pre) / 2.0
        auc = auc_temp / (TP * FP)
        return auc

    def evaluate(self, input_data, split=20, cache=True):
        """
        evaluate the model and return mean loss
        :param data: DataFrame
        :param feature_list: list of features
        :param target_str:
        :param val_size:
        :return: mean loss
        """
        labels_lst = []
        preds_lst = []
        for it, data in enumerate(np.array_split(input_data, split)):
            if cache and it in self.val_datas:
                labels, item_words_indices, item_words_values, user_ids, visual_emb_feat, words_lda, onehots, num_features, face_cols_num = \
                    self.val_datas[it]
            else:
                user_ids = data['user_indices'].values
                visual_emb_feat = data['visual'].tolist()
                words_lda = data['topics'].tolist()
                labels = data['click'].values
                onehots_1 = data['face_cols_01'].tolist()
                # onehots_2 = data[[col for col in data if '_01' in col]].values
                onehots = np.concatenate([onehots_1], axis=1)
                item_words_indices, item_words_values = self._item_words_indices_and_values(data)
                # num_features = data[[col for col in data if '_N' in col]].values
                # num_features = np.asarray(data['context'].tolist())
                num_features = data[ctx_cols].values
                face_cols_num = data['face_cols_num'].tolist()

            feed_dict_ = {
                self.user_indices: user_ids,
                self.visual_emb_feat: visual_emb_feat,
                self.words_lda: words_lda,
                self.item_words_indices_a: item_words_indices,
                self.item_words_values_a: item_words_values,
                self.one_hots_a: onehots,
                self.batch_size: user_ids.shape[0],
                self.num_features: num_features,
                self.face_num: face_cols_num,
                self.dropout_deep: 0,
                self.dropout_emb: 0,
                self.train_phase: False

            }
            if cache:
                self.val_datas[it] = [labels, item_words_indices, item_words_values, user_ids, visual_emb_feat,
                                      words_lda, onehots,
                                      num_features, face_cols_num]
            pred = self.sess.run([self.y_ui_a], feed_dict=feed_dict_)
            labels_lst.extend(labels)
            preds_lst.extend(pred[0])
        if cache:
            for col in input_data:
                if 'click' not in col:
                    del input_data[col]
        return -self.scoreAUC(labels_lst, preds_lst)

    def pred_prob(self, input_data, split=40, cache=True):
        preds_lst = []
        for it, data in enumerate(np.array_split(input_data, split)):
            if cache and it in self.test_datas:
                item_words_indices, item_words_values, user_ids, visual_emb_feat, words_lda, onehots, num_features, face_cols_num = \
                    self.test_datas[
                        it]
            else:
                user_ids = data['user_indices'].values
                visual_emb_feat = data['visual'].tolist()
                onehots_1 = data['face_cols_01'].tolist()
                words_lda = data['topics'].tolist()
                onehots = np.concatenate([onehots_1, ], axis=1)
                # num_features = np.asarray(data['context'].tolist())
                num_features = data[ctx_cols].values
                item_words_indices, item_words_values = self._item_words_indices_and_values(data)
                face_cols_num = data['face_cols_num'].tolist()

            feed_dict_ = {
                self.user_indices: user_ids,
                self.visual_emb_feat: visual_emb_feat,
                self.item_words_indices_a: item_words_indices,
                self.item_words_values_a: item_words_values,
                self.words_lda: words_lda,
                self.one_hots_a: onehots,
                self.batch_size: user_ids.shape[0],
                self.num_features: num_features,
                self.face_num: face_cols_num,
                self.dropout_deep: 0,
                self.dropout_emb: 0,
                self.train_phase: False,
            }
            if cache:
                self.test_datas[it] = [item_words_indices, item_words_values, user_ids, visual_emb_feat,
                                       words_lda,
                                       onehots,
                                       num_features,
                                       face_cols_num]
            pred = self.sess.run([self.y_ui_a], feed_dict=feed_dict_)
            preds_lst.extend(pred[0])
        if cache:
            for col in input_data:
                if col not in ['uid', 'pid']:
                    del input_data[col]
        return preds_lst

    def _save_preds(self, test_data, preds, save_path):
        test_data['preds'] = preds
        test_data['preds'] = test_data['preds'].astype(np.float32)
        test_data[['uid', 'pid', 'preds']].to_pickle(
            os.path.join(save_path, 'deepcross_reg002_hightway_lr00025.pkl'))


if __name__ == '__main__':
    user_embs = pd.read_pickle('../model/user_emb.pkl')
    user_embs = user_embs.sort_values(['user_indices'])
    user_embs = np.array(user_embs['user_emb'].tolist(), np.float32)
    visual_train1 = pd.read_pickle('../data/visual/visual_feature_train_1.pkl')
    visual_train2 = pd.read_pickle('../data/visual/visual_feature_train_2.pkl')
    visual_test = pd.read_pickle('../data/visual/visual_feature_test.pkl')
    visual_train = pd.concat([visual_train1, visual_train2], ignore_index=True, sort=False)

    # visual_test = pd.read_pickle('../data/visual_feature/visual_feature_test.pkl')
    # visual_train = pd.read_pickle('../data/visual_feature/visual_feature_train.pkl')

    print('loaded visual...')
    # print(visual_embs.shape[0], 'photos...')
    # print(visual_embs['pid'].nunique(), 'unique photos...')

    val_data = pd.read_pickle('../data/val_data.pkl')
    train_data = pd.read_pickle('../data/train_data.pkl')
    test_data = pd.read_pickle('../data/test_data.pkl')
    # empty = np.zeros(shape=[6])
    # for df in [train_data, test_data, val_data]:
    #     df['topics'] = df['topics'].apply(lambda lst: empty if pd.isna(lst) is True else lst)

    train_data, val_data = [pd.merge(df, visual_train, 'left', 'pid') for df in
                            [train_data, val_data]]
    test_data = pd.merge(test_data, visual_test, 'left', 'pid')

    one_hots_dims = []
    face_cols = np.array(train_data['face_cols_01'].tolist(), np.uint16)
    one_hots_dims.extend((face_cols.max(axis=0) + 1))
    print('one_hot_dims:', one_hots_dims)

    # dim_num_feat = val_data.ix[0, 'context'].shape[0]
    ctx_cols = [col for col in train_data.columns if 'ctx_' in col and 'ctx_01' not in col]
    dim_num_feat = len(ctx_cols)
    print('ctx_cols:', ctx_cols)

    model_params = {
        'num_user': 37821,
        'num_recent_item': 30,
        'num_words': 152092,
        'dim_num_feat': dim_num_feat,
        'one_hots_dims': one_hots_dims,
        'dim_k': 32,
        'att_dim_k': 8,
        'dim_hidden_out': (256, 256, 256),
        'reg': 0.002,
        'att_reg': 0.2,
        'user_emb_feat': user_embs,
        'dim_lda': 6,
        'lr': 0.00025,
        'prefix': None,
        'seed': 1024,
        'use_deep': True,
        'deep_dims': (512, 512, 512),
        'checkpoint_path': '../model/lda.mdl'
    }
    model = Model(**model_params)
    model.compile(optimizer='adam')
    fit_params = {
        'input_data': train_data,
        'test_data': test_data,
        'batch_size': 8192,
        'epochs': 10,
        'drop_out_deep': 0.5,
        'drop_out_emb': 0.5,
        'validation_data': val_data,
        'shuffle': True,
        'initial_epoch': 0,
        'min_display': 50,
        'max_iter': -1,
        'save_path': '../output/'
    }
    model.fit(**fit_params)


