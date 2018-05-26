import pandas as pd
import tensorflow as tf
import time
import numpy as np
from utils import get_sample_num, new_variable_initializer, sklearn_shuffle
import os
from tensorflow.python.ops import random_ops


class Model(object):
    def __init__(self, num_user, num_recent_item, num_words, one_hots_dims, dim_k, att_dim_k, reg, att_reg, lr, prefix,
                 dim_num_feat,
                 seed=1024,
                 use_deep=True, deep_dims=(256, 128, 64), checkpoint_path=None):
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

        self.recent_words_indices_a = tf.placeholder(shape=[None, 3], dtype=tf.float32,
                                                     name='recent_words_indices_a')  # [num_indices, dim]

        self.recent_words_values_a = tf.placeholder(shape=[None], dtype=tf.float32,
                                                    name='recent_words_values_a')  # [num_indices]

        # 其余物品的one-hot信息
        self.one_hots_a = tf.placeholder(shape=[None, len(self.one_hots_dims)],
                                         dtype=tf.float32, name='one_hots_a')  # [batch_size, num_one_hots]

        self.batch_size = tf.placeholder(shape=[], dtype=tf.int32, name='batch_size')  # num of batch size

        # deep特征
        if self.use_deep:
            self.num_features = tf.placeholder(shape=[None, self.dim_num_feat], dtype=tf.float32,
                                               name='num_features')  # [batch_size, num_features_dims]
            self.dropout_deep = tf.placeholder(shape=[], dtype=tf.float32, name='dropout_deep')

    def _create_weights(self):
        self.Wu_Emb = tf.get_variable(shape=[self.num_user, self.dim_k], initializer=tf.glorot_uniform_initializer(),
                                      dtype=tf.float32, name='user_embedding')

        self.Wwords_Emb = tf.get_variable(shape=[self.num_words, self.dim_k],
                                          initializer=tf.glorot_uniform_initializer(),
                                          dtype=tf.float32, name='words_embedding')

        self.W_one_hots = []
        for i in range(len(self.one_hots_dims)):
            W_temp = tf.get_variable(shape=[self.one_hots_dims[i], self.dim_k],
                                     initializer=tf.glorot_uniform_initializer(),
                                     dtype=tf.float32, name='one_hot_{}'.format(i))
            self.W_one_hots.append(W_temp)

        # Item one-hot features attention
        self.Wu_oh_Att = tf.get_variable(shape=[self.dim_k, self.att_dim_k],
                                         initializer=tf.glorot_uniform_initializer(),
                                         dtype=tf.float32, name='user_attention')
        self.Woh_Att = []
        for i in range(len(self.one_hots_dims)):
            W_temp = tf.get_variable(shape=[self.dim_k, self.att_dim_k],
                                     initializer=tf.glorot_uniform_initializer(),
                                     dtype=tf.float32, name='oh_attention_{}'.format(i))
            self.Woh_Att.append(W_temp)

        self.WW_Att = tf.get_variable(shape=[self.dim_k, self.att_dim_k],
                                      initializer=tf.glorot_uniform_initializer(),
                                      dtype=tf.float32, name='words_attention')

        self.b_oh_Att = tf.get_variable(shape=[self.att_dim_k], initializer=tf.zeros_initializer(),
                                        dtype=tf.float32, name='bias_attention')
        self.w_oh_Att = tf.get_variable(shape=[self.att_dim_k, 1], initializer=tf.glorot_uniform_initializer(),
                                        dtype=tf.float32, name='weight_attention')
        self.c_oh_Att = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(),
                                        dtype=tf.float32, name='bias2_attention')

        self.bias = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(), dtype=tf.float32, name='bias')

        self.bias_u = tf.get_variable(shape=[self.num_user, 1], initializer=tf.zeros_initializer(), dtype=tf.float32,
                                      name='bias_u')

        self.params = [self.Wu_Emb, self.Wwords_Emb, self.bias_u, self.bias, self.b_oh_Att, self.w_oh_Att,
                       self.c_oh_Att] + self.W_one_hots

        self.att_params = [self.WW_Att, self.Wu_oh_Att, ] + self.Woh_Att

        # deep 参数
        if self.use_deep:

            self.Wu_deep_emb = tf.get_variable(shape=[self.num_user, self.dim_k],
                                               initializer=tf.glorot_uniform_initializer(),
                                               dtype=tf.float32, name='user_deep_emb')
            self.Wwords_deep_emb = tf.get_variable(shape=[self.num_words, self.dim_k],
                                                   initializer=tf.glorot_uniform_initializer(),
                                                   dtype=tf.float32, name='words_deep_emb')
            self.W_deep_one_hots = []
            for i in range(len(self.one_hots_dims)):
                W_temp = tf.get_variable(shape=[self.one_hots_dims[i], self.dim_k],
                                         initializer=tf.glorot_uniform_initializer(),
                                         dtype=tf.float32, name='deep_one_hot_{}'.format(i))
                self.W_deep_one_hots.append(W_temp)

    def _forward_pass(self, ):
        # 用户的向量表示
        with tf.name_scope('user_express'):
            # 用户隐向量
            self.Usr_Emb = tf.nn.embedding_lookup(self.Wu_Emb,
                                                  tf.cast(self.user_indices, tf.int32))  # [batch_size, dim_k]

            self.Usr_Expr_a = self.Usr_Emb  # [batch_size, dim_k]

        # 物品的向量表示
        with tf.name_scope('item_express'):
            # self.I_Wds_a = tf.sparse_to_dense(sparse_indices=self.item_words_indices_a,
            #                                   output_shape=[self.batch_size, self.num_words],
            #                                   sparse_values=self.item_words_values_a)  # [batch_size, num_recent_item, num_words]
            self.I_Wds_a = tf.SparseTensor(indices=tf.cast(self.item_words_indices_a, dtype=np.int64),
                                           values=self.item_words_values_a,
                                           dense_shape=[tf.cast(self.batch_size, dtype=np.int64), self.num_words])
            self.att_u_a = tf.matmul(self.Usr_Emb, self.Wu_oh_Att)  # [batch_size, dim_k]
            self.att_oh = []
            self.I_Wds_Emb_a = tf.sparse_tensor_dense_matmul(self.I_Wds_a, self.Wwords_Emb)  # [batch_size, dim_k]
            self.att_I_Wds = tf.matmul(self.I_Wds_Emb_a, self.WW_Att)  # 词的attention
            self.att_I_Wds = tf.nn.relu(self.att_u_a + self.att_I_Wds + self.b_oh_Att)
            self.att_I_Wds = tf.matmul(self.att_I_Wds, self.w_oh_Att) + self.c_oh_Att
            self.att_oh.append(self.att_I_Wds)
            self.I_One_hot_a = []
            for i in range(len(self.one_hots_dims)):
                I_Emb_temp_a = tf.nn.embedding_lookup(self.W_one_hots[i],
                                                      tf.cast(self.one_hots_a[:, i],
                                                              tf.int32))  # [batch_size, dim_k]
                att_oh_temp = tf.matmul(I_Emb_temp_a, self.Woh_Att[i])  # [batch_size, att_dim_k]
                att_temp = tf.nn.relu(self.att_u_a + att_oh_temp + self.b_oh_Att)  # [batch_size, att_dim_k]
                att_temp = tf.matmul(att_temp, self.w_oh_Att) + self.c_oh_Att  # [batch_size, 1]
                self.att_oh.append(att_temp)
                self.I_One_hot_a.append(I_Emb_temp_a)
            self.att_oh = tf.nn.softmax(tf.concat(self.att_oh, axis=1))  # [batch_size, oh_dim] 第一列是词attention
            self.I_Wds_Emb_a = self.I_Wds_Emb_a * self.att_oh[:, 0:1]
            for i in range(1, len(self.one_hots_dims) + 1):
                self.I_One_hot_a[i - 1] = self.I_One_hot_a[i - 1] * self.att_oh[:, i:i + 1]
            self.Item_Expr_a = tf.add_n(self.I_One_hot_a + [self.I_Wds_Emb_a])

        with tf.name_scope('deep'):
            if self.use_deep:
                self.Usr_emb_deep = tf.nn.embedding_lookup(self.Wu_deep_emb,
                                                           tf.cast(self.user_indices, tf.int32))  # [batch_size, dim_k]
                self.I_Wds_emb_deep = tf.sparse_tensor_dense_matmul(self.I_Wds_a,
                                                                    self.Wwords_deep_emb)  # [batch_size, dim_k]
                self.I_one_hot_deep = []
                for i in range(len(self.one_hots_dims)):
                    I_Emb_temp_a = tf.nn.embedding_lookup(self.W_deep_one_hots[i], tf.cast(self.one_hots_a[:, i],
                                                                                           tf.int32))  # [batch_size, dim_k]
                    self.I_one_hot_deep.append(I_Emb_temp_a)

                self.deep_input = tf.concat(
                    [self.Usr_emb_deep, self.I_Wds_emb_deep, self.num_features] + self.I_one_hot_deep,
                    axis=1)  # [batch_size, input_dim]
                self.deep_input = self.num_features
                for deep_dim in self.deep_dims:
                    self.deep_input = tf.layers.dense(inputs=self.deep_input,
                                                      kernel_initializer=tf.glorot_uniform_initializer(),
                                                      units=deep_dim, activation=tf.nn.relu, )
                    fc_mean, fc_var = tf.nn.moments(
                        self.deep_input,
                        axes=[0],  # 想要 normalize 的维度, [0] 代表 batch 维度
                        # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
                    )
                    scale = tf.Variable(tf.ones([deep_dim]))
                    shift = tf.Variable(tf.zeros([deep_dim]))
                    epsilon = 0.001
                    self.deep_input = tf.nn.batch_normalization(self.deep_input, fc_mean, fc_var, shift, scale, epsilon)
                    # 加入dropout
                    self.deep_input = tf.layers.dropout(inputs=self.deep_input, rate=self.dropout_deep)
                self.deep_output = tf.layers.dense(self.deep_input, 1, activation=None)
                self.deep_output = tf.reshape(self.deep_output, [-1])  # [batch_size]

        with tf.name_scope('output'):
            # 输出结果
            self.y_ui_a = tf.reduce_sum(self.Item_Expr_a * self.Usr_Expr_a, axis=1)
            # 用户偏置
            self.y_ui_a = self.y_ui_a + tf.reshape(
                tf.nn.embedding_lookup(self.bias_u, tf.cast(self.user_indices, dtype=tf.int32)), [-1])
            # 整体偏置
            self.y_ui_a = self.y_ui_a + self.bias
            if self.use_deep:
                self.y_ui_a += self.deep_output
            self.y_ui_a = tf.nn.sigmoid(self.y_ui_a)

    def _create_loss(self):
        self.loss = tf.keras.losses.categorical_crossentropy(self.labels, self.y_ui_a)
        # self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=self.labels, logits=self.y_ui_a))
        # 正则项
        for param in self.params:
            self.loss = tf.add(self.loss, self.reg * tf.nn.l2_loss(param))
        for param in self.att_params:
            self.loss = tf.add(self.loss, self.att_reg * tf.nn.l2_loss(param))
        # self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)

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
                          # 'adagradda':tf.train.AdagradDAOptimizer(),
                          'rmsprop': tf.train.RMSPropOptimizer(self.lr),
                          'moment': tf.train.MomentumOptimizer(self.lr, 0.9),
                          'ftrl': tf.train.FtrlOptimizer(self.lr)
                          # tf.train.ProximalAdagradOptimizer#padagrad
                          # tf.train.ProximalGradientDescentOptimizer#pgd
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
        self.saver.save(self.sess, save_path + '.ckpt', self.global_step)

    def load_model(self, meta_graph_path, ckpt_dir=None, ckpt_path=None):
        """
        :meta_graph_path .meta文件路径
        :ckpt_dir 最新的检查点所在目录
        :ckpt_path 指定检查点
        """
        if ckpt_dir is None and ckpt_path is None:
            raise ValueError('Must specify ckpt_dir or ckpt_path')

        restore_saver = tf.train.import_meta_graph(meta_graph_path, )
        if ckpt_path is None:
            ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
            print(ckpt_path)

        restore_saver.restore(self.sess, ckpt_path)

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
        indices = np.asarray(indices)
        values = np.ones([indices.shape[0]])
        return indices, values

    def _recent_words_indices_and_values(self, input_data):
        def func(input):
            for ix, indices in enumerate(input):
                for indice in indices:
                    yield [ix, indice[0], indice[1]]

        indices = list(func(input_data['recent_words']))
        indices = np.asarray(indices)
        values = np.ones([indices.shape[0]])
        return indices, values

    def train_on_batch(self, input_data):  # fit a batch
        user_ids = input_data['user_indices'].as_matrix()
        labels = input_data['click'].as_matrix()
        onehots_1 = np.asarray(input_data['face_cols'].tolist())
        onehots_2 = input_data[[col for col in input_data if '_01' in col]].as_matrix()
        onehots = np.concatenate([onehots_1, onehots_2], axis=1)
        item_words_indices, item_words_values = self._item_words_indices_and_values(input_data)
        # recent_words_indices, recent_words_values = self._recent_words_indices_and_values(input_data)
        num_features = input_data[[col for col in input_data if '_N' in col]].as_matrix()
        feed_dict_ = {
            self.user_indices: user_ids, self.item_words_indices_a: item_words_indices,
            self.item_words_values_a: item_words_values,
            # self.recent_words_indices_a: recent_words_indices,
            # self.recent_words_values_a: recent_words_values,
            self.labels: labels,
            self.one_hots_a: onehots,
            self.batch_size: user_ids.shape[0],
            self.num_features: num_features,
            self.dropout_deep: self.drop_out_deep_on_train
        }
        # batch_size = self.sess.run([self.batch_size], feed_dict=feed_dict_)
        y, loss, _ = self.sess.run([self.y_ui_a, self.loss, self.optimizer], feed_dict=feed_dict_)
        return loss

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

    def evaluate(self, input_data):
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
        for it, data in enumerate(np.array_split(input_data, 10)):
            if it in self.val_datas:
                labels, item_words_indices, item_words_values, user_ids, onehots, num_features = self.val_datas[it]
            else:
                user_ids = data['user_indices'].as_matrix()
                labels = data['click'].as_matrix()
                onehots_1 = np.asarray(data['face_cols'].tolist())
                onehots_2 = data[[col for col in data if '_01' in col]].as_matrix()
                onehots = np.concatenate([onehots_1, onehots_2], axis=1)
                item_words_indices, item_words_values = self._item_words_indices_and_values(data)
                num_features = data[[col for col in data if '_N' in col]].as_matrix()
            feed_dict_ = {
                self.user_indices: user_ids, self.item_words_indices_a: item_words_indices,
                self.item_words_values_a: item_words_values,
                # self.recent_words_indices_a: recent_words_indices,
                # self.recent_words_values_a: recent_words_values,
                self.one_hots_a: onehots,
                self.batch_size: user_ids.shape[0],
                self.num_features: num_features,
                self.dropout_deep: 0

            }
            self.val_datas[it] = [labels, item_words_indices, item_words_values, user_ids, onehots, num_features]
            pred = self.sess.run([self.y_ui_a], feed_dict=feed_dict_)
            labels_lst.extend(labels)
            preds_lst.extend(pred[0])
        return -self.scoreAUC(labels_lst, preds_lst)

    def fit(self, input_data, batch_size=1024, epochs=50, validation_data=None, shuffle=True, initial_epoch=0,
            min_display=50, max_iter=-1, drop_out_deep=0.5, save_path=None, test_data=None, ):

        n_samples = get_sample_num(input_data)
        iters = (n_samples - 1) // batch_size + 1
        self.tr_loss_list = []
        self.val_loss_list = []
        self.drop_out_deep_on_train = drop_out_deep
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
                input_data = sklearn_shuffle(input_data, random_state=None)
            for j in range(iters):
                if isinstance(input_data, list):
                    batch_x = [
                        item[j * batch_size:(j + 1) * batch_size] for item in input_data]
                elif isinstance(input_data, pd.DataFrame):
                    batch_x = input_data.iloc[j * batch_size:(j + 1) * batch_size, :]
                else:
                    batch_x = input_data[j * batch_size:(j + 1) * batch_size]
                loss = self.train_on_batch(
                    batch_x)
                if j % min_display == 0:
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
                            # self.save_model(self.checkpoint_path+'best')
                # self.save_model(self.checkpoint_path)
                if (i * iters) + j == max_iter:
                    stop_flag = True
                    break
            self._save_preds(test_data, self.preds, save_path)
            if stop_flag:
                break

    def pred_prob(self, input_data):
        preds_lst = []
        for it, data in enumerate(np.array_split(input_data, 10)):
            if it in self.test_datas:
                item_words_indices, item_words_values, user_ids, onehots, num_features = self.test_datas[it]
            else:
                user_ids = data['user_indices'].as_matrix()
                onehots_1 = np.asarray(data['face_cols'].tolist())
                onehots_2 = data[[col for col in data if '_01' in col]].as_matrix()
                onehots = np.concatenate([onehots_1, onehots_2], axis=1)
                num_features = data[[col for col in data if '_N' in col]].as_matrix()
                item_words_indices, item_words_values = self._item_words_indices_and_values(data)
            feed_dict_ = {
                self.user_indices: user_ids, self.item_words_indices_a: item_words_indices,
                self.item_words_values_a: item_words_values,
                # self.recent_words_indices_a: recent_words_indices,
                # self.recent_words_values_a: recent_words_values,
                self.one_hots_a: onehots,
                self.batch_size: user_ids.shape[0],
                self.num_features: num_features,
                self.dropout_deep: 0
            }
            self.test_datas[it] = [item_words_indices, item_words_values, user_ids, onehots, num_features]
            pred = self.sess.run([self.y_ui_a], feed_dict=feed_dict_)
            preds_lst.extend(pred[0])
        return preds_lst

    def _save_preds(self, test_data, preds, save_path):
        test_data['preds'] = preds
        test_data[['uid', 'pid', 'preds']].to_pickle(os.path.join(save_path, 'preds_log1.pkl'))
