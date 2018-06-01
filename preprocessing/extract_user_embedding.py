import pandas as pd
import tensorflow as tf
import time
import numpy as np
from utils import get_sample_num, new_variable_initializer, sklearn_shuffle
import os
from tensorflow.python.ops import random_ops


class Model(object):
    def __init__(self, num_user, num_item, num_recent_item, num_words, one_hots_dims, dim_k, att_dim_k, dim_hidden_out,
                 reg,
                 att_reg, lr, prefix,
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
        self.num_item = num_item
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

        self.item_indices = tf.placeholder(shape=[None], dtype=tf.float32, name='item_indices')

        self.batch_size = tf.placeholder(shape=[], dtype=tf.int32, name='batch_size')  # num of batch size

        self.train_phase = tf.placeholder(shape=[], dtype=tf.bool, name='train_phase')

        # embedding的dropout
        self.dropout_emb = tf.placeholder(shape=[], dtype=tf.float32, name='dropout_emb')

        self.dropout_deep = tf.placeholder(shape=[], dtype=tf.float32, name='dropout_deep')

    def _create_weights(self):

        # Embedding
        self.Wu_Emb = tf.get_variable(shape=[self.num_user, self.dim_k], initializer=tf.glorot_uniform_initializer(),
                                      dtype=tf.float32, name='user_embedding')

        self.Wi_Emb = tf.get_variable(shape=[self.num_user, self.dim_k], initializer=tf.glorot_uniform_initializer(),
                                      dtype=tf.float32, name='item_embedding')

        self.bias = tf.get_variable(shape=[1], initializer=tf.zeros_initializer(), dtype=tf.float32, name='bias')

        self.bias_u = tf.get_variable(shape=[self.num_user, 1], initializer=tf.zeros_initializer(), dtype=tf.float32,
                                      name='bias_u')

        self.bias_i = tf.get_variable(shape=[self.num_item, 1], initializer=tf.zeros_initializer(), dtype=tf.float32,
                                      name='bias_i')

        self.params = [self.Wu_Emb, self.Wi_Emb, self.bias_u, self.bias, self.bias_i]

        # self.in_prd_params = [self.W_in_prd]

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

    def _forward_pass(self, ):
        # 用户的向量表示
        with tf.name_scope('user_express'):
            # 用户隐向量
            self.Usr_Emb = tf.nn.embedding_lookup(self.Wu_Emb,
                                                  tf.cast(self.user_indices, tf.int32))  # [batch_size, dim_k]

            self.Usr_Expr_a = self.Usr_Emb  # [batch_size, dim_k]

        with tf.name_scope('item_express'):
            self.Item_Emb = tf.nn.embedding_lookup(self.Wi_Emb,
                                                   tf.cast(self.item_indices, tf.int32))  # [batch_size, dim_k

        with tf.name_scope('output'):
            self.cf_out = tf.layers.dropout(self.Item_Emb * self.Usr_Expr_a, rate=self.dropout_emb)
            self.concated = tf.concat([self.cf_out], axis=1)
            # if self.use_deep:
            #     self.concated = tf.concat([self.cf_out, self.ctx_usr_out, self.ctx_item_out, self.deep_input], axis=1)
            # else:
            #     self.concated = tf.concat([self.cf_out, self.ctx_usr_out, self.ctx_item_out], axis=1)
            self.hidden = tf.layers.dense(inputs=self.concated, kernel_initializer=tf.glorot_uniform_initializer(),
                                          units=self.dim_hidden_out, activation=tf.nn.relu)
            self.hidden = tf.layers.dropout(inputs=self.hidden, rate=self.dropout_deep)
            self.bu = tf.nn.embedding_lookup(self.bias_u, tf.cast(self.user_indices, dtype=tf.int32))
            self.bi = tf.nn.embedding_lookup(self.bias_i, tf.cast(self.item_indices, dtype=tf.int32))
            self.y_ui_a = tf.layers.dense(self.hidden, 1, activation=None) + self.bu + self.bi
            self.y_ui_a = tf.reshape(tf.nn.sigmoid(self.y_ui_a), [-1])

        # with tf.name_scope('output'):
        #     # 输出结果
        #     self.y_ui_a = tf.reduce_sum(self.Item_Expr_a * self.Usr_Expr_a, axis=1)
        #     # 用户偏置
        #     self.y_ui_a = self.y_ui_a + tf.reshape(
        #         tf.nn.embedding_lookup(self.bias_u, tf.cast(self.user_indices, dtype=tf.int32)), [-1])
        #     # 整体偏置
        #     self.y_ui_a = self.y_ui_a + self.bias
        #     if self.use_deep:
        #         self.y_ui_a += self.deep_output
        #     self.y_ui_a = tf.nn.sigmoid(self.y_ui_a)

    def _create_loss(self):
        self.loss = tf.keras.losses.categorical_crossentropy(self.labels, self.y_ui_a)
        # self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=self.labels, logits=self.y_ui_a))
        # 正则项
        for param in self.params:
            self.loss = tf.add(self.loss, self.reg * tf.nn.l2_loss(param))
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

    def train_on_batch(self, input_data):  # fit a batch
        user_ids = input_data['user_indices'].as_matrix()
        item_ids = input_data['photo_indices'].as_matrix()
        labels = input_data['click'].as_matrix()
        feed_dict_ = {
            self.user_indices: user_ids,
            self.item_indices: item_ids,
            self.labels: labels,
            self.batch_size: user_ids.shape[0],
            self.dropout_deep: self.drop_out_deep_on_train,
            self.dropout_emb: self.drop_out_emb_on_train,
            self.train_phase: True,
        }
        # batch_size = self.sess.run([self.batch_size], feed_dict=feed_dict_)
        y, loss, _ = self.sess.run([self.y_ui_a, self.loss, self.optimizer], feed_dict=feed_dict_)
        return loss

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
                labels, user_ids, item_ids = self.val_datas[it]
            else:
                user_ids = data['user_indices'].as_matrix()
                item_ids = data['photo_indices'].as_matrix()
                labels = data['click'].as_matrix()
            feed_dict_ = {
                self.user_indices: user_ids,
                self.item_indices: item_ids,
                self.batch_size: user_ids.shape[0],
                self.dropout_deep: 0,
                self.dropout_emb: 0,
                self.train_phase: False

            }
            self.val_datas[it] = [labels, user_ids, item_ids]
            pred = self.sess.run([self.y_ui_a], feed_dict=feed_dict_)
            labels_lst.extend(labels)
            preds_lst.extend(pred[0])
        return -self.scoreAUC(labels_lst, preds_lst)

    def pred_prob(self, uid_data):
        user_ids = uid_data['user_indices']
        (user_embs,) = self.sess.run([self.Usr_Emb, ], feed_dict={self.user_indices: user_ids})
        return user_embs

    def _save_preds(self, test_data, preds, save_path):
        test_data['user_emb'] = [np.asarray(emb) for emb in preds]
        test_data[['uid', 'user_indices', 'user_emb']].to_pickle(os.path.join(save_path, 'user_emb.pkl'))


if __name__ == '__main__':
    val_data = pd.read_pickle('../data/val_data.pkl')
    train_data = pd.read_pickle('../data/train_data.pkl')
    test_data = train_data[['user_indices', 'uid']].drop_duplicates(['user_indices', 'uid'])
    one_hots_dims = []
    face_cols = np.array(train_data['face_cols_01'].tolist())
    one_hots_dims.extend((face_cols.max(axis=0) + 1))
    # for col in val_data:
    #     if '_01' in col:
    #         one_hots_dims.append(train_data[col].max() + 1)
    dim_num_feat = 0
    for col in val_data:
        if '_N' in col:
            dim_num_feat += 1

    print(one_hots_dims)
    model_params = {
        'num_user': 15141,
        'num_item': 4278686,
        'num_recent_item': 30,
        'num_words': 119637,
        'dim_num_feat': dim_num_feat,
        'one_hots_dims': one_hots_dims,
        'dim_k': 128,
        'att_dim_k': 16,
        'dim_hidden_out': 32,
        'reg': 0.01,
        'att_reg': 0.2,
        'lr': 0.002,
        'prefix': None,
        'seed': 1024,
        'use_deep': True,
        'deep_dims': (512, 256, 64, 32)
    }
    model = Model(**model_params)
    model.compile(optimizer='adam')
    # indices = pd.np.array([[1, 500], [1, 508]])
    #
    # feed_dict_ = {
    #     model.user_indices: [1]*2048, model.item_words_indices_a: indices,
    #     model.item_words_values_a: [1, 1],
    #     model.recent_words_indices_a: [[1,2,1]],
    #     model.recent_words_values_a: [1],
    #     model.labels: [1]*2048,
    #     model.one_hots_a: [[1,23,3]]*2048,
    #     model.batch_size: 2048
    # }
    # a = model.sess.run([model.I_Wds_a], feed_dict=feed_dict_)
    # print(pd.np.array(a)[0, 1:2, 507:509])
    #
    # print(a)
    fit_params = {
        'input_data': train_data,
        'test_data': test_data,
        'batch_size': 4096,
        'epochs': 50,
        'drop_out_deep': 0.5,
        'drop_out_emb': 0.5,
        'validation_data': val_data, 'shuffle': True,
        'initial_epoch': 0,
        'min_display': 10,
        'max_iter': -1,
        'save_path': '../model/'
    }
    model.fit(**fit_params)
