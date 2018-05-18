import tensorflow as tf
import time
import numpy as np
from .utils import (get_sample_num, new_variable_initializer,
                    sigmoid_cross_entropy_with_probs, sklearn_shuffle,
                    sklearn_split)


class Model(object):
    def __init__(self, num_user, num_recent_item, num_words, dim_k, reg, lr, prefix, seed=1024, checkpoint_path=None):
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

        self.num_user = num_user
        self.num_history_item = num_recent_item
        self.num_words = num_words
        self.dim_k = dim_k
        self.reg = reg
        self.lr = lr
        self.prefix = prefix

    def _get_optimizer_loss(self, ):
        return self.loss

    def _build_graph(self):
        self.__create_placeholders()
        self._create_weights()
        self._forward_pass()

    def __create_placeholders(self, ):
        self.user_indices = tf.placeholder(shape=[-1], dtype=tf.float32)  # [batch_size]

        self.item_words_indices_a = tf.placeholder(shape=[-1, 2], dtype=tf.float32)  # [num_indices, dim]

        self.item_words_indices_b = tf.placeholder(shape=[-1, 2], dtype=tf.float32)  # [num_indices, dim]

        self.item_words_values_a = tf.placeholder(shape=[-1], dtype=tf.float32)  # [num_indices]

        self.item_words_values_b = tf.placeholder(shape=[-1], dtype=tf.float32)  # [num_indices]

        self.recent_words_indices_a = tf.placeholder(shape=[-1, 3], dtype=tf.float32)  # [num_indices, dim]

        self.recent_words_values_a = tf.placeholder(shape=[-1], dtype=tf.float32)  # [num_indices]

        self.recent_words_indices_b = tf.placeholder(shape=[-1, 3], dtype=tf.float32)  # [num_indices, dim]

        self.recent_words_values_b = tf.placeholder(shape=[-1], dtype=tf.float32)  # [num_indices]

    def _create_weights(self):
        self.Wu_Emb = tf.get_variable(shape=[self.num_user, self.dim_k], initializer=tf.random_normal_initializer,
                                      dtype=tf.float32, name='user_embedding')

        self.Wwords_Emb = tf.get_variable(shape=[self.num_words, self.dim_k], initializer=tf.random_normal_initializer,
                                          dtype=tf.float32, name='words_embedding')

        self.Wu_Att = tf.get_variable(shape=[self.dim_k, self.dim_k], initializer=tf.contrib.layers.xavier_initializer,
                                      dtype=tf.float32, name='user_attention')

        self.Wwords_Att = tf.get_variable(shape=[self.dim_k, self.dim_k],
                                          initializer=tf.contrib.layers.xavier_initializer,
                                          dtype=tf.float32, name='words_attention')

        self.Wi_Att = tf.get_variable(shape=[self.dim_k, self.dim_k], initializer=tf.contrib.layers.xavier_initializer,
                                      dtype=tf.float32, name='item_attention')

        self.b_Att = tf.get_variable(shape=[self.dim_k], initializer=tf.zeros_initializer,
                                     dtype=tf.float32, name='bias_attention')

        self.w_Att = tf.get_variable(shape=[self.dim_k], initializer=tf.contrib.layers.xavier_initializer,
                                     dtype=tf.float32, name='weight_attention')

        self.c_Att = tf.get_variable(shape=[1], initializer=tf.zeros_initializer,
                                     dtype=tf.float32, name='bias2_attention')

        self.params = [self.Wu_Emb, self.Wwords_Emb, self.Wu_Att, self.Wwords_Att, self.Wi_Att, self.b_Att, self.w_Att,
                       self.c_Att]

    def _forward_pass(self, ):
        # 物品的向量表示
        with tf.name_scope('item_express'):
            I_Wds_a = tf.sparse_to_dense(sparse_indices=self.item_words_indices_a,
                                         output_shape=[-1, self.num_words],
                                         sparse_values=self.item_words_values_a)  # [batch_size, num_recent_item, num_words]
            I_Wds_b = tf.sparse_to_dense(sparse_indices=self.item_words_indices_b,
                                         output_shape=[-1, self.num_words],
                                         sparse_values=self.item_words_values_b)  # [batch_size, num_recent_item, num_words]
            self.I_Wds_Emb_a = tf.matmul(I_Wds_a, self.Wwords_Emb)  # [batch_size, dim_k]
            self.I_Wds_Emb_b = tf.matmul(I_Wds_b, self.Wwords_Emb)  # [batch_size, dim_k]

        # 用户的向量表示
        with tf.name_scope('user_express'):
            # 用户隐向量
            self.Usr_Emb = tf.nn.embedding_lookup(self.Wu_Emb, self.user_indices)  # [batch_size, dim_k]

            # 用户近期历史词隐向量
            Rct_Wds_a = tf.sparse_to_dense(sparse_indices=self.recent_words_indices_a,
                                           output_shape=[-1, self.num_history_item, self.num_words],
                                           sparse_values=self.recent_words_values_a)  # [batch_size, num_recent_item, num_words]
            Rct_Wds_b = tf.sparse_to_dense(sparse_indices=self.recent_words_indices_b,
                                           output_shape=[-1, self.num_history_item, self.num_words],
                                           sparse_values=self.recent_words_values_b)  # [batch_size, num_recent_item, num_words]
            self.Rct_Wds_Emb_a = tf.tensordot(Rct_Wds_a, self.Wwords_Emb,
                                              axes=[[2], [0]])  # [batch_size, num_recent_item, dim_k]
            self.Rct_Wds_Emb_b = tf.tensordot(Rct_Wds_b, self.Wwords_Emb,
                                              axes=[[2], [0]])  # [batch_size, num_recent_item, dim_k]

            # 近期历史Attention系数计算
            att_u_a = tf.matmul(self.Usr_Emb, self.Wu_Att)  # [batch_size, dim_k]
            att_rct_a = tf.tensordot(self.Rct_Wds_Emb_a, self.Wwords_Att,
                                     axes=[[2], [0]])  # [batch_size, num_recent_item, dim_k]
            att_i_a = tf.matmul(self.I_Wds_Emb_a, self.Wi_Att)  # [batch_size, dim_k]
            att_a = tf.nn.relu(att_rct_a + att_u_a + att_i_a + self.b_Att)  # [batch_size, num_recent_item, dim_k]
            att_a = tf.reduce_sum(att_a * self.w_Att, axis=2) + self.c_Att  # [batch_size, num_recent_item]
            self.att_a = tf.nn.softmax(att_a)  # [batch_size, num_recent_item]
            self.att_a = tf.reshape(self.att_a, [-1, self.att_a.shape[1], 1])

            att_u_b = tf.matmul(self.Usr_Emb, self.Wu_Att)  # [batch_size, dim_k]
            att_rct_b = tf.tensordot(self.Rct_Wds_Emb_b, self.Wwords_Att,
                                     axes=[[2], [0]])  # [batch_size, num_recent_item, dim_k]
            att_i_b = tf.matmul(self.I_Wds_Emb_b, self.Wi_Att)  # [batch_size, dim_k]
            att_b = tf.nn.relu(att_rct_b + att_u_b + att_i_b + self.b_Att)  # [batch_size, num_recent_item, dim_k]
            att_b = tf.reduce_sum(att_b * self.w_Att, axis=2) + self.c_Att  # [batch_size, num_recent_item]
            self.att_b = tf.nn.softmax(att_b)  # [batch_size, num_recent_item]
            self.att_b = tf.reshape(self.att_b, [-1, self.att_b.shape[1], 1])

            # 近期物品Text加权求和
            self.Rct_Wds_Expr_a = tf.reduce_sum(self.Rct_Wds_Emb_a * self.att_a, axis=1)  # [batch_size, dim_k]
            self.Rct_Wds_Expr_b = tf.reduce_sum(self.Rct_Wds_Emb_b * self.att_b, axis=1)  # [batch_size, dim_k]

            # 用户的向量表示
            self.Usr_Expr_a = tf.add_n([self.Usr_Emb, self.Rct_Wds_Expr_a])  # [batch_size, dim_k]
            self.Usr_Expr_b = tf.add_n([self.Usr_Emb, self.Rct_Wds_Expr_b])  # [batch_size, dim_k]
        with tf.name_scope('output'):
            # 输出两个结果
            self.y_ui_a = tf.reduce_sum(self.Rct_Wds_Expr_a * self.Usr_Expr_a, axis=1)
            self.y_ui_b = tf.reduce_sum(self.Rct_Wds_Expr_b * self.Usr_Expr_b, axis=1)

    def _create_loss(self):
        self.loss = - tf.reduce_sum(tf.log(tf.nn.sigmoid(self.y_ui_a - self.y_ui_b)))
        # 正则项
        for param in self.params:
            self.loss += self.reg * tf.reduce_sum(param ** 2)

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

        optimizer_dict = {'sgd': tf.train.GradientDescentOptimizer(0.01),
                          'adam': tf.train.AdamOptimizer(0.001),
                          'adagrad': tf.train.AdagradOptimizer(0.01),
                          # 'adagradda':tf.train.AdagradDAOptimizer(),
                          'rmsprop': tf.train.RMSPropOptimizer(0.001),
                          'moment': tf.train.MomentumOptimizer(0.01, 0.9),
                          'ftrl': tf.train.FtrlOptimizer(0.01)
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
            self.metric_list = self._create_metrics(metrics)  # 创建度量列表
            # for the use of BN,tf.get_collection get default Graph
            update_ops = self.graph.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):  # for the use of BN
                self.op = self._create_optimizer(optimizer)
                self.optimizer = self.op.minimize(
                    self._get_optimizer_loss(), global_step=self.global_step)  # 创建优化器
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

    def train_on_batch(self, user_indices, item_words_indices_a, item_words_indices_b, recent_words_indices_a,
                       recent_words_indices_b):  # fit a batch

        feed_dict_ = {self.user_indices: user_indices, self.item_words_indices_a: item_words_indices_a,
                      self.item_words_indices_b: item_words_indices_b,
                      self.item_words_values_a: item_words_values_a,
                      self.recent_words_indices_a: recent_words_indices_a,
                      self.recent_words_values_a: recent_words_values_a,
                      self.recent_words_indices_b: recent_words_indices_b,
                      self.recent_words_values_b: recent_words_values_b}
        self.sess.run([self.optimizer], feed_dict=feed_dict_)



    def fit(self, user_indices, item_words_indices_a, item_words_indices_b, recent_words_indices_a,
                       recent_words_indices_b, batch_size=1024, epochs=50, validation_split=0.0, validation_data=None,
            val_size=2 ** 18, shuffle=True, initial_epoch=0, min_display=50, max_iter=-1):

        n_samples = get_sample_num(user_indices)
        iters = (n_samples - 1) // batch_size + 1
        self.tr_loss_list = []
        self.val_loss_list = []
        print(iters, "steps per epoch")
        print(batch_size, "samples per step")
        start_time = time.time()
        stop_flag = False
        self.best_loss = np.inf
        self.best_ckpt = None
        if not validation_data and validation_split > 0:
            x, val_x, y, val_y = sklearn_split(
                x, y, test_size=validation_split, random_state=self.seed)
            validation_data = [(val_x, val_y)]

        for i in range(epochs):
            if i < initial_epoch:
                continue
            if shuffle:
                x, y = sklearn_shuffle(x, y, random_state=self.seed)
            for j in range(iters):
                if isinstance(x, list):
                    batch_x = [
                        item[j * batch_size:(j + 1) * batch_size] for item in x]
                else:
                    batch_x = x[j * batch_size:(j + 1) * batch_size]
                batch_y = y[j * batch_size:(j + 1) * batch_size]

                self.train_on_batch(
                    batch_x, batch_y)
                if j % min_display == 0:
                    tr_loss = self.evaluate(x, y, val_size, None, None)
                    self.tr_loss_list.append(tr_loss)
                    total_time = time.time() - start_time
                    if validation_data is None:
                        print("Epoch {0: 2d} Step {1: 4d}: tr_loss {2: 0.6f} tr_time {3: 0.1f}".format(i, j, tr_loss,
                                                                                                       total_time))
                    else:
                        val_loss = self.evaluate(
                            validation_data[0][0], validation_data[0][1], val_size)
                        self.val_loss_list.append(val_loss)
                        print(
                            "Epoch {0: 2d} Step {1: 4d}: tr_loss {2: 0.6f} va_loss {3: 0.6f} tr_time {4: 0.1f}".format(
                                i, j, tr_loss, val_loss, total_time))

                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            # self.save_model(self.checkpoint_path+'best')

                # self.save_model(self.checkpoint_path)

                if (i * iters) + j == max_iter:
                    stop_flag = True
                    break
            if stop_flag:
                break