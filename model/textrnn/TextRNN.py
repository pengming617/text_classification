# encoding:utf-8
import tensorflow as tf
import math


class TextRNN(object):

    def __init__(self, max_length, num_classes, vocab_size, embedding_size, hidden_num, hidden_size,
                 l2_reg_lambda, is_training=True):
        # placeholders for input output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, max_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, shape=[None, num_classes], name="input_y")
        self.drop_out_prob = tf.placeholder(tf.float32, name="drop_out_prob")
        self.seq_length = tf.placeholder(tf.int32, [None], name="seq_length")

        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            #                              name="embedding", trainable=True)
            self.embedding = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                         stddev=1.0 / math.sqrt(embedding_size)),
                                         name="embedding", trainable=True)
            self.embedding_chars = tf.nn.embedding_lookup(self.embedding, self.input_x)

        # RNN model
        cells = []
        for _ in range(hidden_size):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_num, forget_bias=0.0, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.drop_out_prob)
            cells.append(lstm_cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
        self.embedding_chars = tf.nn.dropout(self.embedding_chars, self.drop_out_prob)
        outputs, states = tf.nn.dynamic_rnn(cell,
                                            self.embedding_chars,
                                            dtype=tf.float32,
                                            sequence_length=self.seq_length)

        with tf.name_scope("mean_pooling_layer"):
            out_put = tf.reduce_sum(outputs, axis=1) / (tf.cast(tf.expand_dims(self.seq_length, -1), tf.float32))

        # 加入l2正则化
        l1_loss = 0
        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w", [hidden_num, num_classes], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [num_classes], dtype=tf.float32)
            self.logits = tf.nn.xw_plus_b(out_put, softmax_w, softmax_b, name="logits")
            l1_loss += tf.contrib.layers.l2_regularizer(l2_reg_lambda)(softmax_w)

        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.cost = tf.reduce_mean(self.loss) + l1_loss

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.logits, 1, name='prediction')
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # add summary
        loss_summary = tf.summary.scalar("cost", self.cost)
        # add summary
        accuracy_summary = tf.summary.scalar("accuracy_summary", self.accuracy)

        if not is_training:
            return

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.lr = tf.Variable(1e-3, trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        self.summary = tf.summary.merge([loss_summary, accuracy_summary])

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)






