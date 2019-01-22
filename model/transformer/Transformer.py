import tensorflow as tf
import math
import numpy as np
from tensorflow.python.ops import array_ops


class Muti_head_Attention(object):

    def __init__(self, max_length, num_classes, vocab_size, embedding_size, hidden_num, num_blocks, num_heads):
        # placeholders for input output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, max_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, num_classes], name='input_y')
        self.drop_out_prob = tf.placeholder(tf.float32, name='drop_out_keep')

        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedding = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                         stddev=1.0 / math.sqrt(embedding_size)),
                                         name="embedding", trainable=True)
            self.embedding_chars = tf.nn.embedding_lookup(self.embedding, self.input_x)

        # Positional Encoding
        N = array_ops.shape(self.embedding_chars)[0]
        T = max_length
        self.embedding_chars += self.positional_encoding(N, T,
                                                         num_units=hidden_num,
                                                         zero_pad=False,
                                                         scale=False,
                                                         scope="enc_pe")

        # Dropout
        self.enc = tf.layers.dropout(self.embedding_chars, rate=self.drop_out_prob)

        # Blocks
        for i in range(num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i)):
                # Multihead Attention
                self.enc = self.multihead_attention(queries=self.enc,
                                                    keys=self.enc,
                                                    num_units=hidden_num,
                                                    num_heads=num_heads,
                                                    dropout_rate=self.drop_out_prob,
                                                    causality=False)

                # Feed Forward
                self.enc = self.feedforward(self.enc, num_units=[4 * hidden_num, hidden_num])

        # 将特征进行拼接
        self.enc = tf.reshape(self.enc, [-1, max_length * hidden_num, 1])
        self.enc = tf.squeeze(self.enc, -1)

        fc_w = tf.Variable(tf.truncated_normal([max_length * hidden_num, num_classes], stddev=0.1), name='fc_w')
        fc_b = tf.Variable(tf.zeros([num_classes]), name='fc_b')
        # 定义损失函数
        l2_loss = 0
        l2_loss += tf.nn.l2_loss(fc_w)
        l2_loss += tf.nn.l2_loss(fc_b)

        self.logits = tf.matmul(self.enc, fc_w) + fc_b
        self.score = tf.nn.softmax(self.logits, name='score')
        self.predictions = tf.argmax(self.score, 1, name="predictions")

        self.cost = tf.losses.softmax_cross_entropy(self.input_y, self.logits)
        # l2_reg_lambda = 0.01
        # self.cost = self.cost + l2_reg_lambda * l2_loss

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)

        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.accuracy = tf.reduce_mean(
                        tf.cast(tf.equal(tf.argmax(self.input_y, axis=1), tf.argmax(self.score, axis=1)), tf.float32))

    def positional_encoding(self, N, T,
                            num_units,
                            zero_pad=True,
                            scale=True,
                            scope="positional_encoding",
                            reuse=None):
        '''Sinusoidal Positional_Encoding.

        Args:
          inputs: A 2d Tensor with shape of (N, T).
          num_units: Output dimensionality
          zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
          scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
            A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
        '''
        with tf.variable_scope(scope, reuse=reuse):
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
                for pos in range(T)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

            # Convert to a tensor
            lookup_table = tf.convert_to_tensor(position_enc)

            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                          lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

            if scale:
                outputs = outputs * num_units ** 0.5

            outputs = tf.cast(outputs, dtype=tf.float32)

            return outputs

    def multihead_attention(self, queries,
                            keys,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            causality=False,
                            scope="multihead_attention",
                            reuse=None):
        '''Applies multihead attention.

        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          causality: Boolean. If true, units that reference the future are masked.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list[-1]

            # Linear projections
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate)

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.normalize(outputs)  # (N, T_q, C)

        return outputs

    def feedforward(self, inputs,
                    num_units=[2048, 512],
                    scope="multihead_attention",
                    reuse=None):
        '''Point-wise feed forward net.

        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = self.normalize(outputs)

        return outputs

    def normalize(self, inputs,
                  epsilon=1e-8,
                  scope="ln",
                  reuse=None):
        '''Applies layer normalization.

        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
            `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs
