# encoding:utf-8
import tensorflow as tf
from tensorflow.python.ops import array_ops


class TextCNN(object):
    """
        A CNN for text classification.
        Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, max_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda):
        # placeholders for input output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, max_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name='input_y')
        self.drop_keep_prob = tf.placeholder(tf.float32, name="drop_keep_prob")

        # embedding layer
        with tf.device('/gpu:1'), tf.name_scope("embedding"):
            self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                         name='embedding', trainable=True)
            self.embedding_chars = tf.nn.embedding_lookup(self.embedding, self.input_x)
            # 将embedding_chars扩展为四维的
            self.embedding_chars_expand = tf.expand_dims(self.embedding_chars, -1)

        # create a convolution + maxpool layer for each filter size
        pooled_results = []
        for size in filter_sizes:
            # convolution layer
            filter_shape = [size, embedding_size, 1, num_filters]
            # 初始化w
            self.W = tf.Variable(tf.truncated_normal(filter_shape, stddev=.01), name='W')
            self.b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            conv = tf.nn.conv2d(self.embedding_chars_expand, self.W, strides=[1, 1, 1, 1], padding="VALID", name='conv')
            # apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, self.b), name='relu')
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h, [1, max_length-size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')
            pooled_results.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        # 将pooled_results转为[None,1,1,num_filters_total]
        self.h_pool = tf.concat(pooled_results, 3)
        # h_pool装为[None,num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_pool_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.drop_keep_prob)

        # Final (unnormalized) scores and predictions
        l2_loss = 0
        with tf.name_scope("output"):
            W = tf.get_variable(shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer(), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            self.scores = tf.nn.xw_plus_b(self.h_pool_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # 定义损失函数
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        self.focal_loss = self.get_focal_loss(tf.nn.softmax(self.scores, 1), tf.cast(self.input_y, tf.float32))
        self.loss = 0.8 * self.focal_loss + 0.2 * self.loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def get_focal_loss(self, prediction_tensor, target_tensor, weights=None, alpha=0.75, gamma=2):
        # sigmoid_p = tf.nn.sigmoid(prediction_tensor)
        sigmoid_p = prediction_tensor

        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

        # For poitive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

        my_entry_cross = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0))

        return tf.reduce_mean(my_entry_cross)