import tensorflow as tf
import math
import numpy as np
from tensorflow.python.ops import array_ops


class LEAM(object):

    def __init__(self, max_length, num_classes, vocab_size, embedding_size, hidden_num, attn_size):
        # placeholders for input output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, max_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, num_classes], name='input_y')
        self.drop_out_prob = tf.placeholder(tf.float32, name='drop_out_keep')
        self.seq_length = tf.placeholder(tf.int32, [None], name='seq_length')  # 语句的真实长度

        # embedding layer
        with tf.device('/gpu:0'), tf.name_scope("word_embedding"):
            self.embedding = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                                             stddev=1.0 / math.sqrt(embedding_size)),
                                         name="embedding", trainable=True)
            self.embedding_chars = tf.nn.embedding_lookup(self.embedding, self.input_x)

        # label embedding
        with tf.device('/gpu:0'), tf.name_scope("label_embedding"):
            label_emb = tf.Variable(tf.random_uniform(shape=(num_classes, embedding_size), dtype=tf.float32))

        # build model
        cell_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden_num)
        fw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.drop_out_prob)
        cell_bw = tf.nn.rnn_cell.BasicLSTMCell(hidden_num)
        bw_lstm_cell = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.drop_out_prob)

        (outputs, output_states) = tf.nn.bidirectional_dynamic_rnn(fw_lstm_cell, bw_lstm_cell, self.embedding_chars,
                                                                   sequence_length=self.seq_length,
                                                                   dtype=tf.float32)
        sents = tf.concat(outputs, axis=2)
        out = self.label_sent_attention(sents, label_emb, attn_size)

        self.logits = tf.layers.dense(out, num_classes)
        self.score = tf.nn.softmax(self.logits, name='score')
        self.predictions = tf.argmax(self.score, 1, name="predictions")

        self.cost = tf.losses.softmax_cross_entropy(self.input_y, self.logits)

        class_y = tf.constant(name='class_y', shape=[num_classes, num_classes], dtype=tf.float32,
                              value=np.identity(num_classes), )

        logit_label = tf.layers.dense(label_emb, num_classes)
        label_loss = tf.losses.softmax_cross_entropy(onehot_labels=class_y, logits=logit_label)

        self.focal_loss = self.get_focal_loss(self.score, tf.cast(self.input_y, tf.float32))

        self.loss = 0.2 * self.cost + 0.4 * label_loss + 0.4 * self.focal_loss

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)

        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.input_y, axis=1), self.predictions), tf.float32))

    def label_sent_attention(self, sent_encoder, label_emb, att_size):
        sent_encoder = tf.layers.dense(sent_encoder, att_size)
        label_emb = tf.layers.dense(label_emb, att_size)
        tran_label_emb = tf.transpose(label_emb, [1, 0])

        sent_encoder = tf.nn.l2_normalize(sent_encoder, -1)
        tran_label_emb = tf.nn.l2_normalize(tran_label_emb, 0)

        G = tf.einsum('ijk,kl->ijl', sent_encoder, tran_label_emb)

        G = tf.expand_dims(G, -1)
        fliter_w = tf.Variable(tf.random_uniform(shape=(8, 1, 1, 1), dtype=tf.float32))
        max_G = tf.nn.relu(tf.nn.conv2d(G, filter=fliter_w, strides=[1, 1, 1, 1], padding='SAME'))
        max_G = tf.squeeze(max_G, -1)

        max_G = tf.reduce_max(max_G, axis=-1, keep_dims=True)

        mask_G = self.mask(tf.squeeze(max_G, -1), self.seq_length, sent_encoder.shape[1].value)

        soft_mask_G = tf.clip_by_value(tf.nn.softmax(mask_G, 1), 1e-5, 1.0)
        soft_mask_G = tf.expand_dims(soft_mask_G, -1)
        out = tf.einsum('ijk,ijl->ikl', sent_encoder, soft_mask_G)
        out = tf.squeeze(out, -1)

        return out
    
    def mask(self, inputs, seq_len, max_len):
        mask = tf.cast(tf.sequence_mask(seq_len, maxlen=max_len), tf.float32)
        return inputs - (1 - mask) * 1e12

    def get_focal_loss(self, prediction_tensor, target_tensor, weights=None, alpha=0.75, gamma=2):
        # sigmoid_p = tf.nn.sigmoid(prediction_tensor)
        sigmoid_p = prediction_tensor

        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

        # For poitive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

        my_entry_cross = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0))

        return tf.reduce_mean(my_entry_cross)