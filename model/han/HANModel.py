import tensorflow as tf
import math


class HANClassifierModel(object):

    def __init__(self, vocab_size, embedding_size, classes, dropout_keep_prob, sentence_max, word_max,
                 word_hiddencell, sentence_hiddencell, word_attention_size, sentence_attention_size,
                 max_grad_norm, is_training=True, learning_rate=1e-4):

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.classes = classes
        self.dropout_keep_prob = dropout_keep_prob
        self.word_hiddencell = word_hiddencell
        self.sentence_hiddencell = sentence_hiddencell
        self.word_attention_size = word_attention_size
        self.sentence_attention_size = sentence_attention_size
        self.max_grad_norm = max_grad_norm
        self.sentence_max = sentence_max
        self.word_max = word_max

        # placeholders for input output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.sentence_max, self.word_max], name='input_x')
        self.input_y = tf.placeholder(tf.int32, shape=[None, classes], name='input_y')
        self.doc_len = tf.placeholder(tf.int32, shape=[None], name='doc_len') # 每篇文章中含有句子的真实数量
        self.sen_len = tf.placeholder(tf.int32, shape=[None, None], name='sen_len') # 每个句子含有词语的真实数量
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        (self.document_size, self.sentence_size, self.word_size) = tf.unstack(tf.shape(self.input_x))

        # embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.embedding = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                                                             stddev=1.0 / math.sqrt(embedding_size)),
                                         name="embedding", trainable=True)
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("word_layer") as scope:
            word_level_inputs = tf.reshape(self.embedding_inputs, [
                self.document_size * self.sentence_size,
                self.word_size,
                self.embedding_size
            ])
            word_level_lengths = tf.reshape(
                self.sen_len, [self.document_size * self.sentence_size])
            # word encoder
            word_outputs = self.build_bidirectional_rnn(word_level_inputs, word_level_lengths, self.word_hiddencell, scope)
            # word attention
            word_attention = self.build_attention(word_outputs, self.word_attention_size, word_level_lengths, self.word_size)
            # dropout
            word_level_output = tf.nn.dropout(word_attention, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('sentence_layer') as scope:
            sentence_level_inputs = tf.reshape(word_level_output, [self.document_size, self.sentence_size, 2 * self.word_hiddencell])
            # sentence encoder
            sentence_outputs = self.build_bidirectional_rnn(sentence_level_inputs, self.doc_len, self.sentence_hiddencell, scope)
            # sentence attention
            sentence_attention = self.build_attention(sentence_outputs, self.sentence_attention_size, sentence_hiddencell, self.sentence_size)
            # dropout
            sentence_level_output = tf.nn.dropout(sentence_attention, keep_prob=self.dropout_keep_prob)

        with tf.variable_scope('classifier'):
            self.logits = tf.contrib.layers.fully_connected(
                sentence_level_output, self.classes, activation_fn=None)
            self.prediction = tf.argmax(self.logits, axis=-1)

        if not is_training:
            return
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)

            self.loss = tf.reduce_mean(self.cross_entropy)
            tf.summary.scalar('loss', self.loss)

            correct_predictions = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            tf.summary.scalar('accuracy', self.accuracy)

            tvars = tf.trainable_variables()
            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            opt = tf.train.AdamOptimizer(learning_rate)
            self.train_op = opt.apply_gradients(zip(grads, tvars), name='train_op', global_step=self.global_step)
            self.summary_op = tf.summary.merge_all()

    def build_bidirectional_rnn(self, inputs, length, hidden_num, scope):
        cell_fw = tf.contrib.rnn.BasicLSTMCell(hidden_num)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(hidden_num)
        (outputs, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                            cell_bw=cell_bw,
                                                            inputs=inputs,
                                                            sequence_length=length,
                                                            dtype=tf.float32,
                                                            swap_memory=True,
                                                            scope=scope)
        outputs = tf.concat(outputs, 2)
        return outputs

    def build_attention(self, inputs, attention_size, length, max_len):
        # Trainable parameters
        u = tf.Variable(tf.random_normal([attention_size, 1]), trainable='True')
        logit = tf.layers.dense(inputs, attention_size, activation=tf.nn.tanh, use_bias=True)
        logit = tf.einsum('ijk,kl->ijl', logit, u)
        logit = tf.squeeze(logit, -1)
        logit_mask = self.mask(logit, length, max_len)
        soft_logit = tf.nn.softmax(logit_mask, 1)
        attention_out = tf.reduce_sum(inputs * tf.expand_dims(soft_logit, -1), 1)
        return attention_out

    def mask(self, inputs, seq_len, max_len):
        mask = tf.cast(tf.sequence_mask(seq_len, maxlen=max_len), tf.float32)
        return inputs - (1 - mask) * 1e12


if __name__ == '__main__':
    with tf.Session() as session:
        model = HANClassifierModel(
            vocab_size=1000,
            embedding_size=200,
            classes=2,
            sentence_max=10,
            word_max=20,
            word_hiddencell=100,
            sentence_hiddencell=100,
            word_attention_size=100,
            sentence_attention_size=10,
            max_grad_norm=5.0,
            dropout_keep_prob=0.5,
        )
        session.run(tf.global_variables_initializer())

