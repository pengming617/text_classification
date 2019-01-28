import tensorflow.contrib.learn as learn
import tensorflow as tf
import numpy as np
import jieba
import re

# Misc Parameters
dicts = {}
with open("save_model/transformer/labels.txt", 'r', encoding="utf-8") as f:
    for line in f.readlines():
        tag_type = line.replace("\n", "").split(":")
        dicts[int(tag_type[0])] = tag_type[1]


class Infer(object):
    """
        ues RNN model to predict classification.
    """
    def infer(self, sentences):
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore('save_model/transformer/vocab.pickle')
        checkpoint_file = tf.train.latest_checkpoint('save_model/transformer')

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # transfer to vector
                sentence_word = []
                for sentence in sentences:
                    words = jieba.cut(sentence)

                    # 只保留中文文本
                    re_words = re.compile(u"[\u4e00-\u9fa5]+")
                    words = [word for word in words if re_words.search(word)]

                    sentence_word.append(' '.join((list(words))))
                sentences_vectors = np.array(list(vocab_processor.fit_transform(sentence_word)))
                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                drop_keep_prob = graph.get_operation_by_name("drop_out_keep").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("predictions").outputs[0]
                scores = graph.get_operation_by_name("score").outputs[0]

                feed_dict = {
                    input_x: sentences_vectors,
                    drop_keep_prob: 1.0
                }
                y, s = sess.run([predictions, scores], feed_dict)

                # 将数字转换为对应的意图
                labels = [dicts[x] for x in y]
                s = [np.max(x) for x in s]
                return labels, s
