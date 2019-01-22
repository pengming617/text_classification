import tensorflow.contrib.learn as learn
import re
import tensorflow as tf
import numpy as np
import jieba


dicts = {}
with open("save_model/char_cnn/labels.txt", 'r', encoding="utf-8") as f:
    for line in f.readlines():
        tag_type = line.replace("\n", "").split(":")
        dicts[int(tag_type[0])] = tag_type[1]


class Infer(object):
    """
        ues CNN model to predict classification.
    """
    def __init__(self):
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore('save_model/char_cnn/vocab.pickle')
        self.checkpoint_file = tf.train.latest_checkpoint('save_model/char_cnn')
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                saver.restore(self.sess, self.checkpoint_file)

                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("Input-Layer/input_x").outputs[0]
                self.drop_keep_prob = graph.get_operation_by_name("Input-Layer/dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("output_layer/predictions").outputs[0]
                self.score = graph.get_operation_by_name("output_layer/score").outputs[0]

    def infer(self, sentences):
        # transfer to vector
        sentence_word = []
        for sentence in sentences:
            words = [x for x in jieba.cut(sentence) if x != '']
            # 不进行分词
            char_word = []
            for x in words:
                if re.search('[a-zA-Z]', x):
                    char_word.append(x)
                else:
                    char_word.extend([y for y in x])
            sentence_word.append(' '.join(char_word))

        sentences_vectors = np.array(list(self.vocab_processor.fit_transform(sentence_word)))

        feed_dict = {
            self.input_x: sentences_vectors,
            self.drop_keep_prob: 1.0
        }
        y, s = self.sess.run([self.predictions, self.score], feed_dict)
        # 将数字转换为对应的意图
        labels = [dicts[x] for x in y]
        s = [np.max(x) for x in s]
        return labels, s









