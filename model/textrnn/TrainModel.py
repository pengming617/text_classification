# encoding:utf-8
import tensorflow as tf
from model.textrnn.Processing import Processing
from config import Config as Config
from model.textrnn.TextRNN import TextRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os
import numpy as np
from tensorflow.contrib import learn

config = Config.Config()
root_path = os.getcwd()


class TrainModel(object):
    '''
        训练模型
        保存模型
    '''
    def trainModel(self, embedding_dim=100,
                   dropout_keep_prob=0.5,
                   hidden_num=100,
                   hidden_size=1,
                   l2_reg_lambda=0.001):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with tf.name_scope("readfile"):
                processing = Processing()
                articles, tags = processing.loadPracticeFile('data/train.txt')
                self.data_embedding_new, self.tags_new = processing.embedding(articles, tags, config.is_cut)
                X_train, X_val, y_train, y_val = train_test_split(
                    self.data_embedding_new, self.tags_new, test_size=0.1, random_state=0)

            # 加载词典
            vocab = learn.preprocessing.VocabularyProcessor.restore('save_model/text_rnn/vocab.pickle')

            with sess.as_default():
                textRNN = TextRNN(max_length=len(self.data_embedding_new[0]),
                                  num_classes=len(y_train[0]),
                                  vocab_size=len(vocab.vocabulary_),
                                  embedding_size=embedding_dim,
                                  hidden_num=hidden_num,
                                  hidden_size=hidden_size,
                                  l2_reg_lambda=l2_reg_lambda)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                best_accuray = 0.0

                for time in range(config.epoch):
                    batch_size = config.Batch_Size
                    batches = int(len(X_train) / batch_size) + 1
                    cost = 999.99
                    accuracy = 0.0
                    for x in range(batches):
                        if x != batches-1:
                            trainx_batch = X_train[x * batch_size:(x + 1) * batch_size]
                            trainy_batch = y_train[x * batch_size:(x + 1) * batch_size]
                        else:
                            trainx_batch = X_train[x * batch_size:len(articles)]
                            trainy_batch = y_train[x * batch_size:len(articles)]

                        real_length = self.get_length(trainx_batch)
                        feed_dict = {
                            textRNN.input_x: np.array(trainx_batch),
                            textRNN.input_y: np.array(trainy_batch),
                            textRNN.drop_out_prob: dropout_keep_prob,
                            # textRNN.mask_x: np.transpose(np.array(trainX_batch)),
                            textRNN.seq_length: np.array(real_length),
                        }
                        _, cost, accuracy, summary = sess.run([textRNN.train_op, textRNN.cost, textRNN.accuracy,
                                                               textRNN.summary], feed_dict)
                    print("第"+str((time+1))+"次迭代：")
                    print("训练集：loss {:g}, acc {:g}".format(cost, accuracy))

                    def dev_step(dev_x, dev_y):
                        """
                        Evaluates model on a dev set
                        """
                        dev_real_length = self.get_length(dev_x)
                        feed_dict = {
                            textRNN.input_x: np.array(dev_x),
                            textRNN.input_y: np.array(dev_y),
                            textRNN.drop_out_prob: 1.0,
                            # textRNN.mask_x: np.transpose(np.array(dev_x)),
                            textRNN.seq_length: np.array(dev_real_length),
                        }
                        dev_cost, dev_accuracy, predictions = sess.run([textRNN.cost, textRNN.accuracy,
                                                                        textRNN.prediction], feed_dict)
                        y_true = [np.nonzero(x)[0][0] for x in dev_y]
                        f1_scores = f1_score(np.array(y_true), predictions, average='micro')
                        print("验证集：loss {:g}, acc {:g}, f1 {:g}\n".format(dev_cost, dev_accuracy, f1_scores))
                        return dev_accuracy

                    dev_accuracy = dev_step(X_val, y_val)
                    if dev_accuracy > best_accuray:
                        best_accuray = dev_accuracy
                        saver.save(sess, "save_model/text_rnn/TextRNNModel.ckpt")
                        print("saved\n")

    def get_length(self, trainX_batch):
        # sentence length
        lengths = []
        for sample in trainX_batch:
            count = 0
            for index in sample:
                if index != 0:
                    count += 1
                else:
                    break
            lengths.append(count)
        return lengths


if __name__ == '__main__':
    train = TrainModel()
    train.trainModel()