# encoding:utf-8
import tensorflow as tf
from model.char_cnn.Processing import Processing
from config.Config import Config
from model.char_cnn.Char_CNN import CharCNN
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.contrib import learn

# Misc Parameters
allow_soft_placement = True
log_device_placement = False

config = Config()


class TrainModel(object):
    '''
        训练模型
        保存模型
    '''
    def trainModel(self, embedding_dim=256, dropout_keep_prob=0.5,
                   conv_layers=[[256, 5, 3],[256, 5, 3],[256, 1, None],[256, 1, None],[256, 1, None],[256, 1, 3]],
                   fully_layers=[512, 512]):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                          log_device_placement=log_device_placement)
            sess = tf.Session(config=session_conf)
            with tf.name_scope("readfile"):
                processing = Processing()
                articles, tags = processing.loadPracticeFile('data/train_sentiment.txt')
                self.data_embedding_new, self.tags_new = processing.embedding(articles, tags)
                X_train, X_val, y_train, y_val = train_test_split(
                    self.data_embedding_new, self.tags_new, test_size=0.1, random_state=0)

            # 加载词典
            vocab = learn.preprocessing.VocabularyProcessor.restore('save_model/char_cnn/vocab.pickle')

            with sess.as_default():
                charcnn = CharCNN(conv_layers=conv_layers,
                                  fully_layers=fully_layers,
                                  alphabet_size=len(vocab.vocabulary_),
                                  sen_max_length=len(X_train[0]),
                                  class_nums=len(self.tags_new[0]),
                                  embedding_size=embedding_dim)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                best_acc = 0.0

                for time in range(config.epoch):
                    batch_size = config.Batch_Size
                    batches = int(len(X_train) / batch_size) + 1
                    for x in range(batches):
                        if x != batches-1:
                            trainX_batch = X_train[x * batch_size:(x + 1) * batch_size]
                            trainY_batch = y_train[x * batch_size:(x + 1) * batch_size]
                        else:
                            trainX_batch = X_train[x * batch_size:len(articles)]
                            trainY_batch = y_train[x * batch_size:len(articles)]

                        feed_dict = {
                            charcnn.input_x: np.array(trainX_batch),
                            charcnn.input_y: np.array(trainY_batch),
                            charcnn.dropout_keep_prob: dropout_keep_prob
                        }
                        _, cost, accuracy = sess.run([charcnn.train_op, charcnn.loss, charcnn.accuracy], feed_dict)

                    print("第"+str((time+1))+"次迭代的损失为："+str(cost)+";准确率为："+str(accuracy))

                    def dev_step(dev_x, dev_y):
                        """
                        Evaluates model on a dev set
                        """
                        feed_dict = {
                            charcnn.input_x: np.array(dev_x),
                            charcnn.input_y: np.array(dev_y),
                            charcnn.dropout_keep_prob: 1.0,
                        }
                        dev_cost, dev_accuracy, predictions = sess.run([charcnn.loss, charcnn.accuracy, charcnn.predictions], feed_dict)
                        y_true = [np.nonzero(x)[0][0] for x in dev_y]
                        f1 = f1_score(np.array(y_true), predictions, average='micro')
                        print("验证集：loss {:g}, acc {:g}, f1 {:g}\n".format(dev_cost, dev_accuracy, f1))
                        return dev_cost, dev_accuracy

                    dev_cost, dev_accuracy = dev_step(X_val, y_val)

                    if dev_accuracy > best_acc:
                        best_acc = dev_accuracy
                        saver.save(sess, "save_model/char_cnn/charcnn_Model.ckpt")
                        print("Saved model success\n")
