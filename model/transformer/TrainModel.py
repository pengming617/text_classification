# encoding:utf-8
import tensorflow as tf
from model.transformer.Processing import Processing
from model.transformer.Transformer import Muti_head_Attention
from config import Config as Config
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os
import numpy as np
from tensorflow.contrib import learn

# Misc Parameters
allow_soft_placement = True
log_device_placement = False
config = Config.Config()
root_path = os.getcwd()


class TrainModel(object):
    '''
        训练模型
        保存模型
    '''
    def trainModel(self, embedding_dim=200,
                   dropout_keep_prob=0.5,
                   hidden_num=200,
                   num_blocks=3,
                   num_heads=4):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                          log_device_placement=log_device_placement)
            sess = tf.Session(config=session_conf)
            with tf.name_scope("readfile"):
                processing = Processing()
                articles, tags = processing.loadPracticeFile('data/train_sentiment.txt')
                self.data_embedding_new, self.tags_new = processing.embedding(articles, tags, config.is_cut)
                X_train, X_val, y_train, y_val = train_test_split(
                    self.data_embedding_new, self.tags_new, test_size=0.1, random_state=0)

            # 加载词典
            vocab = learn.preprocessing.VocabularyProcessor.restore('save_model/transformer/vocab.pickle')

            with sess.as_default():
                muti_head = Muti_head_Attention(max_length=len(self.data_embedding_new[0]),
                                                num_classes=len(y_train[0]),
                                                vocab_size=len(vocab.vocabulary_),
                                                embedding_size=embedding_dim,
                                                hidden_num=hidden_num,
                                                num_blocks=num_blocks,
                                                num_heads=num_heads)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                best_f1 = 0.0

                for time in range(config.epoch):
                    batch_size = config.Batch_Size
                    batches = int(len(X_train) / batch_size) + 1
                    accuracy_all = []
                    cost_all = []
                    for x in range(batches):
                        if x != batches-1:
                            trainX_batch = X_train[x * batch_size:(x + 1) * batch_size]
                            trainY_batch = y_train[x * batch_size:(x + 1) * batch_size]
                        else:
                            trainX_batch = X_train[x * batch_size:len(articles)]
                            trainY_batch = y_train[x * batch_size:len(articles)]

                        feed_dict = {
                            muti_head.input_x: np.array(trainX_batch),
                            muti_head.input_y: np.array(trainY_batch),
                            muti_head.drop_out_prob: dropout_keep_prob,
                        }
                        _, cost, accuracy = sess.run([muti_head.train_op, muti_head.cost, muti_head.accuracy], feed_dict)
                        accuracy_all.append(accuracy)
                        cost_all.append(cost)

                    print("第" + str((time + 1)) + "次迭代的损失为：" + str(np.mean(np.array(cost_all))) + ";准确率为："
                          + str(np.mean(np.array(accuracy_all))))

                    def dev_step(dev_x, dev_y):
                        """
                        Evaluates model on a dev set
                        """
                        feed_dict = {
                            muti_head.input_x: np.array(dev_x),
                            muti_head.input_y: np.array(dev_y),
                            muti_head.drop_out_prob: 0.0,
                        }
                        dev_cost, dev_accuracy, predictions = sess.run([muti_head.cost, muti_head.accuracy,
                                                                        muti_head.predictions], feed_dict)
                        y_true = [np.nonzero(x)[0][0] for x in dev_y]
                        f1_scores = f1_score(np.array(y_true), predictions, average='micro')
                        print("验证集：loss {:g}, acc {:g}, f1 {:g}\n".format(dev_cost, dev_accuracy, f1_scores))
                        return f1_scores

                    f1_scores = dev_step(X_val, y_val)

                    if f1_scores > best_f1:
                        best_f1 = f1_scores
                        saver.save(sess, "save_model/transformer/transformerModel.ckpt")
                        print("saved\n")
