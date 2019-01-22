# encoding:utf-8
import tensorflow as tf
from model.han.Processing import Processing
from config.Config import Config
from model.han.HANModel import HANClassifierModel
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.contrib import learn

# Misc Parameters
allow_soft_placement = True
log_device_placement = False
sentence_cut_flag = '[,.?!]'  # 句子结尾的标志

config = Config()


class TrainModel(object):
    '''
        训练模型
        保存模型
    '''
    def trainModel(self, embedding_dim=256,
                   dropout_keep_prob=0.5,
                   word_hiddencell=100,
                   sentence_hiddencell=100,
                   word_attention_size=100,
                   sentence_attention_size=100):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                          log_device_placement=log_device_placement)
            sess = tf.Session(config=session_conf)
            with tf.name_scope("readfile"):
                processing = Processing()
                articles, tags = processing.loadPracticeFile('data/train.txt')
                self.data_embedding_new, self.tags_new = processing.embedding(articles, tags, sentence_cut_flag)
                X_train, X_val, y_train, y_val = train_test_split(
                    self.data_embedding_new, self.tags_new, test_size=0.1, random_state=0)

            # 加载词典
            vocab = learn.preprocessing.VocabularyProcessor.restore('save_model/han/vocab.pickle')

            with sess.as_default():
                han_model = HANClassifierModel(
                    vocab_size=len(vocab.vocabulary_),
                    embedding_size=embedding_dim,
                    classes=len(y_train[0]),
                    sentence_max=len(X_train[0]),
                    word_max=len(X_train[0][0]),
                    word_hiddencell=word_hiddencell,
                    sentence_hiddencell=sentence_hiddencell,
                    word_attention_size=word_attention_size,
                    sentence_attention_size=sentence_attention_size,
                    max_grad_norm=5.0,
                    dropout_keep_prob=dropout_keep_prob,
                )

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

                        doc_len, sen_len = self.getdoc_sen_length(trainX_batch)
                        feed_dict = {
                            han_model.input_x: trainX_batch,
                            han_model.input_y: trainY_batch,
                            han_model.doc_len: doc_len,
                            han_model.sen_len: sen_len,
                            han_model.dropout_keep_prob: dropout_keep_prob
                        }
                        _, cost, accuracy = sess.run([han_model.train_op, han_model.loss, han_model.accuracy], feed_dict)

                    print("第"+str((time+1))+"次迭代的损失为："+str(cost)+";准确率为："+str(accuracy))

                    def dev_step(dev_x, dev_y):
                        """
                        Evaluates model on a dev set
                        """
                        doc_len, sen_len = self.getdoc_sen_length(dev_x)
                        feed_dict = {
                            han_model.input_x: np.array(dev_x),
                            han_model.input_y: np.array(dev_y),
                            han_model.doc_len: doc_len,
                            han_model.sen_len: sen_len,
                            han_model.dropout_keep_prob: 1.0,
                        }
                        dev_cost, dev_accuracy, predictions = sess.run([han_model.loss, han_model.accuracy, han_model.prediction], feed_dict)
                        y_true = [np.nonzero(x)[0][0] for x in dev_y]
                        f1 = f1_score(np.array(y_true), predictions, average='micro')
                        print("验证集：loss {:g}, acc {:g}, f1 {:g}\n".format(dev_cost, dev_accuracy, f1))
                        return dev_cost, dev_accuracy

                    dev_cost, dev_accuracy = dev_step(X_val, y_val)

                    if dev_accuracy > best_acc:
                        best_acc = dev_accuracy
                        saver.save(sess, "save_model/han/han_Model.ckpt")
                        print("Saved model success\n")

    def getdoc_sen_length(self,batches):
        doc_len = []
        sen_len = []
        for data in batches:
            x = np.sum(data, 1)
            doc_len.append(len(x[x > 0]))
            lens = []
            for word in data:
                lens.append(len(word[word > 0]))
            sen_len.append(lens)
        return np.array(doc_len), np.array(sen_len)