# encoding:utf-8
import tensorflow as tf
from model.textcnn import TextCNN
from model.textcnn import Processing
from config import Config as Config
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
import numpy as np

config = Config.Config()


class TrainModel(object):
    '''
        训练模型
        保存模型
    '''
    def trainModel(self, embedding_dim=128, filter_sizes='1,2,3', num_filters=128,
                   dropout_keep_prob=0.5, l2_reg_lambda=0.01):

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)

            with tf.name_scope("readfile"):
                processing = Processing.Processing()
                articles, tags = processing.loadPracticeFile("data/train_sentiment.txt")
                self.data_embedding_new, self.tags_new = processing.embedding(articles, tags, config.is_cut)
                X_train, X_val, y_train, y_val = train_test_split(
                    self.data_embedding_new, self.tags_new, test_size=0.1, random_state=0)
            # 加载词典
            vocab = learn.preprocessing.VocabularyProcessor.restore('save_model/text_cnn/vocab.pickle')

            with sess.as_default():
                textcnn = TextCNN.TextCNN(
                    max_length=len(self.data_embedding_new[0]),
                    num_classes=len(y_train[0]),
                    vocab_size=len(vocab.vocabulary_),
                    embedding_size=embedding_dim,
                    filter_sizes=list(map(int, filter_sizes.split(","))),
                    num_filters=num_filters,
                    l2_reg_lambda=l2_reg_lambda)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                # 对var_list中的变量计算loss的梯度 返回一个以元组(gradient, variable)组成的列表
                grads_and_vars = optimizer.compute_gradients(textcnn.loss)
                # 将计算出的梯度应用到变量上
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                best_acc = 0.0

                for time in range(config.epoch):
                    batch_size = config.Batch_Size
                    if int(len(X_train) % batch_size) == 0:
                        batches = int(len(X_train) / batch_size)
                    else:
                        batches = int(len(X_train) / batch_size) + 1
                    for x in range(batches):
                        if x != batches-1:
                            trainX_batch = X_train[x * batch_size:(x + 1) * batch_size]
                            trainY_batch = y_train[x * batch_size:(x + 1) * batch_size]
                        else:
                            trainX_batch = X_train[x * batch_size:len(articles)]
                            trainY_batch = y_train[x * batch_size:len(articles)]
                        feed_dict = {
                            textcnn.input_x: np.array(trainX_batch),
                            textcnn.input_y: np.array(trainY_batch),
                            textcnn.drop_keep_prob: dropout_keep_prob
                        }
                        _, loss, accuracy = sess.run([train_op, textcnn.loss, textcnn.accuracy], feed_dict)

                    print("第"+str((time+1))+"次迭代的损失为："+str(loss)+";准确率为："+str(accuracy))

                    def dev_step(dev_x, dev_y):
                        """
                        Evaluates model on a dev set
                        """
                        feed_dict = {
                            textcnn.input_x: np.array(dev_x),
                            textcnn.input_y: np.array(dev_y),
                            textcnn.drop_keep_prob: 1.0
                        }
                        dev_loss, dev_accuracy = sess.run([textcnn.loss, textcnn.accuracy], feed_dict)
                        print("验证集：loss {:g}, acc {:g}\n".format(dev_loss, dev_accuracy))
                        return dev_loss, dev_accuracy

                    dev_loss, dev_accuracy = dev_step(X_val, y_val)

                    if dev_accuracy > best_acc:
                        best_acc = dev_accuracy
                        saver.save(sess, "save_model/text_cnn/TextCNNModel.ckpt")
                        print("Saved model success\n")
