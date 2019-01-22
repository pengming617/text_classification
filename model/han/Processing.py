# encoding:utf-8
from tensorflow.contrib import learn
import jieba
import numpy as np
import re


class Processing(object):
    '''
        语料的预处理工作
    '''
    def loadPracticeFile(self, filename):
        '''
        :param filename: 文件名
        训练文件格式
        1/t/t全民保/t/t实体
        :return:
        '''
        with open(filename, 'r', encoding='utf-8') as fr:
            articles = []
            tags = []
            for line in fr.readlines():
                data = line.replace("\t\t", "\t").replace("\n", "").split("\t")
                if len(data) == 3:
                    articles.append(data[1].replace("，", ","))
                    tags.append(data[2])
                else:
                    print(line+"------格式错误")
        return articles, tags

    def embedding(self, articles, tags, sentence_cut_flag):
        word_count = []
        articlesWords = []
        sentence_len = []
        for article in articles:
            sentencese = re.split(sentence_cut_flag, article)
            sentence_len.append(len(sentencese))
            for sentence in sentencese:
                words = [x for x in jieba.cut(sentence) if x != '']
                articlesWords.append(' '.join(words))
                word_count.append(len(words))
        max_length = max(word_count)
        max_sentence = max(sentence_len)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
        vocab_processor.fit(articlesWords)
        data_embedding = np.array(list(vocab_processor.fit_transform(articlesWords)))
        vocab_processor.save('save_model/han/vocab.pickle')

        zeros = np.zeros(max_length)
        flag = 0
        data_embedding_x = []
        for x in sentence_len:
            data_embedding_padding = np.empty([0, max_length])
            data_embedding_padding = np.append(data_embedding_padding, data_embedding[flag:flag+x], axis=0)
            flag += x
            if x < max_sentence:
                data_embedding_padding = np.append(data_embedding_padding, np.array([zeros] * (max_sentence - x)), axis=0)
            data_embedding_x.append(data_embedding_padding)

        # 将类型用数字替换
        label_file = open('save_model/han/labels.txt', 'w')
        type = list(set(tags))
        for temp in range(len(type)):
            print(str(temp) + ":" + type[temp])
            label_file.writelines(str(temp) + ":" + type[temp] + "\n")
        tags_new = [type.index(x) for x in tags]
        tags_vec = []
        for x in tags_new:
            temp = [0] * len(type)
            temp[x] = 1
            tags_vec.append(temp)

        return np.array(data_embedding_x), np.array(tags_vec)

