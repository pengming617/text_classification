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

    def embedding(self, articles, tags, is_cut):
        length = []
        articlesWords = []
        for article in articles:
            words = [x for x in jieba.cut(article) if x != '']
            if not is_cut:
                # 不进行分词
                char_word = []
                for x in words:
                    if re.search('[a-zA-Z]', x) or x.isdigit():
                        char_word.append(x)
                    else:
                        char_word.extend([y for y in x])
                articlesWords.append(' '.join(char_word))
                length.append(len(char_word))
            else:
                articlesWords.append(' '.join(words))
                length.append(len(words))
        max_length = max(length)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
        vocab_processor.fit(articlesWords)
        data_embedding = np.array(list(vocab_processor.fit_transform(articlesWords)))
        vocab_processor.save('save_model/birnn_attention/vocab.pickle')

        # 将类型用数字替换
        label_file = open('save_model/birnn_attention/labels.txt', 'w')
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

        return data_embedding, tags_vec

