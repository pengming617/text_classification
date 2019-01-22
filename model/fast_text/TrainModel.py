import fasttext


class TrainModel(object):

    def train(self):
        classifier = fasttext.supervised('data/train_fasttext.txt',
                                         'save_model/fast_text/fasttext_Model', label_prefix='__label__')
        result = classifier.test('data/train_fasttext.txt')
        print("pre:"+str(result.precision))
        print("recall:"+str(result.recall))



