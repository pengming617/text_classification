import fasttext
import jieba


class Infer(object):
    """
        ues RNN model to predict classification.
    """
    def infer(self, sentences):
        words = []
        for text in sentences:
            words.append(' '.join(jieba.cut(text)))
        classifier = fasttext.load_model('save_model/fast_text/fasttext_Model' + '.bin')
        predicts = classifier.predict(words)
        return predicts