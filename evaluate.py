import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np


tf.app.flags.DEFINE_string("model_type", "transformer", "默认为cnn")
FLAGS = tf.app.flags.FLAGS
model_type = FLAGS.model_type

with open("data/test_sentiment.txt", 'r', encoding='utf-8') as fr:
    articles = []
    tags = []
    for line in fr.readlines():
        data = line.replace("\t\t", "\t").replace("\n", "").split("\t")
        if len(data) == 3:
            articles.append(data[1].replace("，", ","))
            tags.append(data[2])
        else:
            print(line + "------格式错误")

infer = None
if model_type == 'textcnn':
    import model.textcnn.Infer as textcnn_infer
    infer = textcnn_infer.Infer()
elif model_type == 'charcnn':
    import model.char_cnn.Infer as char_cnn_infer
    infer = char_cnn_infer.Infer()
elif model_type == 'fasttext':
    import model.fast_text.Infer as fasttext_infer
    infer = fasttext_infer.Infer()
elif model_type == 'textrnn':
    import model.textrnn.Infer as textrnn_infer
    infer = textrnn_infer.Infer()
elif model_type == 'birnn_attention':
    import model.birnn_attention.Infer as birnn_attention_infer
    infer = birnn_attention_infer.Infer()
elif model_type == 'leam':
    import model.leam.Infer as leam_infer
    infer = leam_infer.Infer()
elif model_type == 'transformer':
    import model.transformer.Infer as transformer_infer
    infer = transformer_infer.Infer()
else:
    print("do not exist this model")

predicts, scores = infer.infer(articles)
f1 = f1_score(np.array(tags), np.array(predicts), average='micro')
print("evaluate over f1:{}".format(f1))

