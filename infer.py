import model.textrnn.Infer as textrnn_infer
import model.textcnn.Infer as textcnn_infer
import model.birnn_attention.Infer as birnn_attention_infer
import model.char_cnn.Infer as char_cnn_infer
import model.leam.Infer as leam_infer
import model.fast_text.Infer as fasttext_infer
import model.transformer.Infer as transformer_infer
import tensorflow as tf


tf.app.flags.DEFINE_string("model_type", "charcnn", "默认为cnn")
tf.app.flags.DEFINE_string("sentence", "微信可以登录吗", "默认为cnn")
FLAGS = tf.app.flags.FLAGS
model_type = FLAGS.model_type
sentences = FLAGS.sentence

infer = None
if model_type == 'textcnn':
    infer = textcnn_infer.Infer()
elif model_type == 'charcnn':
    infer = char_cnn_infer.Infer()
elif model_type == 'fasttext':
    infer = fasttext_infer.Infer()
elif model_type == 'textrnn':
    infer = textrnn_infer.Infer()
elif model_type == 'birnn_attention':
    infer = birnn_attention_infer.Infer()
elif model_type == 'leam':
    infer = leam_infer.Infer()
elif model_type == 'transformer':
    infer = transformer_infer.Infer()
else:
    print("do not exist this model")
print(infer.infer([sentences]))