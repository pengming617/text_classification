import model.textcnn.TrainModel as textcnn_train
import model.textrnn.TrainModel as textrnn_train
import model.birnn_attention.TrainModel as birnn_attention_train
import model.char_cnn.TrainModel as charcnn_train
import model.fast_text.TrainModel as fasttext_train
import model.han.TrainModel as han_train
import model.transformer.TrainModel as transformer_train
import model.leam.TrainModel as leam_train
import tensorflow as tf
import json

tf.app.flags.DEFINE_string("model_type", "charcnn", "默认为cnn")
FLAGS = tf.app.flags.FLAGS
model_type = FLAGS.model_type

with open("config/config.json", "r", encoding='utf-8') as f:
    data = json.loads(f.read())
    model_parameters = data['model'][model_type]['model_parameters']

# is_cut是否对语句进行分词
# model_type is one of the ["textcnn","charcnn","fasttext","textrnn","birnn_attention","han","leam","transformer"]
train = None
if model_type == 'textcnn':
    train = textcnn_train.TrainModel()
    embedding_dim = model_parameters['embedding_dim']
    filter_sizes = model_parameters['filter_sizes']
    num_filters = model_parameters['num_filters']
    dropout_keep_prob = model_parameters['dropout_keep_prob']
    l2_reg_lambda = model_parameters['l2_reg_lambda']
    train.trainModel(embedding_dim, filter_sizes, num_filters, dropout_keep_prob, l2_reg_lambda)
elif model_type == 'charcnn':
    embedding_dim = model_parameters['embedding_dim']
    dropout_keep_prob = model_parameters['dropout_keep_prob']
    conv_layers = model_parameters['conv_layers']
    fully_layers = model_parameters['fully_layers']
    train = charcnn_train.TrainModel()
    train.trainModel(embedding_dim, dropout_keep_prob, conv_layers, fully_layers)
elif model_type == 'fasttext':
    train = fasttext_train.TrainModel()
    train.train()
elif model_type == 'textrnn':
    train = textrnn_train.TrainModel()
    embedding_dim = model_parameters['embedding_dim']
    dropout_keep_prob = model_parameters['dropout_keep_prob']
    hidden_num = model_parameters['hidden_num']
    hidden_size = model_parameters['hidden_size']
    l2_reg_lambda = model_parameters['l2_reg_lambda']
    train.trainModel(embedding_dim, dropout_keep_prob, hidden_num, hidden_size, l2_reg_lambda)
elif model_type == 'birnn_attention':
    train = birnn_attention_train.TrainModel()
    embedding_dim = model_parameters['embedding_dim']
    dropout_keep_prob = model_parameters['dropout_keep_prob']
    hidden_num = model_parameters['hidden_num']
    attn_size = model_parameters['attn_size']
    train.trainModel(embedding_dim, dropout_keep_prob, hidden_num, attn_size)
elif model_type == 'han':
    train = han_train.TrainModel()
    embedding_dim = model_parameters['embedding_dim']
    dropout_keep_prob = model_parameters['dropout_keep_prob']
    word_hiddencell = model_parameters['word_hiddencell']
    sentence_hiddencell = model_parameters['sentence_hiddencell']
    word_attention_size = model_parameters['word_attention_size']
    sentence_attention_size = model_parameters['sentence_attention_size']
    train.trainModel(embedding_dim, dropout_keep_prob, word_hiddencell, sentence_hiddencell,
                     word_attention_size, sentence_attention_size)
elif model_type == 'leam':
    train = leam_train.TrainModel()
    embedding_dim = model_parameters['embedding_dim']
    dropout_keep_prob = model_parameters['dropout_keep_prob']
    hidden_num = model_parameters['hidden_num']
    attn_size = model_parameters['attn_size']
    l2_reg_lambda = model_parameters['l2_reg_lambda']
    train.trainModel(embedding_dim, dropout_keep_prob, hidden_num, attn_size, l2_reg_lambda)
elif model_type == 'transformer':
    train = transformer_train.TrainModel()
    embedding_dim = model_parameters['embedding_dim']
    dropout_keep_prob = model_parameters['dropout_keep_prob']
    hidden_num = model_parameters['hidden_num']
    num_blocks = model_parameters['num_blocks']
    num_heads = model_parameters['num_heads']
    train.trainModel(embedding_dim, dropout_keep_prob, hidden_num, num_blocks, num_heads)
else:
    print("do not exist this model")
