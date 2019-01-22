import fastai
from fastai import *
from fastai.text import *
import pandas as pd
import numpy as np
from functools import partial
import io
import os

from sklearn.datasets import fetch_20newsgroups
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data
print(dataset.target_names)

df = pd.DataFrame({'label': dataset.target, 'text': dataset.data})
print(df.shape)
df = df[df['label'].isin([1,10])]
df = df.reset_index(drop=True)
print(df['label'].value_counts())

# Preprocessing
df['text'] = df['text'].str.replace("[^a-zA-Z]", " ")
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# tokenization
tokenized_doc = df['text'].apply(lambda x: x.split())
# remove stop-words
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
# de-tokenization
detokenized_doc = []
for i in range(len(df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
df['text'] = detokenized_doc

from sklearn.model_selection import train_test_split

# split data into training and validation set
df_trn, df_val = train_test_split(df, stratify=df['label'], test_size=0.4, random_state=12)
print(df_trn.shape, df_val.shape)

# Data Preparation
# Language model data
data_lm = TextLMDataBunch.from_df(train_df=df_trn, valid_df=df_val, path="")

# Classifier model data
data_clas = TextClasDataBunch.from_df(path="", train_df=df_trn, valid_df=df_val, vocab=data_lm.train_ds.vocab, bs=32)

learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.7)

# train the learner object
learn.fit_one_cycle(1, 1e-2)

