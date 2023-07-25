
from text_preprocessing import text_processing

import string
import sys

from pathlib import Path
from gensim import corpora, models
import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import scipy.cluster.hierarchy as sch
from nltk.corpus import *
from nltk.tokenize import *
import contractions
from sklearn.feature_extraction import *
from sklearn.metrics import *
import re
import nltk


from pathlib import Path
import os


def return_train_valid_files(path):
    train_file_list = []

    for o in os.listdir(path):
        if ('valid' in o.lower() or 'train' in o.lower()) and 'features' not in o.lower():
            train_file_list.append(path/o)

    return train_file_list



def return_dict_corpus(train_df):
    texts = list(train_df.lemmatize_title_w_pos.values)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    print(f'number of unique tokens: {len(dictionary)}')
    print(f'number of documents : {len(corpus)}')

    return texts, dictionary, corpus


path = Path(r'/home/sirakr/src/LDA_HC/LHC-SE/Tawosi_Dataset')


training_files = return_train_valid_files(path)

training_files = [pd.read_csv(o) for o in training_files]

train = pd.concat(training_files)

train.to_csv('train.csv')

texts = []

train = text_processing(train)

texts, dictionary, corpus = return_dict_corpus(train)

num_topics = 100

lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=100,
    chunksize=2000,
    iterations=400,
    passes=20,
    per_word_topics=True,
    alpha='auto',
    eta='auto',
    eval_every=True
)


# from gensim.test.utils import datapath
# temp_file = datapath('model')
lda_model.save('lda.model')