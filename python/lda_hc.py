
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



def return_dict_corpus(train_df):
    texts = list(train_df.lemmatize_title_w_pos.values)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    print(f'number of unique tokens: {len(dictionary)}')
    print(f'number of documents : {len(corpus)}')

    return texts, dictionary, corpus


path = Path(r'../Tawosi_Dataset')

train, valid, test = pd.read_csv(path / 'DM-train.csv'), pd.read_csv(path / 'DM-valid.csv'), pd.read_csv(
    path / 'DM-test.csv')

data = pd.concat([train, valid])

texts = []

# print(train)

train = text_processing(train)

texts, dictionary, corpus = return_dict_corpus(train)

num_topics = 20

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

topics = lda_model[corpus]

train_ik = train.issuekey.values


test_prob = {}

topics_number = set(f'topic_{i}' for i in range(num_topics))

for key, prob in zip(train_ik, topics):
    top_preds = {}

    for (topic_no, value) in prob[0]:
        top_preds[f'topic_{topic_no}'] = value


    for tn in topics_number:
        if not top_preds.get(tn, None):
            top_preds[tn] = 0.0

    test_prob[key] = top_preds


prob_df_cols = sorted(list(topics_number), key=lambda x: int(x.split('_')[1]))

prob_df = pd.DataFrame.from_dict(test_prob, orient='index')

prob_df.index.name = 'issuekey'

prob_df.to_csv('prob_df.csv')

dendrogram = sch.dendrogram(sch.linkage(prob_df.values, method='ward'), no_plot=True)

cn = len(set(dendrogram['color_list'])) - 1

print(f"cluster no: {cn}")

ac_m = AgglomerativeClustering(n_clusters=cn, affinity='euclidean', linkage='ward')

preds = ac_m.fit_predict(prob_df.values)

prob_df['labels'] = preds

print(prob_df.labels.value_counts())