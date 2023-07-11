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


# nltk.download('stopwords')
# nltk.download('brown')
# nltk.download('punkt')
# nltk.download('wordnet')

def remove_url(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_non_ascii(text):
    return re.sub(
        r"[^\x00-\x7f]", r'', text
    )


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def lemmatize_word(text):
    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(word, tag) for word, tag in text]
    return lemma

def remove_emoji(text):
    emoji_pattern = re.compile(
        u'(\U0001F1F2\U0001F1F4)|'       # Macau flag
        u'([\U0001F1E6-\U0001F1FF]{2})|' # flags
        u'([\U0001F600-\U0001F64F])'     # emoticons
        "+", flags=re.UNICODE)

    return emoji_pattern.sub('', text)

wordnet_map = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'J': wordnet.ADJ,
        'R': wordnet.ADV
    }

train_senst = brown.tagged_sents(categories='news')
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(train_senst, backoff=t0)
t2 = nltk.BigramTagger(train_senst, backoff=t1)



def pos_tag_wordnet(text, post_tag_type='pos_tag'):


    pos_tagged_text = t2.tag(text)

    pos_tagged_text = [
        (word, wordnet_map.get(pos_tag[0])) if pos_tag[0] in wordnet_map.keys() else (word, wordnet.NOUN) for
        (word, pos_tag) in pos_tagged_text
    ]

    return pos_tagged_text


stop = set(stopwords.words('english'))



def text_processing(df):

    df['title_clean'] = df.title.apply(lambda x: x.lower())
    df['title_clean'] = df['title_clean'].apply(lambda x: contractions.fix(x))
    df['title_clean'] = df['title_clean'].apply(lambda x: remove_url(x))
    df['title_clean'] = df['title_clean'].apply(lambda x: remove_non_ascii(x))
    df['title_clean'] = df['title_clean'].apply(lambda x: remove_emoji(x))
    df['title_clean'] = df['title_clean'].apply(lambda x: remove_punctuation(x))

    df['title_tokenized'] = df.title_clean.apply(word_tokenize)

    df['title_stopwords_removed'] = df.title_tokenized.apply(
        lambda x: [word for word in x if word not in stop]
    )

    df['title_pos_tag_wordnet'] = df.title_stopwords_removed.apply(lambda x: pos_tag_wordnet(x))

    df['lemmatize_title_w_pos'] = df.title_pos_tag_wordnet.apply(lambda x: lemmatize_word(x))

    df['lemmatize_title_w_pos'] = df['lemmatize_title_w_pos'].apply(lambda x: [word for word in x if word not in stop])

    df['lemmatize_title'] = [" ".join(map(str, l)) for l in df.lemmatize_title_w_pos]

    return df
