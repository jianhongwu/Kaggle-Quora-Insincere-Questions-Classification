# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:00:56 2019

@author: WJH
"""
import pandas as pd
import numpy as np
import re
import time
import gc

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from cfg import * 
from sklearn import metrics
#%%

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", 
                "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", 
                "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", 
                "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", 
                "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 
                "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", 
                "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", 
                "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", 
                "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", 
                "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", 
                "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is",
                "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", 
                "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", 
                "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  
                "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", 
                "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", 
                "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",
                "y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", 
                "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 
                'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 
                'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 
                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 
                'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 
                'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 
                'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 
                'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 
                'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 
                'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 
                'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

def clean_text(x):
    """
    清洗标点
    """
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def clean_numbers(x):
    """
    清洗数字
    """
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]
    return mispellings_re.sub(replace, text)

def add_features(df):
    """
    增加一些统计特征
    """
    df['question_text'] = df['question_text'].apply(lambda x:str(x))
    df['total_length'] = df['question_text'].apply(len)
    df['capitals'] = df['question_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),
                                axis=1)
    df['num_words'] = df.question_text.str.count('\S+')
    df['num_unique_words'] = df['question_text'].apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']  

    return df

def load_and_prec():
    """
    读取数据与初处理
    """
    start_time = time.time()
    print("+++++loading and precessing data+++++")
    
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    # lower
    train_df["question_text"] = train_df["question_text"].apply(lambda x: x.lower())
    test_df["question_text"] = test_df["question_text"].apply(lambda x: x.lower())
    
    # Clean the text
    train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))
    
    # Clean numbers
    train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_numbers(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_numbers(x))
    
    # Clean speelings
    train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: replace_typical_misspell(x))
    
    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values
    
    ## add features
#    train = add_features(train_df)
#    test = add_features(test_df)

#    features = train[['caps_vs_length', 'words_vs_unique']].fillna(0)
#    test_features = test[['caps_vs_length', 'words_vs_unique']].fillna(0)

#    ss = StandardScaler()
#    ss.fit(np.vstack((features, test_features)))
#    features = ss.transform(features)
#    test_features = ss.transform(test_features)

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=max_len)
    test_X = pad_sequences(test_X, maxlen=max_len)

    ## Get the target values
    train_y = train_df['target'].values
    
    #shuffling the data
    np.random.seed(SEED)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    
    print(f"+++++loading and precossing finished: {elapsed(time.time()-start_time)}+++++")
    
    return train_X, test_X, train_y, tokenizer.word_index


def load_word_embeddings(word_index,name='glove'):
    start_time = time.time()
    print(f"+++++loading {name}+++++")
    
    def get_coefs(word,*arr):
        return word, np.asarray(arr,dtype='float32')
    
    if name =='glove':
        EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)
    elif name == 'fasttext':
        EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)
    elif name == 'paragram':
        EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)
    elif name =='word2vec':
        word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin", binary=True)
        embeddings_index={}
        for word in word2vecDict.wv.vocab:
            embeddings_index[word] = word2vecDict.word_vec(word)
    else:
        raise NameError
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(),all_embs.std()
    embed_size = all_embs.shape[1]
    
    num_words = min(max_features,len(word_index))
    embedding_matrix = np.random.normal(emb_mean,emb_std,(num_words, embed_size))
    for word,i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print(f"+++++loading {name} finished: {elapsed(time.time()-start_time)}+++++")
    del(embeddings_index)
    gc.collect()
    return embedding_matrix


def read_data_from_path():
    seed_everything()

    train_X, test_X, train_y, word_index = load_and_prec()

    #平均两个词向量
    embedding_matrix_glove = load_word_embeddings(word_index,'glove')
    embedding_matrix_paragram = load_word_embeddings(word_index,'paragram')
    embedding_matrix_mean = np.mean([embedding_matrix_glove, embedding_matrix_paragram], axis=0)
    
#    save_data_to_disk(train_X, test_X, train_y, features, test_features, word_index,embedding_matrix_glove,embedding_matrix_paragram,embedding_matrix_mean)
    
    gc.collect()
    
    return  train_X, test_X, train_y, embedding_matrix_mean

def save_data_to_disk(train_X, test_X, train_y, features, test_features, word_index,embedding_matrix_glove,embedding_matrix_paragram,embedding_matrix_mean):
    np.save("train_X",train_X)
    np.save("test_X",test_X)
    np.save("train_y",train_y)

    np.save("features",features)
    np.save("test_features",test_features)
    np.save("word_index.npy",word_index)
    np.save("embedding_matrix_mean",embedding_matrix_mean)
    np.save("embedding_matrix_glove",embedding_matrix_glove)
    np.save("embedding_matrix_paragram",embedding_matrix_paragram)
    
    
def read_data_from_disk():
    train_X = np.load("../input/final-version/train_X.npy")
    test_X = np.load("../input/final-version/test_X.npy")
    train_y = np.load("../input/final-version/train_y.npy")
    features = np.load("../input/final-version/features.npy")
    test_features = np.load("../input/final-version/test_features.npy")
    #word_index = np.load("word_index.npy").item()
    embedding_matrix_mean = np.load("../input/final-version/embedding_matrix_mean.npy")
    # embedding_matrix_glove = np.load("embedding_matrix_glove.npy")
    # embedding_matrix_paragram = np.load("embedding_matrix_paragram.npy")

    print(f"train_X shape: {train_X.shape}")
    print(f"test_X shape: {test_X.shape}")
    print(f"train_y shape: {train_y.shape}")
    print(f"embedding_matrix_mean shape: {embedding_matrix_mean.shape}")
    
    return train_X,test_X,train_y,features,test_features,embedding_matrix_mean

    
    
    

