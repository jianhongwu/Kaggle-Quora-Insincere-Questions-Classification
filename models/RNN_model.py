# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:58:25 2019

@author: WJH
"""
#%%
from models import Attention
from keras.layers import Input, Embedding, Bidirectional, CuDNNGRU, GlobalAveragePooling1D
from keras.layers import GlobalMaxPooling1D, concatenate, Dense, Dropout, SpatialDropout1D
from keras.models import Model
#%%
def model_rnn_gru1_pool(embedding_matrix, max_len=70, max_features=120000, embed_size=300):
    """
    TextRNN1
    """
    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable=True)(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def model_rnn_gru3_atten(embedding_matrix, max_len=70, max_features=120000, embed_size=300):
    """
    TextRNN2 with attentation
    """
    inp = Input(shape=(max_len,))
    x = Embedding(max_features,embed_size,weights=[embedding_matrix],trainable=True)(inp)
    x = Bidirectional(CuDNNGRU(128,return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(64,return_sequences=True))(x)
    x = Bidirectional(CuDNNGRU(32,return_sequences=True))(x)
    x = Attention(max_len)(x)
    x = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=inp,outputs=x)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    return model


def model_rnn_pool_atten(embedding_matrix, max_len=70, max_features=120000, embed_size=300):
    """
    TextRNN3 RNN1和RNN2的结合体
    """
    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(CuDNNGRU(60, return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(120, return_sequences=True))(x)
    
    atten_1 = Attention(max_len)(x) # skip connect
    atten_2 = Attention(max_len)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)
    
    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(16, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)    

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model