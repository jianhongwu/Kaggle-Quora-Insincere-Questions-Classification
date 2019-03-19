# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:46:49 2019

@author: WJH
"""
from keras.layers import Input, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape
from keras.layers import Conv2D, MaxPool2D, Concatenate, Dense, Dropout, Flatten
from keras.models import Model

def model_fasttext(embedding_matrix_mean, max_len=70, max_features=120000, embed_size=300):
    '''
    FastText
    '''
    
    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embed_size,weights=[embedding_matrix_mean],trainable=True)(inp)
    x1 = GlobalAveragePooling1D()(x)
    x2 = GlobalMaxPooling1D()(x)
    x3 = Reshape((max_len, embed_size, 1))(x)
    x3 = Conv2D(42, kernel_size=(3, embed_size),
                                 kernel_initializer='he_normal', activation='tanh')(x3)
    x3 =  MaxPool2D(pool_size=(max_len - 3 + 1, 1))(x3)
    x3 = Reshape((42,))(x3)
    x = Concatenate(axis=1)([x1,x2,x3])
    x = Dense(128,activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(64,activation="relu")(x)
    x = Dropout(0.1)(x)
    outp = Dense(1,activation="sigmoid")(x)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model_cnn(embedding_matrix,max_len=70, max_features=120000, embed_size=300):
    '''
    TextCNN
    '''
    filter_sizes = [1,2,3,5]
    num_filters = 42

    inp = Input(shape=(max_len,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable=True)(inp)
    x = Reshape((max_len, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                     kernel_initializer='he_normal',activation='tanh')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(max_len - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model