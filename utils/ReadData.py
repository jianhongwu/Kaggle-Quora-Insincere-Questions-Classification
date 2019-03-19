# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:21:23 2019

@author: WJH
"""
import numpy as np
import fire

def read_data_from_disk():
    '''
    从磁盘中读取数据
    '''
    
    train_X = np.load("/home/ubuntu/kaggle/data/x_train.npy")
    test_X = np.load("/home/ubuntu/kaggle/data/x_test.npy")
    train_y = np.load("/home/ubuntu/kaggle/data/y_train.npy")
    features = np.load("/home/ubuntu/kaggle/data/features.npy")
    test_features = np.load("/home/ubuntu/kaggle/data/test_features.npy")
    embedding_matrix_mean = np.load("/home/ubuntu/kaggle/data/embedding_matrix.npy")
    # word_index = np.load("word_index.npy").item()
    # embedding_matrix_glove = np.load("embedding_matrix_glove.npy")
    # embedding_matrix_paragram = np.load("embedding_matrix_paragram.npy")

    print(f"train_X shape: {train_X.shape}")
    print(f"test_X shape: {test_X.shape}")
    print(f"train_y shape: {train_y.shape}")
    print(f"embedding_matrix_mean shape: {embedding_matrix_mean.shape}")
    
    return train_X,test_X,train_y,features,test_features,embedding_matrix_mean #,word_index,embedding_matrix_glove,embedding_matrix_paragram

if __name__ == '__main__':
    fire.Fire()