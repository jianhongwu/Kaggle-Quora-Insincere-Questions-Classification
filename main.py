# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:27:25 2019

@author: WJH
"""
#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import numpy as np
import pandas as pd
from utils import read_data_from_disk
from models import model_fasttext, model_cnn
from models import model_rnn_gru1_pool, model_rnn_gru3_atten, model_rnn_pool_atten
from sklearn.model_selection import train_test_split
from models import train_pred, find_best_thresh

#%% Read data
start_time = time.time()
train_X,test_X,train_y,features,test_features,embedding_matrix_mean = read_data_from_disk()
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, 
                                                  test_size=0.1, random_state = SEED, 
                                                  stratify= train_y)

#%% Train and Predict
model = []
outputs = []
model_name = ["model_cnn","model_fasttext","model_rnn_gru1_pool","model_rnn_gru3_atten","model_rnn_pool_atten"]


model.append(model_cnn(embedding_matrix_mean)) #epoch=1, 5min
model.append(model_fasttext(embedding_matrix_mean)) #epoch=1,3min
model.append(model_rnn_gru1_pool(embedding_matrix_mean)) #epoch=1,4min
model.append(model_rnn_gru3_atten(embedding_matrix_mean)) #epoch=1,8.5min
model.append(model_rnn_pool_atten(embedding_matrix_mean)) #epoch=1,5.5min


for i in range(len(model)):
    pred_val_y, pred_test_y = train_pred(model[i],train_X, train_y,val_X,val_y,test_X, epochs=2)
    best_thresh,best_score = find_best_thresh(val_y, pred_val_y,verbose=0)
    outputs.append([pred_val_y,pred_test_y,best_score,best_thresh,model_name[i]])
    print(f"{model_name[i]} finished")
    print("")
    
#%%
outputs.sort(key = lambda x : x[2])
for output in outputs:
    print(output[2],output[3],output[4])
    
from sklearn.linear_model import LinearRegression
X = np.asarray([outputs[i][0] for i in range(len(outputs))])
X = X[...,0]
reg = LinearRegression().fit(X.T, val_y)
print(reg.score(X.T, val_y),reg.coef_)
weights = reg.coef_
print(weights)

pred_val_y = np.mean([outputs[i][0] * weights[i] for i in range(len(outputs))], axis = 0)
best_thresh,best_score = find_best_thresh(val_y, pred_val_y)
pre_test_y = np.mean([outputs[i][1] * weights[i] for i in range(len(outputs))], axis = 0)
test_predict = ((pre_test_y)>best_thresh).astype(int)

confusion_mtx = confusion_matrix(val_y, ((pred_val_y)>best_thresh).astype(int))
plot_confusion_matrix(confusion_mtx, classes = range(2),normalize=True)
print(confusion_mtx)

sub = pd.read_csv('../input/sample_submission.csv')
sub['prediction'] = test_predict
sub.to_csv('submission.csv', index=False)
print("Overall time:",elapsed(time.time()-stage1_time))




































