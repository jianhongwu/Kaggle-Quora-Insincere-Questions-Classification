# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:09:26 2019

@author: WJH
"""
#%%
import time
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

def elapsed(sec):
    """
    时间计算
    """
    if sec<60:
        return str(sec)+"sec"
    elif sec<(60*60):
        return str(np.round(sec/60.0,2)) + "min"

def train_pred(model, train_X, train_y, val_X, val_y, test_X,epochs=2):
    """
    模型训练
    """
    print("training begin.......")
    pred_val_y = np.zeros((val_y.shape[0],1))
    pred_test_y = np.zeros((test_X.shape[0],1))
    for e in range(epochs):
        start_time = time.time()
        model.fit(train_X,train_y,batch_size=512,epochs=1,validation_data=(val_X,val_y))
        temp_val_y = model.predict([val_X],batch_size=1024,verbose=1)
        pred_test_y += model.predict([test_X],batch_size=1024,verbose=1)/epochs
        
        best_thresh = 0.5
        best_score = 0.0
        for thresh in np.arange(0.1, 0.501, 0.01):
            thresh = np.round(thresh, 2)
            score = metrics.f1_score(val_y, (temp_val_y > thresh).astype(int))
            if score > best_score:
                best_thresh = thresh
                best_score = score

        print("Best Val F1 Score at {} is: {:.4f}".format(best_thresh,best_score))
        print(f"epoch finished: {elapsed(time.time()-start_time)}")
        
        pred_val_y += temp_val_y/epochs
        
    return pred_val_y, pred_test_y

def train_corss_val(model,train_X,train_y,test_X,epochs=2):
    spliter = StratifiedKFold(n_splits=4, shuffle=False)
    train_predict = np.zeros(train_y.shape)
    test_predict = np.zeros(test_X.shape[0])
    
    stage3_time = time.time()
    for fold_id,(train_idx,val_idx) in enumerate(spliter.split(train_X,train_y)):
        print('FOLD:',fold_id)
        X_train = train_X[train_idx]
        y_train = train_y[train_idx]
        X_val = train_X[val_idx]
        y_val = train_y[val_idx]
        
        pred_val_y, pred_test_y = train_pred(model, X_train, y_train, X_val, y_val,test_X,epochs = 5)
        train_predict[val_idx] = pred_val_y.reshape(-1)
        test_predict += pred_test_y
        
    print("Cross Training model used time:", elapsed(time.time()-stage3_time))
    return train_predict,test_predict
