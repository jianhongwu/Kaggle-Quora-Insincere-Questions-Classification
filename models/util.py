# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 21:01:46 2019

@author: WJH
"""
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from keras import backend as K
#%%

def f1(y_true, y_pred):
    '''
    metric from here 
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm,4)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def find_best_thresh(val_y, pred_val_y,verbose=1):
    """
    阈值选择
    """
    _threshold=[]
    for thresh in np.arange(0.0,0.5,0.01):
        thresh = np.round(thresh,2)
        _threshold.append([thresh,metrics.f1_score(val_y,(pred_val_y>thresh).astype(int))])
        if verbose:
            print("F1 score at threshold {0} is {1}".format(thresh,metrics.f1_score(val_y,(pred_val_y>thresh).astype(int))))
        
    _threshold = np.array(_threshold)
    best_id = _threshold[:,1].argmax()
    best_thresh = _threshold[best_id][0]
    best_score = _threshold[best_id][1]
    print()
    print("Best threshold is {},Best score is {}".format(best_thresh,best_score))
    return best_thresh,best_score
    
def submit(test_predict):
    """
    提交结果
    """
    sub = pd.read_csv('../input/sample_submission.csv')
    sub['prediction'] = test_predict.astype(int)
    sub.to_csv('submission.csv', index=False)
    print("Submit Finished")
    
    
def elapsed(sec):
    """
    时间计算
    """
    if sec<60:
        return str(sec)+"sec"
    elif sec<(60*60):
        return str(np.round(sec/60.0,2)) + "min"

