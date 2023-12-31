# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:42:36 2019

@author: SY
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from bsgnn.parsing import parse_predict_args, modify_train_args
from bsgnn.train import make_predictions
from bsgnn.data import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,accuracy_score
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math



if __name__ == '__main__':
    args = parse_predict_args()
    df = pd.read_csv(args.test_path, index_col=0)
    pred, smiles = make_predictions(args, df.index.tolist())
    hard_preds = [1 if p[0] > 0.5 else 0 for p in pred]
    df_pred_l = pd.DataFrame()
    pred1 = []
    for i in range(len(pred)):
        pred1.append(pred[i][0])
    y_pred = np.argmax(pred,axis=1)
    true1 = []
    predict_result = pd.DataFrame()
    for i in range(len(smiles)):
        predict_result.loc[smiles[i],'pred'] = 2**pred1[i]
    predict_result.to_csv(args.preds_path)