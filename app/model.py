# -*- coding: UTF-8 -*-
import pickle
import gzip
import numpy as np

# 載入Model
with gzip.open('app/model/random_forest_model_north.pgz', 'r') as f:
    rfModel_north = pickle.load(f)

with gzip.open('app/model/min_max_scaler_north.pgz', 'r') as f:
    scaler_north = pickle.load(f)

with gzip.open('app/model/random_forest_model_middle.pgz', 'r') as f:
    rfModel_middle = pickle.load(f)

with gzip.open('app/model/min_max_scaler_middle.pgz', 'r') as f:
    scaler_middle = pickle.load(f)

def predictNorth(input):
    input_without_month = input[:, :-1]
    input_scaled = scaler_north.transform(input_without_month)
    input_data = np.hstack((input_scaled, input[:, -1:]))
    pred=rfModel_north.predict(input_data)[0]
    print(pred)
    return pred

def predictMiddle(input):
    input_without_month = input[:, :-1]
    input_scaled = scaler_middle.transform(input_without_month)
    input_data = np.hstack((input_scaled, input[:, -1:]))
    pred=rfModel_middle.predict(input_data)[0]
    print(pred)
    return pred
