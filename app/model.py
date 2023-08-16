# -*- coding: UTF-8 -*-
import pickle
import gzip
import numpy as np

with gzip.open('app/model/random_forest_model.pgz', 'r') as f:
    rfModel = pickle.load(f)

with gzip.open('app/model/min_max_scaler.pgz', 'r') as f:
    scaler = pickle.load(f)

def predict(input):
    input_without_month = input[:, :-1]
    input_scaled = scaler.transform(input_without_month)
    input_data = np.hstack((input_scaled, input[:, -1:]))
    pred=rfModel.predict(input_data)[0]
    print(pred)
    return pred

