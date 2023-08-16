# -*- coding: UTF-8 -*-
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import app.model as model

app = Flask(__name__)
CORS(app)

@app.route('/test', methods=['GET'])
def getResult():
    input = np.array([[1014.0, 27.1, 20.9, 77, 4.5, 7.8, 12]])
    result = model.predict(input)
    return jsonify({'result':str(result)})

@app.route('/predict', methods=['POST'])
def postInput():
    #取得前端傳的值
    insertValues = request.get_json()
    x1=insertValues['氣壓']
    x2=insertValues['最高溫度']
    x3=insertValues['最低溫度']
    x4=insertValues['相對濕度']
    x5=insertValues['風速']
    x6=insertValues['雲量']
    x7=insertValues['月份']
    input = np.array([[x1,x2,x3,x4,x5,x6,x7]])
    
    result = model.predict(input)
    print(input)

    return jsonify({'return':str(result)})
