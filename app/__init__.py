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

@app.route('/predictNorth', methods=['POST'])
def postInput():
    # 取得前端傳的值
    forecastData = request.get_json()

    # 遍歷預報數據
    results = []
    for forecast in forecastData:
        x1 = forecast['氣壓']
        x2 = forecast['最高溫度']
        x3 = forecast['最低溫度']
        x4 = forecast['相對濕度']
        x5 = forecast['風速']
        x6 = forecast['雲量']
        x7 = forecast['月份']
        
        input_data = np.array([[x1, x2, x3, x4, x5, x6, x7]])
        result = model.predictNorth(input_data)
        results.append({'result': str(result)})

    return jsonify(results)


@app.route('/predictMiddle', methods=['POST'])
def postInput():
    # 取得前端傳的值
    forecastData = request.get_json()

    # 遍歷預報數據
    results = []
    for forecast in forecastData:
        x1 = forecast['氣壓']
        x2 = forecast['最高溫度']
        x3 = forecast['最低溫度']
        x4 = forecast['相對濕度']
        x5 = forecast['風速']
        x6 = forecast['雲量']
        x7 = forecast['月份']
        
        input_data = np.array([[x1, x2, x3, x4, x5, x6, x7]])
        result = model.predictMiddle(input_data)
        results.append({'result': str(result)})

    return jsonify(results)
