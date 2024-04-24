from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline

application = Flask(__name__)

app = application

# Route for a home page


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:

        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        date, bias = predict_pipeline.predict()
        print("after Prediction")
        return render_template('home.html', date=date, bias=bias)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
