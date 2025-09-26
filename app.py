import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import logging

application = Flask(__name__)
app = application

## Route for a home page


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    app.logger.info("✅ /predict route triggered")
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = request.form
        custom_data = CustomData(
            gender=data.get("gender"),
            ethnicity=data.get("ethnicity"),
            parental_level_of_education=data.get("parental_level_of_education"),
            lunch=data.get("lunch"),
            test_preparation_course=data.get("test_preparation_course"),
            reading_score=float(data.get("reading_score")),
            writing_score=float(data.get("writing_score")),
        )
        custom_data_df = custom_data.get_data_as_dataframe()
        print(custom_data_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(custom_data_df)
        # app.logger.info(f"Prediction result: {results[0]}")
        print("Prediction result:", results[0], flush=True)
        return render_template("home.html", results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
    # app.run(debug=True)
