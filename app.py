from flask import Flask, request, render_template
import pandas as pd
from src.utils import load_object
from src.exception import CustomException
import sys

app = Flask(__name__)

# Load preprocessor and model 
try:
    preprocessor = load_object("artifacts/preprocessor.pkl")
    model = load_object("artifacts/best_model.pkl")
except Exception as e:
    raise CustomException(e, sys)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            income = float(request.form["income"])
            age = float(request.form["age"])
            loan = float(request.form["loan"])

            # DataFrame and transform
            input_df = pd.DataFrame([[income, age, loan]], columns=["income", "age", "loan"])
            X_transformed = preprocessor.transform(input_df)

            # Predict
            pred = model.predict(X_transformed)
            prediction = int(pred[0])
        except Exception as e:
            raise CustomException(e, sys)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0",port=5000)
    