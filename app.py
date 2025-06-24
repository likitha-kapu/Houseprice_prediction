from flask import Flask, render_template, request
import joblib
import numpy as np
app = Flask(__name__)
model = joblib.load("model.pkl")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(features)])
    return render_template("index.html", prediction_text=f"Predicted Price: {prediction[0]:.2f} per unit area")
if __name__ == "__main__":
    app.run(debug=True)
