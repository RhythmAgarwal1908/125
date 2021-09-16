from re import U
from flask import Flask,jsonify,request
from werkzeug.exceptions import MethodNotAllowed
from classifier import get_prediction
app = Flask(__name__)
@app.route('/predict-digit',methods=["POST"])
def predict_data():
    image=request.files.get("DIGIT")
    prediction=get_prediction(image)
    return jsonify({"Prediction":prediction}),200
if __name__ == "__main__":
    app.run(debug=True)