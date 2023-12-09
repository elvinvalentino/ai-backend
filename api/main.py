from flask import Flask, request
from controllers.PredictController import PredictController
from flask_cors import CORS
import os

UPLOAD_FOLDER =  os.path.join(
        os.path.dirname(__file__), "..", "uploads"
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

@app.route("/predict", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return {"data": "hello world"}
    if request.method == "POST":
        return PredictController.predict(app)
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)