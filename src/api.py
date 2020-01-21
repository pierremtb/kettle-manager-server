import os
import flask
import numpy as np
from flask import request, send_file
import tensorflow as tf

from model import BoilModel

app = flask.Flask(__name__)
app.config["DEBUG"] = True

temp_tfl_path = os.path.join("temp", "tfl_model.h5")

def get_model_path(device_id):
    return os.path.join("data", str(device_id))

@app.route('/', methods=['GET'])
def home():
    return "<h1>Kettle Manager Server</h1><p>This site is a prototype API for server neural networks training.</p>"

@app.route('/data', methods=['POST'])
def post_data():
    device_id = request.args["device_id"]
    content = request.files["out.csv"]
    Xy = np.loadtxt(content, delimiter=",")
    X = Xy[:, 0:2]
    y = Xy[:, 2].reshape(-1, 1)
    model_path = get_model_path(device_id)
    boil_model = BoilModel(model_path)
    boil_model.train(X, y)
    return "Training done"

@app.route('/model', methods=['GET'])
def get_model():
    device_id = request.args["device_id"]

    model_path = get_model_path(device_id)
    model = BoilModel(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model.model)
    tflite_model = converter.convert()
    with open(temp_tfl_path, "wb") as f:
        f.write(tflite_model)
    return send_file(os.path.join("..", temp_tfl_path), as_attachment=True) 

app.run(host="0.0.0.0", port=80)