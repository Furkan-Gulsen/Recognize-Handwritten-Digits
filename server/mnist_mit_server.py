# import the necessary libraries
from tensorflow.keras.models import load_model
from flask import Response
from flask import request
from flask import Flask
import numpy as np
import base64
import cv2


app = Flask(__name__)
model = None

@app.route('/predict', methods=['POST'])
def predict():
    content = request.form['data']
    img = np.fromstring(base64.b64decode(content[22:]), dtype=np.uint8)
    character = cv2.imdecode(img, 0)
    resized_character = cv2.resize(character, (28, 28)).astype('float32') / 255
    number = model.predict_classes(resized_character.reshape((1, 784)))[0]
    resp = Response(str(number))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route('/sample_post', methods=['POST'])
def sample_post():
    content = request.form['data']
    return content


@app.route('/')
def hello_world():
    character = cv2.imread('3.png', 0)
    resized_character = cv2.resize(character, (28, 28)).astype('float32') / 255
    number = model.predict_classes(resized_character.reshape((1, 28 * 28)))[0]
    resp = Response(str(number))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == '__main__':
    model = load_model("model.h5")
    app.run(debug=True, host='0.0.0.0', port=8888)