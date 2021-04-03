import sys
import flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin
from Classifier.image_features import Decode_Extract_Features

app = flask.Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
def hello_world():
    print(request.json)
    return "Welcome to Banana Project API"

@app.route('/api/classifier', methods=['POST', 'OPTIONS'])
@cross_origin()
def get_image():
    base64_img = request.json.get('base64')
    features = Decode_Extract_Features(base64_img)
    print("features: ", features.shape)
    return "IMAGE RECEIVED"