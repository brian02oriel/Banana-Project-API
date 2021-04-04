import base64
import io
import cv2
from imageio import imread
import numpy as np
from joblib import load
import pickle
from sklearn.metrics.pairwise import euclidean_distances

def get_features(img, net):
  blob = cv2.dnn.blobFromImage(np.asarray(img), 1, (224, 224), (104, 117, 123))
  net.setInput(blob)
  preds = net.forward(outputName='pool5')
  return preds[0]

def Decode_Extract_Features(base64_img):
    img = imread(io.BytesIO(base64.b64decode(base64_img)))
    img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    img = cv2.resize(img, (150, 150))
    model_file = "Classifier/Models/ResNet-50-model.caffemodel"
    deploy_prototxt = "Classifier/Models/ResNet-50-deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(deploy_prototxt, model_file)
    features = get_features(img, net)
    print(features.shape)
    features = features.reshape(-1, 1)
    print(features.shape)
    #kmeans = load("Classifier/Models/KMEANS_MODEL.joblib")
    with open("Classifier/Models/KMEANS_MODEL.pkl", "rb") as file:
      kmeans = pickle.load(file)
    kmeans_prediction = kmeans.predict(features)
    kmeans_centers = kmeans.cluster_centers_
    prediction_center = kmeans_centers[kmeans_prediction]
    kmeans_distance_from_center = euclidean_distances([features], [prediction_center])
    print(kmeans_prediction)
    print(kmeans_distance_from_center)
    return features
