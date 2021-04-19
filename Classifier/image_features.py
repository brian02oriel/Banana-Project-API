import base64
import io
import cv2
from imageio import imread
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import joblib
import math
#from Classifier.LocalBinaryPattern import *

def get_features(img, net):
  blob = cv2.dnn.blobFromImage(np.asarray(img), 1, (224, 224), (104, 117, 123))
  net.setInput(blob)
  preds = net.forward(outputName='pool5')
  return preds[0]

def check_banana(input_image):
  radius = 3
  no_points = 8 * radius
  desc = LocalBinaryPatterns(no_points, radius)
  image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
  hist = desc.describe(image)
  hist = np.array(hist)
  model = joblib.load("Classifier/Models/IS_BANANA_MODEL.joblib")
  prediction = model.predict(hist.reshape(1, -1))
  return prediction



def Decode_Extract_Features(base64_img):
  img = imread(io.BytesIO(base64.b64decode(base64_img)))
  #is_banana = check_banana(img)
  #if(is_banana):
  #  print("IS BANANA")
  #else:
  #  print("NOT BANANA")
  img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
  img = cv2.resize(img, (150, 150))
  model_file = "Classifier/Models/ResNet-50-model.caffemodel"
  deploy_prototxt = "Classifier/Models/ResNet-50-deploy.prototxt"
  net = cv2.dnn.readNetFromCaffe(deploy_prototxt, model_file)
  features = get_features(img, net)
  features = features.reshape(-1)
  features = features.reshape(1, -1)
  kmeans = joblib.load("Classifier/Models/KMEANS_MODEL.joblib")
  kmeans_prediction = kmeans.predict(features)
  print("kmeans prediction: ",kmeans_prediction[0])
  kmeans_centers = kmeans.cluster_centers_
  prediction_center = kmeans_centers[kmeans_prediction]
  kmeans_distance_from_center = euclidean_distances(features, prediction_center)
  kmeans_distance_from_center = kmeans_distance_from_center.reshape(-1)
  knn = joblib.load("Classifier/Models/BANANA_RIPENESS_CLASSIFIER.joblib")
  knn_prediction = knn.predict(features)
  print("knn prediction: ", knn_prediction[0])
  rf = joblib.load("Classifier/Models/BANANA_REMAINING_DAYS_RF_REGRESSOR.joblib")
  regression_params = [[knn_prediction[0], kmeans_distance_from_center[0]]]
  rf_prediction = math.floor(rf.predict(regression_params))
  mlp = joblib.load("Classifier/Models/BANANA_REMAINING_DAYS_MLP_REGRESSOR.joblib")
  mlp_prediction = math.floor(mlp.predict(regression_params))
  svr = joblib.load("Classifier/Models/BANANA_REMAINING_DAYS_SVR_REGRESSOR.joblib")
  svr_prediction = math.floor(svr.predict(regression_params))
  print("RandomForest: ", rf_prediction, "MLP: ", mlp_prediction, "SVR: ", svr_prediction)
  regressions = [rf_prediction, mlp_prediction, svr_prediction]
  regressions.sort()
  response = {
    "group": str(knn_prediction[0]),
    "days_higher": str(regressions[2]),
    "days_lower": str(regressions[0])
  }
  return response

