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
  model = joblib.load("Classifier/Models/Banana_bin_classifier.joblib")
  prediction = model.predict(hist.reshape(1, -1))

  return prediction

def get_features_rgb(src):
    src_row1 = cv2.hconcat([src, src, src, src])
    src_row2 = cv2.hconcat([src, src, src, src])
    src_row3 = cv2.hconcat([src, src, src, src])
    src_row4 = cv2.hconcat([src, src, src, src])
    src = cv2.vconcat([src_row1, src_row2, src_row3, src_row4])
    src_count = src.size
    histSize = 256
    histRange = (0, 256)
    accumulate = False
    b_hist = cv2.calcHist(src, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist(src, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist(src, [2], None, [histSize], histRange, accumulate=accumulate)
    b_hp = [b * 100 / src_count for b in b_hist]
    g_hp = [g * 100 / src_count for g in g_hist]
    r_hp = [r * 100 / src_count for r in r_hist]
    return b_hp, g_hp, r_hp


def Decode_Extract_Features(base64_img):
  img = imread(io.BytesIO(base64.b64decode(base64_img)))
  img = cv2.resize(img, (150, 150))
  is_banana = check_banana(img)
  if(is_banana):
    print("IS BANANA")
  else:
    print("NOT BANANA")
  
  b, g, r = get_features_rgb(img)
  features = r + g + b
  rf = joblib.load("Classifier/Models/rf_regressor.joblib")
  rf_prediction = math.floor(rf.predict(features))
  mlp = joblib.load("Classifier/Models/mlp_regressor.joblib")
  mlp_prediction = math.floor(mlp.predict(features))
  svr = joblib.load("Classifier/Models/svr_regressor.joblib")
  svr_prediction = math.floor(svr.predict(features))
  print("RandomForest: ", rf_prediction, "MLP: ", mlp_prediction, "SVR: ", svr_prediction)
  regressions = [rf_prediction, mlp_prediction, svr_prediction]
  regressions.sort()
  response = {
    "days_higher": str(regressions[2]),
    "days_lower": str(regressions[0])
  }
  return response

