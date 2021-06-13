import base64
import io
import cv2
from imageio import imread
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import joblib
import math
from Classifier.LocalBinaryPattern import *

def get_image_center_sample(img):
  box_w = 65
  box_h = 65
  (h, w) = img.shape[:2]
  (cX, cY) = (w // 2, h // 2)
  x = cX - box_w/2
  y = cY - box_h/2
  crop_img = img[int(y):int(y+box_h), int(x):int(x+box_w)]
  return crop_img

def binMask(img):
    img = cv2.resize(img, (350, 350))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
    canny = cv2.Canny(blur, 50, 150)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pts = np.argwhere(canny > 0)
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)
    cropped = img[y1:y2, x1:x2]
   
    return cropped

def check_banana(img):
  radius = 3
  no_points = 8 * radius
  desc = LocalBinaryPatterns(no_points, radius)
  img = cv2.resize(img, (150, 150))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  hist = desc.describe(img)
  hist = np.array(hist)
  model = joblib.load("Classifier/Models/Banana_bin_classifier.joblib")
  prediction = model.predict(hist.reshape(1, -1))

  return prediction

def get_features_rgb(src):
    src = cv2.resize(src, (150, 150))
    src_row1 = cv2.hconcat([src, src, src, src])
    src_row2 = cv2.hconcat([src, src, src, src])
    src_row3 = cv2.hconcat([src, src, src, src])
    src_row4 = cv2.hconcat([src, src, src, src])
    src = cv2.vconcat([src_row1, src_row2, src_row3, src_row4])
    cv2.imshow("input framed", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
  print(img.size)
  img = cv2.resize(img, (300, 400))
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  is_banana = check_banana(img)
  if(is_banana):
    print("IS BANANA")
  else:
    print("NOT BANANA")
  
  img = binMask(img)
  #cv2.imshow("input", img)

  #img = get_image_center_sample(img)
  #cv2.imshow("input cropped", img)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()

  r, g, b = get_features_rgb(img)
  features = b + g + r
  features = np.array(features)
  features = np.squeeze(features)
  features = features.reshape(1, -1)
  rf = joblib.load("Classifier/Models/rf_regressor.joblib")
  rf_prediction = math.floor(rf.predict(features))
  mlp = joblib.load("Classifier/Models/mlp_regressor.joblib")
  mlp_prediction = math.floor(mlp.predict(features))
  svr = joblib.load("Classifier/Models/svr_regressor.joblib")
  svr_prediction = math.floor(svr.predict(features))
  print("RandomForest: ", rf_prediction, "MLP: ", mlp_prediction, "SVR: ", svr_prediction)
  regressions = [rf_prediction, svr_prediction, mlp_prediction]
  regressions.sort()
  response = {
    "days_higher": regressions[1],
    "days_lower": regressions[0]
  }
  return response

