import base64
import io
import cv2
from imageio import imread

def get_features(img, net):
  layers = net.getLayerNames() 
  blob = cv2.dnn.blobFromImage(np.asarray(img), 1, (224, 224), (104, 117, 123))
  net.setInput(blob)
  preds = net.forward(outputName='pool5')
  return preds[0]

def Decode_Extract_Features(base64_img):
    img = imread(io.BytesIO(base64.b64decode(base64_img)))
    img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    img = cv2.resize(img, (150, 150))
    model_file = "../ModelsResNet-50-model.caffemodel"
    deploy_prototxt = "../Models/ResNet-50-deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(deploy_prototxt, model_file)
    features = get_features(img, net)
    print(features.shape)
    return features
