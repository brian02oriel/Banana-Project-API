import base64
import io
import cv2
from imageio import imread

def Decode_Extract_Features(base64_img):
    img = imread(io.BytesIO(base64.b64decode(base64_img)))
    print(type(img))
    return img
