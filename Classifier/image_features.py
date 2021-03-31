import base64
import io
import cv2

def Decode_Extract_Features(base64_img):
    img = cv2.imread(io.BytesIO(base64.b64decode(base64_img)))
    print(img)
    return img
