import base64
import io
import cv2

def Decode_Extract_Features(base64):
    b64_bytes = base64.b64encode(data)
    b64_string = b64_bytes.decode()
    img = cv2.imread(io.BytesIO(base64.b64decode(b64_string)))
    print(img)
    return img
