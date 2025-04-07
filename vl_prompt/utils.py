import numpy as np
import cv2
import base64
def decode_base64(base64_data):
    with open('./base64.jpg', 'wb') as file:
        img = base64.b64decode(base64_data)
        file.write(img)
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_gpt4v(image):
    return 'data:image/jpeg;base64,' + encode_image(image)