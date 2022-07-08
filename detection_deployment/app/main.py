from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, Body
import numpy as np
from starlette.requests import Request
from fastapi.responses import JSONResponse, Response
from fastapi.encoders import jsonable_encoder
import os
import cv2
import time
import datetime
import json
import base64
import io
from urllib.request import urlopen
from PIL import Image, ImageDraw, ImageColor, ImageFont
import boto3
import onnxruntime as rt
import boto3
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing import image
import requests


s3 = boto3.client('s3')


app = FastAPI()


PATH_TO_CKPT = 'model.onnx'
export_path = "multilabel"

label_map = { '1' : 'RC' } 
classes = [['Card', 'DigiLocker', 'Front','Page','back']]

model_multi = tf.compat.v1.keras.experimental.load_from_saved_model(export_path, custom_objects={'KerasLayer':hub.KerasLayer})

IMG_SIZE = 224 # Specify height and width of image to match the input format of the model
CHANNELS = 3 # Keep RGB color channels to match the input format of the model



NUM_CLASSES = 1
threshold = 0.5

sess = rt.InferenceSession(PATH_TO_CKPT)



def sess_run(img_data, sess):
    # we want the outputs in this order
    outputs = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
    result = sess.run(outputs, {"input_tensor": img_data})
    num_detections, detection_boxes, detection_scores, detection_classes = result
    return result

def draw_output(top, left, bottom, right, output2, img_arr):
    """Draw box and label for 1 detection."""
    draw = ImageDraw.Draw(img_arr)
    label = str(output2)
#     fontsize = 1  # starting font size
#     font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", fontsize, encoding="unic")
#     # portion of image width you want text width to be
#     img_fraction = 0.3
#     while font.getsize(label)[0] < img_fraction*img_arr.size[0]:
#     # iterate until the text size is just larger than the criteria
#         fontsize += 1
#         font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", fontsize, encoding="unic")

#     # optionally de-increment to be sure it is less than criteria
#     fontsize -= 1
#     font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", fontsize, encoding="unic")

    label_size = draw.textsize(label)
    print(label_size)
    if top - label_size[1] >= 0:
        text_origin = tuple(np.array([left, top - label_size[1]]))
    else:
        text_origin = tuple(np.array([left, top + 1]))
    color = ImageColor.getrgb("green")
    thickness = 3
    draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness], outline=color)
    draw.text(text_origin, label, fill=color)
    img = np.array(img_arr)
    return img

def coordinates(width, height, d):
    print('width :' , width)
    print('height :' , height)
    # the box is relative to the image size so we multiply with height and width to get pixels.
    top = d[0] * height
    left = d[1] * width
    bottom = d[2] * height
    right = d[3] * width
    top = int(max(0, np.floor(top + 0.5).astype('int32')))
    left = int(max(0, np.floor(left + 0.5).astype('int32')))
    bottom = int(min(height, np.floor(bottom + 0.5).astype('int32')))
    right = int(min(width, np.floor(right + 0.5).astype('int32')))
    return top, left, bottom, right


def crop_save(top, left, bottom, right, img_arr):
    img = img_arr
    im1 = img.crop((left, top, right, bottom))
    img = np.array(im1)
    cv2.imwrite("cropped.jpg", img)
    return im1

def bbox_class_name(c, s, dict_output):
    """Draw box and label for 1 detection."""
    c = str(c)
    c = c[0]
    print(c)
    label = label_map[c]
    dict_output[label] =  float("{:.3f}".format(s))
    #print(dict)
    return dict_output


def multilabel_class(model, classes, dict_output):
    
    img_path =  'cropped.jpg'
    # Read and prepare image
    img = image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE,CHANNELS))
    img = image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)
    data = model.predict(img)
    
    #print(data[0])
    conf = []
    for i in range(len(data[0])):
        if data[0][i]> 0.5:
            conf.append(data[0][i])
            #print(i, data[0][i])
    prediction = (model.predict(img) > 0.5).astype('int')
    #print('prediction_out :',prediction)
    prediction = pd.Series(prediction[0])
    
    mlb = MultiLabelBinarizer()
    mlb.fit(classes)
    # Loop over all labels and show them
    N_LABELS = len(mlb.classes_)
    prediction.index = mlb.classes_
    prediction = prediction[prediction==1].index.values
    for i in range(len(prediction)):
        #conf1 = round(conf[i],2)
        #print(conf1)
        dict_output[prediction[i]] = float("{:.3f}".format(conf[i]))
    return dict_output

def img_inference(img):
    dict_output = {}
    height, width, channels = img.shape
    img_arr = Image.fromarray(img)
    #image_np = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    img_data = np.expand_dims(img.astype(np.uint8), axis=0)
    result = sess_run(img_data, sess)
    num_detections = result[0]
    detection_boxes = result[1]
    detection_scores = result[2]
    detection_classes = result[3]
    batch_size = num_detections.shape[0]
    for batch in range(0, batch_size):
        for detection in range(0, int(num_detections[batch])):
            if detection_scores[0][detection] > threshold:
                c = str(detection_classes[batch][detection])
                d = detection_boxes[batch][detection]
                s = detection_scores[0][detection]
                dict_output = bbox_class_name(c,s, dict_output)
                top, left, bottom, right = coordinates( width,height, d)
                crop_save(top, left, bottom, right, img_arr)
                dict_output = multilabel_class(model_multi, classes, dict_output)
                img = draw_output(top, left, bottom, right, dict_output, img_arr)
    return img, dict_output
  


                
@app.post("/predict/document_image")
def image_meta_gen(request: Request,userPhoto: Optional[bytes] = File(None), url: Optional[str] = Body(None)):
    if url is not None: 
        image_r = requests.get(url)
        fetch_status = image_r.status_code
        if fetch_status == 200:
            image = image_r.content
            img_np = cv2.imdecode(np.asarray(bytearray(image), dtype=np.uint8), 1)
    elif userPhoto is not None:
        nparr = np.fromstring(userPhoto, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        return {"response":"Please provide url or image"}
  
    s3_push_img, dict_output = img_inference(img_np)
    image_string = cv2.imencode('.jpg', s3_push_img)[1].tostring()
    s3.put_object(Bucket="rz-motor-images", Key = "dre_api/document_temp.jpg", Body=image_string)
    dict_output["s3_url"] = "https://rz-motor-images.s3.amazonaws.com/dre_api/document_temp.jpg"
    return dict_output

    
    
    
    
    

    
   

