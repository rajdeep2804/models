import numpy as np
from PIL import Image, ImageDraw, ImageColor, ImageFont
import os
import cv2
import time
import onnxruntime as rt
import pandas as pd

from os import listdir
from os.path import isfile, join
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.preprocessing import image


#image_path = 'new_data_back/0014e8-c59b-49ad-8898-a328dfea8d_800.jpg'
dict = {}
PATH_TO_CKPT = 'model.onnx'
export_path = "bce_model_docType_docSide_docState"

label_map = { '1' : 'RC' } 
classes = [['Card', 'Delhi', 'DigiLocker', 'Front', 'Haryana','Other','Page','Uttar_Pradesh','back']]

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

def draw_output(top, left, bottom, right, output2, img_arr, width):
    """Draw box and label for 1 detection."""
    draw = ImageDraw.Draw(img_arr)
    label = str(output2)
    font_width = int(right - left)
    fontsize = int(0.03*font_width) # starting font size
    
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", fontsize, encoding="unic")
    thickness = 5
    label_size = draw.textsize(label)
    print(label_size)
    label_size_top = int(1.1*label_size[1])
    label_size_left = int(1.1*label_size[0])
    print(label_size_top)
    print(label_size_left)
    if top - label_size_top >= 0:
        text_origin = tuple(np.array([left, top - 40 -label_size_top]))
    else:
        text_origin = tuple(np.array([left, top - 1]))
    color = ImageColor.getrgb("green")
    color_text = ImageColor.getrgb("blue")
    
    draw.rectangle([left - thickness, top - thickness, right + thickness, bottom + thickness], width = 5 , outline=color)
    draw.text(text_origin, label, fill=color_text, font = font)
    
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
    return img

def bbox_class_name(c, s):
        """Draw box and label for 1 detection."""
        c = str(c)
        c = c[0]
        print(c)
        label = label_map[c]
        s = str(round(s, 3))
        dict[label] = s
        #print(dict)
        return dict


def multilabel_class(crop_img, model, classes):
    
    #img_path =  'cropped.jpg'
    # Read and prepare image
    #res = image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE,CHANNELS))
    #print("crop_img : ", crop_img)
    start_classification = time.time()
    res = cv2.resize(crop_img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    img = image.img_to_array(res)
    #img = Image.fromarray(res)
    
    img = img/255
    #img = np.asarray(img) / 255
    #mean = np.mean(img)
    #std_dv = np.std( img )
    #img = (img - mean)/std_dv
    img = np.expand_dims(img, axis=0)
    data = model.predict(img)
    
    #print(data[0])
    conf = []
    for i in range(len(data[0])):
        if data[0][i]> 0.5:
            conf.append(data[0][i])
            #print(i, data[0][i])
    prediction = (model.predict(img) > 0.5).astype('int')
    end_classification = time.time()-start_classification
    print('classification infernce time : ',end_classification)
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
        dict[prediction[i]] = round(conf[i],3)
    return dict

def img_inference(image_path):
    start = time.time()
    print(image_path)
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    img_arr = Image.fromarray(img)
    image_dir, image_name = os.path.split(image_path)
    #image_np = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    img_data = np.expand_dims(img.astype(np.uint8), axis=0)
    
    result = sess_run(img_data, sess)
    end_inference_time = time.time() - start
    print('inference_time_detection : ', end_inference_time)
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
                output1 = bbox_class_name(c,s)
                top, left, bottom, right = coordinates( width,height, d)
                crop_img = crop_save(top, left, bottom, right, img_arr)
                output2 = multilabel_class(crop_img, model_multi, classes)
                print('output2 : ', output2)
                img = draw_output(top, left, bottom, right, output2, img_arr, width)
                ress = 'card_output/'+image_name
                cv2.imwrite(ress, img)
    print("time : ", time.time()-start)
  

PATH_TO_TEST_IMAGES_DIR = 'new_data_back'
image_name=[]

from natsort import natsorted
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, f) for f in listdir(PATH_TO_TEST_IMAGES_DIR) if isfile(os.path.join(PATH_TO_TEST_IMAGES_DIR, f))]
TEST_IMAGE_PATHS = sorted(TEST_IMAGE_PATHS)
TEST_IMAGE_PATHS = natsorted(TEST_IMAGE_PATHS)
#print('TEST_IMAGE_PATHS :',TEST_IMAGE_PATHS)

image_name=[f for f in listdir(PATH_TO_TEST_IMAGES_DIR) if isfile(os.path.join(PATH_TO_TEST_IMAGES_DIR, f))]
image_name = sorted(image_name)
image_name = natsorted(image_name)
print(image_name)

for file in os.scandir(PATH_TO_TEST_IMAGES_DIR):
    if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')) :
        image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, file.name)
        #class_id = 1
        print(image_path)
        img_inference(image_path)

