from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

cfg_save_path = "IS_cfg.pickle"


with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)
    
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") #load model weight path of custom dataset we trained

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
predictor = DefaultPredictor(cfg)

image_path = "try_out/test1.jpg"
#image_path = "card_data/card_images/94b90d82-44d6-44eb-9f2b-7695635928d7_819350.jpg"
#image_path = "card_data/card_images/09e865b3-d19f-44d1-9abe-7c5a8d54ccde_818835.jpg"
#image_path = "card_data/card_images/8a382223-a4a3-4c36-8828-2eda38bf5111_820140.jpg"
#image_path = "card_data/card_images/880d816e-492c-4ab8-b907-00439d054b76_819511.jpg"
#image_path = "card_data/card_images/8149bd3f-c4d8-4baa-870e-e6d93a501dcf_817508.jpg"
#image_path = "card_data/card_images/008e6476-ec4d-428a-ab68-69b022286d08_820298.jpg"

on_image(image_path, predictor)
#convex_hull(canny_path)
#fill_boundary(convexhull_path)
#transformations(opening_path, image_path)
#on_image1(image_path, predictor)