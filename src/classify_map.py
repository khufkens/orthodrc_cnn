import os, sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# set TF log level (suppress verbose output)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import segmentation_models as sm
from data_prep_synth import *

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

def mask2img(mask):
    palette = {
        0: (0, 0, 0), # forest
        1: (255, 0, 0), # dist
        2: (0, 255, 0), # background
        3: (0, 0, 255),
        4: (0, 255, 255),
    }
    
    mask = np.argmax(mask, axis = -1)[0]
    rows = mask.shape[0]
    cols = mask.shape[1]
    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for j in range(rows):
        for i in range(cols):
            image[j, i] = palette[mask[j, i]]
    return image
    
def pad(image):
    aug = A.PadIfNeeded(320, 320)
    return aug(image=image)['image']

def predict_classes(image):
    image = pad(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    pr_mask = model.predict(image).round()
    pr_mask = np.argmax(pr_mask, axis = -1)[0]
    return pr_mask

def run_model(image):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    output = np.zeros((iH, iW), dtype="int8")
    
    # set kernel width (default training data size)
    kW = 320
    pad = kW // 2
    step = kW // 3
    
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REFLECT)
    output = cv2.copyMakeBorder(output, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	
    # implement skipping window
    for y in np.arange(pad, iH + pad, step):
       for x in np.arange(pad, iW + pad, step):
         roi = image[y - pad:y + pad, x - pad:x + pad]
         seg_map = predict_classes(roi)
         seg_map = output[y - pad:y + pad, x - pad:x + pad] + seg_map
         output[y - pad:y + pad, x - pad:x + pad] = seg_map
			  
    output = output[pad:iH+pad,pad:iW+pad]
    return output

if __name__ == "__main__":

    # load model
    # Model setup
    BACKBONE = 'efficientnetb3'
    BATCH_SIZE = 1
    CLASSES = ['forest','dist']
    LR = 0.0001
    EPOCHS = 5
    DATA_DIR = './data/'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    
    # read main image
    #orig = cv2.imread('./data/test_images/test_2.jpg')
    #orig = cv2.imread('./data/test_images/58-05-008.png')
    orig = cv2.imread('./data/test_images/geo-eye.tif')
    #image = cv2.merge((orig, orig, orig))
    
    #orig = cv2.imread('./data/test_images/yangambi_orthomosaic_modified.jpg')
    image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'
    
    #create model
    model = sm.Unet(
      BACKBONE,
      classes=n_classes,
      activation=activation,
      #encoder_weights='imagenet'
    )
    
    # define optomizer
    optim = keras.optimizers.Adam(LR)
    
    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    
    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    # load weights
    model.load_weights('./src/forest_model.h5')
    
    # predict results
    pr_mask = run_model(image)
    
    # remove padding and write to disk
    cv2.imwrite('./data/forest_cover.png', pr_mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
      
