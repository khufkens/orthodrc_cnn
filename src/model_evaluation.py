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

if __name__ == "__main__":

    # load model
    # Model setup
    BACKBONE = 'efficientnetb3'
    BATCH_SIZE = 1
    CLASSES = ['forest','dist']
    LR = 0.0001
    #EPOCHS = 5
    DATA_DIR = './data/synth/'
    
    x_test_dir = os.path.join(DATA_DIR, 'images')
    y_test_dir = os.path.join(DATA_DIR, 'labels')

    preprocess_input = sm.get_preprocessing(BACKBONE)
    
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
    
    # define optimizer
    optim = keras.optimizers.Adam(LR)
    
    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
    
    # compile keras model with defined optimizer, loss and metrics
    model.compile(optim, total_loss, metrics)

    # load weights
    model.load_weights('forest_model.h5')  
    
    test_dataset = Dataset(
      x_test_dir,
      y_test_dir,
      classes=CLASSES,
      augmentation=get_validation_augmentation(),
      preprocessing=get_preprocessing(preprocess_input),
      subset = "test"
    )

    test_dataloader = Dataloader(test_dataset, batch_size=1, shuffle=False)
    
    # check shapes for errors
    print(test_dataloader[0][0].shape)
    print(test_dataloader[0][1].shape)
    assert test_dataloader[0][0].shape == (BATCH_SIZE, 320, 320, 3)
    assert test_dataloader[0][1].shape == (BATCH_SIZE, 320, 320, n_classes)
    
    # load best weights
    model.load_weights('forest_model.h5')

    scores = model.evaluate_generator(test_dataloader)

    print("Loss: {:.5}".format(scores[0]))
    for metric, value in zip(metrics, scores[1:]):
     print("mean {}: {:.5}".format(metric.__name__, value))

    # visualize results
    n = 5
    ids = np.random.choice(np.arange(len(test_dataset)-1), size=n)

    for i in ids:
      image, gt_mask = test_dataset[i]
      image = np.expand_dims(image, axis=0)
      pr_mask = model.predict(image).round()
    
      visualize(
          image=denormalize(image.squeeze()),
          gt_mask=gt_mask[..., 0].squeeze(),
          pr_mask=pr_mask[..., 0].squeeze(),
      )
