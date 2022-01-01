from arch import unet
import tensorflow as tf
from keras.models import load_model
from utils.util import *
from utils.labels import map_class_to_rgb
import cv2
import numpy as np
import os
import segmentation_models as sm


if __name__ == "__main__":
    # model = load_model("seg_model_100_epoch.h5", custom_objects={
    #                    "jaccard_distance": jaccard_distance})

    # constants
    base_data_dir = 'original'
    saved_models = 'docs_models'

    iti_data_dir = os.path.join(base_data_dir, 'aroundITI')
    single_frames_dir = os.path.join(base_data_dir, 'single_frames')
    videos_dir = os.path.join(base_data_dir, 'videos')

    output_images_dir = os.path.join(base_data_dir, 'output_images')
    output_videos_dir = os.path.join(base_data_dir, 'output_videos')

    model_name = "deeplabv3_epoch_001_valLoss_0.4450.h5"
    DIM = 300
    output_width, output_height = 300, 300

    used_model = os.path.join(saved_models, model_name)

    model = load_model(used_model, custom_objects={
        "categorical_crossentropy_plus_jaccard_loss": sm.losses.cce_jaccard_loss, "iou_score": sm.metrics.iou_score})

    img_paths = []

    img_list = os.listdir(single_frames_dir)

    predict_images(model, img_list, DIM, output_width,
                   output_height, directory=single_frames_dir)
