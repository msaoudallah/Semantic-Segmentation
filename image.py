from arch import unet
import tensorflow as tf
from keras.models import load_model
from utils.util import *
from utils.labels import map_class_to_rgb
import cv2
import numpy as np
import os
import segmentation_models as sm
import argparse


if __name__ == "__main__":
    # model = load_model("seg_model_100_epoch.h5", custom_objects={
    #                    "jaccard_distance": jaccard_distance})

    # constants

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-model', type=str, required=False,
                        help='model name', default='')

    # parser.add_argument('-path', type=str, required=False,
    #                     help='input images path', default='')

    parser.add_argument('-dim', type=int, required=False,
                        help='dimension used in training model', default=300)

    parser.add_argument('-outputwidth', type=int, required=False,
                        help='dimension used in training model', default=300)
    parser.add_argument('-outputheight', type=int, required=False,
                        help='dimension used in training model', default=300)

    args = parser.parse_args()

    base_data_dir = 'dataset'
    saved_models = 'docs_models'

    output_images_dir = 'output_images'

    single_frames_dir = os.path.join(base_data_dir, 'test')
    videos_dir = os.path.join(base_data_dir, 'videos')

    # model_name = "deeplabv3_fulltraining_epoch_074_valLoss_0.5576.h5"
    model_name = args.model

    DIM = args.dim
    output_width, output_height = args.outputwidth, args.outputheight

    used_model = os.path.join(saved_models, model_name)

    model = load_model(used_model, custom_objects={
        "categorical_crossentropy_plus_jaccard_loss": sm.losses.cce_jaccard_loss, "iou_score": sm.metrics.iou_score})

    img_paths = []

    img_list = os.listdir(single_frames_dir)

    predict_images(model, img_list, DIM, output_width,
                   output_height, output_images_dir, directory=single_frames_dir)
