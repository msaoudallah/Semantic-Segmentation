from arch import unet
import tensorflow as tf
from keras.models import load_model
from utils.util import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import segmentation_models as sm
from utils.labels import map_class_to_rgb
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-model', type=str, required=False,
                        help='model name', default='')

    parser.add_argument('-dim', type=int, required=False,
                        help='dimension used in training model', default=300)

    parser.add_argument('-videopath', type=str, required=False,
                        help='input video path', default='')

    args = parser.parse_args()

    base_data_dir = 'dataset'
    saved_models = 'docs_models'
    output_videos_dir = 'output_videos'

    classes_path = os.path.join(base_data_dir, 'classes.csv')

    model_name = args.model

    DIM = args.dim
    used_model = os.path.join(saved_models, model_name)

    model = load_model(used_model, custom_objects={
        "categorical_crossentropy_plus_jaccard_loss": sm.losses.cce_jaccard_loss, "iou_score": sm.metrics.iou_score})

    classes = pd.read_csv(classes_path, index_col='name')

    sample_video = args.videopath

    cap = cv2.VideoCapture(sample_video)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    vid_name = sample_video.split("\\")[-1]
    vid_path = os.path.join(output_videos_dir, "prediction_"+vid_name)
    out = cv2.VideoWriter(vid_path, fourcc, 15.0, (DIM, DIM))
    while (cap.isOpened()):
        ret, frame = cap.read()

        if frame is None:
            break

        frame = cv2.resize(frame, (DIM, DIM))
        cv2.imshow('video', frame)

        x = frame
        x = x/255.
        x = np.expand_dims(x, 0)
        pred = model.predict(x)

        rgb_mask = np.apply_along_axis(
            map_class_to_rgb, -1, np.expand_dims(np.argmax(pred[0], axis=-1), -1))
        rgb_mask = rgb_mask.astype('uint8')
        rgb_mask = cv2.resize(rgb_mask, (DIM, DIM))
        rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', rgb_mask)

        overlay = cv2.addWeighted(frame, 0.7, rgb_mask, 0.5, 1)
        cv2.imshow('overlayed', overlay)

        out.write(overlay)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # release the video capture object
    cap.release()
    out.release()

    # Closes all the windows currently opened.
    cv2.destroyAllWindows()
