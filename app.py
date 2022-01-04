import os
import cv2
from flask import Flask, request, render_template, send_from_directory

import base64
from keras.models import load_model

import tensorflow as tf
import numpy as np
import segmentation_models as sm
import pandas as pd

import cv2
import os

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


models = [
    {
        'model_path': 'docs_models\deeplabv3_epoch_001_valLoss_0.4450.h5',
        'model_short_name': 'deeplab_512',
        'model_id': 'deeplabv3plus_512',
        'model_input_size': 512,
        'model': None
    },
    {
        'model_path': 'docs_models\deeplabv3_fulltraining_epoch_173_valLoss_0.4395.h5',
        'model_short_name': 'deeplab_300',
        'model_id': 'deeplabv3plus_300',
        'model_input_size': 300,
        'model': None
    },



    {
        'model_path': 'docs_models\sm_psp_epoch_165_valLoss_0.7048.h5',
        'model_short_name': 'PSP',
        'model_id': 'PSP_156',
        'model_input_size': 288,
        'model': None
    },
    {
        'model_path': 'docs_models\sm_unet_resnet101_epoch_148_valLoss_0.6643.h5',
        'model_short_name': 'UNET',
        'model_id': 'UNET_0148',
        'model_input_size': 256,
        'model': None
    }



]


data = []

for model in models:
    print('loading model : ' + model['model_id'])
    model['model'] = load_model(
        model['model_path'],
        custom_objects={
            "categorical_crossentropy_plus_jaccard_loss": sm.losses.cce_jaccard_loss,
            "iou_score": sm.metrics.iou_score
        }
    )
    data.append({'name': model['model_short_name']})


filename = ''
original_image_path = ''


def segmentImage(path, model, dim, classes_csv_path):
    # read image
    im = cv2.imread(path)

    # convert colors
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    hight, width, ch = im.shape

    # resize image to fit the input of the model
    image = tf.image.resize(images=im, size=[dim, dim])
    image = image * 1.0 / 255

    # add the patch size dimension in the first (1 , 512 , 512 , 3)
    image = tf.expand_dims(image, 0)

    # make prediction the output will be of shape (1 , 512 , 512 , 34) contains the probability of each class ( pre pixel )
    pred = model.predict(image)

    classes = pd.read_csv('classes.csv', index_col='name')
    cls2rgb = {cl: list(classes.loc[cl, :]) for cl in classes.index}
    idx2rgb = {idx: np.array(rgb)
               for idx, (cl, rgb) in enumerate(cls2rgb.items())}

    def map_class_to_rgb(p):
        return idx2rgb[p[0]]

    # categorize each pixel by selecting the index of the pixel with highst class propability
    rgb_mask = np.apply_along_axis(
        map_class_to_rgb, -1, np.expand_dims(pred.argmax(-1), -1))

    # resize the image to match the original one and convert it to integer
    seg = tf.image.resize(images=rgb_mask[0], size=(hight, width))
    seg = tf.cast(seg, np.uint8)

    # convert `tensor a` to a proto tensor
    proto_tensor = tf.make_tensor_proto(seg)
    seg = tf.make_ndarray(proto_tensor)

    oi = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    si = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    mi = cv2.addWeighted(oi, 0.5, si, 0.9, 0)

    return oi, si, mi


@app.route("/")
def main():
    global data
    '''
    return : rendered index.html web page
    '''
    return render_template('index.html', data=data)


@app.route('/static/images/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


@app.route("/upload", methods=["POST"])
def upload():
    global original_image_path
    global filename
    global models

    target = os.path.join(APP_ROOT, 'static/images')
    if not os.path.isdir(target):
        os.mkdir(target)

    upload = request.files.getlist("file")[0]
    filename = upload.filename

    original_image_path = 'static/images/' + filename

    print(original_image_path)

    upload.save(original_image_path)
    return render_template('index.html', original_image=original_image_path, data=data)


def cvImage2Base64(image):
    retval, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer)
    return "data:image/png;base64, " + jpg_as_text.decode("utf-8")

# -----------------------------------------------------------------------------------------
@app.route("/test", methods=['GET', 'POST'])
def test():
    global data
    global filename
    global models
    select = request.form.get('comp_select')
    print(select)
    oi, si, mi = None, None, None
    for model in models:
        if select == model['model_short_name']:
            oi, si, mi = segmentImage(
                path=original_image_path,
                model=model['model'],
                dim=model['model_input_size'],
                classes_csv_path='classes.csv'
            )
            break

    original_image = oi
    mixed_image = cvImage2Base64(mi)
    segmented_image = cvImage2Base64(si)

    return render_template('index.html',
                           model_name=select,
                           original_image=original_image_path,
                           mixed_image=mixed_image,
                           segmented_image=segmented_image,
                           data=data
                           )


# -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run()
