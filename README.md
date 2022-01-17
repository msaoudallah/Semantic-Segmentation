# Semantic-Segmentation:

semantic segmentation project on kitti data set as part of ITI AI-Pro internship graduation project
![alt text](https://nanonets.com/blog/content/images/2020/08/1_wninXztJ90h3ZHtKXCNKFA.jpeg)

# Refrences:

- [latest papers in segmentation | tracking | detection](https://www.linkedin.com/company/argo-vision)
- [PSPNet implementation](https://medium.com/analytics-vidhya/semantic-segmentation-in-pspnet-with-implementation-in-keras-4843d05fc025)
- [PSPNet explained](https://developers.arcgis.com/python/guide/how-pspnet-works/)
- [Image Segmentation Keras : Implementation of Segnet, FCN, UNet, PSPNet and other models in Keras](https://github.com/divamgupta/image-segmentation-keras)
- [Transfer Learning and Pre-trained ConvNets | Ahmad El Sallab](https://www.youtube.com/watch?v=5Wb6C-d1W-s&list=PLX2D7RnWrLv5f13RK5XvjZ_BMDKBqWriD&index=7)
- [Semantic Segmentation Models](https://paperswithcode.com/methods/category/segmentation-models)

# Team Members:

- Mahmoud Said
- Ahmed Mohsen
- Mohamed abdelbaset
- Mahmoud Osama
- Mohamed salah-el-den

# Repo structure

- dataset/ (contains all data resources)
  - data/ (contain train , validation data)
    - train/
      - images/
      - labels/
    - valid/
      - images/
      - labels/
  - classes.csv (contains labels and their RGB)
  - test/ ( contains images to test)
- output_images/ (contains output images from prediction )
- output_videos/ (contains output videos from prediction )
- docs_models/ (contains saved models after training)
- arch/ (contains different models to be used in training)
- utils/ (contains necessary functions to be used for training and inference)
- static/ , templates/ (directiories used by the flask web app)
- image.py (inference script for images)
- video.py (inference script for videos)
- sample_training_notebook.ipynb (notebook for training models)
- app.py ( flask application python file)

## Required Packages:

- [Anaconda distribution for python] (https://www.anaconda.com/)
- Tensorflow 2.X for GPU
- [Segmentation models] (https://github.com/qubvel/segmentation_models)
- [open cv] (https://pypi.org/project/opencv-contrib-python/)
- [Flask] (https://flask.palletsprojects.com/en/2.0.x/)

# How to run notebook for training

follow instruction in the notebook for training

# How to inference script for images

put images you want to test inside "dataset/test" folder and run image.py from terminal and provide the following parameters:

- model : provide model name e.g "example.h5" model should be in "docs_models" folder
- dim : proivde dimension you used in training your model
- outputwidhth (optional): provide the width of the output segmented image
- outputheight (optional): provide the height of the output segmented image
  script will go through all images in "dataset/test" and output will be saved to "output_images" folder

# How to inference script for vidoes

run vdieo.py from terminal and provide the following parameters:

- model : provide model name e.g "example.h5" model should be in "docs_models" folder
- dim : proivde dimension you used in training your model
- videopath : provide the full path of the video to be segmented

script will load the video and output will be saved to "output_videos" folder

# Application

-from terminal run app.py and wait for server to start
-open [localhost](http://127.0.0.1:5000/) and start using the application
