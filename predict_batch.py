######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import argparse
import glob
from tqdm import tqdm
# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
tf.logging.set_verbosity(tf.logging.ERROR)

def prediction(img):
    fname = os.path.basename(img)
    print('processing image: {0}'.format(fname))

    # Number of classes the object detector can identify
    NUM_CLASSES = 1

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value

    image = cv2.imread(img)
    newImage = image.copy()
    image_expanded = np.expand_dims(image, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.90)


    cv2.imwrite(os.path.join(CWD_PATH,args.output_path,fname),image)
    boxes_reshape = np.reshape(boxes,[300,4])
    scores_reshape = np.reshape(scores,[300,1])
    # Extract meter reading bounding box

    if scores_reshape[0] > 0.90:
        ymin,xmin,ymax,xmax = boxes_reshape[0]
        (im_height,im_width) = (np.shape(image)[0],np.shape(image)[1])
        ymin = boxes_reshape[0][0]*im_height
        xmin = boxes_reshape[0][1]*im_width
        ymax = boxes_reshape[0][2]*im_height
        xmax = boxes_reshape[0][3]*im_width

        print ('Top left')
        print (xmin,ymin,)
        print ('Bottom right')
        print (xmax,ymax)

        bbox = newImage[int(ymin):int(ymax),int(xmin):int(xmax)]
        cv2.imwrite(os.path.join(args.output_path,os.path.splitext(fname)[0] + '_bbox.jpg'),bbox)

if __name__ == '__main__':

    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', action='store', dest='model_directory',
                        help='path of the model directory')
    parser.add_argument('-i', action='store', dest='images_path',
                        help='path of the test image')
    parser.add_argument('-l', action='store', dest='label_path',
                        help='path of the labels')
    parser.add_argument('-o',action='store',dest='output_path',help='path to store output')
    args = parser.parse_args()

    IMAGES_PATH = args.images_path

    # Path to image
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IMAGES_PATH)

    all_imgs = glob.glob(PATH_TO_IMAGES + '/*.jpg')
    print ('total number of images: {0}'.format(len(all_imgs)))

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = args.model_directory
    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    #PATH_TO_LABELS = os.path.join(CWD_PATH,'/object_detection/','labelmap.pbtxt')
    PATH_TO_LABELS = args.label_path

    for img in tqdm(all_imgs):
        prediction(img)
    




