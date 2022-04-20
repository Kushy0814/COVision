# -*- coding: utf-8 -*-
# Imports for Tensorflow and Matplotlib
import tensorflow as tf
from matplotlib import pyplot 
import numpy as np
import random
from random import shuffle 
import cv2
import os
from tqdm import tqdm

# Load the model and store the weights
model = tf.keras.models.load_model('ct_model.h5')
filters, biases = model.layers[0].get_weights()

# Iterate through the 64 kernels, creating visualizations and saving them as png files
n_filters, ix = 64, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	pyplot.xticks([])
	pyplot.yticks([])
	pyplot.imshow(f[:, :, 0])
	pyplot.savefig("drive/MyDrive/COVision Visualizations/Convolutional Filters/filters" + str(i+1) + ".png")
	ix += 1

# Imports for specific modules within Tensorflow and Matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.python.keras.utils.data_utils import Sequence

# Binarize the predictions using 0.5 as a threshold
def decode_prediction(pred):
  pred = tf.where(pred < 0.5, 0, 1)
  return pred.numpy()

# Process the txt file and strip out the path of each image and its corresponding label
imagePathsAll = []
def load_labels(label_file):
    paths, labels = [], []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            path, label, xmin, ymin, xmax, ymax = line.strip('\n').split()
            paths.append(path)
            imagePathsAll.append(path)
            labels.append(int(label))
    return paths, labels

# Take a random sample from the image data and preprocess the images and labels
# [Code Hidden]
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

trainPaths, trainLabels = load_labels('/content/train_COVIDx_CT-2A.txt')
trainData, trainLabels = process_images(trainPaths, trainLabels, 1000)


class_info = ["Normal", "Pneumonia", "COVID-19"]
pred_raw = model.predict(trainData)

# Plot the original CT Scans 
plt.figure(figsize=(10, 10))
for image in trainData:
  plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Create a normalized heatmap using the gradient of the output predictions and actiations of the last convolutional layer
# [Code Hidden]
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 


# Superimpose heatmaps over CT scans through Matplotlib with a jet color scheme
# [Code Hidden]
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
#
# 
# 


# Plot and save images
i = 0
for image in trainData:
  if (i > 500):
    break
  img_array = np.expand_dims(image, axis=0)
  preds = model.predict(img_array)
  heatmap = make_gradcam_heatmap(img_array, "conv2d_2")
  plt.matshow(heatmap)
  plt.xticks([])
  plt.yticks([])
  # plt.title("Convolutional Filter of " + class_info[trainLabels[i]] + " CT")
  plt.colorbar();
  plt.savefig("drive/MyDrive/COVision Visualizations/Grad-Cams/" + class_info[trainLabels[i]] + "/scale_ct_scan_" + str(i+1) + ".png")
  plt.show()
  i += 1

save_and_display_gradcam(img_array, heatmap)