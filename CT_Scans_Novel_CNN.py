# -*- coding: utf-8 -*-
# Imports for Tensorflow and Matplotlib
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, applications
import tensorflow_addons as tfa
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
print("Imports Completed")

# Command-line arguments for altering training parameters
epochAmount = 0;
batchSize = 32;
trainingSize = 2100;
testingSize = 900;
imageSize = 256;
if __name__ == "__main__":
    for arg in sys.argv[1:]:
        try:
            name, value = arg.split('=',1)
        except:
            print("Error parsing argument. No '=' found.")

        if name == "--epoch":
            epochAmount = int(float(value)) 
        if name == "--batch_size":
            batchSize = int(float(value)) 
        if name == "--train_size":
            trainingSize = int(float(value)) 
        if name == "--test_size":
            testingSize = int(float(value))
        if name == "--image_size":
            imageSize = int(float(value)) 
print("Epoch Amount: " + str(epochAmount))
print("Batch Size: " + str(batchSize))
print("Training Size: " + str(trainingSize))
print("Testing Size: " + str(testingSize))
print("Image Size: " + str(imageSize))

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
# 
# 
# 


trainPaths, trainLabels = load_labels('train_COVIDx_CT-2A.txt')
testPaths, testLabels = load_labels('test_COVIDx_CT-2A.txt')
print("Labels Processed")
trainData, trainLabels = process_images(trainPaths, trainLabels, trainingSize)
testData, testLabels = process_images(testPaths, testLabels, testingSize)
trainLabels = tf.keras.utils.to_categorical(trainLabels)
print("Images Processed")

# Instantiate the architecture for the model,
# compile the model with Adam's optimizer and CCE loss,
# and train the model with the given parameters
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
# 
# 
# 
# 
# 

# Test the model and store its predictions
testPredictions = model.predict(testData)
testPredictionLabels = []
for i in range(len(testPredictions)):
  testPredictionLabels.append(np.argmax(testPredictions[i]))
print("Model Tested")

# Using Matplotlib to make a graph of accuracy vs epoch and loss vs epoch from the training 
history_dict = history.history
print(history_dict.keys())

acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(epochAmount)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')
plt.title('Training Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()

# Calculation for the metrics of the model 
confusion_mtx = tf.math.confusion_matrix(testLabels, testPredictionLabels)
matrix_string = ""
for row in confusion_mtx:
  for col in row:
    matrix_string += str(col)
  matrix_string += '\n'

metric = tf.keras.metrics.Accuracy()
metric.update_state(testLabels, testPredictionLabels)
acc = metric.result().numpy()

metric = tfa.metrics.CohenKappa(num_classes=3, sparse_labels=True)
metric.update_state(testLabels, testPredictionLabels)
cohen = metric.result().numpy()

testLabels = tf.keras.utils.to_categorical(testLabels, num_classes=3)
testPredictionLabels = tf.keras.utils.to_categorical(testPredictionLabels, num_classes=3)

metric = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=3)
metric.update_state(testLabels, testPredictionLabels)
mcc = metric.result().numpy()

metric = tfa.metrics.F1Score(num_classes=3)
metric.update_state(testLabels, testPredictionLabels)
f1 = metric.result().numpy()

metric = tf.keras.metrics.AUC()
metric.update_state(testLabels, testPredictionLabels)
auc = metric.result().numpy()

loss = tf.keras.losses.CategoricalCrossentropy()
lossCCE = loss(testLabels, testPredictions).numpy()

# Writing a txt file to store the calculated metrics
metricFile = open("Model_Metrics.txt", "w")
metricFile.write("Accuracy: " + str(acc))
metricFile.write("\nCohen's Kappa: " + str(cohen))
metricFile.write("\nMatthews Correlation Coefficient: " + str(mcc))
metricFile.write("\nF-1 Score: " + str(f1))
metricFile.write("\nAUROC: " + str(auc))
metricFile.write("\nLoss: " + str(lossCCE))
metricFile.write("\nConfusion Matrix: " + matrix_string)
metricFile.close()
print("Metrics Calculated")

# Saving the trained model as an .h5 file
model.save('ct_model.h5')
print("Model Saved")
