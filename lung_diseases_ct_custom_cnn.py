# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, applications
import tensorflow_addons as tfa
import numpy as np

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

import random
directory = ''
def process_images(paths, labels, bound):
  arr = random.sample(range(len(paths)), len(paths))
  imageData = []
  boundedLabels = []
  imageCount = [0, 0, 0]
  imageDone = [False, False, False]
  for i in range(len(paths)):
    if (imageCount[labels[arr[i]]] < bound):
      imageCount[labels[arr[i]]] += 1
      imgTensor = tf.image.decode_image(tf.io.read_file(directory + paths[arr[i]]))
      resizedImage = tf.image.resize(imgTensor, [128, 128], method='lanczos3')/255
      imageData.append(resizedImage)
      boundedLabels.append(labels[arr[i]])
    else:
      imageDone[labels[arr[i]]] = True
      if (imageDone[0] and imageDone[1] and imageDone[2]):
        break
  return tf.convert_to_tensor(imageData), boundedLabels

trainPaths, trainLabels = load_labels('/content/train_COVIDx_CT-2A.txt')
testPaths, testLabels = load_labels('/content/test_COVIDx_CT-2A.txt')
trainData, trainLabels = process_images(trainPaths, trainLabels, 0)
testData, testLabels = process_images(testPaths, testLabels, 2)
trainLabels = tf.keras.utils.to_categorical(trainLabels)
print("Images Processed")

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(3, activation='softmax'))
model.summary()
print("Model Created")

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
print("Model Compiled")

history = model.fit(trainData, trainLabels, batch_size=64, epochs=2, validation_data=(testData, testLabels))
print("Model Trained")


testPredictions = model.predict(testData)
testPredictionLabels = []
for i in range(len(testPredictions)):
  testPredictionLabels.append(np.argmax(testPredictions[i]))

confusion_mtx = tf.math.confusion_matrix(testLabels, testPredictionLabels)

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

print("Accuracy: " + str(acc))
print("Cohen's Kappa: " + str(cohen))
print("Matthews Correlation Coefficient: " + str(mcc))
print("F-1 Score: " + str(f1))
print("AUROC: " + str(auc))
print("Loss: " + str(lossCCE))
print("Confusion Matrix: ")
print(confusion_mtx)

print("Model Tested & Metrics Calculated")

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, show_layer_activations=True)