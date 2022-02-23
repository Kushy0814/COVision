# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, applications
import tensorflow_addons as tfa
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
print("Imports Completed")

epochAmount = 0;
batchSize = 64;
trainingSize = 35000;
testingSize = 15000;
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
print("Epoch Amount: " + str(epochAmount))
print("Batch Size: " + str(batchSize))
print("Training Size: " + str(trainingSize))
print("Testing Size: " + str(testingSize))

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

directory = '2A_images/'
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
print("Functions Declared")

trainPaths, trainLabels = load_labels('train_COVIDx_CT-2A.txt')
testPaths, testLabels = load_labels('test_COVIDx_CT-2A.txt')
print("Labels Processed")
trainData, trainLabels = process_images(trainPaths, trainLabels, trainingSize)
testData, testLabels = process_images(testPaths, testLabels, testingSize)
trainLabels = tf.keras.utils.to_categorical(trainLabels)
print("Images Processed")

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))
print("Model Created")

model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
print("Model Compiled")

history = model.fit(trainData, trainLabels, batch_size=batchSize, epochs=epochAmount)
print("Model Trained")

testPredictions = model.predict(testData)
testPredictionLabels = []
for i in range(len(testPredictions)):
  testPredictionLabels.append(np.argmax(testPredictions[i]))
print("Model Tested")

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

metricFile = open("Model_Metrics.txt", "w")
metricFile.write("Accuracy: " + str(acc))
metricFile.write("Cohen's Kappa: " + str(cohen))
metricFile.write("Matthews Correlation Coefficient: " + str(mcc))
metricFile.write("F-1 Score: " + str(f1))
metricFile.write("AUROC: " + str(auc))
metricFile.write("Loss: " + str(lossCCE))
metricFile.write("Confusion Matrix: " + matrix_string)
metricFile.close()
print("Metrics Calculated")

# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False, show_layer_activations=True)

model.save('ct_model.h5')
print("Model Saved")

