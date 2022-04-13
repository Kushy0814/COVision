# -*- coding: utf-8 -*-
# Imports for Tensorflow and Matplotlib
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

# Process csv data into array
import csv
cf_data = []
with open('/content/combined_cf_data.csv', 'r') as f:
    reader = csv.reader(f)
    i = 0
    for line in reader:
      if (i > 20000):
        break
      i += 1
      cf_data.append(line)
for row in cf_data[1:]:
  for i in range(len(row)):
    row[i] = int(row[i])

# Split the data into a 23:7 train-to-test ratio
trainIndices, testIndices = [], []
for i in range(1, 18001):
  if ((i-1) % 30 <= 22):
    trainIndices.append(i)
  else:
    testIndices.append(i)

# Process data from arrays
trainData, testData = [], []
trainLabels, testLabels = [], []
for num in trainIndices:
  trainData.append(cf_data[num][1:8])
  trainLabels.append(cf_data[num][8])
for num in testIndices:
  testData.append(cf_data[num][1:8])
  testLabels.append(cf_data[num][8])


# Oversample the minority classes of COVID-19 and Pneumonia
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
trainData, trainLabels = oversample.fit_resample(trainData, trainLabels)
testData, testLabels = oversample.fit_resample(testData, testLabels)
trainData, trainLabels = oversample.fit_resample(trainData, trainLabels)
testData, testLabels = oversample.fit_resample(testData, testLabels)
trainData = tf.convert_to_tensor(trainData)
testData = tf.convert_to_tensor(testData)
trainLabels = tf.keras.utils.to_categorical(trainLabels)

# Instantiate the architecture for the model 
model = models.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# Compile the model with Adam's optimizer and CCE loss
model.compile(optimizer='Adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

# Train the model with the given parameters

history = model.fit(trainData, trainLabels, epochs=10)
# Using Matplotlib to make a graph of accuracy vs epoch and loss vs epoch from the training 
acc = history.history['accuracy']
loss = history.history['loss']
epochs_range = range(10)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()

# Test the model and store its predictions
testPredictions = model.predict(testData)
testPredictionLabels = []
testAccuracy = 0
for i in range(len(testPredictions)):
  testPredictionLabels.append(np.argmax(testPredictions[i]))
  if (testPredictionLabels[i] == testLabels[i]):
    testAccuracy += 1
print(testAccuracy/len(testPredictionLabels))

# Calculate confusion matrix and loss for the model
confusion_mtx = tf.math.confusion_matrix(testLabels, testPredictionLabels, 3)
print(confusion_mtx)

testLabels = tf.keras.utils.to_categorical(testLabels, num_classes=3)
testPredictionLabels = tf.keras.utils.to_categorical(testPredictionLabels, num_classes=3)

loss = tf.keras.losses.CategoricalCrossentropy()
lossCCE = loss(testLabels, testPredictions).numpy()
print("\nLoss: " + str(lossCCE))

# Saving the trained model as an .h5 file
model.save("normalCF.h5")

# Extract weights from trained model
weights = []
for i in range(7):
  print(i)
  print(model.layers[0].get_weights()[0][i])
  weights.append(np.mean(model.layers[0].get_weights()[0][i]))
data = np.array(weights)
print(1-(data - np.min(data)) / (np.max(data) - np.min(data)))