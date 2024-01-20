import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Montowanie dysku Google Drive
from google.colab import drive
drive.mount('/content/drive')

DATADIR = "/content/drive/My Drive/Fruits_identyfication/archive_kopia/Fruits Classification/train/"
CATEGORIES = ["Strawberry", "Banana"]


IMG_SIZE = 60
training_data = []

def set_training_data():
  for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])

set_training_data()


import random

random.shuffle(training_data)

X = []
y = []


for features, label in training_data:
  X.append(features)
  y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()


pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import csv

NAME = "Strawberies-vs-Banana-cnn-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# Wczytanie danych
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

X = X / 255.0

print(X.shape[1:])

# model.add(Flatten(input_shape=X.shape[1:]))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dense(1, activation='sigmoid'))

# Utworzenie modelu
model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Flatten())
model.add(Dense(128))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
history = model.fit(np.array(X), np.array(y), batch_size=32, epochs = 15, validation_split = 0.1, callbacks=[tensorboard])


#history = model.fit(np.array(X), np.array(y), batch_size=32, epochs=30, validation_split=0.3, callbacks=[tensorboard])

# Wizualizacja obrazu z zbioru uczącego
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(X[0], cmap='gray')
#plt.title('Przykładowy obraz z zbioru uczącego')
plt.axis('on')  # Włączenie osi

# Wizualizacja rozkładu klas w zbiorze uczącym
plt.figure(figsize=(10, 4))
plt.subplot(1, 1, 1)
plt.bar(['0', '1'], height=np.bincount(y), color=['blue', 'orange'])
#plt.title('Rozkład klas w zbiorze uczącym')

plt.show()


# Zapisanie wag po pierwszej epoce nauki
# history = model.fit(np.array(X), np.array(y), batch_size=32, epochs=1, validation_split=0.25)

layer_weights_first_epoch = model.layers[3].get_weights()[0]

# Proces uczenia (reszta epok)
#history = model.fit(np.array(X), np.array(y), batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])

# Wizualizacja błędów
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Dokładność (accuracy)')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
plt.title('Dokładność w trakcie uczenia')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Strata (loss)')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.title('Strata w trakcie uczenia')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.show()

with open('historia_treningu_strata.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoka', 'Strata', 'Strata walidacyjna'])

    for epoch, loss, val_loss in zip(range(1, len(history.history['loss']) + 1), history.history['loss'], history.history['val_loss']):
        writer.writerow([epoch, loss, val_loss])

with open('historia_treningu_dokladnosc.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoka', 'Dokładność', 'Dokładność walidacyjna'])

    for epoch, loss, val_loss in zip(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], history.history['val_accuracy']):
        writer.writerow([epoch, loss, val_loss])


# # Wizualizacja wartości wag na początku uczenia
# plt.figure(figsize=(12, 4))
# for i in range(layer_weights_first_epoch.shape[1]):
#     plt.plot(layer_weights_first_epoch[:, i], label=f'Neuron {i}')
# plt.title('Wartości wag w warstwie ukrytej - Po pierwszej epoce')
# plt.xlabel('Indeks wagi')
# plt.ylabel('Wartość wagi')
# plt.legend()
# plt.show()


# # Wizualizacja wartości wag na końcu uczenia
# layer_weights_end = model.layers[3].get_weights()[0]
# plt.figure(figsize=(12, 4))
# for i in range(layer_weights_end.shape[1]):
#     plt.plot(layer_weights_end[:, i], label=f'Neuron {i}')
# plt.title('Wartości wag w warstwie ukrytej - Koniec uczenia')
# plt.xlabel('Indeks wagi')
# plt.ylabel('Wartość wagi')
# plt.legend()
# plt.show()
