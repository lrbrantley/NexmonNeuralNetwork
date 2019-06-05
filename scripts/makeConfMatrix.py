#!/usr/bin/python3
import sys
import imageio
import os

# Disables messages about sub-optimal compiler options and "Using ... Backend."
# God, I hate this hack
def import_keras_silently():
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
   temp = sys.stderr
   sys.stderr = open("/dev/null", "w")
   from keras import backend as K
   sys.stderr = temp

import numpy as np
import_keras_silently()
from keras.models import Sequential #, load_model
from tensorflow.keras.models import load_model
from keras.layers.core import Flatten, Reshape, Permute
from keras.layers import TimeDistributed, Conv2D, Dense, Dropout, Activation, \
   LSTM, MaxPooling2D, GRU, ConvLSTM2D, Bidirectional
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization

nn = load_model(sys.argv[1])
image = sys.argv[2]
labels = sys.argv[3].split("\n")

image = imageio.imread(image)
image = np.reshape(image, (1, -1, 56, 1))

p = nn.predict(image)

for val in p:
   print(labels[np.argmax(val)])
