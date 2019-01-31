import sys
import imageio

import numpy as np
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Reshape, Permute
from keras.layers import TimeDistributed, Conv2D, Dense, Dropout, Activation, LSTM, MaxPooling2D, GRU, ConvLSTM2D, Bidirectional
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization

nn = load_model(sys.argv[1])
image = sys.argv[2]

image = imageio.imread(image)
image = np.reshape(image, (1, 255, 56, 1))

p = nn.predict(image)
labels = ['amanda', 'andreas', 'empty', 'lucy', 'robert']

for val in p:
   print(labels[np.argmax(val)])
