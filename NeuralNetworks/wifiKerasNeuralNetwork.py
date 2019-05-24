#!/usr/bin/python

import numpy as np
from keras import backend as K
from keras.models import Sequential
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
import sys
import os
import argparse

def train_dir(data_dir):
    return '%s/train' % data_dir

def test_dir(data_dir):
    return '%s/validation' % data_dir

def color_mode(b):
    return 'rgb' if b else 'grayscale'

def data_dir_path(string):
    if not os.path.isdir(train_dir(string)):
        raise NotADirectoryError(train_dir(string))
    elif not os.path.isdir(test_dir(string)):
        raise NotADirectoryError(test_dir(string))
    else:
        return string

def data_generator(args, directory):
    return train_datagen.flow_from_directory(directory,
                                             color_mode=color_mode(args.color),
                                             target_size=(args.rows, args.cols),
                                             batch_size=args.batch,
                                             class_mode='categorical')

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-r', '--rows', default=400, type=int,
                        help='Number of rows per image')
    parser.add_argument('-c', '--cols', default=56, type=int,
                        help='Number of columns per image')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='Number of training iterations')
    parser.add_argument('-b', '--batch', default=30, type=int,
                        help='Size of a training batch')
    parser.add_argument('-1', '--convolution1', default=32, type=int,
                        help='Number of convolutional layers')
    parser.add_argument('-2', '--convolution2', default=128, type=int,
                        help='Number of convolutional layers')
    parser.add_argument('-o', '--color', action='store_true',
                        help='Data is color image')
    parser.add_argument('data_path', nargs='?',
                        type=data_dir_path, help='Data directory',
                        default="%s/../data" % os.path.dirname(sys.argv[0]))
    parser.add_argument('--version', action='version', version='%(prog)s 1.1')
    return parser.parse_args()

#Start
args = parse_args()
bins = os.listdir(train_dir(args.data_path))
train_dir_path = train_dir(args.data_path)
test_dir_path = test_dir(args.data_path)
num_of_train_samples = 0
num_of_test_samples = 0
for b in bins:
    num_of_train_samples += len(os.listdir("%s/%s" % (train_dir_path, b)))
    num_of_test_samples += len(os.listdir("%s/%s" % (test_dir_path, b)))

input_shape = (args.rows, args.cols, 1)

#Image Generator
train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()
train_generator = data_generator(args, train_dir_path)
validation_generator = data_generator(args, test_dir_path)

# Build model
model = Sequential()

model.add(Conv2D(filters=args.convolution1,
                 kernel_size=(5,5),# TODO: Parameterize
                 input_shape=input_shape,
                 padding='valid',
                 activation='tanh',# TODO: Parameterize
                 strides=1))

model.add(MaxPooling2D(pool_size=(2,2)))# TODO: Parameterize
model.add(Dropout(.15))# TODO: Parameterize

model.add(Conv2D(filters=args.convolution2,
                 kernel_size=(5,5),# TODO: Parameterize
                 padding='valid',
                 activation='tanh',# TODO: Parameterize
                 strides=1))

model.add(MaxPooling2D(pool_size=(2,2)))# TODO: Parameterize
model.add(Dropout(.10))# TODO: Parameterize

model.add(Reshape((args.convolution2,-1)))
model.add(Permute((2,1)))
model.add(Bidirectional(LSTM(128)))# TODO: Parameterize
model.add(Dense(len(bins)), activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',# TODO: Parameterize
              metrics=['accuracy'])

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')
callbacks_list=[checkpoint]

plot_model(model, to_file='model_plot.png', show_shapes=True,
           show_layer_names=True)

#Train
history = model.fit_generator(train_generator,
                    steps_per_epoch=num_of_train_samples // args.batch,
                    epochs=args.epochs,
                    callbacks=callbacks_list,
                    validation_data=validation_generator,
                    validation_steps=num_of_test_samples // args.batch)

#History for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#History for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
