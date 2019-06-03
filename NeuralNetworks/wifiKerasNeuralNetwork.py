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
from collections import Iterable

optimizers = ["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"]

activate_fns = ["softmax", "elu", "selu", "softplus", "softsign", "relu",
                "tanh", "sigmoid", "hard_sigmoid", "exponential", "linear"]

def train_dir(data_dir):
    return '%s/train' % data_dir

def test_dir(data_dir):
    return '%s/validation' % data_dir

def color_mode(b):
    return 'rgb' if b else 'grayscale'

def data_dir_path(string):
    if not os.path.isdir(train_dir(string)):
        raise ValueError("No such directory %s" % train_dir(string))
    elif not os.path.isdir(test_dir(string)):
        raise ValueError("No such directory %s" % test_dir(string))
    else:
        return string

def data_generator(args, directory):
    return ImageDataGenerator() \
        .flow_from_directory(directory,
                             color_mode=color_mode(args.color),
                             target_size=(args.rows, args.cols),
                             batch_size=args.batch,
                             class_mode='categorical')

def get_arg_index(flag, longFlag=None, argv=sys.argv):
    try:
        flagI = argv.index(flag)
        try:
            argv.index(longFlag)
            raise ValueError("Flags %s and %s are mutually exclusive." %
                             (flag, longFlag))
        except:
            return flagI
    except:
        try:
            return argv.index(longFlag)
        except:
            return -1

def match_tuple(arg, convolutionFlags):
    for t in convolutionFlags:
        if arg in t[:2]:
            return t
    return None

# TODO: This function is rather messy
def get_convolution_args(out, flag, longFlag, convFlags, argv=sys.argv):
    i = get_arg_index(flag, longFlag, argv)
    n = argparse.Namespace()
    d = vars(n)
    for t in convFlags:
        if len(t) >= 5:
            d[t[2]] = t[4]
        else:
            d[t[2]] = None
    if i != -1:
        argv = argv[:i]+argv[i+1:]
        t = match_tuple(argv[i], convFlags)
        while t is not None:
            if callable(t[3]):
                d[t[2]] = t[3](argv[i+1])
            elif isinstance(t[3], Iterable):
                if argv[i+1] in t[3]:
                    d[t[2]] = argv[i+1]
                else:
                    raise ValueError("%s is not a valid argument for %s" %
                                     (argv[i+1], argv[i]))
            else:
                d[t[2]] = argv[i+1]
            argv = argv[:i] + argv[i+2:]
            if i < len(argv):
                t = match_tuple(argv[i], convFlags)
            else:
                t = None
    vars(out)[longFlag[2:]] = n
    return argv

def to_tuple(s):
    try:
        return int(s)
    except:
        v = s.split(",")
        if len(v) != 2:
            raise ValueError("%s is not a valid kernel size" % s)
        return (int(v[0]), int(v[1]))

convArgHelp="""convolution arguments:
  each convolution can be provided additional arguments

  convolutions:
    -1 --conv1 [convolution options] first convolution layer
    -2 --conv2 [convolution options] second convolution layer

  convolution options:
    -l --filters FILTERS number of convolution filters (default 30/128)
    -a --activation FN   activation function {"softmax", "elu",
       "selu", "softplus", "softsign", "relu", "tanh", "sigmoid",
       "hard_sigmoid", "exponential", "linear"} (default: tanh)
    -k --kernel SIZE     size of kernel (default: 5,5)
    -p --pool SIZE       size of pool (default: 2,2)
    -d --dropout RATE    dropout percentage (default: 0.15/0.10)
"""
def parse_args(argv):
    convFlags = [
        ("-l", "--filters", "layers", int, 32),
        ("-a", "--activation", "activeFn", activate_fns, "tanh"),
        ("-k", "--kernel", "kernel", to_tuple, (5, 5)),
        ("-p", "--pool", "pool", to_tuple, (2, 2)),
        ("-d", "--dropout", "dropout", float, 0.15)
    ]
    parser = argparse.ArgumentParser(epilog=convArgHelp,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--flags', action="store_true",
                        help='print arguments')
    parser.add_argument('-r', '--rows', default=400, type=int,
                        help='number of rows per image')
    parser.add_argument('-c', '--cols', default=56, type=int,
                        help='number of columns per image')
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='number of training iterations')
    parser.add_argument('-b', '--batch', default=30, type=int,
                        help='size of a training batch')
    parser.add_argument('--color', action='store_true', help='Data is color image')
    parser.add_argument('-o', '--optimizer', choices=optimizers, default='rmsprop',
                        help='optimize function')
    parser.add_argument('-w', '--weights', type=argparse.FileType('w'),
                        default='weights.best.hdf5',
    parser.add_argument('-m', '--model', type=argparse.FileType('w'),
                        default='model_plot.png',
                        help='Output file for model.')
    parser.add_argument('-g', '--no-graphs', dest="graphs", action='store_false',
                        help='do not show graphs once processing completes.')
    parser.add_argument('-v', '--verbose', action='count',
                        help='print more or less information.')
    parser.add_argument('data_path', nargs='?',
                        type=data_dir_path, help='Data directory',
                        default="%s/../data" % os.path.dirname(sys.argv[0]))
    parser.add_argument('--version', action='version', version='%(prog)s 1.1')
    n = argparse.Namespace()
    argv = get_convolution_args(n, "-1", "--conv1", convFlags, argv)
    convFlags[0] = ("-l", "--layers", "layers", int, 128)
    convFlags[4] = ("-d", "--dropout", "dropout", float, 0.1)
    argv = get_convolution_args(n, "-2", "--conv2", convFlags, argv)
    return parser.parse_args(args=argv[1:], namespace=n)

def build_model(args):
    # Build model
    model = Sequential()
    model.add(Conv2D(filters=args.conv1.layers,
                     kernel_size=args.conv1.kernel,
                     input_shape=(args.rows, args.cols, 1),
                     padding='valid',
                     activation=args.conv1.activeFn,
                     strides=1))

    if args.conv1.pool != (1, 1):
        model.add(MaxPooling2D(pool_size=args.conv1.pool))
    if args.conv1.dropout != 0:
        model.add(Dropout(args.conv1.dropout))

    if args.conv2.layers != 0:
        model.add(Conv2D(filters=args.conv2.layers,
                         kernel_size=args.conv2.kernel,
                         padding='valid',
                         activation=args.conv2.activeFn,
                         strides=1))
        if args.conv2.pool != (1, 1):
            model.add(MaxPooling2D(pool_size=args.conv2.pool))
        if args.conv2.dropout != 0:
            model.add(Dropout(args.conv2.dropout))
        model.add(Reshape((args.conv2.layers,-1)))
    else:
        model.add(Reshape((args.conv1.layers,-1)))
    model.add(Permute((2,1)))
    model.add(Bidirectional(LSTM(128)))# TODO: Parameterize
    model.add(Dense(len(os.listdir(train_dir(args.data_path))),
                    activation='softmax'))
    return model
#Start
def run(argv=sys.argv):
    args = parse_args(argv)
    if args.flags:
        print(' '.join(argv[1:]))
    bins = os.listdir(train_dir(args.data_path))
    train_dir_path = train_dir(args.data_path)
    test_dir_path = test_dir(args.data_path)
    num_of_train_samples = 0
    num_of_test_samples = 0
    for b in bins:
        num_of_train_samples += len(os.listdir("%s/%s" % (train_dir_path, b)))
        num_of_test_samples += len(os.listdir("%s/%s" % (test_dir_path, b)))

    #Image Generator
    train_generator = data_generator(args, train_dir_path)
    validation_generator = data_generator(args, test_dir_path)
    
    # Build model
    model = build_model(args)

    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=args.optimizer,
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(args.weights, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list=[checkpoint]
    
    plot_model(model, to_file=args.model, show_shapes=True,
               show_layer_names=True)

    #Train
    history = model\
              .fit_generator(train_generator,
                             steps_per_epoch=num_of_train_samples // args.batch,
                             epochs=args.epochs,
                             verbose=args.verbose,
                             callbacks=callbacks_list,
                             validation_data=validation_generator,
                             validation_steps=num_of_test_samples // args.batch)

    if args.graphs:
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
    
    #print("\nFinal score: %f" % score[1])
    # TODO: Fix hardcoded index to use model.metrics_names
    return model.evaluate_generator(validation_generator,
                                    num_of_test_samples,
                                    use_multiprocessing=True)[1]
if __name__ == "__main__":
    run()
