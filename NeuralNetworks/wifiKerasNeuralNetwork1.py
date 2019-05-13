import numpy as np
import pandas as pd
import os
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Reshape, Permute
from keras.layers import TimeDistributed, Conv2D, Dense, Dropout, Activation, LSTM, MaxPooling2D, GRU, ConvLSTM2D, Bidirectional
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization

#Start
train_data_path = '../data/train'
test_data_path = '../data/validation'
img_rows = 51
img_cols = 56
epochs = 30
batch_size = 30
num_of_train_samples = 366
num_of_test_samples = 144
convFilter1 = 128
maxRows = 500
# input_shape = (img_rows, img_cols, 1)
input_shape = (maxRows, 256, 1)

#Data

def get_target(entry):
    if entry.find("amanda", 0, 10) >= 0:
        return 0
    elif entry.find("andreas", 0, 10) >= 0:
        return 1
    elif entry.find("empty", 0, 10) >= 0:
        return 2
    elif entry.find("lucy", 0, 10) >= 0:
        return 3
    else: # robert
        return 4

labels = []
train_data = pd.DataFrame()
for entry in os.listdir('../csv-data'):
    print(entry)
    tmp_df = pd.read_csv('../csv-data/' + entry, header=0, index_col=0)
    if tmp_df.shape[0] > maxRows:
        train_data = train_data.append(tmp_df[:maxRows])
    elif tmp_df.shape[0] > 0:
        train_data = train_data.append(tmp_df)
        # fill zeros for empty rows for uniform # of rows
        train_data = train_data.append(pd.DataFrame(np.full((maxRows-tmp_df.shape[0], 256), '(0+0j)'), columns=train_data.columns))
    tmp = np.zeros(5)
    tmp[get_target(entry)] = 1
    labels.append(tmp[:])
    # labels.append(np.eye(5)[get_target(entry)])
train_data = train_data.apply(lambda col: col.apply(lambda val: complex(val.strip('()')).real if isinstance(val, str) else val))


labels = np.asarray(labels, dtype=np.uint8)
train_data = np.reshape(train_data.values, (train_data.shape[0] // maxRows, maxRows, 256, 1)) # df float64 to ndarray float32, reshaped
print(train_data.shape)
print(labels.shape)

'''
mean = df.mean()
df -= mean                  # Shift so average is zero
std = df.std()
df /= std
'''

'''
#Image Generator
train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(train_data_path,
                                                    color_mode='grayscale',
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(test_data_path,
                                                        color_mode='grayscale',
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
'''
# Build model
model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(5,5),
                 input_shape=input_shape,
                 padding='valid',
                 activation='tanh',
                 strides=1))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.15))

model.add(Conv2D(filters=convFilter1,
                 kernel_size=(5,5),
                 padding='valid',
                 activation='tanh',
                 strides=1))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(.10))

model.add(Reshape((convFilter1,-1)))
model.add(Permute((2,1)))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(5, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list=[checkpoint]

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#Train
'''
history = model.fit_generator(train_generator,
                    steps_per_epoch=num_of_train_samples // batch_size,
                    epochs=epochs,
                    callbacks=callbacks_list,
                    validation_data=validation_generator,
                    validation_steps=num_of_test_samples // batch_size)
'''
history = model.fit(train_data, labels, batch_size=4, epochs=1) # , validation_split=0.25
'''
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
plt.show() ''' 
