import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Reshape, Permute
from keras.layers import TimeDistributed, Conv2D, Dense, Dropout, Activation, LSTM, MaxPooling2D, GRU, ConvLSTM2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers.normalization import BatchNormalization

#Start
train_data_path = './data/train'
test_data_path = './data/validation'
img_rows = 255
img_cols = 56
epochs = 10
batch_size = 15
num_of_train_samples = 750
num_of_test_samples = 250
convFilter1 = 64

input_shape = (img_rows, img_cols, 1)

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

# Build model
model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(5,5),
                 input_shape=input_shape,
                 padding='valid',
                 activation='tanh',
                 strides=1))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=convFilter1,
                 kernel_size=(3,3),
                 padding='valid',
                 activation='tanh',
                 strides=1))

model.add(Reshape((convFilter1,-1)))
model.add(Permute((2,1)))
#for x in train_generator:
#              print(x[0].reshape((191,)))

#model.add(Reshape((None, 191,), input_shape=(1,1,1,191,1))) 
#model.add(ConvLSTM2D(filters=16, kernel_size=(1,7), padding='valid', activation='tanh'))
model.add(LSTM(128))
#model.add(Flatten())

#print(cnn.layers[-1].output_shape)

#model = Sequential()
#model.add(TimeDistributed(cnn, input_shape=cnn.layers[-1].output_shape))
#model.add(LSTM(16))
#model.add(Dense(32, activation='tanh'))
model.add(Dense(5, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Train
model.fit_generator(train_generator,
                    steps_per_epoch=num_of_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_of_test_samples // batch_size)

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['amanda', 'andreas', 'empty', 'lucy', 'robert']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
