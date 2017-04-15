import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import Activation
from keras.layers.advanced_activations import *
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

def read_data(filename, label=True, width=48, height=48):
    width = height = 48
    with open(filename, 'r') as f:
        data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
        data = np.array(data)
        X = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1, width, height, 1)).astype('float')
        Y = data[::width*height+1].astype('int')

        X /= 255

        if label:
            return X, Y
        else:
            return X

def main(args):
    width = height = 48

    print('read data')
    X, Y = read_data(args[1], label=True, width=width, height=height)

    input_shape = (width, height, 1)
    num_classes = int(np.max(Y) + 1)
    batch_size = 128
    epochs = 100000000

    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    np.save('attr.npy', [mean, std])

    X = (X - mean) / (std + 1e-20)

    X = np.concatenate((X, X[:, :, ::-1]), axis=0)
    Y = np.concatenate((Y, Y), axis=0)

    Y = to_categorical(Y, num_classes)

    print('input_shape: {}, num_classes: {}'.format(input_shape, num_classes))

    datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.8, 1.2],
            shear_range=0.2,
            horizontal_flip=True)

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=input_shape, padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(5, 5), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.35))

    model.add(Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='glorot_normal'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    callbacks = []
    callbacks.append(ModelCheckpoint('ckpt/model-{epoch:05d}.h5', monitor='loss', period=1))
    #callbacks = [EarlyStopping(monitor='val_loss', patience=312)]
    #model.fit(X, Y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    
    model.fit_generator(
            datagen.flow(X, Y, batch_size=batch_size), 
            steps_per_epoch=5*len(X)//batch_size,
            epochs=epochs,
            callbacks=callbacks
            )

    model.save(args[2])    

if __name__ == '__main__':
    main(sys.argv)
