import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers.advanced_activations import *
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
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
    epochs = 300

    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    X = (X - mean) / (std + 1e-20)
    np.save('attr.npy', [mean, std])

    print('input_shape: {}, num_classes: {}'.format(input_shape, num_classes))

    datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=[0.8, 1.2],
            horizontal_flip=True)

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, kernel_size=(3, 3), kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(Conv2D(64, kernel_size=(3, 3), kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, kernel_size=(3, 3), kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(Conv2D(128, kernel_size=(3, 3), kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())

    model.add(Dense(2048, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='glorot_normal'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    #callbacks = [EarlyStopping(monitor='val_loss', patience=312)]
    callbacks = None
    #model.fit(X, Y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    
    model.fit_generator(
            datagen.flow(X, Y, batch_size=batch_size), 
            steps_per_epoch=len(X) // batch_size,
            epochs=epochs)

    model.save(args[2])    

if __name__ == '__main__':
    main(sys.argv)
