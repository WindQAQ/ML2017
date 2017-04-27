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
from keras.callbacks import CSVLogger
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

    X = (X - mean) / (std + 1e-20)

    X_train, X_valid = X[:-5000], X[-5000:]
    Y_train, Y_valid = Y[:-5000], Y[-5000:]

    X_train = np.concatenate((X_train, X_train[:, :, ::-1]), axis=0)
    Y_train = np.concatenate((Y_train, Y_train), axis=0)

    order = np.random.permutation(X_train.shape[0])
    X_train, Y_train = X_train[order], Y_train[order]
    R = 0.20
    num_unlabeled = int(X_train.shape[0] * R)

    Y_train = to_categorical(Y_train, num_classes)
    Y_valid = to_categorical(Y_valid, num_classes)

    X_labeled, Y_labeled = X_train[:-num_unlabeled], Y_train[:-num_unlabeled]
    X_unlabeled = X_train[-num_unlabeled:]

    print('input_shape: {}, num_classes: {}'.format(input_shape, num_classes))

    datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.8, 1.2],
            shear_range=0.2,
            horizontal_flip=True)

    model = Sequential()

    model.add(Conv2D(64, kernel_size=(5, 5), input_shape=input_shape, padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
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

    #callbacks = []
    #callbacks.append(ModelCheckpoint('ckpt/model-phase1-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1))
    #callbakcs.append(CSVLogger('log/self-training-phase1'))

    phase_epochs = 10

    i = 0
    confidence = 0.7
    X_train, Y_train = X_labeled, Y_labeled
    while len(X_unlabeled) > 0:
        callbacks = []
        callbacks.append(CSVLogger('log/self-training-phase{}'.format(i)))

        model.fit_generator(
                datagen.flow(X_train, Y_train, batch_size=batch_size), 
                steps_per_epoch=1852,
                epochs=phase_epochs,
                validation_data=(X_valid, Y_valid),
                callbacks=callbacks
            )

        pred_unlabeled = model.predict(X_unlabeled)
        pred_proba = np.max(pred_unlabeled, axis=-1)
        pred_class = np.argmax(pred_unlabeled, axis=-1).flatten()

        index = np.argwhere(pred_proba >= confidence).flatten()
        if len(index) == 0:
            continue

        print(X_unlabeled[index].shape, X_train.shape, index.shape, pred_class.shape)

        X_train = np.concatenate((X_train, X_unlabeled[index]), axis=0)
        Y_train = np.concatenate((Y_train, to_categorical(pred_class[index], num_classes)), axis=0)
        X_unlabeled = np.delete(X_unlabeled, index, axis=0)

        print('Phase {}, remain {} unlabeled data'.format(i, len(X_unlabeled)))
        i += 1

    callbacks = []
    callbacks.append(CSVLogger('log/self-training-phase{}'.format(i)))
    model.fit_generator(
            datagen.flow(X_train, Y_train, batch_size=batch_size), 
            steps_per_epoch=1852,
            epochs=phase_epochs,
            validation_data=(X_valid, Y_valid),
            callbacks=callbacks
        )

if __name__ == '__main__':
    main(sys.argv)
