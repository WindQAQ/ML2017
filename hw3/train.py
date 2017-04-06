import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
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
    X_train, Y_train = read_data(args[1], label=True, width=width, height=height)

    input_shape = (width, height, 1)
    num_classes = int(np.max(Y_train) + 1)
    batch_size = 256
    epochs = 50

    Y_train = to_categorical(Y_train, num_classes)

    print('input_shape: {}, num_classes: {}'.format(input_shape, num_classes))

    model = Sequential()

    model.add(Conv2D(25, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(50, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    score = model.evaluate(X_train, Y_train)
    print('Training accuracy: {}'.format(score))

    model.save(args[2])    

if __name__ == '__main__':
    main(sys.argv)
