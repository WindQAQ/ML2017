import sys
import numpy as np
from keras.models import load_model

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

def submit(pred, filename):
    with open(filename, 'w') as f:
        print('id,label', file=f)
        print('\n'.join(['{},{}'.format(i, p) for (i, p) in enumerate(pred)]), file=f)

def main(args):
    width = height = 48
    X_test = read_data(args[1], label=False, width=width, height=height)
    attr = np.load(args[4])

    X_test = (X_test - attr[0]) / (attr[1] + 1e-20)

    model = load_model(args[3])
    pred = model.predict_classes(X_test)
    
    submit(pred, args[2])

if __name__ == '__main__':
    main(sys.argv)
