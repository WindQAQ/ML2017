import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix

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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.3f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main(args):
    class_names = ['Angry', 'Digust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    width = height = 48
    X, Y = read_data(args[1], label=True, width=width, height=height)
    attr = np.load(args[3])

    X = (X - attr[0]) / (attr[1] + 1e-20)

    X, Y = X[-5000:], Y[-5000:]

    model = load_model(args[2])
    pred = model.predict_classes(X)

    cnf_matrix = confusion_matrix(pred, Y) 

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, 
            title='Confusion Matrix')

    plt.show()

if __name__ == '__main__':
    main(sys.argv)
