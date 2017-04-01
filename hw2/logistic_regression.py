import sys
import numpy as np
import pandas as pd

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def read_data(filename, label=False):
    if label:
        data = pd.read_csv(filename, header=None)
    else:
        data = pd.read_csv(filename)
        data['square_hours'] = data['hours_per_week'] ** 2
        data['square_gain'] = data['capital_gain'] ** 2
        data['square_loss'] = data['capital_loss'] ** 2
        data['cubic_hours'] = data['hours_per_week'] ** 3

    return data.as_matrix().astype('float')
    
class Logistic_Regression():
    def __init__(self):
        pass

    def _init_para(self):
        self.W = np.zeros((self.feature_dim + 1, 1))
        self._W_lr = 0.0

    def fit(self, X, Y, max_epochs=2000, lr=0.05, C=0.0):
        assert X.shape[0] == Y.shape[0]
        self.feature_dim = X.shape[1]
        self._init_lr = self._lr = lr
        self.C = C

        self._init_para()
        
        self._mean, self._std = self._get_attr(X)
        X = self._scale(X)
        X = self._add_bias(X)

        for epoch in range(1, max_epochs+1):
            self._step(X, Y)

            if epoch % 100 == 0:
                loss = self._loss(X, Y)
                acc = self.evaluate(X, Y)
                print('[Epoch {:5d}] - training loss: {:.5f}, accuracy: {:.5f}'.format(epoch, loss, acc))

    def predict(self, X, test=False):
        # sigmoid(X * W)
        if test:
            X = self._scale(X)
            X = self._add_bias(X)
        return sigmoid(np.dot(X, self.W))

    def _get_attr(self, X):
        return np.mean(X, axis=0), np.std(X, axis=0) 

    def _scale(self, X):
        return (X - self._mean) / (self._std + 1e-20)

    def _step(self, X, Y):
        pred = self.predict(X)
        self._update(X, Y, pred)

    def _update(self, X, Y, pred):
        grad = self._gradient(X, Y, pred)

        self._W_lr = self._W_lr + grad ** 2
        self._lr = self._init_lr / np.sqrt(self._W_lr)

        self.W = self.W - self._lr * (grad + self.C * np.sum(self.W))

    def _gradient(self, X, Y, pred):
        return -np.dot(X.T, (Y - pred))

    def _loss(self, X, Y, pred=None):
        # y_hat: prediction of model
        # -mean(y * log(y_hat) + ((1 - y) * log((1-y_hat))))
        if pred is None:
            pred = self.predict(X)
        return -np.mean(Y * np.log(pred + 1e-20) + (1 - Y) * np.log((1 - pred + 1e-20)))

    def evaluate(self, X, Y):
        pred = self.predict(X)
        p = pred
        p[pred < 0.5] = 0.0
        p[pred >= 0.5] = 1.0
        return np.mean(1 - np.abs(Y - p))

    def _add_bias(self, X):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

def main(args):
    X = read_data(args[1])
    Y = read_data(args[2], label=True)    
    X_test = read_data(args[3])

    model = Logistic_Regression()
    model.fit(X, Y, C=0.0, max_epochs=2000, lr=0.05)

    with open(args[4], 'w') as fout:
        print('id,label', file=fout)
        pred = model.predict(X_test, test=True)
        for (i, v) in enumerate(pred.flatten()):
            print('{},{}'.format(i+1, 1 if v >= 0.5 else 0), file=fout)

if __name__ == '__main__':
    main(sys.argv)
