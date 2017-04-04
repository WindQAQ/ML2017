import sys
import numpy as np
import pandas as pd

def read_data(filename, label=False):
    if label:
        data = pd.read_csv(filename, header=None)
        dtype = 'int'
    else:
        data = pd.read_csv(filename)
        dtype = 'float'

    return data.as_matrix().astype(dtype)
    
def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)

class GenerativeModel():
    def __init__(self):
        pass

    def _get_attr(self, X):
        return 0, 1
        #return np.mean(X, axis=0), np.std(X, axis=0) 

    def _scale(self, X):
        return (X - self._mean) / (self._std + 1e-20)

    def fit(self, X, Y, valid=None, num_classes=2):
        assert X.shape[0] == Y.shape[0]

        self.num_classes = num_classes
        self._feature_dim = X.shape[1]
        self._mean, self._std = self._get_attr(X)
        X = self._scale(X)
        if valid is not None:
            X_valid = self._scale(valid[0])
            Y_valid = valid[1]
        
        self._num, self._mu = [], []
        self._share_cov = 0.0
        for i in range(num_classes):
            C = X[(Y == i).flatten()]
            num = C.shape[0]
            mu = np.mean(C, axis=0)
            cov = self._covariance(C, mu)
            self._num.append(num)
            self._mu.append(mu)
            self._share_cov += (num / X.shape[0]) * cov

        self._inv_cov = np.linalg.inv(self._share_cov)
        print('training accuracy: {:.5f}'.format(self.evaluate(X, Y, test=False)))
        if valid is not None:
            print('valid accuracy: {:.5f}'.format(self.evaluate(X_valid, Y_valid, test=False)))

    def evaluate(self, X, Y, test=True):
        pred = self.predict(X, test=test)
        pred = np.around(1.0-pred)
        result = (Y.flatten() == pred)
        return np.mean(result)

    def _p(self, x, mu, invcov, num):
        return np.exp(-1./2 * self._vec_inner_prod(np.dot((x-mu), invcov), (x-mu).T)) * num

    def _covariance(self, X, mu):
        return np.mean([(X[i]-mu).reshape((-1, 1)) * (X[i]-mu).reshape((1, -1)) for i in range(X.shape[0])], axis=0)

    def _vec_inner_prod(self, A, B):
        return np.array([np.dot(A[i, :], B[:, i]) for i in range(A.shape[0])])

    def predict(self, X, test=True):
        if test:
            X = self._scale(X)

        W = np.dot( (self._mu[0]-self._mu[1]), self._inv_cov)
        X = X.T
        B = (-0.5) * np.dot(np.dot([self._mu[0]], self._inv_cov), self._mu[0]) + (0.5) * np.dot(np.dot([self._mu[1]], self._inv_cov), self._mu[1]) + np.log(self._num[0]/self._num[1])
        a = np.dot(W, X) + B
        y = sigmoid(a)
        return y

def main(args):
    X = read_data(args[1])
    Y = read_data(args[2], label=True)    
    X_test = read_data(args[3])

    valid = None
    if len(args) == 6:
        valid_num = int(args[5])
        X_train, Y_train = X[:-valid_num], Y[:-valid_num]
        valid = (X[-valid_num:], Y[-valid_num:])
    else:
        X_train, Y_train = X, Y

    model = GenerativeModel()
    model.fit(X_train, Y_train, valid=valid)
    with open(args[4], 'w') as fout:
        print('id,label', file=fout)
        pred = model.predict(X_test, test=True)
        for (i, v) in enumerate(pred.flatten()):
            print('{},{}'.format(i+1, 0 if v >= 0.5 else 1), file=fout)

if __name__ == '__main__':
    main(sys.argv)
