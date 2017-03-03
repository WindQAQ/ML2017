import sys
import pandas as pd
import numpy as np

def ReadTrainData(filename):
    raw_data = pd.read_csv(filename, encoding='big5').as_matrix()
    data = raw_data[:, 3:] # 12 months, 20 days per month, 18 features per day. shape: (4320 , 24)
    data[data == 'NR'] = 0.0
    data = data.astype('float')

    X, Y = [], []
    for i in range(0, data.shape[0], 18*20):
        # i: start of each month
        days = np.vsplit(data[i:i+18*20], 20) # shape: 20 * (18, 24)
        concat = np.concatenate(days, axis=1) # shape: (18, 480)
        for j in range(0, concat.shape[1]-10):
            X.append(concat[:, j:j+9].flatten())
            Y.append([concat[9, j+10]])

    return np.array(X), np.array(Y)

def ReadTestData(filename):
    raw_data = pd.read_csv(filename, header=None, encoding='big5').as_matrix()
    data = raw_data[:, 2:]
    data[data == 'NR'] = 0.0
    data = data.astype('float')

    obs = np.vsplit(data, data.shape[0]/18)
    X = []
    for i in obs:
        X.append(i.flatten())

    return np.array(X)

class Linear_Regression():
    def __init__(self):
        pass

    def _error(self, X, Y):
        return Y - self.predict(X)

    def _loss(self, X, Y):
        return np.sqrt(np.mean(np.square(self._error(X, Y))))

    def _init_parameters(self):
        self.B = np.random.rand() - 0.5
        self.W = np.random.rand(self.feature_dim, 1) - 0.5

    def _scale(self, X, istrain=True):
        if istrain:
            self.max = np.max(X, axis=0)
            self.min = np.min(X, axis=0)
        return (X - self.min) / (self.max - self.min)

    def fit(self, _X, Y, max_epoch=100000, lr=0.1):
        assert _X.shape[0] == Y.shape[0]
        N = _X.shape[0]
        self.feature_dim = feature_dim = _X.shape[1]

        X = self._scale(_X)

        self._init_parameters()

        B_lr = 1e-20
        W_lr = np.full((feature_dim, 1), 1e-20)
        for epoch in range(1, max_epoch+1):

            B_grad = np.random.rand() - 0.5
            W_grad = np.random.rand(feature_dim, 1) - 0.5

            error = self._error(X, Y)

            B_grad = B_grad - 2.0 * np.sum(error) * 1.0
            W_grad = W_grad - 2.0 * np.dot(X.T, error)

            B_lr = B_lr + B_grad ** 2
            W_lr = W_lr + W_grad ** 2

            self.B = self.B - lr / np.sqrt(B_lr) * B_grad
            self.W = self.W - lr / np.sqrt(W_lr) * W_grad

            if epoch % 100 == 0:
                print('[Epoch {}]: loss: {}'.format(epoch, self._loss(X, Y)))

    def predict(self, X):
        _X = X.reshape((-1, self.feature_dim))
        return np.dot(_X, self.W) + self.B
 
    def predict_test(self, X):
        _X = self._scale(X, istrain=False)
        _X = _X.reshape((-1, self.feature_dim))
        return np.dot(_X, self.W) + self.B

def main(args):
    X, Y = ReadTrainData(args[1])

    model = Linear_Regression()
    model.fit(X, Y, max_epoch=100000, lr=10)

    X_test = ReadTestData(args[2])
    predict = model.predict_test(X_test)

    with open(args[3], 'w') as f:
        print('id,value', file=f)
        for (i, p) in enumerate(predict) :
            print('id_{},{}'.format(i, p[0]), file=f)

if __name__ == '__main__':
    main(sys.argv)
