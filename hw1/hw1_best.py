import sys
import math
import pandas as pd
import numpy as np

attrs = ['AMB', 'CH4', 'CO', 'NMHC', 'NO', 'NO2',
        'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH',
        'SO2', 'THC', 'WD_HR', 'WIND_DIR', 'WIND_SPEED', 'WS_HR']

attr_range = {}

for i, attr in enumerate(attrs):
    attr_range[attr] = list(range(9*i, 9*i+9))

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
        for j in range(0, concat.shape[1]-9):
            X.append(concat[:, j:j+9].flatten())
            Y.append([concat[9, j+9]])

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
        return np.sqrt(np.mean(self._error(X, Y) ** 2))

    def _scale(self, X, istrain=True):
        if istrain:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0) + 1e-20
        return (X - self.mean) / self.std

    def _add_bias(self, X):
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def fit(self, _X, Y, valid):
        assert _X.shape[0] == Y.shape[0]
        N = _X.shape[0]
        self.feature_dim = feature_dim = _X.shape[1]

        X = self._scale(_X)
        X_bias = self._add_bias(X)

        self.W = np.linalg.lstsq(X_bias, Y)[0]
        print('training loss: {}'.format(self._loss(X, Y)))

        if valid is not None:
            X_valid, Y_valid = valid
            X_valid = self._scale(X_valid, istrain=False)
            print('valid loss: {}'.format(self._loss(X_valid, Y_valid)))
            

    def predict(self, X):
        _X = np.reshape(X, (-1, self.feature_dim))
        _X = self._add_bias(_X)
        return np.dot(_X, self.W)
 
    def predict_test(self, X):
        _X = self._scale(X, istrain=False)
        _X = _X.reshape((-1, self.feature_dim))
        _X = self._add_bias(_X)
        return np.dot(_X, self.W)

def main(args):
    X, Y = ReadTrainData(args[1])
    X_test = ReadTestData(args[2])

    select_attr = attrs
    select_attr = ['PM10', 'PM2.5', 'O3', 'WIND_DIR', 'WIND_SPEED', 'WD_HR', 'WS_HR', 'RAINFALL']
    select_range = []
    for attr in select_attr:
        select_range += attr_range[attr]

    X = X[:, select_range]
    X_test = X_test[:, select_range]

    X = np.concatenate((X, X[:, 0:18] ** 3), axis=1)
    X_test = np.concatenate((X_test, X_test[:, 0:18] ** 3), axis=1)

    valid = None
    try:
        valid_num = int(args[4])
        order = np.random.permutation(X.shape[0])
    except:
        coef = np.loadtxt(args[4])
        valid_num = coef[0].astype('int')
        order = coef[1:].astype('int')
    X, Y = X[order], Y[order]
    valid = X[:valid_num], Y[:valid_num]
    X, Y = X[valid_num:], Y[valid_num:]

    model = Linear_Regression()
    model.fit(X, Y, valid=valid)

    predict = model.predict_test(X_test)
    with open(args[3], 'w') as f:
        print('id,value', file=f)
        for (i, p) in enumerate(predict) :
            print('id_{},{}'.format(i, p[0]), file=f)

    np.savetxt('coef.txt', np.concatenate((np.asarray(valid_num).reshape((1, )), order)), fmt='%d')

if __name__ == '__main__':
    main(sys.argv)
