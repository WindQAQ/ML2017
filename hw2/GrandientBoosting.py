import sys
import pandas as pd
import numpy as np
import pickle

class DecisionNode():
    def __init__(self, attr=None, thres=None, left=None, right=None, pred=None):
        self.attr = attr
        self.thres = thres
        self.left = left
        self.right = right
        self.pred = pred

class DecisionTree():
    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float('inf')):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self._impurity_on_node = None
        self._impurity = None
        self._leaf_pred = None

    def fit(self, X, Y):
        self.root = self._build(X, Y)

    def _build(self, X, Y, depth=0):
        min_impurity = init_impurity = 1e20
        best_criteria = None

        n_samples, n_features = X.shape

        size = len(Y)
        if n_samples >= self.min_samples_split and depth <= self.max_depth:
            for attr in range(n_features):
                sorted_indices = np.argsort(X[:, attr])
                X, Y = X[sorted_indices], Y[sorted_indices]

                for i in range(0, n_samples-1):
                    if X[i, attr] < X[i+1, attr]:
                        impurity = (
                                self._impurity_on_node(Y[:i+1]) * (i+1) + 
                                self._impurity_on_node(Y[i+1:]) * (size-i-1)
                                ) / size
                        if impurity < min_impurity:
                            thres = (X[i, attr] + X[i+1, attr]) / 2
                            min_impurity = impurity
                            best_criteria = (attr, thres)

        impurity_on_node = self._impurity_on_node(Y)
        if  impurity_on_node < self.min_impurity or min_impurity == init_impurity or depth > self.max_depth:
            pred = self._leaf_pred(Y)
            print('predict {} with impurity {}'.format(pred, impurity_on_node))
            return DecisionNode(pred=pred)
        else:
            attr, thres = best_criteria
            print('split at attr {} with threshold {}'.format(attr, thres))
            l, r = (X[:, attr] < thres), (X[:, attr] >= thres)
            X_l, Y_l, X_r, Y_r = X[l], Y[l], X[r], Y[r]
            left = self._build(X_l, Y_l, depth=depth+1)
            right = self._build(X_r, Y_r, depth=depth+1)
            return DecisionNode(attr=attr, thres=thres, left=left, right=right)

    def predict(self, X):
        pred = []
        for x in X:
            pred.append(self._predict_one(x, node=self.root))
        return np.array(pred)

    def _predict_one(self, x, node=None):
        if node.pred is not None:
            return node.pred

        f = x[node.attr]
        child = node.left
        if f >= node.thres:
            child = node.right

        return self._predict_one(x, node=child)

class RegressionTree(DecisionTree):
    def _cal_var(self, Y):
        mean = np.ones(np.shape(Y)) * Y.mean(0)
        n_samples = np.shape(Y)[0]
        var = (1 / n_samples) * np.diag((Y-mean).T.dot(Y-mean))
        return var

    def _variance_reduce(self, Y, Y_l, Y_r):
        var = self._cal_var(Y)
        var_l = self._cal_var(Y_l)
        var_r = self._cal_var(Y_r)
      
        return np.sum(var - var_l * (len(Y_l)/len(Y)) - var_r * (len(Y_r)/len(Y)))

    def _variance_on_node(self, Y):
        return np.sum(self._cal_var(Y))

    def _mean(self, Y):
        return np.mean(Y, axis=0)

    def fit(self, X, Y):
        self._impurity_on_node = self._variance_on_node
        self._impurity = self._variance_reduce
        self._leaf_pred = self._mean
        super(RegressionTree, self).fit(X, Y)

class GradientBoostClassifier():
    def __init__(self, n_estimators=20, learning_rate=0.1, 
            max_depth=3, min_samples_split=2, min_impurity=1e-7):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity

        self.estimators = []
        for i in range(self.n_estimators):
            est = RegressionTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=self.min_impurity,
                    max_depth=self.max_depth)
            self.estimators.append(est)

    def fit(self, X, Y):
        pred = np.full(np.shape(Y), np.mean(Y, axis=0))
        for i, est in enumerate(self.estimators):
            print('build tree {}'.format(i))
            grad = self._gradient(Y, pred)
            est.fit(X, grad)
            update = est.predict(X)

            pred -= self.learning_rate * update

    def predict(self, X):
        pred = 0.0
        for est in self.estimators:
            update = est.predict(X)
            pred -= self.learning_rate * update

        return np.argmax(pred, axis=1)

    def _gradient(self, y, pred):
        return -(y - 1. / (1. + np.exp(-pred)))

def read_data(filename, label=False):
    if label:
        data = pd.read_csv(filename, header=None)
    else:
        data = pd.read_csv(filename)

    return data.as_matrix().astype('float')
    
def one_hot(Y):
    _Y = np.zeros((Y.shape[0], 2))
    _Y[np.arange(Y.shape[0]), Y.flatten()] = 1

    return _Y

def main(args):
    X = read_data(args[1])
    Y = read_data(args[2], label=True).astype('int') 
    X_test = read_data(args[3])

    Y = one_hot(Y)

    valid = None
    if len(args) >= 6:
        valid_num = int(args[5])
        X_train, Y_train = X[:-valid_num], Y[:-valid_num]
        valid = (X[-valid_num:], Y[-valid_num:])
    else:
        X_train, Y_train = X, Y

    if len(args) == 7:
        # load model
        print('load model, path {}'.format(args[6]))
        with open(args[6], 'rb') as fmodel:
            model = pickle.load(fmodel)
    else:
        # train model
        model = GradientBoostClassifier(n_estimators=1, learning_rate=0.5, max_depth=3)
        model.fit(X_train, Y_train)

    try:
        pred = model.predict(X_train)
        train_acc = np.mean(pred.flatten() == np.argmax(Y_train, axis=1).flatten())
        print('training accuracy: {}'.format(train_acc))
        if valid is not None:
            pred = model.predict(valid[0])
            valid_acc = np.mean(pred.flatten() == np.argmax(valid[1], axis=1).flatten())
            print('valid accuracy: {}'.format(valid_acc))
    except:
        pass

    if len(args) < 7:
        with open('model', 'wb') as fmodel:
            pickle.dump(model, fmodel, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args[4], 'w') as fout:
        print('id,label', file=fout)
        pred = model.predict(X_test)
        for (i, v) in enumerate(pred.flatten()):
            print('{},{}'.format(i+1, 1 if v >= 0.5 else 0), file=fout)

if __name__ == '__main__':
    main(sys.argv)
