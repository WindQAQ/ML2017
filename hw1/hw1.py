import argparse
import pandas as pd
import numpy as np

def parse_arg():
	parser = argparse.ArgumentParser(description='ML hw1: Linear Regression.')
	parser.add_argument('-train', action='store', type=str, dest='train_data', required=True)
	parser.add_argument('-epoch', action='store', type=int, dest='epoch', default=1000)
	parser.add_argument('-lr', action='store', type=float, dest='lr', default=0.05)
	parser.add_argument('-test', action='store', type=str, dest='test_data', required=True)
	parser.add_argument('-output', action='store', type=str, dest='output', required=True)	

	return parser.parse_args()

def read_data(filename, istrain=True):
	hr = 0
	if istrain:
		raw_data = pd.read_csv(filename, encoding='big5').as_matrix()

		_X = raw_data[:, 3:]
		_X[_X == 'NR'] = 0.
		_X = _X.astype('float')

		_X = _X.reshape(12, -1, 24)
		_X = _X.reshape(12, -1, 18, 24)
		_X = _X.swapaxes(1, 2).reshape(12, 18, -1)

		X, Y = [], []
		for m in range(_X.shape[0]):
			for i in range(0, _X.shape[2]-10):
				X.append(_X[m, :, i+hr:i+9].flatten())
				Y.append(_X[m, 9, i+10])

		X, Y = np.asarray(X), np.asarray(Y)
		return X, Y
	else:
		raw_data = pd.read_csv(filename, header=None, encoding='big5').as_matrix()
		X = raw_data[:, 2+hr:]
		X[X == 'NR'] = 0.
		X = X.astype('float')
		X = np.reshape(X, (-1, 18*(9-hr)))
		return X

def feature_scale(X, X_test):
	for i in range(X.shape[1]):
		# if i in range(90, 100):
		# 	X[:, i] = np.square(X[:, i])
		# 	X_test[:, i] = np.square(X_test[:, i])
		tmp = X[:, i]
		# M, m = tmp.max(), tmp.min()
		# X[:, i] = (X[:, i] - m) / (M-m + 1e-20)
		# X_test[:, i] = (X_test[:, i] - m) / (M-m + 1e-20)

		mean, std = tmp.mean(), tmp.std()
		X[:, i] = (X[:, i] - mean) / std
		X_test[:, i] = (X_test[:, i] - mean) / (std + 1e-20)

	return X, X_test

def linear_regression(X, Y, lr, max_epoch):
	N = X.shape[0] # number of data

	P = X.shape[1] # number of features
	W = np.random.rand(P, 1) - 0.5 # weight matrix

	print('Number of data: ', N)
	print('Number of features: ', P)

	W_lr = np.zeros((P, 1))
	for epoch in range(1, max_epoch+1):
		H = predict(W, X).flatten()
		diff = H - Y
		loss = np.sqrt((1. / N) * np.sum(np.square(diff)))
		if epoch % 100 == 0:
			print('Epoch: {}, loss: {}'.format(epoch, loss))

		dW = np.zeros((P, ))
		for i in range(N):
			dW = dW + diff[i] * X[i, :]

		dW = 2 * dW.reshape((P, 1))
		W_lr = W_lr + np.square(dW)
		W = W - lr / np.sqrt(W_lr) * dW
		# W = W - lr * dW

	return W

def predict(W, X):
	return np.dot(X, W)

def main(args):
	X, Y = read_data(args.train_data)
	X_test = read_data(args.test_data, istrain=False)

	X, X_test = feature_scale(X, X_test)
	X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1) # concat ones
	X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1) # concat ones

	W = linear_regression(X, Y, args.lr, args.epoch)
	H_test = predict(W, X_test)

	with open(args.output, 'w') as f:
		print('id,value', file=f)
		for (i, v) in enumerate(H_test):
			print('id_{},{}'.format(i, v[0]), file=f)

if __name__ == '__main__':
	args = parse_arg()
	main(args)