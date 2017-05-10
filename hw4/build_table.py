import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors

def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)

def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)

def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)

    return out

def gen_data(N, dim, layer_dims):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data @ w + b
    return gen_data

if __name__ == '__main__':
    # if we want to generate data with intrinsic dimension of 10
    # the hidden dimension is randomly chosen from [60, 79] uniformly

    std_table, mean_table = [], []
    for dim in range(1, 61):
        print('Process dim {}'.format(dim))
        stds, means, coef_vars = [], [], []
        for i in range(100):
            N = np.random.randint(1, 11) * 10000
            layer_dims = [np.random.randint(60, 80), 100]
            data = gen_data(N, dim, layer_dims)
            data = data[:5000]
            nbrs = NearestNeighbors(n_jobs=-1, n_neighbors=2).fit(data)
            distances, indices = nbrs.kneighbors(data)
            knn = distances[:, 1]
            std = np.std(knn)
            mean = np.mean(knn)
            stds.append(std)
            means.append(mean)
            print('\tRound {}, N: {}, h: {}, std: {}, mean: {}'.format(i, N, layer_dims[0], std, mean))
        std_table.append(np.mean(stds))
        mean_table.append(np.mean(means))
        print('\toverall std: {}, mean: {}'.format(np.mean(stds), np.mean(means)))

    np.save('std.npy', std_table)
    np.save('mean.npy', mean_table)
