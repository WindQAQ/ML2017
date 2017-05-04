import sys
import multiprocessing
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_data(filename):
    return np.load(filename)

def LL(d, p, k, N):
    return N*np.log2(k) + N*np.log2(d) + (d - 1) * np.sum(np.log2(p + 1e-20)) + (k - 1) * np.sum(np.log2(1 - p**d + 1e-20))

def main():
    data = load_data(sys.argv[1])
    
    n_jobs = np.round(3*multiprocessing.cpu_count()/4).astype(int)
    D = 60
    k = 10 # for MiND
    k1, k2 = 6, 20 # for MLE
    ans = []
    for id in range(200):
        print('Process set id {}'.format(id))

        X = data[str(id)]
        X = X[:10000]
        N = X.shape[0]

        print('{} data points'.format(N))

        nbrs = NearestNeighbors(n_neighbors=k2+1, n_jobs=n_jobs, algorithm='kd_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        #knn = distances[:, 1:k2+1]

        # MiND
        knn = distances[:, 1:k+1]
        p = np.min(knn, axis=-1) / np.max(knn, axis=-1)
        d_i = [LL(d, p, k, N) for d in range(1, D+1)]
        d_hat = np.argmax(d_i) + 1
        print('MiND estimator: {}'.format(d_hat))

        # MLE
        if d_hat >= 16:
            knn = 0.5 * np.log(distances[:, 1:k2+1])
            S = np.cumsum(knn, axis=-1)
            idk = np.arange(k1, k2+1)
            d_hat = -(idk-2) / (S[:, k1-1:k2] - knn[:, k1-1:k2] * idk)
            d_hat = np.mean(d_hat)
            d_hat = np.round(d_hat).astype(int)
            print('MLE estimator: {}'.format(d_hat))

        ans.append(d_hat)

        print('Estimation of intrinsic dimension: {}'.format(d_hat))

    with open(sys.argv[2], 'w') as fout:
        print('SetId,LogDim', file=fout)
        print('\n'.join(['{},{}'.format(i, np.log(d)) for (i, d) in enumerate(ans)]), file=fout)

if __name__ == '__main__':
    main()
