import sys
import multiprocessing
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_data(filename):
    return np.load(filename)

def LL(d, p, k, N):
    return N*np.log(k) + N*np.log(d) + (d - 1) * np.sum(np.log(p + 1e-20)) + (k - 1) * np.sum(np.log(1 - p**d + 1e-20))

def main():
    data = load_data(sys.argv[1])
    table = np.load(sys.argv[2])

    n_jobs = multiprocessing.cpu_count()
    D = 60
    k = 10 # for MiND_ML
    k1, k2 = 3, 10 # for MLE
    ans = []
    for id in range(200):
        print('Process set id {}'.format(id))

        X = data[str(id)]
        X = X[:5000]
        N = X.shape[0]

        print('\t{} data points'.format(N))

        nbrs = NearestNeighbors(n_neighbors=k2+1, n_jobs=n_jobs, algorithm='kd_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)

        # MiND_ML
        knn = distances[:, 1:k+1]
        p = np.min(knn, axis=-1) / np.max(knn, axis=-1)
        d_i = [LL(d, p, k, N) for d in range(1, D+1)]
        d_hat = np.argmax(d_i) + 1
        print('\tMiND_ML estimator: {}'.format(d_hat))

        # IDEA
        #knn = distances[:, 1:k+1]
        #S = np.cumsum(knn, axis=-1)
        #m = np.sum(S[:, -2] / knn[:, -1])
        #m *= 1.0 / (N*(k-1))
        #d_hat = np.round(m / (1-m)).astype(int)
        #print('\tIDEA estimator: {}'.format(d_hat))

        # MLE
        #if d_hat >= 17:
        #    knn = 0.5 * np.log(distances[:, 1:k2+1])
        #    S = np.cumsum(knn, axis=-1)
        #    idk = np.arange(k1, k2+1)
        #    d_hat = -(idk-2) / (S[:, k1-1:k2] - knn[:, k1-1:k2] * idk)
        #    d_hat = np.mean(d_hat)
        #    d_hat = np.round(d_hat).astype(int)
        #    print('\tMLE estimator: {}'.format(d_hat))
        
        if d_hat >= 14:
            knn = distances[:, 1]
            std = np.std(knn)
            diff = np.abs(table-std)
            #if d_hat < np.argmin(diff) + 1 + 41:
            d_hat = np.argmin(diff) + 1
            print('\tlook up table estimator: {}'.format(d_hat))

        ans.append(d_hat)

        print('\tEstimation of intrinsic dimension: {}'.format(d_hat))
        print()

    with open(sys.argv[3], 'w') as fout:
        print('SetId,LogDim', file=fout)
        print('\n'.join(['{},{}'.format(i, np.log(d)) for (i, d) in enumerate(ans)]), file=fout)

if __name__ == '__main__':
    main()
