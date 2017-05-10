import sys
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_data(filename):
    return np.load(filename)

def parse_args():
    parser = argparse.ArgumentParser(description='intrinsic dimension')
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mean')
    parser.add_argument('--std')
    parser.add_argument('--mle', nargs=2, type=int)
    parser.add_argument('--idea', type=int)
    parser.add_argument('--mind_ml', type=int)
    parser.add_argument('--mind_kl', type=int)
    return parser.parse_args()

def randSphere(d, N, r=1):
    y = np.random.normal(0., 1., size=(N, d))
    l2 = np.linalg.norm(y, axis=-1, keepdims=True)
    y = y * (np.random.rand(N, d) ** (1/d)) / l2
    return y

def LL(d, p, k, N):
    return N*np.log(k) + N*np.log(d) + (d - 1) * np.sum(np.log(p + 1e-20)) + (k - 1) * np.sum(np.log(1 - p**d + 1e-20))

def MiND_MLk(D, N, k, distances):
    knn = distances[:, 1:k+1]
    p = np.min(knn, axis=-1) / np.max(knn, axis=-1)
    d_i = [LL(d, p, k, N) for d in range(1, D+1)]
    d_hat = np.argmax(d_i) + 1

    return d_hat

def build_sphere(D, N, k):
    pExp = {}
    for d in range(1, D+1):
        y = randSphere(d, N)
        nbrs = NearestNeighbors(n_neighbors=k+1, n_jobs=-1, algorithm='kd_tree').fit(y)
        distancesExp, _ = nbrs.kneighbors(y)
        knnExp = distancesExp[:, 1:k+1]
        pExp[d] = np.min(knnExp, axis=-1) / np.max(knnExp, axis=-1)

    return pExp

def KL(p1, p2):
    N = len(p1)
    p1 = np.sort(p1)
    nn1 = np.insert(p1, [0, p1.shape[0]], [-np.inf, np.inf])
    nn1 = np.abs(nn1[1:] - nn1[:-1])
    nn1 = np.minimum(nn1[:-1], nn1[1:])
    nn2 = np.min(np.abs(p1.reshape((-1, 1)) - p2.reshape((1, -1))), axis=-1)

    div = np.abs(np.log(N / (N-1.)) + (1./N)*np.sum(np.log(nn2/(nn1+1e-20))))

    return div

def MiND_KL(D, N, k, distances, pExp):
    knn = distances[:, 1:k+1]
    p = np.min(knn, axis=-1) / np.max(knn, axis=-1)

    kl = []
    for d in range(1, D+1):
        '''
        y = randSphere(d, N)
        nbrs = NearestNeighbors(n_neighbors=k+1, n_jobs=-1, algorithm='kd_tree').fit(y)
        distancesExp, _ = nbrs.kneighbors(y)
        knnExp = distancesExp[:, 1:k+1]
        pExp = np.min(knnExp, axis=-1) / np.max(knnExp, axis=-1)
        '''
        kl.append(KL(pExp[d], p))

    return np.argmin(kl) + 1

def IDEA(N, k, distances):
    knn = distances[:, 1:k+1]
    S = np.cumsum(knn, axis=-1)
    m = np.sum(S[:, -2] / knn[:, -1])
    m *= 1.0 / (N*(k-1))
    d_hat = np.round(m / (1-m)).astype(int)

    return d_hat

def MLE(k1, k2, distances):
    knn = 0.5 * np.log(distances[:, 1:k2+1])
    S = np.cumsum(knn, axis=-1)
    idk = np.arange(k1, k2+1)
    d_hat = -(idk-2) / (S[:, k1-1:k2] - knn[:, k1-1:k2] * idk)
    d_hat = np.mean(d_hat)
    d_hat = np.round(d_hat).astype(int)

    return d_hat

def look_up(table, distances, type):
    knn = distances[:, 1]
    if type == 'std':
        std = np.std(knn)
        diff = np.abs(table-std)
    elif type == 'mean':    
        mean = np.mean(knn)
        diff = np.abs(table-mean)
    else:
        mean = np.mean(knn)
        std = np.std(knn)
        coef_var = std / mean
        diff = np.abs(table-coef_var)

    d_hat = np.argmin(diff) + 1

    return d_hat

def main(args):
    data = load_data(args.data)
    if args.std is not None:
        std_table = np.load(args.std)
    if args.mean is not None:
        mean_table = np.load(args.mean)

    if args.mean is not None or args.std is not None:
        if args.mean is not None and args.std is not None:
            table = std_table / mean_table
            type = 'coef_var'
        elif args.mean is not None:
            table = mean_table
            type = 'mean'
        else:
            table = std_table
            type = 'std'

    N = 5000
    D = 60
    k_max = 20
    ans = []
    
    if args.mind_kl:
        pExp = build_sphere(D, N, args.mind_kl)

    for id in range(200):
        print('Process set id {}'.format(id))

        X_ = data[str(id)]
        X = X_[:N]

        print('\t{} data points'.format(N))

        nbrs = NearestNeighbors(n_neighbors=k_max+1, n_jobs=-1, algorithm='kd_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)

        # MiND_ML
        if args.mind_ml is not None:
            d_hat = MiND_MLk(D, N, args.mind_ml, distances)
            print('\tMiND_MLk estimator: {}'.format(d_hat))

        # IDEA
        if args.idea is not None:
            d_hat = IDEA(N, args.idea, distances)
            print('\tIDEA estimator: {}'.format(d_hat))

        # MLE
        if args.mle is not None:
            d_hat = MLE(args.mle[0], args.mle[1], distances)
            print('\tMLE estimator: {}'.format(d_hat))
        
        # MiND_KL
        if args.mind_kl is not None:
            d_hat = MiND_KL(D, N, args.mind_kl, distances, pExp)
            print('\tMiND_KL estimator: {}'.format(d_hat))

        # look up table
        if args.mean is not None or args.std is not None:
            d_hat = look_up(table, distances, type=type)
            print('\tlook up table estimator: {}'.format(d_hat))

        ans.append(d_hat)

        print('\tEstimation of intrinsic dimension: {}\n'.format(d_hat))

    with open(args.output, 'w') as fout:
        print('SetId,LogDim', file=fout)
        print('\n'.join(['{},{}'.format(i, np.log(d)) for (i, d) in enumerate(ans)]), file=fout)

if __name__ == '__main__':
    args = parse_args()
    main(args)
