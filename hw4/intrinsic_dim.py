import sys
import numpy as np
from sklearn import decomposition

def load_data(filename):
    return np.load(filename)

def main():
    data = load_data(sys.argv[1])

    ans = []
    for i in range(200):
        print('Process set id {}'.format(i))
        X = data[str(i)]
        pca = decomposition.PCA()
        pca.fit(X)
        cum = np.cumsum(pca.explained_variance_ratio_)
        ans.append(np.sum(cum <= 0.95))

    with open(sys.argv[2], 'w') as fout:
        print('SetId,LogDim', file=fout)
        print('\n'.join(['{},{}'.format(i, np.log(d)) for (i, d) in enumerate(ans)]), file=fout)

if __name__ == '__main__':
    main()
