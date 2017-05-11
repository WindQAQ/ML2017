import os
import argparse
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

def print_func(func):
    def wrapper(*args, **kwargs):
        print('plot {}'.format(func.__name__))
        return func(*args, **kwargs)
    return wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='PCA - eigenfaces')
    parser.add_argument('--data_dir', type=str, metavar='<#data>', required=True)
    parser.add_argument('--avg', action='store_true')
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--findk', action='store_true')
    parser.add_argument('--eigenface', type=int, metavar='<#eigenface>')
    parser.add_argument('--reconstruct', type=int, metavar='<#reconstruct>')
    return parser.parse_args()

def read_data(dir):
    X = []
    size = None
    for file in sorted(os.listdir(dir)):
        if file.endswith('.bmp'):
            img = scipy.misc.imread(os.path.join(dir, file), flatten=True)
            size = img.shape
            X.append(img.flatten())

    return np.array(X), size

@print_func
def average_face(X, size=(64, 64)):
    mean = np.mean(X, axis=0)
    scipy.misc.imsave('avg_face.png', mean.reshape(size))

@print_func
def eigenface(X, size=(64, 64), top=9):
    mean = np.mean(X, axis=0)
    X_center = X - mean
    U, S, V = np.linalg.svd(X_center.T, full_matrices=False)

    sub_width, sub_height = 3, np.ceil(top / 3).astype(int)
    fig = plt.figure(figsize=(9, 9))
    for i in range(top):
        ax = fig.add_subplot(sub_height, sub_width, i+1)
        ax.imshow(U[:, i].reshape(size), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    
    fig.savefig('eigenface.png')

@print_func
def original_face(X, size=(64, 64)):
    fig = plt.figure(figsize=(10, 10))
    for (i, img) in enumerate(X):
        ax = fig.add_subplot(10, 10, i+1)
        ax.imshow(img.reshape(size), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    
    fig.savefig('original_face.png')

@print_func
def reconstruct_face(X, size=(64, 64), top=5):
    mean = np.mean(X, axis=0)
    X_center = X - mean
    U, S, V = np.linalg.svd(X_center.T, full_matrices=False)
    weights = np.dot(X_center, U)

    fig = plt.figure(figsize=(10, 10))
    for (i, _) in enumerate(X):
        recon = mean + np.dot(weights[i, :top], U[:, :top].T)
        ax = fig.add_subplot(10, 10, i+1)
        ax.imshow(recon.reshape(size), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
    
    fig.savefig('reconstruct_face.png')    

def findk(X, size=(64, 64)):
    mean = np.mean(X, axis=0)
    X_center = X - mean
    U, S, V = np.linalg.svd(X_center.T, full_matrices=False)
    weights = np.dot(X_center, U)

    for k in range(1, U.shape[1]):
        rmse = []
        for (i, img) in enumerate(X):
            recon = mean + np.dot(weights[i, :k], U[:, :k].T)
            err = ((recon - img) / 256) ** 2
            rmse.append(err)
        rmse = np.sqrt(np.mean(rmse))
        print('k: {:3d}, RMSE: {:.8f}'.format(k, rmse))
        if rmse <= 1e-2:
            return k

def main(args):
    data_dir = args.data_dir
    X, size = read_data(data_dir)

    if args.avg:
        average_face(X, size=size)
    
    if args.eigenface is not None:
        eigenface(X, size=size, top=args.eigenface)

    if args.reconstruct is not None:
        reconstruct_face(X, size=size, top=args.reconstruct)

    if args.original:
        original_face(X, size=size)

    if args.findk:
        findk(X, size=size)

if __name__ == '__main__':
    args = parse_args()
    main(args)
