import argparse
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser('Matrix Factorization.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--user2id', required=True)
    parser.add_argument('--movie2id', required=True)

    return parser.parse_args()


def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))


def read_data(filename, user2id, movie2id):
    df = pd.read_csv(filename)

    df['UserID'] = df['UserID'].apply(lambda x: user2id[x])
    df['MovieID'] = df['MovieID'].apply(lambda x: movie2id[x])

    df = df.drop(['TestDataID'], axis=1)

    return df[['UserID', 'MovieID']].values


def submit(filename, pred):
    df = pd.DataFrame({'TestDataID': list(range(1, len(pred)+1)), 'Rating': pred}, columns=('TestDataID', 'Rating'))
    df.to_csv(filename, index=False, columns=('TestDataID', 'Rating'))


def main(args):
    user2id = np.load(args.user2id)[()]
    movie2id = np.load(args.movie2id)[()]
    X_test = read_data(args.test, user2id, movie2id)

    model = load_model(args.model, custom_objects={'rmse': rmse})
    
    pred = model.predict([X_test[:, 0], X_test[:, 1]]).squeeze()

    pred = pred.clip(1.0, 5.0)
    submit(args.output, pred)
     

if __name__ == '__main__':
    args = parse_args()
    main(args)
