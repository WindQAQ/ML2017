import argparse
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import load_model
from keras.engine.topology import Layer


class WeightedAvgOverTime(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedAvgOverTime, self).__init__(**kwargs)
   
    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, axis=-1)
            s = K.sum(mask, axis=1)
            if K.equal(s, K.zeros_like(s)) is None:
                return K.mean(x, axis=1)
            else:
                return K.cast(K.sum(x * mask, axis=1) / K.sqrt(s), K.floatx())
        else:
            return K.mean(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, x, mask=None):
        return None

    def get_config(self):
        base_config = super(WeightedAvgOverTime, self).get_config()
        return dict(list(base_config.items()))


def parse_args():
    parser = argparse.ArgumentParser('Matrix Factorization.')
    parser.add_argument('--model', nargs='+', required=True)
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

    return df['TestDataID'], df[['UserID', 'MovieID']].values


def submit(filename, id, pred):
    df = pd.DataFrame({'TestDataID': id, 'Rating': pred}, columns=('TestDataID', 'Rating'))
    df.to_csv(filename, index=False, columns=('TestDataID', 'Rating'))


def main(args):
    user2id = np.load(args.user2id)[()]
    movie2id = np.load(args.movie2id)[()]
    id, X_test = read_data(args.test, user2id, movie2id)

    pred_en = []
    for fmodel in args.model:
        model = load_model(fmodel, custom_objects={'rmse': rmse, 'WeightedAvgOverTime': WeightedAvgOverTime})
        pred = model.predict([X_test[:, 0], X_test[:, 1]]).squeeze()
        pred = pred.clip(1.0, 5.0)
        pred_en.append(pred)
    
    pred = np.mean(pred_en, axis=0)

    submit(args.output, id, pred)
     

if __name__ == '__main__':
    args = parse_args()
    main(args)
