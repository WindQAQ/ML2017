import argparse
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import add
from keras.layers import Dot
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Embedding
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


def parse_args():
    parser = argparse.ArgumentParser('Matrix Factorization.')
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--dnn', type=int, nargs='*')

    return parser.parse_args()


def read_data(trainfile, testfile):
    traindf, testdf = pd.read_csv(trainfile), pd.read_csv(testfile)

    traindf['test'] = 0
    testdf['test'] = 1

    df = pd.concat([traindf, testdf])

    id2user = df['UserID'].unique()
    id2movie = df['MovieID'].unique()

    user2id = {k: id for id, k in enumerate(id2user)}
    movie2id = {k: id for id, k in enumerate(id2movie)}

    df['UserID'] = df['UserID'].apply(lambda x: user2id[x])
    df['MovieID'] = df['MovieID'].apply(lambda x: movie2id[x])

    df = df.loc[df['test'] == 0]

    return df[['UserID', 'MovieID']].values, df['Rating'].values, user2id, movie2id


def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))


def build(num_users, num_movies, dim, dnn=None):
    u_input = Input(shape=(1,))
    U = Embedding(num_users, dim, embeddings_regularizer=l2(0.00001))(u_input)
    U = Reshape((dim,))(U)
    U = Dropout(0.1)(U)

    m_input = Input(shape=(1,))
    M = Embedding(num_movies, dim, embeddings_regularizer=l2(0.00001))(m_input)
    M = Reshape((dim,))(M)
    M = Dropout(0.1)(M)

    if dnn is None:
        pred = Dot(axes=-1)([U, M])
        U_bias = Reshape((1,))(Embedding(num_users, 1, embeddings_regularizer=l2(0.00001))(u_input))
        M_bias = Reshape((1,))(Embedding(num_users, 1, embeddings_regularizer=l2(0.00001))(m_input))

        pred = add([pred, U_bias, M_bias])
        pred = Lambda(lambda x: x + K.constant(3.5817, dtype=K.floatx()))(pred)

    else:
        pred = Concatenate()([U, M])
        for units in dnn:
            pred = Dense(units, activation='relu')(pred)
            pred = Dropout(0.3)(pred)

        pred = Dense(1, activation='relu')(pred)
        
    return Model(inputs=[u_input, m_input], outputs=[pred])


def main(args):
    X_train, Y_train, user2id, movie2id = read_data(args.train, args.test)

    np.save('user2id', user2id)
    np.save('movie2id', movie2id)

    np.random.seed(5)
    indices = np.random.permutation(len(X_train))
    X_train, Y_train = X_train[indices], Y_train[indices]
    
    num_users, num_movies = len(user2id), len(movie2id)
    dim = args.dim
    
    model = build(num_users, num_movies, dim, dnn=args.dnn)
    model.summary()

    callbacks = []
    callbacks.append(EarlyStopping(monitor='val_rmse', patience=10))
    callbacks.append(ModelCheckpoint('model.h5', monitor='val_rmse', save_best_only=True))

    model.compile(loss='mse', optimizer='adam', metrics=[rmse])
    model.fit([X_train[:, 0], X_train[:, 1]], Y_train, epochs=1000, batch_size=128, validation_split=0.1, callbacks=callbacks) 


if __name__ == '__main__':
    args = parse_args()
    main(args)
