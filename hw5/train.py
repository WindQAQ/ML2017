import pickle
import argparse
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Sequential
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import AveragePooling1D
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer

def fmeasure(y_true, y_pred):
    thres = 0.4
    y_pred = K.cast(K.greater(y_pred, thres), dtype='float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=-1)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=-1)
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return K.mean(f)


class ArticleClassifier():
    def __init__(self, tokenizer, mlb, embedding, maxlen=400):
        self.tokenizer = tokenizer
        self.mlb = mlb
        self.num_classes = len(mlb.classes_)
        self.vocab_size = len(tokenizer.word_index)
        self.maxlen = maxlen
        self.embedding = embedding
        self.embedding_size = embedding.shape[1]
        print('#classes: {}'.format(self.num_classes))

    def build(self, rnn_size=256, num_layer=1):
        model = Sequential()
        model.add(Embedding(
            self.vocab_size,
            self.embedding_size,
            input_length=self.maxlen,
            weights=[self.embedding],
            trainable=False))

        ''' 
        model.add(Conv1D(filters=64, kernel_size=5, padding='same'))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Conv1D(filters=64, kernel_size=5, padding='same'))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        #model.add(AveragePooling1D(pool_size=2, padding='same'))
        '''

        for i in range(num_layer):
            model.add(GRU(rnn_size, activation='tanh', return_sequences=(i != num_layer-1), dropout=0.3))
        
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))    

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(self.num_classes, activation='sigmoid'))
        self.model = model

    def summary(self):
        self.model.summary()

    def compile(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[fmeasure])

    def fit(self, X, Y, **kwargs):
        self.model.fit(X, Y, **kwargs)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)

    def save(self):
        pickle.dump(self.tokenizer, open('word_index', 'wb'))
        pickle.dump(self.mlb, open('label_mapping', 'wb'))

def print_func(func):
    def wrapper(*args, **kwargs):
        print(func.__name__)
        return func(*args, **kwargs)
    return wrapper


@print_func
def read_data(filename, isTrain=True):
    texts, labels = [], []
    with open(filename, encoding='latin1') as f:
        f.readline()
        if isTrain:
            for line in f:
                _, label, text = line.strip('\r\n').split('"', 2)
                texts.append(text[1:])
                labels.append(label.split(' '))
            return texts, labels
        else:
            for line in f:
                _, text = line.strip('\r\n').split(',', 1)
                texts.append(text)
            return texts


@print_func
def preprocess(train, test, maxlen=400, split=" ", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\'\t\n'):
    tokenizer = Tokenizer(filters=filters, split=split)
    tokenizer.fit_on_texts(train + test)
    sequences_train = tokenizer.texts_to_sequences(train)
    sequences_test = tokenizer.texts_to_sequences(test)
    tokenizer.word_index['<PAD>'] = 0
    return pad_sequences(sequences_train, maxlen=maxlen), pad_sequences(sequences_test, maxlen=maxlen), tokenizer


@print_func
def parse_args():
    parser = argparse.ArgumentParser(description='Article Classification.')
    parser.add_argument('--train')
    parser.add_argument('--test')
    parser.add_argument('--embedding')
    return parser.parse_args()


@print_func
def read_embedding(filename, word2id, embedding_size=100):
    embedding = np.zeros((len(word2id), embedding_size))
    with open(filename) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                embedding[word2id[word]] = np.array(vec.split(' ')).astype(float)
        
        return embedding


def main(args):

    if args.train is not None:
        texts_train, labels = read_data(args.train)
        texts_test = read_data(args.test, isTrain=False)
        sequences, _, tokenizer = preprocess(texts_train, texts_test)
        embedding = read_embedding(args.embedding, tokenizer.word_index)

        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)


        for i in range(5, 10):
            model = ArticleClassifier(tokenizer=tokenizer, mlb=mlb, embedding=embedding)

            model.build(num_layer=1, rnn_size=128)
            model.compile()
            model.summary()

            callbacks = [EarlyStopping(monitor='val_fmeasure', patience=10, verbose=1, mode='max'),
                    ModelCheckpoint('model-{}.h5'.format(i), monitor='val_fmeasure', verbose=1, save_best_only=True, mode='max')]
           
            np.random.seed(i)
            order = np.random.permutation(len(sequences))
            sequences, labels = sequences[order], labels[order]

            sequences_train, labels_train = sequences[490:], labels[490:]
            sequences_valid, labels_valid = sequences[:490], labels[:490]

            model.fit(sequences_train, labels_train, epochs=50000, batch_size=128, validation_data=(sequences_valid, labels_valid), callbacks=callbacks)
            model.save()

if __name__ == '__main__':
    args = parse_args()
    main(args)
