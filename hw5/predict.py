import pickle
import argparse
import numpy as np
import keras.backend as K
from scipy.stats import mode
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

thres = 0.4

def fmeasure(y_true, y_pred):
    y_pred = K.cast(K.greater(y_pred, thres), dtype='float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=-1)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=-1)
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return K.mean(f)


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


def submit(filename, labels, default):
    with open(filename, 'w') as fout:
        print('"id","tags"',file=fout)
        for id, label in enumerate(labels):
            if len(label) == 0:
                print(id, 'NOTHING')
                print(default[id])
                print('"{}","{}"'.format(id, ' '.join(default[id])), file=fout)
            else:
                print('"{}","{}"'.format(id,' '.join([l for l in label])), file=fout)


def parse_args():
    parser = argparse.ArgumentParser('Predict.')
    parser.add_argument('--model', nargs='+')
    parser.add_argument('--default_pred', nargs='+')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--mlb', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tfidf', action='store_true')
    return parser.parse_args()


def main(args):
    tokenizer = pickle.load(open(args.tokenizer, 'rb'))
    mlb = pickle.load(open(args.mlb, 'rb'))

    texts_test = read_data(args.test, isTrain=False)    
    sequences_test = tokenizer.texts_to_sequences(texts_test)
    sequences_test = pad_sequences(sequences_test, maxlen=400)
    texts_train, labels = read_data(args.train, isTrain=True)    

    labels = mlb.transform(labels)

    pred = []

    if args.tfidf:
        for max_features in [20000, 40000, 60000, 100000]:
            tfidf = Pipeline([
                ('vectorizer', TfidfVectorizer(stop_words='english', sublinear_tf=True, ngram_range=(1, 3), max_features=max_features)),
                ('clf', OneVsRestClassifier(LinearSVC(C=0.0005, class_weight='balanced', random_state=42)))])
            
            tfidf.fit(texts_train, labels)
            pred.append(tfidf.predict(texts_test))

    rnn_default = []
    for mfile in args.model:
        print('model path - {}'.format(mfile))
        model = load_model(mfile, custom_objects={'fmeasure': fmeasure})
        proba = model.predict(sequences_test)
       
        if mfile in args.default_pred:
            rnn_default.append(proba)

        proba[proba >= thres] = 1
        proba[proba < thres] = 0
        pred.append(proba)

    rnn_default = np.mean(rnn_default, axis=0)
    rnn_default[rnn_default >= thres] = 1
    rnn_default[rnn_default < thres] = 0
    rnn_default = mlb.inverse_transform(rnn_default)

    pred = mode(pred, axis=0)[0].squeeze().astype(int)
    pred = mlb.inverse_transform(pred)
    submit(args.output, pred, default=rnn_default)


if __name__ == '__main__':
    args = parse_args()
    main(args)
