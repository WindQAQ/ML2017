import argparse
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

def print_func(func):
    def wrapper(*args, **kwargs):
        print(func.__name__)
        return func(*args, **kwargs)
    return wrapper


@print_func
def parse_args():
    parser = argparse.ArgumentParser(description='Article Classification.')
    parser.add_argument('--train')
    parser.add_argument('--test')
    parser.add_argument('--output')
    return parser.parse_args()


@print_func
def read_data(filename, isTrain=True):
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\'\t\n'
    sentences, labels = [], []
    with open(filename, encoding='latin1') as f:
        f.readline()
        if isTrain:
            for line in f:
                id, label, text = line.strip('\r\n').split('"', 2)
                id = id[:-1]
                sentences.append(text[1:])
                labels.append(label.split(','))
            return sentences, labels
        else:
            for line in f:
                id, text = line.strip('\r\n').split(',', 1)
                sentences.append(text[1:])
            return sentences


@print_func
def submit(filename, labels):
    with open(filename, 'w') as fout:
        print('id,tags',file=fout)
        for id, label in enumerate(labels):
            if len(label) == 0:
                print('{},"FICTION"'.format(id), file=fout)
            else:
                print('{},"{}"'.format(id,','.join([l for l in label])), file=fout)


def main(args):
    if args.train is not None:
        X_train, labels = read_data(args.train)
        X_test = read_data(args.test, isTrain=False)
        
        mlb = MultiLabelBinarizer()
        labels = mlb.fit_transform(labels)

        model = Pipeline([
            ('vectorizer', CountVectorizer(stop_words='english')),
            ('tfidf', TfidfTransformer()),
            ('clf', OneVsRestClassifier(LinearSVC(C=0.001, class_weight='balanced')))])
        
        '''
        model.fit(X_train[:-400], labels[:-400])
        pred = model.predict(X_train[:-400])
        print(f1_score(labels[:-400], pred, average='micro'))
        pred = model.predict(X_train[-400:])
        print(f1_score(labels[-400:], pred, average='micro'))
        '''

        '''
        model.fit(X_train[400:], labels[400:])
        pred = model.predict(X_train[400:])
        print(f1_score(labels[400:], pred, average='micro'))
        pred = model.predict(X_train[:400])
        print(f1_score(labels[:400], pred, average='micro'))
        '''

        model.fit(X_train, labels)
        pred = model.predict(X_train)
        print('train f1 score', f1_score(labels, pred, average='micro'))
    
        pred = model.predict(X_test)
        pred = mlb.inverse_transform(pred)
        if args.output:
            submit(args.output, pred)


if __name__ == '__main__':
    args = parse_args()
    main(args)
