import pickle
import argparse
import numpy as np
from gensim.models import doc2vec
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import text_to_word_sequence

def print_func(func):
    def wrapper(*args, **kwargs):
        print(func.__name__)
        return func(*args, **kwargs)
    return wrapper


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
                sentence = doc2vec.LabeledSentence(
                        words=text_to_word_sequence(text[1:], filters=filters),
                        tags=['TRAIN_'+id]
                        )
                sentences.append(sentence)
                labels.append(label.split(','))
            return sentences, labels
        else:
            for line in f:
                id, text = line.strip('\r\n').split(',', 1)
                sentence = doc2vec.LabeledSentence(
                        words=text_to_word_sequence(text, filters=filters),
                        tags=['TEST_'+id]
                        )
                sentences.append(sentence)
            return sentences


def submit(filename, labels):
    with open(filename, 'w') as fout:
        print('id,tags',file=fout)
        for id, label in enumerate(labels):
            if len(label) == 0:
                print('{},"FICTION"'.format(id), file=fout)
            else:
                print('{},"{}"'.format(id,','.join([l for l in label])), file=fout)


@print_func
def parse_args():
    parser = argparse.ArgumentParser(description='Article Classification.')
    parser.add_argument('--train')
    parser.add_argument('--test')
    parser.add_argument('--save')
    parser.add_argument('--output')
    parser.add_argument('--model')
    return parser.parse_args()


class ArticleClassifier():
    def __init__(self, sentences, *args, **kwargs):
        self.d2v = doc2vec.Doc2Vec(sentences, **kwargs)
        self.OvR = OneVsRestClassifier(SVC(C=25., class_weight='balanced'), n_jobs=-1)
        self.mlb = MultiLabelBinarizer()

    def train_doc2vec(self, sentences, epochs):
        total_words = sum([len(sen.words) for sen in sentences])
        self.d2v.train(sentences, total_words=total_words, epochs=epochs)

    def save_doc2vec(self, path):
        self.d2v.save(path)

    def get_doc2vec(self, tags):
        return self.d2v.docvecs[tags].squeeze()

    def fit_OvR(self, X, Y):
        self.OvR.fit(X, Y)

    def predict(self, X):
        return self.OvR.predict(X)

    def fit_transform_mlb(self, labels):
        return self.mlb.fit_transform(labels)

    def inverse_transform_mlb(self, labels):
        return self.mlb.inverse_transform(labels)

def main(args):
    if args.train is not None:
        sentences_train, labels = read_data(args.train)
        sentences_test = read_data(args.test, isTrain=False)
        sentences = sentences_train + sentences_test

        model = ArticleClassifier(sentences,size=256, window=10, negative=10, min_count=3, alpha=0.025, min_alpha=0.025, workers=8)
        
        print('doc2vec')
        model.train_doc2vec(sentences, epochs=10)

        X_train = []
        for sentence in sentences_train:
            X_train.append(model.get_doc2vec(sentence.tags))

        X_test = []
        for sentence in sentences_test:
            X_test.append(model.get_doc2vec(sentence.tags))

        labels = model.fit_transform_mlb(labels)

        print('multilabel')
        model.fit_OvR(X_train, labels)

        pickle.dump(model, open(args.save, 'wb'))
    else:
        model = pickle.load(open(args.model, 'rb'))

        sentences_test = read_data(args.test, isTrain=False)
        X_test = []
        for sentence in sentences_test:
            X_test.append(model.get_doc2vec(sentence.tags))


    pred = model.predict(X_test)
    pred = model.inverse_transform_mlb(pred)
         
    if args.output:
        submit(args.output, pred)


if __name__ == '__main__':
    args = parse_args()
    main(args)
