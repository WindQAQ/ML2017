import pickle
import argparse
import numpy as np
import keras.backend as K
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


def submit(filename, labels):
    with open(filename, 'w') as fout:
        print('"id","tags"',file=fout)
        for id, label in enumerate(labels):
            if len(label) == 0:
                print('"{}","SPECULATIVE-FICTION"'.format(id), file=fout)
            else:
                print('"{}","{}"'.format(id,' '.join([l for l in label])), file=fout)


def parse_args():
    parser = argparse.ArgumentParser('Predict.')
    parser.add_argument('--model', nargs='+')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--mlb', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def main(args):
    tokenizer = pickle.load(open(args.tokenizer, 'rb'))
    mlb = pickle.load(open(args.mlb, 'rb'))
    texts = read_data(args.test, isTrain=False)    
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = pad_sequences(sequences, maxlen=400)

    
    pred = []
    for mfile in args.model:
        print('model path - {}'.format(mfile))
        model = load_model(mfile, custom_objects={'fmeasure': fmeasure})
        pred.append(model.predict(sequences))
        
    pred = np.mean(pred, axis=0)
    pred[pred >= thres] = 1
    pred[pred < thres] = 0
    pred = mlb.inverse_transform(pred)
    submit(args.output, pred)

if __name__ == '__main__':
    args = parse_args()
    main(args)
