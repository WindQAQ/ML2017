import nltk
import argparse
import word2vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from adjustText import adjust_text
from collections import OrderedDict

def print_func(func):
    def wrapper(*args, **kwargs):
        print(func.__name__)
        return func(*args, **kwargs)
    return wrapper

def parse_args():
    parser = argparse.ArgumentParser(description='word2vec')
    parser.add_argument('--data')
    parser.add_argument('--model')
    return parser.parse_args()

@print_func
def train(filename):
    save_path = '{}.bin'.format(filename)
    word2vec.word2vec(
            train=filename, 
            output=save_path, 
            size=128,
            window=5,
            sample=1e-3,
            hs=1,
            negative=5,
            min_count=3,
            alpha=0.025,
            verbose=True)

@print_func
def load(model_path):
    return word2vec.load(model_path)

@print_func
def plot(model, top=1000):
    pos_tags = ['JJ', 'NNP', 'NN', 'NNS']
    punctuations = ['“', '”', ',', '.', ':', ';', '’', '!', '?', '`']

    plot_vocabs = []
    for vocab, tag in nltk.pos_tag(model.vocab[:top]):
        plot_vocabs.append([vocab, model[vocab], tag])

    vocabs, vectors, tags = list(zip(*plot_vocabs))

    tsne = TSNE(n_components=2)
    points = tsne.fit_transform(list(vectors))
    xs, ys = points[:, 0], points[:, 1]
    xs, ys = xs * 10000, ys * 10000

    plt.figure(figsize=(80, 60), dpi=150)
    plt.xticks([])
    plt.yticks([])

    texts = []
    colors = {'JJ': 'r', 'NNP': 'g', 'NN': 'b', 'NNS': 'y'}
    for x, y, vocab, tag in zip(xs, ys, vocabs, tags):
        if tag not in pos_tags or len(vocab) == 1 or any(p in vocab for p in punctuations):
            continue
        plt.scatter(x, y, color=colors[tag], label=tag, s=60)
        texts.append(plt.text(x, y, vocab, size=40))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
    
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    
    plt.savefig('scatter.png')

def main(args):
    data = args.data
    model_path = args.model
    if data is not None:
        train(data)
    
    if model_path is not None:
        model = load(model_path)
        plot(model)

if __name__ == '__main__':
    args = parse_args()
    main(args)
