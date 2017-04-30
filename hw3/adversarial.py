#!/usr/bin/env python

import os
import pickle
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage.filters import median_filter
from scipy.ndimage.filters import gaussian_filter
from keras.models import load_model
from keras import backend as K
from utils import *

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def deprocess(x):
    x = (x - x.mean()) / (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def main():
    parser = argparse.ArgumentParser(prog='maximize_class.py',
            description='ML-Assignment3 image that maximize class.')
    parser.add_argument('--model', type=str, metavar='<#model>', required=True)
    parser.add_argument('--data', type=str, metavar='<#data>', required=True)
    parser.add_argument('--attr', type=str, metavar='<#attr>', required=True)
    parser.add_argument('--lr', type=float, metavar='<#lr>', default=8000.)
    parser.add_argument('--step', type=int, metavar='<#step>', default=1000)
    parser.add_argument('--filter_dir', type=str, metavar='<#filter_dir>', default='./image/filter/')
    args = parser.parse_args()

    lr = args.lr
    data_name = args.data
    num_step = args.step
    attr_name = args.attr
    model_name = args.model
    filter_dir = args.filter_dir

    if not os.path.isdir(filter_dir):
        os.mkdir(filter_dir)

    adver_dir = os.path.join(filter_dir, 'adver')
    if not os.path.isdir(adver_dir):
        os.mkdir(adver_dir)

    print('load data')
    data = pickle.load(open(data_name, 'rb'))
    X, Y, _X = data[0], data[1], data[2]

    print('load attr')
    attr = np.load(attr_name)
    mean, std = attr[0], attr[1]

    print('load model')
    emotion_classifier = load_model(model_name)

    class_names = ['angry', 'digust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    img_idx = -4987
    class_index = 5
    class_name = class_names[class_index]

    with Image.new('L', (48, 48)) as img:
        img.putdata(_X[img_idx].flatten())
        img = img.resize((128, 128), Image.ANTIALIAS)
        img.save(os.path.join(adver_dir, 'adversarial-{}.png'.format(img_idx)))

    print('Process class {}'.format(class_name))
    input_img = emotion_classifier.input
    output_class = emotion_classifier.output[:, class_index]
    input_img_data = X[img_idx].reshape(1, 48, 48, 1)

    print('Label: {}'.format(class_names[Y[img_idx]]))
    print(' '.join([str(p) for p in emotion_classifier.predict(input_img_data).flatten()]))

    target = K.mean(output_class)
    grads = K.gradients(target, input_img)[0]
    iterate = K.function([input_img, K.learning_phase()], [target, grads])

    for i in range(num_step):
        input_img_data *= 0.99

        if (i+1) % 3 == 0:
            input_img_data = median_filter(input_img_data, size=(1, 1, 3, 3))

        if (i+1) % 7 == 0:
            input_img_data = gaussian_filter(input_img_data, sigma=[0, 0, 0.3, 0.3])

        target_val, grads_val = iterate([input_img_data, 0])
        print('Step {}, Current target: {}'.format(i, target_val))

        input_img_data += grads_val * lr

    print(' '.join([str(p) for p in emotion_classifier.predict(input_img_data).flatten()]))

    input_img_data = input_img_data.reshape(48, 48, 1)
    input_img_data = (input_img_data * std + mean) * 255.
    input_img_data = np.clip(input_img_data, 0, 255).astype('uint8')
    with Image.new('L', (48, 48)) as img:
        img.putdata(input_img_data.flatten())
        img = img.resize((128, 128), Image.ANTIALIAS)
        img.save(os.path.join(adver_dir, 'adversarial-{}-{}.png'.format(class_name, img_idx)))

if __name__ == "__main__":
    main()
