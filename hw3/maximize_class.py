#!/usr/bin/env python

import os
import pickle
import argparse
import numpy as np
from PIL import Image
from keras.models import load_model
from keras import backend as K
from utils import *
#from marcos import *

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
    parser.add_argument('--attr', type=str, metavar='<#attr>', required=True)
    parser.add_argument('--lr', type=float, metavar='<#lr>', default=1000.)
    parser.add_argument('--filter_dir', type=str, metavar='<#filter_dir>', default='./image/filter')
    args = parser.parse_args()

    lr = args.lr
    attr_name = args.attr
    model_name = args.model
    filter_dir = args.filter_dir

    if not os.path.isdir(filter_dir):
        os.mkdir(filter_dir)

    max_class_dir = os.path.join(filter_dir, 'max_class')
    if not os.path.isdir(max_class_dir):
        os.mkdir(max_class_dir)

    print('load attr')
    attr = np.load(attr_name)
    mean, std = attr[0], attr[1]

    print('load model')
    emotion_classifier = load_model(model_name)

    class_names = ['Angry', 'Digust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    for (class_index, class_name) in enumerate(class_names):
        print('Process class {}'.format(class_name))
        input_img = emotion_classifier.input
        output_class = emotion_classifier.output[:, class_index]

        input_img_data = np.random.random((1, 48, 48, 1)) # random noise

        input_img_data = (input_img_data - mean) / (std + 1e-20)
        target = K.mean(output_class)
        grads = normalize(K.gradients(target, input_img)[0])
        iterate = K.function([input_img, K.learning_phase()], [target, grads])
        adagrad = 0.0
        while True:
            target_val, grads_val = iterate([input_img_data, 0])
            print('Current target: {}'.format(target_val))

            if target_val >= 0.99999:
                break

            adagrad += grads_val ** 2
            input_img_data += grads_val * lr / np.sqrt(adagrad)

        print(' '.join([str(p) for p in emotion_classifier.predict(input_img_data).flatten()]))

        input_img_data = input_img_data.reshape(48, 48, 1)
        input_img_data = (input_img_data * std + mean) * 255.
        input_img_data = np.clip(input_img_data, 0, 255).astype('uint8')
        with Image.new('L', (48, 48)) as img:
            img.putdata(input_img_data.flatten())
            img = img.resize((128, 128), Image.ANTIALIAS)
            img.save(os.path.join(max_class_dir, '{}-max.png'.format(class_name)))

if __name__ == "__main__":
    main()
