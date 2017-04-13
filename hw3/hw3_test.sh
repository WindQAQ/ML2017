#!/usr/bin/env bash

cat model/model-part* > model.h5

python3.6 test.py $1 $2 model.h5 attr.npy
