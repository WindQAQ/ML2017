#/usr/bin/env bash

python3.6 predict.py --test $1/test.csv --output $2 --user2id best/user2id.npy --movie2id best/movie2id.npy --model best/dual-svdpp-model-14.h5 best/dual-svdpp-model-5.h5
