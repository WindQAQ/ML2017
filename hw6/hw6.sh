#/usr/bin/env bash

python3.6 predict.py --test $1/test.csv --output $2 --user2id best/user2id.npy --movie2id best/movie2id.npy --model best/mean-bias-model.h5
