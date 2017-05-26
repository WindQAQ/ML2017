#!/usr/bin/env bash

python3 predict.py --model best/model-5.h5 \
                   --default_pred best/model-5.h5 \
                   --train best/model \
                   --tokenizer best/word_index \
                   --mlb best/label_mapping \
                   --test $1 \
                   --output $2
