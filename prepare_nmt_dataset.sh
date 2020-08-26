#!/usr/bin/env bash

DATASET=$1

if [ "$DATASET"  == "" ]; then
    echo "Please specify the dataset name. Supported choices: wmt14_en_de, wmt14_en_fr"
    exit
fi

##
## WMT14 En2De dataset
##
if [ "$DATASET" == "wmt14_en_de" ]; then
    # Download and prepare the data
    cd examples/translation/
        bash prepare_wmt14_en2de.sh
    cd ../..

    # Preprocess/binarize the data
    TEXT=examples/translation/wmt14_en_de
    fairseq-preprocess \
        --source-lang en --target-lang de \
        --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
        --destdir data-bin/wmt14_en_de --thresholdtgt 0 --thresholdsrc 0 \
        --workers 20 --joined-dictionary
##
## WMT14 En2Fr dataset
##
elif [ "$DATASET" == "wmt14_en_fr" ]; then
    cd examples/translation/
        bash prepare_wmt14_en2fr.sh
    cd ../..

    # Binarize the dataset
    TEXT=examples/translation/wmt14_en_fr
    fairseq-preprocess \
        --source-lang en --target-lang fr \
        --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
        --destdir data-bin/wmt14_en_fr --thresholdtgt 0 --thresholdsrc 0 \
        --workers 60 --joined-dictionary
else
    echo "Only these datasets are supported (wmt14_en_de, wmt14_en_fr)"
fi
