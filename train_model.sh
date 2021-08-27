#!/bin/bash

usage="USAGE: bash train_normalizer.sh [-h] [-o output_dir] [-s source_suffix] [-t target_suffix] [-v ngram_test_file] [-f] INPUT_FILE CONFIG_FILE"

while getopts ":ho:s:t:v:f" opt; do
    case $opt in
        h ) echo $usage
            ;;
        o ) outdir=$OPTARG
            ;;
        s ) src=$OPTARG
            ;;
        t ) tgt=$OPTARG
            ;;
        v ) ngram_test=$OPTARG
            ;;
        f ) force_overwrite=true
            ;;
        \? ) echo "$usage"
            ;;
    esac
done

shift "$(( OPTIND - 1 ))"

arch=transformer
input=$1
config=$2
outdir=${outdir:-textnorm}
src=${src:-fnhd}
tgt=${tgt:-nhd}
force_overwrite=${force_overwrite:-false}
ngram_test=${ngram_test:-""}

if [[ -e ${outdir} ]] && [[ ${force_overwrite} = false ]]; then
    echo "The directory ${outdir} already exists. Choose another directory name with the option -o, or use the option -f to force overwriting."
    exit 1;
fi

if [[ -e ${outdir} ]] && [[ ${force_overwrite} = true ]]; then
    rm -r ${outdir}
    rm -r data-bin/${outdir}
fi

mkdir -p ${outdir}

# generate parallel corpus
echo "Prepare parallel corpus ..."
python3 scripts/clean_target_corpus.py ${input} ${outdir}/corpus.${tgt} ${config}
python3 scripts/generate_source_corpus.py ${outdir}/corpus.${tgt} ${outdir}/corpus.${src} ${config}

# cinduct ngram test
if ! [[ -z ${ngram_test} ]]; then
    echo "Conduct ngram test ..."
    python3 scripts/ngram_test.py ${outdir}/corpus.${src} ${ngram_test} ngram_report.txt
    if [[ -e ngram_report.txt ]]; then
        echo "WARNING: The ngram test has found some uncovered ngrams. Training script is stopped."
        exit 1;
    else
        echo "Ngram test was successful. Continuing with data preparation ..."
    fi
fi

# apply encoding
python3 scripts/prepare_encoding.py ${outdir}/corpus.${src} ${outdir}/corpus.${tgt} ${outdir}/corpus.enc.${src} ${outdir}/corpus.enc.${tgt}

# split data into train/valid/test sets
echo "Split training data into train/valid/test sets ..."
python3 scripts/split_training_data.py ${outdir}/corpus.enc.${src} ${outdir}/corpus.enc.${tgt} ${outdir} ${src} ${tgt}
echo "Done."

# binarize training data
fairseq-preprocess --joined-dictionary \
                   --source-lang ${src} \
                   --target-lang ${tgt} \
                   --trainpref ${outdir}/train \
                   --validpref ${outdir}/valid \
                   --testpref ${outdir}/test \
                   --destdir data-bin/${outdir} \
                   --thresholdtgt 0 \
                   --thresholdsrc 0 \
                   --workers 25

# train model
fairseq-train data-bin/${outdir} \
          --source-lang ${src} \
          --target-lang ${tgt} \
          --arch ${arch} \
          --share-all-embeddings \
          --dropout 0.3 \
          --weight-decay 0.0 \
          --criterion label_smoothed_cross_entropy \
          --label-smoothing 0.1 \
          --optimizer adam \
          --adam-betas '(0.9, 0.98)' \
          --clip-norm 0.0 \
          --lr 0.001 \
          --lr-scheduler inverse_sqrt \
          --warmup-updates 4000 \
          --max-tokens 4096 \
          --update-freq 16 \
          --max-update 100000 \
          --max-epoch 10 \
          --save-dir checkpoints/${outdir} \
          --skip-invalid-size-inputs-valid-test
