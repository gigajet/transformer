#!/usr/bin/sh
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt

# See Heaps'law, we take K=10 and beta=0.4, roughly rounded
BPE_TOKENS_cz=15000 
BPE_TOKENS=25000 

echo 'Downloading datasets...'
gdown https://drive.google.com/uc?id=1OndWoLIdoGPLZVeb50EW0euFm6iZhc4t
gdown https://drive.google.com/uc?id=1nDVshzeZnQ2-0JXOWq1IIVxpNzqdueHq
gdown https://drive.google.com/uc?id=1UUrRRBScglj0PgyN-A96biSDjlGdMQWo
gdown https://drive.google.com/uc?id=1k935uagEg2D32cCD0AwgYLOjNMWrYdpH
tar xzvf cs-en.tgz
tar xzvf de-en.tgz
tar xzvf fr-en.tgz
unzip -o testsets.zip

# We have datasets in the same directory as this script.
# We wanna preprocess each "cs-en", "de-en", "fr-en" dataset
# and throw them into respective folder.

src1=cs
src2=de
src3=fr
tgt=en
tmp=$prep/tmp
orig=.

for src in $src1 $src2 $src3; do
    folder=$src-$tgt
    echo "Preprocessing $folder"
    mkdir $folder
    echo "Tokenizing..."
    cat europarl-v7.$folder.$src | \
    perl $TOKENIZER -threads 8 -l $src > europarl-v7.$folder.tok.$src

    cat europarl-v7.$folder.$tgt | \
    perl $TOKENIZER -threads 8 -l $tgt > europarl-v7.$folder.tok.$tgt

    perl $CLEAN -ratio 1.8 europarl-v7.$folder.tok $src $tgt europarl-v7.$folder.clean 1 175
    for l in $src $tgt; do
        perl $LC < europarl-v7.$folder.clean.$l > europarl-v7.$folder.$l.lowercased
    done

    echo "Splitting train and valid..."
    for l in $src $tgt; do
        awk '{if (NR%23 == 0)  print $0; }' europarl-v7.$folder.$l.lowercased > $folder/valid.$l
        awk '{if (NR%23 != 0)  print $0; }' europarl-v7.$folder.$l.lowercased > $folder/train.$l
    done

    echo "Initializing test set..."
    if [ "$src" = "cs" ]; then
        cat nc-devtest2007.cz-en.cz \
            nc-dev2007.cz \
            > test-set.$folder.$src

        cat nc-devtest2007.cz-en.en \
            nc-dev2007.en \
            > test-set.$folder.$tgt
        
        cat test-set.$folder.$src | \
        perl $TOKENIZER -threads 8 -l $src > test-set.$folder.tok.$src

        cat test-set.$folder.$tgt | \
        perl $TOKENIZER -threads 8 -l $tgt > test-set.$folder.tok.$tgt

        perl $CLEAN -ratio 1.8 test-set.$folder.tok $src $tgt test-set.$folder.clean 1 175
        for l in $src $tgt; do
            perl $LC < test-set.$folder.clean.$l > $folder/test.$l
        done
    else
        for l in $src $tgt; do
            cat nc-devtest2007.$l \
                nc-dev2007.$l \
                > test-set.$folder.$l

            cat test-set.$folder.$l | \
            perl $TOKENIZER -threads 8 -l $l > test-set.$folder.tok.$l
        done

        perl $CLEAN -ratio 1.8 test-set.$folder.tok $src $tgt test-set.$folder.clean 1 175

        for l in $src $tgt; do
            perl $LC < test-set.$folder.clean.$l > $folder/test.$l
        done
    fi

    echo 'Learning bpe...'
    TRAIN=$folder/train.$folder
    BPE_CODE=$folder/bpe_code
    rm -f $TRAIN
    for l in $src $tgt; do
        cat $folder/train.$l >> $TRAIN
    done

    if [ "$src" = "cs" ]; then
        python $BPEROOT/learn_bpe.py -s $BPE_TOKENS_cz < $TRAIN > $BPE_CODE
    else
        python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE
    fi
    rm -f $TRAIN

    echo 'Applying learned bpe...'
    for L in $src $tgt; do
        for f in train.$L valid.$L test.$L; do
            echo "apply_bpe.py to ${folder}/${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE < $folder/$f > $folder/$f.bpe
            rm -f $folder/$f
            mv $folder/$f.bpe $folder/$f
        done
    done
done
