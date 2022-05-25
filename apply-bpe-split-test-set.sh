
src=en
tgt=vi
BPE_CODE=$prep/code
prep=iwslt15.tokenized.en-vi
tmp=$prep/tmp
for L in $src $tgt; do
    for f in test.00-10.$L test.11-20.$L test.21-30.$L test.31-40.$L test.41-.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done