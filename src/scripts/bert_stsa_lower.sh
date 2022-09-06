#!/usr/bin/env bash

SRC=/home/jungmin/readability/text_augmentation/TransformersDataAugmentation/src
CACHE=/home/jungmin/readability/text_augmentation/TransformersDataAugmentation/CACHE

TASK=stsa

for NUMEXAMPLES in 10;
do
    for i in {0..14};
        do
        RAWDATADIR=$SRC/utils/datasets/${TASK}/exp_${i}_${NUMEXAMPLES}

       # Baseline classifier
        python $SRC/bert_aug/bert_classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i} --cache $CACHE > $RAWDATADIR/bert_baseline.log

   # #######################
   # # CBERT Classifier
   # #######################

    CBERTDIR=$RAWDATADIR/cbert
    mkdir $CBERTDIR
    python $SRC/bert_aug/cbert.py --data_dir $RAWDATADIR --output_dir $CBERTDIR --task_name $TASK  --num_train_epochs 10 --seed ${i}  --cache $CACHE > $RAWDATADIR/cbert.log
    cat $RAWDATADIR/train.tsv $CBERTDIR/cbert_aug.tsv > $CBERTDIR/train.tsv
    cp $RAWDATADIR/test.tsv $CBERTDIR/test.tsv
    cp $RAWDATADIR/dev.tsv $CBERTDIR/dev.tsv
    python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $CBERTDIR --seed ${i} --cache $CACHE > $RAWDATADIR/bert_cbert.log

   # #######################
   # # CMODBERT Classifier
   # ######################

    CMODBERTDIR=$RAWDATADIR/cmodbert
    mkdir $CMODBERTDIR
    python $SRC/bert_aug/cmodbert.py --data_dir $RAWDATADIR --output_dir $CMODBERTDIR --task_name $TASK  --num_train_epochs 150 --learning_rate 0.00015 --seed ${i} --cache $CACHE > $RAWDATADIR/cmodbert.log
    cat $RAWDATADIR/train.tsv $CMODBERTDIR/cmodbert_aug.tsv > $CMODBERTDIR/train.tsv
    cp $RAWDATADIR/test.tsv $CMODBERTDIR/test.tsv
    cp $RAWDATADIR/dev.tsv $CMODBERTDIR/dev.tsv
    python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $CMODBERTDIR --seed ${i} --cache $CACHE > $RAWDATADIR/bert_cmodbert.log

   # #######################
   # # CMODBERTP Classifier
   # ######################

    CMODBERTPDIR=$RAWDATADIR/cmodbertp
    mkdir $CMODBERTPDIR
    python $SRC/bert_aug/cmodbertp.py --data_dir $RAWDATADIR --output_dir $CMODBERTPDIR --task_name $TASK  --num_train_epochs 10 --seed ${i} --cache $CACHE > $RAWDATADIR/cmodbertp.log
    cat $RAWDATADIR/train.tsv $CMODBERTPDIR/cmodbertp_aug.tsv > $CMODBERTPDIR/train.tsv
    cp $RAWDATADIR/test.tsv $CMODBERTPDIR/test.tsv
    cp $RAWDATADIR/dev.tsv $CMODBERTPDIR/dev.tsv
    python $SRC/bert_aug/bert_classifier.py --task $TASK --data_dir $CMODBERTPDIR --seed ${i}  --cache $CACHE > $RAWDATADIR/bert_cmodbertp.log

    done
done


