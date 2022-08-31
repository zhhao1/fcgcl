#!/bin/bash


. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

#data prep is done using espnet script

#train
device=${3}
nj=32
ngpu=1 #decode, 1 or 0, multi gpu is not supported
version=${1}
src_case=lc.rm
tgt_case=tc
tgt_lang=de
criterion=acc
average_number=5

decode_config=conf/decode.yaml
preprocess_conf=
slurm_conf=conf/slurm.conf

name=MUST-C/${2} #libri_trans or MUST-C/en-de
data_name=MUST-C/en-de
bpemode=bpe
nbpe=8000
dict=../data/MUST-C/en-de/lang_1spm/train_sp.en-de.de_bpe8000_units_tc.txt
#trans_set="test.fr dev.fr"  #for libri_trans dataset
#trans_set="tst-COMMON tst-HE" #for MUST-C/en-de dataset
trans_set="tst-COMMON"

remove_nonverbal=true
hparam=logs/${name}/version_${version}/hparams.yaml
bpemodel=../data/MUST-C/en-de/lang_1spm/train_sp.en-de.de_bpe8000_tc
#decode
python local/average_checkpoints.py --model logs/${name}/version_${version}/checkpoints/epoch=* \
                                    --out logs/${name}/version_${version}/checkpoints/average_top_${average_number}_${criterion}.ckpt \
                                    --num ${average_number}

average_model=average_top_${average_number}_${criterion}.ckpt
best_model=epoch=00-val_acc_epoch=0.6175.ckpt
#if false;then
for ttask in ${trans_set};do
    mkdir -p logs/${name}/version_${version}/decode/${ttask}
    mkdir -p logs/${name}/version_${version}/results/${ttask}
    python local/splitjson.py --parts ${nj} ../data/${data_name}/wav2vec/data/tst-COMMON/data_bpe8000.lc.rm_tc.json
    for type in average;do
        mkdir -p logs/${name}/version_${version}/decode/${ttask}/${type} 
        mkdir -p logs/${name}/version_${version}/results/${ttask}/${type}
        if [ "${type}" = "average" ];then
            model=${average_model}
        else
            model=${best_model}
        fi
        echo "in version_${version} using ${model} decoding ${ttask}"

        ${decode_cmd} --gpu ${ngpu} JOB=1:${nj} logs/${name}/version_${version}/decode/${ttask}/${type}/decode.JOB.log  CUDA_VISIBLE_DEVICES=${device} python local/st_trans.py \
                    --config ${decode_config} \
                    --ngpu ${ngpu} \
                    --hparam ${hparam} \
                    --trans-json ../data/${data_name}/wav2vec/data/tst-COMMON/split${nj}utt/data_bpe8000.JOB.json \
                    --result-label logs/${name}/version_${version}/results/${ttask}/${type}/data.JOB.json \
                    --model logs/${name}/version_${version}/checkpoints/${model} \
                    --prefix 'model' \
                    --task 'ST' \
                    --initilize-module 'encoder' 'speech_embed' 'decoder'
    done
done
#fi
##bleu score
for ttask in ${trans_set};do
    for type in average best;do
        local/score_bleu.sh --case tc --bpe ${nbpe} --bpemodel ${bpemodel}.model \
                       --remove_nonverbal ${remove_nonverbal} \
                       logs/${name}/version_${version}/results/${ttask}/${type} ${tgt_lang} ${dict}
    done
done

