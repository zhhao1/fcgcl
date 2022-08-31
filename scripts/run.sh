conda activate huggingface
# train
. ./wav2vec.sh 0 1000 1 1000 1 train # 0:for scale JOB number, 1000:total scp number, 1:JOB start, 1000:JOB end, 1:i, train:split 
. ./wav2vec.sh 1 2000 1 1000 1 train # 1001 2000
. ./wav2vec.sh 2 2297 1 297 1 train # 

nj=2297
split=train
for n in $(seq $nj); do
  cat ../feat/$split/feats.$n.scp
done > ../feat/$split/feats.scp
"""
# dev
. ./wav2vec.sh 0 15 1 15 0 dev

# tst-COMMON
. ./wav2vec.sh 0 27 1 27 3 tst-COMMON

# tst-HE
. ./wav2vec.sh 0 6 1 6 4 tst-HE

# train_sp
. ./wav2vec.sh 0 1000 1 1000 2 train_sp
. ./wav2vec.sh 1 2000 1 1000 2 train_sp
. ./wav2vec.sh 2 3000 1 1000 2 train_sp
. ./wav2vec.sh 3 4000 1 1000 2 train_sp
echo "ok"
. ./wav2vec.sh 4 5000 1 1000 2 train_sp
. ./wav2vec.sh 5 6000 1 1000 2 train_sp
. ./wav2vec.sh 6 6891 1 891 2 train_sp
nj=6891
split=train_sp
for n in $(seq $nj); do
  cat ../feat/$split/feats.$n.scp
done > ../feat/$split/feats.scp
"""

#####reduce feat.scp, in order to have the same number utt, because wav2vec extract feat from source wav, not remove too long or too short utt
#this should do in project dir, such as st_espnet, to using data2json.sh
src_dir=/home/zhhao/pytorch-lightning/data/MUST-C/en-de/st/data
wav2vec_dir=/home/zhhao/pytorch-lightning/data/MUST-C/en-de/wav2vec
split_src=dev.en-de
split_wav2vec=train #train dev tst-COMMON train_sp

awk -F ' ' '{if(ARGIND==1){val[$1]} else{if($1 in val) {print $0}}}' $src_dir/$split_src.de/feats.scp $wav2vec_dir/feat/$split_wav2vec/feats.scp | sort > $wav2vec_dir/data/$split_wav2vec/feats.scp
echo $(wc -l $src_dir/$split_src.de/feats.scp)  #src
echo $(wc -l $wav2vec_dir/feat/$split_wav2vec/feats.scp) #wav2vec 
echo $(wc -l $wav2vec_dir/data/$split_wav2vec/feats.scp) #reduced_wav2vec
awk -F ' ' '{if(ARGIND==1){val[$1]} else{if($1 in val) {print $0}}}' $wav2vec_dir/data/$split_wav2vec/feats.scp $src_dir/$split_src.de/utt2spk | sort > $wav2vec_dir/data/$split_wav2vec/utt2spk
awk -F ' ' '{if(ARGIND==1){val[$1]} else{if($1 in val) {print $0}}}' $wav2vec_dir/data/$split_wav2vec/feats.scp $src_dir/$split_src.de/text.tc | sort > $wav2vec_dir/data/$split_wav2vec/text.tc
awk -F ' ' '{if(ARGIND==1){val[$1]} else{if($1 in val) {print $0}}}' $wav2vec_dir/data/$split_wav2vec/feats.scp $src_dir/$split_src.en/text.lc.rm | sort > $wav2vec_dir/data/$split_wav2vec/text.lc.rm


#data2json and add source text
bpemodel=$src_dir/lang_1spm/train_sp.en-de.de_bpe8000_tc
dict=$src_dir/lang_1spm/train_sp.en-de.de_bpe8000_units_tc.txt

###for wav2vec      
data2json.sh --nj 16 --feat $wav2vec_dir/data/$split_wav2vec/feats.scp --text $wav2vec_dir/data/$split_wav2vec/text.tc --bpecode ${bpemodel}.model --lang de \
        $wav2vec_dir/data/$split_wav2vec ${dict} > $wav2vec_dir/data/$split_wav2vec/data_bpe8000.lc.rm_tc.json

update_json.sh --text $wav2vec_dir/data/$split_wav2vec/text.lc.rm --bpecode ${bpemodel}.model \
            $wav2vec_dir/data/$split_wav2vec/data_bpe8000.lc.rm_tc.json $wav2vec_dir/data/$split_wav2vec ${dict}
