conda activate huggingface
#####reduce feat.scp, in order to have the same number utt, because wav2vec extract feat from source wav, not remove too long or too short utt
#this should do in project dir, such as st_espnet, to using data2json.sh
src_dir=/home/zhhao/pytorch-lightning/data/MUST-C/en-de/st/data
wav2vec_dir=/home/zhhao/pytorch-lightning/data/MUST-C/en-de/wav2vec
split_wav2vec=train #train dev tst-COMMON train_sp

awk -F ' ' '{if(ARGIND==1){val[$1]} else{if($1 in val) {print $0}}}' $wav2vec_dir/data/$split_wav2vec/feats.scp $wav2vec_dir/data/$split_wav2vec/wav.scp | sort > $wav2vec_dir/data/$split_wav2vec/wav_reduce.scp

#data2json and add source text
bpemodel=$src_dir/lang_1spm/train_sp.en-de.de_bpe8000_tc
dict=$src_dir/lang_1spm/train_sp.en-de.de_bpe8000_units_tc.txt

###for wav2vec
data2json.sh --nj 16 --feat $wav2vec_dir/data/$split_wav2vec/feats.scp --wavpath $wav2vec_dir/data/$split_wav2vec/ --text $wav2vec_dir/data/$split_wav2vec/text.tc --bpecode ${bpemodel}.model --lang de \
        $wav2vec_dir/data/$split_wav2vec ${dict} > $wav2vec_dir/data/$split_wav2vec/data_bpe8000.lc.rm_tc_w2v.json 

update_json.sh --text $wav2vec_dir/data/$split_wav2vec/text.lc.rm --bpecode ${bpemodel}.model \
            $wav2vec_dir/data/$split_wav2vec/data_bpe8000.lc.rm_tc_w2v.json $wav2vec_dir/data/$split_wav2vec ${dict}      
