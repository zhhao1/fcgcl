#0 dev:1423 
#1 train:229696 
#2 train_sp:689088 
#3 tst-COMMON:2641 
#4 tst-HE:600
conda activate huggingface
ngpu=1
i=$5
split=$6
k=$1
steps=100
slurm_conf=./slurm.conf
./slurm.pl --config ${slurm_conf} --gpu ${ngpu} JOB=$3:$4  ../logs/$split/feature_extract_$k.JOB.log \
python ./wav2vec.py $i JOB $steps $k 

