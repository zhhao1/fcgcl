import os, shutil, sys
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL)

data_path = '/home/zhhao/MUST-C/en-de/data'
file_path = '/home/zhhao/pytorch-lightning/data/MUST-C/en-de/st/data'
splits = ['dev.en-de', 'train.en-de', 'train_sp.en-de', 'tst-COMMON.en-de', 'tst-HE.en-de']
dst_dir = '/home/zhhao/pytorch-lightning/data/MUST-C/en-de/wav2vec'

for i in splits:
    os.makedirs(os.path.join(dst_dir, 'wav_segment', i.split('.')[0]), exist_ok=True)

split = splits[4]
data_split = splits[4].split('.')[0]

with open(os.path.join(file_path, split, 'wav.scp')) as wavs:
    wavs = wavs.readlines()
    wav_dict = {}
    for wav in wavs:
        key = wav.split()[0]
        value = " ".join(wav.split()[1:])
        wav_dict[key] = value

with open(os.path.join(file_path, split, 'segments')) as segments:
    segments = segments.readlines()
    segments = [i.split() for i in segments]  
    for segment in segments:
        target = os.path.join(dst_dir, 'wav_segment', data_split, segment[0])
        cmd1 = wav_dict[segment[1]]
        cmd2 = " sox - " + target + '.wav trim ' + segment[2] + ' =' + segment[3]
        cmd = cmd1 + cmd2
        os.system(cmd1+cmd2)
        
