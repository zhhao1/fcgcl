import re, sys, os
import pickle
import torch
import torchaudio
from kaldiio import WriteHelper
from datasets import Dataset, load_metric
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import pandas as pd

dst_dir = '/home/zhhao/pytorch-lightning/data/MUST-C/en-de/wav2vec'
file_path = '/home/zhhao/pytorch-lightning/data/MUST-C/en-de/st/data'
splits = ['dev.en-de', 'train.en-de', 'train_sp.en-de', 'tst-COMMON.en-de', 'tst-HE.en-de']

for i in splits:
    os.makedirs(os.path.join(dst_dir, 'feat', i.split('.')[0]), exist_ok=True)

split = splits[int(sys.argv[1])]
data_split = splits[int(sys.argv[1])].split('.')[0] 
   
with open(os.path.join(file_path, split, 'segments')) as segments:
    segments = segments.readlines()
    segments = [i.split() for i in segments] 
    paths = [os.path.join(dst_dir, 'wav_segment', data_split, i[0])+'.wav' for i in segments] 

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, 
                                             padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)
                                                                                         
model = Wav2Vec2Model.from_pretrained('/home/zhhao/wav2vec-large/960', output_hidden_states=True)
model.to("cuda")
outputs = []
wav_names = []

start = ( int(sys.argv[2])-1 + 1000*int(sys.argv[4])) * int(sys.argv[3])
#end = (int(sys.argv[2])+ 1000*int(sys.argv[4])) * int(sys.argv[3])
end = start + int(sys.argv[3])
if end > len(segments):
    end = len(segments)

with torch.no_grad():
    for i in range(start, end):
        speech_array, sampling_rate = torchaudio.load(paths[i])
        speech_array = speech_array.squeeze().numpy()
        inputs = feature_extractor(speech_array, sampling_rate=16_000, return_tensors="pt")
        output = model(inputs.input_values.to("cuda")).hidden_states[-1].squeeze(0).cpu().detach().numpy()
        outputs.append(output)
        wav_names.append(segments[i][0])
        #writer(segments[i][0], output)

tgt = os.path.join(dst_dir, 'feat', data_split)
save_number = str(int(sys.argv[2]) + 1000*int(sys.argv[4]))
to_write = 'ark,scp:' + tgt + '/feats.' + save_number + '.ark,' + tgt + '/feats.' + save_number + '.scp'

with WriteHelper(to_write) as writer:
    for i in range(len(outputs)):
        writer(wav_names[i], outputs[i])        

"""
temp = kaldiio.load_mat('file.ark:26')
print(temp.shape)
"""
