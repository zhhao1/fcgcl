temp = 'tst-HE'
dataset = 'MUST-C/en-de'
split = '/st/data/' + temp + '.en-de.de'
wav2vec_path = '/home/zhhao/pytorch-lightning/data/' + dataset + '/wav2vec/wav_segment/' + temp

with open('/home/zhhao/pytorch-lightning/data/' + dataset + '/wav2vec/data/' + temp + '/wav.scp', 'w') as f, open('/home/zhhao/pytorch-lightning/data/' + dataset + split + '/segments', 'r') as s:
    segments = s.readlines()
    wavnames = [i.split()[0] for i in segments]
    wav_path = [wavname + ' ' + wav2vec_path + '/' + wavname + '.wav' + '\n' for wavname in wavnames] 
    f.writelines(wav_path)  
    
