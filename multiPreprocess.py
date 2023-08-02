
import torch
import librosa
import soundfile as sf
import skimage.measure
import numpy as np

from torch import nn
import torch.nn as nn
import torch
import torch.nn.functional as F

import pandas as pd

import tqdm

import soundfile as sf

import multiprocessing
import concurrent.futures

import os

def get_sig(filename):
    y, sr = librosa.load(filename,sr=44100, offset=0, duration=30)
    return y
    
def createMonoSpectrogram(filename, n_mels=128, n_fft=2048, hop_length=512):
  sig, sr = sf.read(filename)
  sig = sig[:,0]
  S = librosa.feature.melspectrogram(y=sig, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
  return S

def createMonoSpectrogram(sig, sr, n_mels=128, n_fft=2048, hop_length=512):
  sig = sig[:,0]

  S = librosa.feature.melspectrogram(y=sig, sr=sr, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
  return S

def createHPSS(y, n_fft=2544, hop_length=520):

  D = librosa.stft(y,n_fft=2544,hop_length=520)
  D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=10)
  D_harmonic_downsampled = skimage.measure.block_reduce(D_harmonic, (5,10), func=np.max, cval=0)
  D_percussive_downsampled = skimage.measure.block_reduce(D_percussive, (5,10), func=np.max, cval=0)
  S = np.array([D_harmonic_downsampled,D_percussive_downsampled])
  return S

test_df = pd.read_csv(r"D:\git\Acoustic-Scene-Classification-and-Time-of-Day-Estimation\TestSet\labels.csv")
train_df = pd.read_csv(r"D:\git\Acoustic-Scene-Classification-and-Time-of-Day-Estimation\DataSet\labels.csv")
train_df['name'] = train_df['filename'].apply(lambda x:x.split('.')[0])
test_df['name'] = test_df['filename'].apply(lambda x:x.split('.')[0])
from pathlib import Path

files = [str(x) for x in Path(r'F:\bird identification2\Quarantine recordings Tel Aviv (Yoel)').rglob('*.wav')]

df_lst = [(x,x.split('\\')[-3].split(' ')[1],x.split('\\')[-1].split('.')[0]) for x in files]

files_df = pd.DataFrame(df_lst,columns=['path','spot','name'])
files_df.spot = files_df.spot.astype('int64')
merged_train = train_df.merge(files_df,'inner',left_on=['name','label'],right_on=['name','spot'])
merged_test = test_df.merge(files_df,'inner',left_on=['name','label'],right_on=['name','spot'])
def filename_to_time(file_name) :  
    timestam_str = file_name.split("_", 1)[1]
    timestam_str = (timestam_str.split(".", 1)[0]).split(" ",1)[0]
    timestamp = int(timestam_str)/1000 # quantization of 30-minutes
    return timestamp
files_df['time_of_the_day'] = files_df.name.apply(lambda x: filename_to_time(x)) 
s = set(merged_train.name.values )
files_df['in_train'] = files_df['name'].apply(lambda x: x in s )
s = set(merged_test.name.values )
files_df['in_test'] = files_df['name'].apply(lambda x: x in s )

def helper(index):

    try:
        file_name = files_df.path[index]
        new_path = fr'D:\Birds_audio\{files_df.spot[index]}_{files_df.name[index]}.flac'
        if os.path.exists(new_path):
           return 1
        sig = get_sig(file_name)
        # S = createHPSS(sig)
        # Im_hpss = torch.from_numpy(S).float()

        # torch.save(Im_hpss, fr'D:\Birds_audio\{files_df.spot[index]}_{files_df.name[index]}.hpss')
        # pred_times.append(hpss_model(hpss.cuda().unsqueeze(0), timestamp=None).item())
        # dy = librosa.resample(sig,orig_sr=44100,target_sr=16000).astype('float32')
        sf.write(new_path, sig.astype('float32'),44100)

        return 1
    except:
        import traceback
        print(traceback.format_exc())
        return 0
    

if __name__ == '__main__':
    p = concurrent.futures.ProcessPoolExecutor(4)
    np.seterr(all='ignore')
    res = list(tqdm.tqdm(p.map(helper,range(len(files_df))),total=len(files_df)))