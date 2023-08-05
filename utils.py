
import pandas as pd
import librosa 
import torch 

import torch
import soundfile as sf
import skimage.measure
import numpy as np

import tqdm

def get_sig(filename, sample_rate=44100):
    y, sr = librosa.load(filename,sr=sample_rate, offset=0, duration=30)
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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, sample_rate=44100):
        'Initialization'

        self.df = df
        self.sample_rate = sample_rate
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)
    print('reversing dataset')
    def __getitem__(self, index):
        try:
          'Generates one sample of data'
          file_name = self.df.path[index]
          name = self.df.name[index]
          spot = self.df.spot[index]
          sig = get_sig(file_name, self.sample_rate)
          # S = createHPSS(sig)
          #torch.from_numpy(S).float()

          return spot,name,sig, self.df.time_of_the_day[index]
        except:
           import traceback
           print(traceback.format_exc())
           return torch.zeros(1),torch.zeros(1),torch.zeros(1)


test_df = pd.read_csv("/content/drive/MyDrive/Acoustic-Scene-Classification-and-Time-of-Day-Estimation/TestSet/labels.csv")
train_df = pd.read_csv("/content/drive/MyDrive/Acoustic-Scene-Classification-and-Time-of-Day-Estimation/DataSet/labels.csv")
train_df['name'] = train_df['filename'].apply(lambda x:x.split('.')[0])
test_df['name'] = test_df['filename'].apply(lambda x:x.split('.')[0])
from pathlib import Path

files = [str(x) for x in Path("/content/drive/MyDrive/birds_project/bird identification2_old").rglob('*.WAV')]
files.reverse()


df_lst = [(x,x.split('/')[-3].split(' ')[1],x.split('/')[-3].split(' ')[1]+'_'+x.split('/')[-1].split('.')[0]) for x in files]

ignore_dir = "/content/drive/MyDrive/Acoustic-Scene-Classification-and-Time-of-Day-Estimation/embeds2"
ignore_lst = set([str(x).split('/')[-1].split('.')[0] for x in Path(ignore_dir).rglob('*')])

print(f'{len(ignore_lst)} in ignore list filtering....')
filter_counter = 0
filtered_df_lst = []
for d in tqdm.tqdm(df_lst):
  if d[-1] in ignore_lst:
    filter_counter+=1
    continue
  else:
    filtered_df_lst.append(d)

print(f'\nfiltered {filter_counter}')

files_df = pd.DataFrame(filtered_df_lst,columns=['path','spot','name'])
files_df.spot = files_df.spot.astype('int64')
merged_train = train_df.merge(files_df,'inner',left_on=['name','label'],right_on=['name','spot'])
merged_test = test_df.merge(files_df,'inner',left_on=['name','label'],right_on=['name','spot'])
def filename_to_time(file_name) :  
    timestam_str = file_name.split("_", 1)[-1]
    timestam_str = (timestam_str.split(".", 1)[0]).split(" ",1)[0]
    timestamp = int(timestam_str)/1000 # quantization of 30-minutes
    return timestamp
files_df['time_of_the_day'] = files_df.name.apply(lambda x: filename_to_time(x)) 
s = set(merged_train.name.values )
files_df['in_train'] = files_df['name'].apply(lambda x: x in s )
s = set(merged_test.name.values )
files_df['in_test'] = files_df['name'].apply(lambda x: x in s )
