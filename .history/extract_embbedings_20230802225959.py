



from utils import files_df,Dataset
import torch
import tqdm
import pickle

import sys
sys.path.append(r'D:\git\PaSST')
from models.passt import get_model
from models.preprocess import AugmentMelSTFT

from torch.utils.data import DataLoader


if __name__ == '__main__':

    mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024,freqm=0,timem=0, #freqm=48,timem=192,
                            htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                            fmax_aug_range=2000)

    model  = get_model(arch="passt_s_swa_p16_128_ap476", pretrained=True, n_classes=527, in_channels=1,
                    fstride=10, tstride=10,input_fdim=128, input_tdim=998,
                    u_patchout=0, s_patchout_t=40, s_patchout_f=4)

    model.eval()
    model = model.cuda()

    ds = Dataset(files_df,32000)
    dl = DataLoader(ds,num_workers=8,prefetch_factor=10)

    embeds = []
    for a in tqdm.tqdm(dl,total=len(ds)):
        with torch.no_grad():
            melspec = mel(torch.tensor(a[0])).unfold(-1,998,998).permute(2,0,1,3)
            embed = model(melspec.cuda())

    with open('./embeds.pkl','wb') as f:
        pickle.dump(embed)
