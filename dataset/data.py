import torch
from torch.utils.data import Dataset
import soundfile
import numpy as np
import random
import pandas as pd
import glob
import librosa
from utils.utils import mixing
from avse4_utils import *
from .int_partition import random_occlude
        
class inf_dataset(Dataset):
    def __init__(self, mix_scp, lip_scp, expr_scp, fs):
        self.mix = {x.split()[0]:x.split()[1] for x in open(mix_scp)}
        self.lip = {x.split()[0]:x.split()[1] for x in open(lip_scp)}
        self.expr = {x.split()[0]:x.split()[1] for x in open(expr_scp)}
       
        mix_id = []
       
        for l in open(mix_scp):
            mix_id.append(l.split()[0])
        
        self.mix_id = mix_id
        self.fs = fs
        self.len = len(self.mix)
        print(self.len)
    
    def _trun_wav(self, y, tlen, offset=0):
        if y.shape[-1] < tlen:
            npad = tlen - y.shape[-1]
            y = np.pad(y, [(0, 0)]*(y.ndim-1)+[(0, npad)], mode='constant', constant_values=0)
        else:
            y = y[..., offset:offset+tlen]
        return y 
   
    
    def __getitem__(self, sample_idx):
        if isinstance(sample_idx, int):
            index, tlen = sample_idx, None
        elif len(sample_idx) == 2:
            index, tlen = sample_idx
        else:
            raise AssertionError
        mix_utt = self.mix_id[index]
       
        lip_utt = mix_utt
        expr_utt = mix_utt
        
        mix_wav_path = self.mix[mix_utt]
        lip_path = self.lip[lip_utt]
        expr_path = self.expr[expr_utt]
        
        mix_wav, _ = librosa.load(mix_wav_path, sr=self.fs, mono=False) # [T]
        ilen = mix_wav.shape[-1]

        lip_emb = np.load(lip_path)["data"].squeeze(0)
        expr_emb = np.load(expr_path)["data"]

        switch = True
        while ilen > len(lip_emb) * 640:
            if switch:
                lip_emb = np.insert(lip_emb, -1, lip_emb[-1,...], axis=0)
                expr_emb = np.insert(expr_emb, -1, expr_emb[-1,...], axis=0)
                switch = False
            else:
                lip_emb = np.insert(lip_emb, 0, lip_emb[0,...], axis=0)
                expr_emb = np.insert(expr_emb, 0, expr_emb[0,...], axis=0)
                switch = True
        mix_wav = self._trun_wav(mix_wav, len(lip_emb) * 640)
        
        assert mix_wav.shape[-1] == len(lip_emb) * 640, '{} mix {}, frame len {}'.format(mix_utt, mix_wav.shape, len(lip_emb))
        
        mix_wav = torch.from_numpy(mix_wav) # [M, L] or [L]
        ilen = np.array([ilen])
        ilen = torch.from_numpy(ilen)
        lip_emb = torch.from_numpy(lip_emb)
        expr_emb = torch.from_numpy(expr_emb)
        return mix_utt, mix_wav,  ilen, lip_emb, expr_emb
    
    def __len__(self):
        return self.len