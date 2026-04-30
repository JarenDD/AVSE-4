import os
import torch
import numpy as np
import argparse
import yaml
import soundfile as sf
from collections import OrderedDict
from tqdm import tqdm
import librosa
import sys
from utils.utils import get_instance
import models.tfgridnet_separator as module_model
from utils.loss.pit_criterion import *
from models.skipConvNet.dataprep import spectralImages_1D_joint
from librosa.core import stft, istft
from torch.nn import functional as F

import dataset
import dataloader

def load_pretrained_modules(model, ckpt_path):
    model_info = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in model_info['model_state_dict'].items():
        # print(k)
        name = k.replace("module.", "")    # remove 'module.'
        state_dict[name] = v
    model.load_state_dict(state_dict)
    del model_info

    return model

def main(config):
    inferset = get_instance(dataset, config['inferset'])
    inferloader = get_instance(dataloader, config['inferloader'], inferset)

    model = module_model.AVGridNetJoint(**config['AVGridNetJoint_kwargs'])
    model = load_pretrained_modules(model, '/public/home/qinxy/jarend/avse4_joint/ckpt/joint/temp_best.pth.tar')
    model.cuda()
    model.eval()

    dest_path_est = ''
   
    os.makedirs(dest_path_est, exist_ok=True)

    count = torch.tensor(0.).cuda() 
     # 创建一个文本文件用于保存输出
    log_file = open('console_output.log', 'w')
    sys.stdout = log_file

    with torch.no_grad():
        prog_bar = tqdm(enumerate(inferloader))
        for i, (mix_utt, mix_wav, ilens, lip_emb, expr_emb) in prog_bar:
            utt_id = mix_utt[0]
            output_path = os.path.join(dest_path_est, f'{utt_id}.wav')

            # 检查输出文件是否存在
            if os.path.exists(output_path):
                print(f"跳过已存在的文件: {utt_id}")
                log_file.flush()
                continue

            mix_wav = mix_wav.cuda()
            ilens = ilens.squeeze(-1).cuda()
            lip_emb = lip_emb.cuda()
            expr_emb = expr_emb.cuda()
            
            print("mix_utt",mix_utt[0])
            print('mix_wav shape:',mix_wav.shape )   #[B,C,T] [1,2,T] 
            print('ilen shape',ilens.shape) #[1]
            #print('tar_wav shape:',tar_wav.shape )
            print('lip_emb shape:',lip_emb.shape )   #[B,T2,F2] [1,T2,512]
            print('expr_emb shape:',expr_emb.shape ) #[B,T3,F3] [1,T3,2048]
         
            log_file.flush()
            
            est_list = model(mix_wav.transpose(1, -1), ilens, lip_emb,expr_emb)[0] # [B, C, T]

            #去混响处理
            output = est_list[-1].squeeze()
            print('output',output.shape) #[T]
            output_np = output.detach().cpu().numpy()
            audioName = ""
            PSD_frames = spectralImages_1D_joint(audioName = audioName, audio=output_np)
            nframes = len([key for key in PSD_frames if 'Phase' in key])
            audio = {}
            noisy_mag = []; noisy_phase = []; noisy_norm = [];
            clean_mag = []; clean_phase = []; clean_norm = [];

            for k in range(nframes):
                uttname = 'MagdB_'+audioName+'_frame_'+str(k)
                noisy_mag.append(PSD_frames[uttname])
                noisy_phase.append(PSD_frames[uttname.replace('MagdB', 'Phase')])

            noisy_norm = PSD_frames[uttname.replace('MagdB', 'Norm').split('_frame')[0]]
            samples    = PSD_frames[uttname.replace('MagdB', 'Samples').split('_frame')[0]]
            
            audio['noisy_mag']   = torch.from_numpy(np.expand_dims(noisy_mag, axis=1))
            audio['noisy_phase'] = np.hstack(noisy_phase)
            audio['noisy_norm']  = noisy_norm
            audio['utt_samples'] = int(samples)
            audio['uttname']     = audioName

            input_mag     = audio['noisy_mag'].float().cuda()
            print('input_mag',input_mag.shape) #[2, 1, 256, 256]
            enhanced_mag  = model.dereverb(input_mag)
            print('enhanced_mag',enhanced_mag.shape) #[2, 1, 256, 256]
            enhanced_mag = enhanced_mag.detach().cpu().numpy()
            if enhanced_mag.shape[0]>1:
                enhanced_mag  = np.hstack(np.squeeze(enhanced_mag))
            else:
                enhanced_mag  = np.squeeze(enhanced_mag)
            noisy_mag = np.hstack(np.squeeze(audio['noisy_mag'].numpy()))
            noisy_mag = np.interp(noisy_mag, [-1,1], audio['noisy_norm'])
            enhanced_mag = np.interp(enhanced_mag, [-1,1],audio['noisy_norm'])

            temp = np.zeros((257, enhanced_mag.shape[1])) + 1j*np.zeros((257, enhanced_mag.shape[1]))
            temp[:-1,:] = 10**(enhanced_mag/20) * (np.cos(audio['noisy_phase']) + np.sin(audio['noisy_phase'])*1j)
            enhanced_audio = istft(temp)
            enhanced_audio = 0.98*enhanced_audio/np.max(np.abs(enhanced_audio))
            enhanced_audio = enhanced_audio[:audio['utt_samples']]
            print('enhanced_audio',enhanced_audio.shape) #[T]
            sf.write(os.path.join(dest_path_est, '{}.wav'.format(mix_utt[0])), enhanced_audio,   16000)    

            count += 1
            
        print('总条数:',count)
        log_file.flush() 
        log_file.close()
      
        sys.stdout = sys.__stdout__
        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speech Separation')
    parser.add_argument('--config', default='config/std_train.yml', type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()
   
    assert os.path.isfile(args.config), "No such file: %s" % args.config
    with open(args.config) as rfile:
        config = yaml.safe_load(rfile)

    main(config)