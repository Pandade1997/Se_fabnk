import os
import argparse
import sys

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR

import tensorboardX
from tensorboardX import SummaryWriter

from scipy.io import wavfile
import librosa

import soundfile as sf
from pystoi.stoi import stoi
from pypesq import pesq

from tqdm import tqdm
from models.layers.istft import ISTFT
import train_utils
from load_dataset import AudioDataset
from models.attention import AttentionModel

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=8, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
parser.add_argument('--dropout_p', default=0.2, type=float, help='Attention model drop out rate')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='Learning rate')
parser.add_argument('--stacked_encoder', default=True, type=bool)
parser.add_argument('--attn_len', default=5, type=int)
parser.add_argument('--hidden_size', default=448, type=int)
parser.add_argument('--ck_dir', default='ckpt_dir', help='ck path')
parser.add_argument('--ck_name', help='ck file', default='T_dataset.pt')
parser.add_argument('--test_set', help='test', default='test')
parser.add_argument('--attn_use', default=True, type=bool)
args = parser.parse_args()

n_fft, hop_length = 512, 128
window = torch.hann_window(n_fft).cuda()
# STFT
stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
# ISTFT
istft = ISTFT(n_fft, hop_length, window='hanning').cuda()


def main():
    test_dataset = AudioDataset(data_type=args.test_set)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate,
                                  shuffle=False, num_workers=0)

    # model select
    print('Model initializing\n')
    net = torch.nn.DataParallel(
        AttentionModel(257, hidden_size=args.hidden_size, dropout_p=args.dropout_p, use_attn=args.attn_use,
                       stacked_encoder=args.stacked_encoder, attn_len=args.attn_len))
    # net = AttentionModel(257, 112, dropout_p = args.dropout_p, use_attn = args.attn_use)
    net = net.cuda()
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    # Check point load

    print('Trying Checkpoint Load\n')
    ckpt_dir = args.ck_dir
    best_PESQ = 0.
    best_STOI = 0.
    ckpt_path = os.path.join(ckpt_dir, args.ck_name)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        try:
            net.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            # best_STOI = ckpt['best_STOI']

            print('checkpoint is loaded !')
            # print('current best loss : %.4f' % best_loss)
        except RuntimeError as e:
            print('wrong checkpoint\n')
    else:
        print('checkpoint not exist!')
        # print('current best loss : %.4f' % best_loss)

    # test phase
    n = 0
    avg_test_loss = 0

    net.eval()
    with torch.no_grad():
        audio_config = {
            "frame_length": 25,
            "frame_shift": 10,
        }
        test_bar = tqdm(test_data_loader)
        for input in test_bar:
            test_mixed, test_clean, seq_len = map(lambda x: x.cuda(), input)
            lt_test_mixed_vstack = []
            lt_test_clean_vstack = []
            for i in range(len(test_mixed)):
                test_mixed_vstack = torchaudio.compliance.kaldi.fbank(test_mixed[i].unsqueeze(0), num_mel_bins=40,
                                                                      channel=-1,
                                                                      sample_frequency=16000,
                                                                      **audio_config)
                test_clean_vstack = torchaudio.compliance.kaldi.fbank(test_clean[i].unsqueeze(0), num_mel_bins=40,
                                                                      channel=-1,
                                                                      sample_frequency=16000,
                                                                      **audio_config)

                test_mixed_vstack_data = test_mixed_vstack.transpose(0, 1).unsqueeze(0).detach()
                test_clean_vstack_data = test_clean_vstack.transpose(0, 1).unsqueeze(0).detach()

                lt_test_mixed_vstack.append(test_mixed_vstack_data)
                lt_test_clean_vstack.append(test_clean_vstack_data)

            lt_test_mixed_feat = torch.cat(lt_test_mixed_vstack, dim=0).transpose(1, 2)
            lt_test_clean_feat = torch.cat(lt_test_clean_vstack, dim=0).transpose(1, 2)

            out_lt_test_mixed_feat, attn_weight = net(lt_test_mixed_feat)

            test_loss = F.mse_loss(out_lt_test_mixed_feat, lt_test_clean_feat, True)

            for i in range(len(test_mixed)):
                librosa.output.write_wav('test_out.wav',
                                         logits_audio[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()], 16000)

            avg_test_loss += test_loss
            n += 1


        avg_test_loss /= n
        print(
            'test loss : {:.4f} '.format(avg_test_loss, ))


if __name__ == '__main__':
    main()
