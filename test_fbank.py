import os
import argparse
import sys

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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
parser.add_argument('--hidden_size', default=112, type=int)
parser.add_argument('--ck_dir', default='ckpt_dir', help='ck path')
parser.add_argument('--ck_name', help='ck file', default='se_fbank.pt')
parser.add_argument('--test_set', help='test', default='test')
parser.add_argument('--attn_use', default=True, type=bool)
parser.add_argument('--out_path',
                    default='/data01/AuFast/origin_dataset/dataset/LibriSpeech/test_dataset/gen/gen_test/',
                    type=str)
args = parser.parse_args()

num_fbank = 40
sr1 = 16000
window_size = 25  # int, window size for FFT (ms)
stride = 10
ws = int(sr1 * 0.001 * window_size)
st = int(sr1 * 0.001 * stride)


def main():
    test_dataset = AudioDataset(data_type=args.test_set)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate,
                                  shuffle=False, num_workers=0)

    # model select
    print('Model initializing\n')
    net = torch.nn.DataParallel(
        AttentionModel(40, hidden_size=args.hidden_size, dropout_p=args.dropout_p, use_attn=args.attn_use,
                       stacked_encoder=args.stacked_encoder, attn_len=args.attn_len))
    # net = AttentionModel(257, 112, dropout_p = args.dropout_p, use_attn = args.attn_use)
    net = net.cuda()
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    # Check point load

    print('Trying Checkpoint Load\n')
    ckpt_dir = args.ck_dir
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
    name_sum = 0
    net.eval()
    with torch.no_grad():
        test_bar = tqdm(test_data_loader)
        for input in test_bar:
            test_mixed, test_clean, seq_len = map(lambda x: x.cuda(), input)

            test_mixed_feat = test_mixed.resize(len(test_mixed), int(len(test_mixed[0]) / num_fbank),
                                                num_fbank).to(
                device='cuda')

            # feed data
            out_test_mixed_feat, attn_weight = net(test_mixed_feat)

            for i in range(len(out_test_mixed_feat)):
                feat = librosa.feature.inverse.mel_to_audio(
                    M=out_test_mixed_feat[i].cuda().data.cpu().numpy().transpose(1, 0), sr=16000, n_fft=ws,
                    hop_length=st)
                name = str(test_dataset.file_names[name_sum]).split('/', 9)[9]
                librosa.output.write_wav(args.out_path + name, feat, 16000)
                name_sum += 1


if __name__ == '__main__':
    main()
