# -*- coding: utf-8 -*-

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.optim.lr_scheduler import ExponentialLR
from scipy.io import wavfile
import librosa
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import librosa
import csv
import torch
from torch.utils import data
import subprocess

# dataset_path = os.path.join(gdrive_root, 'dataset')
# dataset_path = "/data01/AuFast/Pan_dataset/Exp3_SE/data"
# dataset_path = "/home/pjh/dataset/T_dataset"
# dataset_path = "/home/panjiahui/dataset/SE_asr/test1/se_fbank_dataset"
# dataset_path = "/data01/AuFast/Pan_dataset/Exp3_SE/T_dataset"
# dataset_path = "/data01/AuFast/Pan_dataset/Exp3_SE/data3/SNR-5"
# dataset_path = "/data01/AuFast/Pan_dataset/Exp3_SE/data3/SNR0"
# dataset_path = "/data01/AuFast/Pan_dataset/Exp3_SE/data3/SNR5"
dataset_path = "/data01/AuFast/Pan_dataset/SE_asr/test1/se_fbank_dataset"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

train_tar_path = os.path.join(dataset_path, "train")
valid_tar_path = os.path.join(dataset_path, "valid")
test1_tar_path = os.path.join(dataset_path, "test")


# DATA LOADING - LOAD FILE LISTS
def load_data_list(folder=dataset_path, setname='train'):
    assert (setname in ['train', 'valid', 'test', 'test2', 'test3', 'test4'])

    dataset = {}

    if "test" in setname:
        clean_foldername = folder + '/testset'
    else:
        clean_foldername = folder + '/' + setname + "set"
    noisy_foldername = folder + '/' + setname + "set"

    print("Loading files...")
    dataset['innames'] = []
    dataset['outnames'] = []
    dataset['shortnames'] = []

    noisy_filelist = os.listdir("%s_noisy" % (noisy_foldername))
    noisy_filelist.sort()
    # filelist = [f for f in filelist if f.endswith(".wav")]
    for i in tqdm(noisy_filelist):
        dataset['innames'].append("%s_noisy/%s" % (noisy_foldername, i))
        dataset['shortnames'].append("%s" % (i))

    clean_filelist = os.listdir("%s_clean" % (clean_foldername))
    clean_filelist.sort()
    for i in tqdm(clean_filelist):
        dataset['outnames'].append("%s_clean/%s" % (clean_foldername, i))

    return dataset


# DATA LOADING - LOAD FILE DATA
def load_data(dataset):
    dataset['inaudio'] = [None] * len(dataset['innames'])
    dataset['outaudio'] = [None] * len(dataset['outnames'])

    for id in tqdm(range(len(dataset['innames']))):
        audio_config = {
            "frame_length": 25,
            "frame_shift": 10,
        }

        if dataset['inaudio'][id] is None:
            # inputData, sr = librosa.load(dataset['innames'][id], sr=None)
            # outputData, sr = librosa.load(dataset['outnames'][id], sr=None)

            inputData, sample_rate = torchaudio.load(dataset['innames'][id])
            outputData, sample_rate = torchaudio.load(dataset['outnames'][id])

            inputData_vstack = torchaudio.compliance.kaldi.fbank(inputData, num_mel_bins=40,
                                                                 channel=-1,
                                                                 sample_frequency=16000,
                                                                 **audio_config)
            # [5552,40]

            outputData_vstack = torchaudio.compliance.kaldi.fbank(outputData, num_mel_bins=40,
                                                                  channel=-1,
                                                                  sample_frequency=16000,
                                                                  **audio_config)
            inputData_vstack_feat = inputData_vstack.numpy().flatten('A')
            outputData_vstack_feat = outputData_vstack.numpy().flatten('A')

            shape = np.shape(inputData)

            dataset['inaudio'][id] = np.float32(inputData_vstack_feat)
            dataset['outaudio'][id] = np.float32(outputData_vstack_feat)

    return dataset


class AudioDataset(data.Dataset):
    """
    Audio sample reader.
    """

    def __init__(self, data_type):
        dataset = load_data_list(setname=data_type)
        self.dataset = load_data(dataset)

        self.file_names = dataset['innames']

    def __getitem__(self, idx):
        mixed = torch.from_numpy(self.dataset['inaudio'][idx]).type(torch.FloatTensor)
        clean = torch.from_numpy(self.dataset['outaudio'][idx]).type(torch.FloatTensor)

        return mixed, clean

    def __len__(self):
        return len(self.file_names)

    def zero_pad_concat(self, inputs):
        max_t = max(inp.shape[0] for inp in inputs)
        shape = (len(inputs), max_t)
        input_mat = np.zeros(shape, dtype=np.float32)
        for e, inp in enumerate(inputs):
            input_mat[e, :inp.shape[0]] = inp
        return input_mat

    def collate(self, inputs):
        mixeds, cleans = zip(*inputs)
        seq_lens = torch.IntTensor([i.shape[0] for i in mixeds])

        x = torch.FloatTensor(self.zero_pad_concat(mixeds))
        y = torch.FloatTensor(self.zero_pad_concat(cleans))

        batch = [x, y, seq_lens]
        return batch

# Below is how to use data loader

# train_dataset = AudioDataset(data_type='train')
# train_data_loader = DataLoader(dataset=train_dataset, batch_size=4,
#                                collate_fn=train_dataset.collate, shuffle=True, num_workers=4)
#
# valid_dataset = AudioDataset(data_type='valid')
# valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=4,
#                                collate_fn=valid_dataset.collate, shuffle=False, num_workers=4)
# train_bar = tqdm(train_data_loader)
#
# test_dataset = AudioDataset(data_type='test')
# test_data_loader = DataLoader(dataset=test_dataset, batch_size=4,
#                               collate_fn=test_dataset.collate, shuffle=True, num_workers=4)

# test_dataset2 = AudioDataset(data_type='test2')
# test_data_loader2 = DataLoader(dataset=test_dataset2, batch_size=4,
#        collate_fn=test_dataset2.collate, shuffle=True, num_workers=4)
