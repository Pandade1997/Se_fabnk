import os

from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset

# Additional (official) text src provided
OFFICIAL_TXT_SRC = ['librispeech-lm-norm.txt']
# Remove longest N sentence in librispeech-lm-norm.txt
REMOVE_TOP_N_TXT = 5000000
# Default num. of threads used for loading LibriSpeech
READ_FILE_THREADS = 1


class LibriDataset(Dataset):
    def __init__(self, path, split, bucket_size, ascending=False):
        # Setup
        self.path = path
        self.bucket_size = bucket_size

        # List all wave files
        file_list = []
        for s in split:
            split_list = list(Path(join(path, s)).rglob("*.wav"))
            assert len(split_list) > 0, "No data found @ {}".format(join(path, s))
            file_list += split_list

        # Sort dataset by text length
        # file_len = Parallel(n_jobs=READ_FILE_THREADS)(delayed(getsize)(f) for f in file_list)
        self.file_list = [(f_name)
                          for f_name in
                          sorted(zip(file_list), reverse=not ascending)]

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list) - self.bucket_size, index)
            return [(f_path) for f_path in
                    zip(self.file_list[index:index + self.bucket_size])]
        else:
            return self.file_list[index]

    def __len__(self):
        return len(self.file_list)


class LibriNoisyDataset(Dataset):
    def __init__(self, job, input, mode, path, split, bucket_size, ascending=False):
        # Setup
        split = input
        self.path = path
        self.bucket_size = bucket_size
        if job == 'train':
            # path = path + "/train-noisy-100/"
            path = path + "/trainset_noisy/"
        elif job == 'dev':
            # path = path + "/dev-noisy/"
            path = path + "/validset_noisy/"
        elif job == 'test':
            # path = path + "/test-noisy/"
            path = path + "/testset_noisy/"

        # List all wave files
        file_list = []
        for s in split:
            split_list = path + "/" + s + ".wav"
            file_list.append(split_list)

        # Sort dataset by text length
        # file_len = Parallel(n_jobs=READ_FILE_THREADS)(delayed(getsize)(f) for f in file_list)
        self.file_list = [(f_name)
                          for f_name in
                          sorted(zip(file_list), reverse=not ascending)]

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list) - self.bucket_size, index)
            return [(f_path) for f_path in
                    zip(self.file_list[index:index + self.bucket_size])]
        else:
            return self.file_list[index]

    def __len__(self):
        return len(self.file_list)
