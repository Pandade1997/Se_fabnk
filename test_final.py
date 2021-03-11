import os
import argparse
import torch
import torch.optim as optim
import yaml
import librosa
from tqdm import tqdm
from load_dataset import AudioDataset
from models.attention import AttentionModel

from src.data_final import load_dataset, load_noisy_dataset
from scipy.io import savemat
import scipy.io as scio
import numpy as np
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=8, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
parser.add_argument('--dropout_p', default=0.2, type=float, help='Attention model drop out rate')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='Learning rate')
parser.add_argument('--stacked_encoder', default=True, type=bool)
parser.add_argument('--attn_len', default=5, type=int)
parser.add_argument('--hidden_size', default=448, type=int)
parser.add_argument('--ck_dir', default='ckpt_dir', help='ck path')
parser.add_argument('--ck_name', help='ck file', default='final_batch.pt')
parser.add_argument('--test_set', help='test', default='test')
parser.add_argument('--attn_use', default=True, type=bool)

parser.add_argument('--njobs', default=0, type=int,
                    help='Number of threads for dataloader/decoding.', required=False)
parser.add_argument('--config', type=str, help='Path to experiment config.', default="config/asr_example.yaml")
parser.add_argument('--no-pin', action='store_true',
                    help='Disable pin-memory for dataloader')
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')

parser.add_argument('--out_path',
                    default='/data01/AuFast/Pan_dataset/SE_asr/finaltest/gen_mat_batch/train_mat/',
                    type=str)

args = parser.parse_args()
setattr(args, 'gpu', not args.cpu)
setattr(args, 'pin_memory', not args.no_pin)
setattr(args, 'verbose', not args.no_msg)
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)


def verbose(msg):
    ''' Verbose function for print information to stdout'''
    if args.verbose:
        if type(msg) == list:
            for m in msg:
                print('[INFO]', m.ljust(100))
        else:
            print('[INFO]', msg.ljust(100))


def fetch_data(data):
    ''' Move data to device and compute text seq. length'''
    name, feat, feat_len = data
    feat = feat.to(torch.device('cuda'))
    feat_len = feat_len.to(torch.device('cuda'))

    return feat, feat_len


def main():
    _, tt_set, feat_dim, msg = load_dataset(args.njobs, args.gpu, args.pin_memory,
                                            config['hparas']['curriculum'] > 0,
                                            **config['data'])
    verbose(msg)

    # model select
    print('Model initializing\n')
    net = torch.nn.DataParallel(
        AttentionModel(120, hidden_size=args.hidden_size, dropout_p=args.dropout_p, use_attn=args.attn_use,
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
            print('checkpoint is loaded !')
        except RuntimeError as e:
            print('wrong checkpoint\n')
    else:
        print('checkpoint not exist!')

    # test phase
    net.eval()
    with torch.no_grad():
        sr1 = 16000
        window_size = 25  # int, window size for FFT (ms)
        stride = 10
        ws = int(sr1 * 0.001 * window_size)
        st = int(sr1 * 0.001 * stride)
        name_sum = 0
        for input in tqdm(tt_set):
            tt_noisy_set, feat_dim = load_noisy_dataset("test", input[0], args.njobs,
                                                        args.gpu,
                                                        args.pin_memory,
                                                        config['hparas']['curriculum'] > 0,
                                                        **config['data_noisy'])
            for input_noisy in tt_noisy_set:
                test_noisy_feat, feat_len = fetch_data(input_noisy)

                # test_noisy_feat = input_noisy[1].to(device='cuda')
                # feed data
                test_mixed_feat, attn_weight = net(test_noisy_feat)

                for i in range(len(test_mixed_feat)):
                    name = args.out_path + input_noisy[0][i] + '.mat'
                    feat = test_mixed_feat[i].to(device='cpu').numpy()
                    scio.savemat(name, {'feat': feat})


if __name__ == '__main__':
    main()
