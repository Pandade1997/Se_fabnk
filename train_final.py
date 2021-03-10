# --batch_size=64 --dropout_p=0.2 --attn_use=True --stacked_encoder=True --attn_len=5 --hidden_size=448 --num_epochs=61
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models.attention import AttentionModel
from src.data_final import load_dataset, load_noisy_dataset

import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiment/SE_model.json', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--batch_size', default=8, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
parser.add_argument('--dropout_p', default=0.2, type=float, help='Attention model drop out rate')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='Learning rate')
parser.add_argument('--attn_use', default=True, type=bool)
parser.add_argument('--stacked_encoder', default=True, type=bool)
parser.add_argument('--attn_len', default=5, type=int)
parser.add_argument('--hidden_size', default=448, type=int)
parser.add_argument('--ck_name', default='final_batch.pt')

parser.add_argument('--njobs', default=16, type=int,
                    help='Number of threads for dataloader/decoding.', required=False)
parser.add_argument('--config', type=str, help='Path to experiment config.', default="config/asr_example.yaml")
parser.add_argument('--no-pin', action='store_true',
                    help='Disable pin-memory for dataloader')
parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')

args = parser.parse_args()
setattr(args, 'gpu', not args.cpu)
setattr(args, 'pin_memory', not args.no_pin)
setattr(args, 'verbose', not args.no_msg)
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

num_fbank = 40


def fetch_data(data):
    ''' Move data to device and compute text seq. length'''
    name, feat, feat_len = data
    feat = feat.to(torch.device('cuda'))
    feat_len = feat_len.to(torch.device('cuda'))

    return feat, feat_len


def verbose(msg):
    ''' Verbose function for print information to stdout'''
    if args.verbose:
        if type(msg) == list:
            for m in msg:
                print('[INFO]', m.ljust(100))
        else:
            print('[INFO]', msg.ljust(100))


def main():
    summary = SummaryWriter('./log')
    tr_set, dv_set, feat_dim, msg = load_dataset(args.njobs, args.gpu, args.pin_memory,
                                                 config['hparas']['curriculum'] > 0,
                                                 **config['data'])

    verbose(msg)
    # model select
    print('Model initializing\n')
    net = torch.nn.DataParallel(
        AttentionModel(120, hidden_size=args.hidden_size, dropout_p=args.dropout_p, use_attn=args.attn_use,
                       stacked_encoder=args.stacked_encoder, attn_len=args.attn_len))
    # net = AttentionModel(257, 112, dropout_p = args.dropout_p, use_attn = arg0s.attn_use)
    net = net.cuda()
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    scheduler = ExponentialLR(optimizer, 0.5)

    # check point load
    # Check point load

    print('Trying Checkpoint Load\n')
    ckpt_dir = 'ckpt_dir'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    best_loss = 200000.
    ckpt_path = os.path.join(ckpt_dir, args.ck_name)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        try:
            net.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            best_loss = ckpt['best_loss']

            print('checkpoint is loaded !')
            print('current best loss : %.4f' % best_loss)
        except RuntimeError as e:
            print('wrong checkpoint\n')
    else:
        print('checkpoint not exist!')
        print('current best loss : %.4f' % best_loss)

    print('Training Start!')
    # train
    iteration = 0
    train_losses = []
    test_losses = []
    for epoch in range(args.num_epochs):
        n = 0
        avg_loss = 0
        net.train()
        for input in tqdm(tr_set):
            tr_noisy_set, feat_dim = load_noisy_dataset("train", input[0], args.njobs,
                                                        args.gpu,
                                                        args.pin_memory,
                                                        config['hparas']['curriculum'] > 0,
                                                        **config['data_noisy'])
            for input_noisy in tr_noisy_set:
                train_clean_feat, feat_len = fetch_data(input)
                train_noisy_feat, feat_len = fetch_data(input_noisy)

                iteration += 1

                # feed data
                train_mixed_feat, attn_weight = net(train_noisy_feat)
                if train_mixed_feat.shape == train_clean_feat.shape:
                    loss = F.mse_loss(train_mixed_feat, train_clean_feat, True)

                    if torch.any(torch.isnan(loss)):
                        torch.save(
                            {'clean_mag': train_clean_feat, 'noisy_mag': train_noisy_feat, 'out_mag': train_mixed_feat},
                            'nan_mag')
                        raise ('loss is NaN')
                    avg_loss += loss.item()

                    n += 1
                    # gradient optimizer
                    optimizer.zero_grad()

                    loss.backward()

                    # update weight
                    optimizer.step()

        avg_loss /= n
        print('result:')
        print('[epoch: {}, iteration: {}] avg_loss : {:.4f}'.format(epoch, iteration, avg_loss))

        summary.add_scalar('Train Loss', avg_loss, iteration)

        train_losses.append(avg_loss)
        if (len(train_losses) > 2) and (train_losses[-2] < avg_loss):
            print("Learning rate Decay")
            scheduler.step()

        # test phase
        n = 0
        avg_test_loss = 0
        net.eval()
        with torch.no_grad():
            for input in tqdm(dv_set):
                dv_noisy_set, feat_dim = load_noisy_dataset("dev", input[0], args.njobs,
                                                            args.gpu,
                                                            args.pin_memory,
                                                            config['hparas']['curriculum'] > 0,
                                                            **config['data_noisy'])
                for input_noisy in dv_noisy_set:
                    test_clean_feat = input[1].to(device='cuda')
                    test_noisy_feat = input_noisy[1].to(device='cuda')

                    test_mixed_feat, logits_attn_weight = net(test_noisy_feat)
                    if test_mixed_feat.shape == test_clean_feat.shape:
                        test_loss = F.mse_loss(test_mixed_feat, test_clean_feat, True)

                        avg_test_loss += test_loss.item()
                        n += 1

            avg_test_loss /= n

            test_losses.append(avg_test_loss)
            summary.add_scalar('Test Loss', avg_test_loss, iteration)

            print('[epoch: {}, iteration: {}] test loss : {:.4f} '.format(epoch, iteration, avg_test_loss))
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                # Note: optimizer also has states ! don't forget to save them as well.
                ckpt = {'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_loss': best_loss}
                torch.save(ckpt, ckpt_path)
                print('checkpoint is saved !')


if __name__ == '__main__':
    main()
