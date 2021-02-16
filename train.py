# --batch_size=64 --dropout_p=0.2 --attn_use=True --stacked_encoder=True --attn_len=5 --hidden_size=448 --num_epochs=61
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models.layers.istft import ISTFT
from load_dataset import AudioDataset
from models.attention import AttentionModel

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
parser.add_argument('--hidden_size', default=112, type=int)
parser.add_argument('--ck_name', default='se_fbank.pt')
args = parser.parse_args()

n_fft, hop_length = 512, 128
window = torch.hann_window(n_fft).cuda()
# STFT
stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
# ISTFT
istft = ISTFT(n_fft, hop_length, window='hanning').cuda()
num_fbank = 40


def normalized(tensor):
    output = [[] for i in range(len(tensor))]

    for i in range(len(tensor)):
        nummer = tensor[i] - torch.min(tensor[i])
        denomi = torch.max(tensor[i]) - torch.min(tensor[i])

        output[i] = (nummer / (denomi + 1e-5)).tolist()

    return torch.tensor(output)


def main():
    summary = SummaryWriter('./log')
    # os.system('tensorboard --logdir=log')
    #
    # set Hyper parameter
    # json_path = os.path.join(args.model_dir)
    # params = train_utils.Params(json_path)

    # data loader
    train_dataset = AudioDataset(data_type='train')
    # modify:num_workers=4
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate,
                                   shuffle=True, num_workers=4)
    test_dataset = AudioDataset(data_type='valid')
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate,
                                  shuffle=False, num_workers=4)

    # # data loader
    # train_dataset = AudioDataset(data_type='test')
    # # modify:num_workers=4
    # train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate,
    #                                shuffle=True, num_workers=0)
    # test_dataset = AudioDataset(data_type='test')
    # test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate,
    #                               shuffle=False, num_workers=0)

    # model select
    print('Model initializing\n')
    net = torch.nn.DataParallel(
        AttentionModel(40, hidden_size=args.hidden_size, dropout_p=args.dropout_p, use_attn=args.attn_use,
                       stacked_encoder=args.stacked_encoder, attn_len=args.attn_len))
    # net = AttentionModel(257, 112, dropout_p = args.dropout_p, use_attn = arg0s.attn_use)
    net = net.cuda()
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    scheduler = ExponentialLR(optimizer, 0.5)

    # check point load
    # Check point load

    print('Trying Checkpoint Load\n')
    # ckpt_dir = 'ckpt_dir_stoi'
    ckpt_dir = 'ckpt_dir'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    best_PESQ = 0.
    best_STOI = 0.
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
        train_bar = tqdm(train_data_loader)
        audio_config = {
            "frame_length": 25,
            "frame_shift": 10,
        }
        # train_bar = train_data_loader876\
        n = 0
        avg_loss = 0
        net.train()
        for input in train_bar:
            iteration += 1
            # load data
            train_mixed, train_clean, seq_len = map(lambda x: x.cuda(), input)
            # -----------------------------------
            # lt_train_mixed_vstack = []
            # lt_train_clean_vstack = []
            # for i in range(len(train_mixed)):
            #     train_mixed_vstack = torchaudio.compliance.kaldi.fbank(train_mixed[i].unsqueeze(0), num_mel_bins=40,
            #                                                            channel=-1,
            #                                                            sample_frequency=16000,
            #                                                            **audio_config)
            #     train_clean_vstack = torchaudio.compliance.kaldi.fbank(train_clean[i].unsqueeze(0), num_mel_bins=40,
            #                                                            channel=-1,
            #                                                            sample_frequency=16000,
            #                                                            **audio_config)
            #
            #     train_mixed_vstack_data = train_mixed_vstack.transpose(0, 1).unsqueeze(0).detach()
            #     train_clean_vstack_data = train_clean_vstack.transpose(0, 1).unsqueeze(0).detach()
            #
            #     lt_train_mixed_vstack.append(train_mixed_vstack_data)
            #     lt_train_clean_vstack.append(train_clean_vstack_data)
            #
            # lt_train_mixed_feat = torch.cat(lt_train_mixed_vstack, dim=0).transpose(1, 2)
            # lt_train_clean_feat = torch.cat(lt_train_clean_vstack, dim=0).transpose(1, 2)

            # ------------------------

            train_mixed_feat = train_mixed.reshape(len(train_mixed), int(len(train_mixed[0]) / num_fbank),
                                                   num_fbank).to(
                device='cuda')
            train_clean_feat = train_clean.reshape(len(train_clean), int(len(train_clean[0]) / num_fbank),
                                                   num_fbank).to(
                device='cuda')

            # feed data
            out_train_mixed_feat, attn_weight = net(train_mixed_feat)

            # # feed data
            # out_lt_train_mixed_feat, attn_weight = net(lt_train_mixed_feat)

            loss = F.mse_loss(out_train_mixed_feat, train_clean_feat, True)
            if torch.any(torch.isnan(loss)):
                torch.save(
                    {'clean_mag': train_clean_feat, 'out_mag': train_mixed_feat, 'mag': out_train_mixed_feat},
                    'nan_mag')
                raise ('loss is NaN')
            avg_loss += loss.item()
            n += 1
            # gradient optimizer
            optimizer.zero_grad()

            # backpropagate LOSS20+

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
        test_bar = tqdm(test_data_loader)

        net.eval()
        with torch.no_grad():
            for input in test_bar:
                test_mixed, test_clean, seq_len = map(lambda x: x.cuda(), input)
                test_mixed_feat = test_mixed.reshape(len(test_mixed), int(len(test_mixed[0]) / num_fbank),
                                                     num_fbank).to(
                    device='cuda')
                test_clean_feat = test_clean.reshape(len(test_clean), int(len(test_clean[0]) / num_fbank),
                                                     num_fbank).to(
                    device='cuda')

                logits_test_mixed_feat, logits_attn_weight = net(test_mixed_feat)

                test_loss = F.mse_loss(logits_test_mixed_feat, test_clean_feat, True)

                # lt_test_mixed_vstack = []
                # lt_test_clean_vstack = []
                # for i in range(len(test_mixed)):
                #     test_mixed_vstack = torchaudio.compliance.kaldi.fbank(test_mixed[i].unsqueeze(0), num_mel_bins=40,
                #                                                           channel=-1,
                #                                                           sample_frequency=16000,
                #                                                           **audio_config)
                #     test_clean_vstack = torchaudio.compliance.kaldi.fbank(test_clean[i].unsqueeze(0), num_mel_bins=40,
                #                                                           channel=-1,
                #                                                           sample_frequency=16000,
                #                                                           **audio_config)
                #
                #     test_mixed_vstack_data = test_mixed_vstack.transpose(0, 1).unsqueeze(0).detach()
                #     test_clean_vstack_data = test_clean_vstack.transpose(0, 1).unsqueeze(0).detach()
                #
                #     lt_test_mixed_vstack.append(test_mixed_vstack_data)
                #     lt_test_clean_vstack.append(test_clean_vstack_data)
                #
                # lt_test_mixed_feat = torch.cat(lt_test_mixed_vstack, dim=0).transpose(1, 2)
                # lt_test_clean_feat = torch.cat(lt_test_clean_vstack, dim=0).transpose(1, 2)
                #
                # out_lt_test_mixed_feat, attn_weight = net(lt_test_mixed_feat)

                # test_loss = F.mse_loss(out_lt_test_mixed_feat, lt_test_clean_feat, True)

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
