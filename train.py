import os
import argparse

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from warpctc_pytorch import CTCLoss

from utils.trainer import SolverWrapper
from utils.converter import LabelConverter
from utils.dataset import digitsDataset, Normalize, Resize
from utils.dataset import ToTensor, ToTensorRGBFlatten

from models.lstm import LSTMFeatures
from models.cnn import DenseNetFeature

from easydict import EasyDict
gpu_id = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
import warnings
warnings.filterwarnings("always")

def parse_args():
    '''Parse input arguments.'''
    parser = argparse.ArgumentParser(description='Digit Recognition')
    parser.add_argument('--train_root_path', type=str,
                        default='./data/train.data', help='train dataset path')
    parser.add_argument('--val_root_path', type=str,
                        default='./data/val.data', help='valuate dataset path')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu called when train')
    parser.add_argument('--alphabet', type=str, default='0123456789',
                        help='label alphabet')
    parser.add_argument('--log_path', type=str, default='./log',
                        help='output folder path')
    parser.add_argument('--max_epoch', type=int, default='500',
                        help='train epoch')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size to train a model')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--decay_steps', type=int, default=10000,
                        help='decay step')
    parser.add_argument('--decay_rate', type=float, default=0.9,
                        help='decay rate')
    parser.add_argument('--print_freq', type=int, default=20,
                        help='print frequency (default: 100)')
    parser.add_argument('--validate_interval', type=int, default=500,
                        help='Interval to be displayed')
    parser.add_argument('--save_interval', type=int, default=10000,
                        help='save the model')
    parser.add_argument('--expr_path', type=str, default='./data/experiment',
                        help='Where to store samples and models')
    parser.add_argument('--rnn', action='store_true',
                        help='Train the model with model of rnn')
    args = parser.parse_args()
    return args

def main(args):
    if args.rnn:
        transform = transforms.Compose([Normalize([0.3956, 0.5763, 0.5616],
                                                  [0.1535, 0.1278, 0.1299]),
                                        Resize((204, 32)),
                                        ToTensorRGBFlatten()])
    else:
        transform = transforms.Compose([Normalize([0.3956, 0.5763, 0.5616],
                                                  [0.1535, 0.1278, 0.1299]),
                                        Resize((204, 32)),
                                        ToTensor()])
    train_set = digitsDataset(args.train_root_path, transform=transform)
    val_set = digitsDataset(args.val_root_path, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=4,
                            pin_memory=True)
    # trainer parameters
    params = EasyDict()
    params.max_epoch = args.max_epoch
    params.print_freq = args.print_freq
    params.validate_interval = args.validate_interval
    params.save_interval = args.save_interval
    params.expr_path = args.expr_path
    params.rnn = args.rnn
    device = torch.device("cuda")

    # train engine
    ntoken = len(args.alphabet) + 1
    if args.rnn:
        input_dim = 96
        model = LSTMFeatures(input_dim, args.batch_size, ntoken)
    else:
        model = DenseNetFeature(num_classes=ntoken)
    model = model.to(device)
    criterion = CTCLoss()
    criterion = criterion.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    converter = LabelConverter(args.alphabet)

    solver = SolverWrapper(params)
    # train
    solver.train(train_loader, val_loader, model, criterion,
                 optimizer, device, converter)

if __name__ == '__main__':
    main(parse_args())
