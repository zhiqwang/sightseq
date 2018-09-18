import os
import argparse
import numpy as np
import torch
from models.lstm_feature import LSTMFeatures
from utils.converter import LabelConverter
import skimage
from skimage import io
from torchvision import transforms
from utils.dataset import Normalize, Resize, ToTensor, ToTensorRGBFlatten

from models.lstm_feature import LSTMFeatures
from models.cnn_feature import CNNFeature

def parse_args():
    '''Parse input arguments.'''
    parser = argparse.ArgumentParser(description='Digit Recognition')
    parser.add_argument('--input', type=str,
                        default='./data/train.data', help='image path')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu called when train')
    parser.add_argument('--alphabet', type=str, default='0123456789',
                        help='label alphabet')
    parser.add_argument('--model_path', type=str, default='./data/experiment',
                        help='Where to store samples and models')
    parser.add_argument('--rnn', action='store_true',
                        help='Train the model with model of rnn')
    args = parser.parse_args()
    return args

def main(args):
    img_name = './data/val.data/00036_99727091586974.jpg'
    model_path = './data/model/lstm_ctc_demon.pth'
    device = torch.device("cuda")
    img = io.imread(img_name)
    img = skimage.img_as_float32(img)

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

    img = transform(img)

    alphabet = args.alphabet
    converter = LabelConverter(alphabet)
    ntoken = len(alphabet) + 1

    N = 1
    size = img.shape
    img = img.reshape(1, *size)
    img = img.to(device)

    if args.rnn:
        input_dim = 96
        model = LSTMFeatures(input_dim, N, ntoken, nhid=512, nlayers=2)
        model.hidden = model.init_hidden(N)
    else:
        model = CNNFeature(ntoken)

    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        if args.rnn:
            model.hidden = model.init_hidden(N)
        print('image name: {}'.format(img_name))
        outputs = model(img)
        preds = converter.best_path_decode(outputs)
        print('preds: {}'.format(preds))

if __name__ == '__main__':
    main(parse_args())