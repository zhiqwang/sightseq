import os
import numpy as np
import torch
from models.lstm_feature import LSTMFeatures
from utils.converter import LabelConverter
import skimage
from skimage import io
from torchvision import transforms
from utils.dataset import Normalize, Resize, ToTensor

img_name = './data/val.data/00036_99727091586974.jpg'
model_path = './data/model/lstm_ctc_demon.pth'
device = torch.device("cuda")
img = io.imread(img_name)
img = skimage.img_as_float32(img)
transform = transforms.Compose([Normalize([0.3956, 0.5763, 0.5616],
                                          [0.1535, 0.1278, 0.1299]),
                                Resize((204, 32)),
                                ToTensor()])
# transform = ToTensor()
img = transform(img)

alphabet = '0123456789'
converter = LabelConverter(alphabet)
ntoken = len(alphabet) + 1
input_dim = 96

N = 1
size = img.shape
img = img.reshape(1, *size)
img = img.to(device)

model = LSTMFeatures(input_dim, N, ntoken, nhid=512, nlayers=2)
model.hidden = model.init_hidden(N)
model = model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

with torch.no_grad():
    model.hidden = model.init_hidden(N)
    print('image name: {}'.format(img_name))
    outputs = model(img)
    preds = converter.best_path_decode(outputs)
    print('preds: {}'.format(preds))
