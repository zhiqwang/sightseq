import torch.nn as nn
import torch.nn.functional as F

class LSTMFeatures(nn.Module):
    """Get the lstm features
    """
    def __init__(self, input_dim, batch_size, ntoken, nhid=512, nlayers=2):
        super().__init__()
        self.nhid = nhid
        self.nlayers = nlayers

        # The LSTM takes
        self.lstm = nn.LSTM(input_dim, self.nhid, nlayers)
        self.fc = nn.Linear(self.nhid, ntoken)
        self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        # The axes semantics are (nlayers, minibatch_size, nhid)
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                weight.new_zeros(self.nlayers, batch_size, self.nhid))

    def forward(self, images):
        # images is batch_size x seq_len x input_dim
        seq_len = images.shape[1]
        images = images.permute(1, 0, 2).contiguous()
        # lstm output is seq_len x batch_size x nhid
        out, self.hidden = self.lstm(images, self.hidden)
        # feature (out) is seq_len x batch_size x ntoken
        out = self.fc(out)
        out = F.log_softmax(out, dim=2)
        return out
