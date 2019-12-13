import torch.nn as nn
import torch
from torch import optim


class DQN(nn.Module):

    def __init__(self, nlp, img, outputs):
        super(DQN, self).__init__()
        img_linear_input_size = self._get_size(list(img.size())[-1], list(img.size())[-2])
        self.seq_img = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Linear(img_linear_input_size, 100)
        )
        size = list(nlp.size())[-1]*list(nlp.size())[-2]*18
        self.nlpRNN = nn.GRU(input_size=size, hidden_size=500, num_layers=4, bidirectional=True)
        self.seq_nlp = nn.Sequential(
            nn.Linear(size, size*2),
            nn.BatchNorm1d(size*2),
            nn.ReLU(),
            nn.Linear(size*2, size*2),
            nn.BatchNorm1d(size*2),
            nn.ReLU(),
            nn.Linear(size*2, size*2),
            nn.BatchNorm1d(size*2),
            nn.ReLU(),
            nn.Linear(size*2, size*2),
            nn.BatchNorm1d(size*2),
            nn.ReLU(),
            nn.Linear(size*2, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
        ).cuda()
        self.GAMMA = 0.999
        # self.head_conv = nn.Conv2d(64, 32, kernel_size=1, stride=1)


        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

        self.update_output(outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def conv2d_size_out(self, size, kernel_size=3, stride=2):
        toreturn = size - (kernel_size - 1)
        toreturn = toreturn - 1
        toreturn = toreturn // stride
        toreturn += 1
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x_nlp = x[0]
        x_img = x[1]
        sizes = list(x_nlp.size())
        num_filters = list(x_nlp.size())[-3]
        x_nlp = x_nlp.view((sizes[0], num_filters * sizes[-1] * sizes[-2])).unsqueeze(0)
        nlp_output = self.seq_nlp(x_nlp.cuda())
        img_output = self.seq_img(x_img.cuda())
        catted = torch.cat((nlp_output, img_output), -1)
        toreturn = self.head(catted)
        return toreturn


    def update_output(self,outputs):
        self.head = nn.Linear(200, outputs).cuda()

    def _get_size(self, w, h):
        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(w)))
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(h)))
        return convw * convh * 32

