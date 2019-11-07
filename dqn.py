import torch.nn as nn
import torch
from torch import optim


class DQN(nn.Module):

    def __init__(self, nlp, img, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.seq_img = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        size = list(nlp.size())[-1]*list(nlp.size())[-2]
        self.nlpRNN = nn.GRU(input_size=size, hidden_size=1000, num_layers=4, bidirectional=True)
        self.GAMMA = 0.999
        self.head_conv = nn.Conv2d(64, 32, kernel_size=1, stride=1)


        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

        self.update_output(nlp, img, outputs)

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
        x_nlp = x_nlp.view((sizes[0], sizes[-1] * sizes[-2])).unsqueeze(0)
        self.conv1 = nn.Conv2d(num_filters, 32, kernel_size=3, stride=2).cuda()
        nlp_rnn_output = self.nlpRNN(x_nlp.cuda())[0]
        img_seq_output = self.seq_img(self.conv1(x_img.cuda()))
        nlp_output = self.head_nlp(nlp_rnn_output.view(nlp_rnn_output.size(0), -1))
        img_output = self.head_img(img_seq_output.view(img_seq_output.size(0), -1))
        catted = torch.cat((nlp_output, img_output), 0)
        toreturn = self.head(catted.view(-1)).cpu().detach().numpy()
        return toreturn.argmax()

    def update_output(self, nlp, img, outputs):

        img_linear_input_size = self._get_size(list(img.size())[-1], list(img.size())[-2])
        self.head_img = nn.Linear(img_linear_input_size, outputs).cuda()
        self.head_nlp = nn.Linear(2000 * list(nlp.size())[-3], outputs).cuda()
        self.head = nn.Linear(outputs * 2, outputs).cuda()

    def _get_size(self, w, h):
        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(w)))
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(h)))
        return convw * convh * 32

