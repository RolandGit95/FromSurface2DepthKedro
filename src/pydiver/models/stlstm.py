# %%
import math
import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt

# %%   

class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=(3, 3), padding=(1, 1), name='CLSTM-Cell'):

        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wx = nn.Conv2d(input_size, hidden_size*3, kernel_size, 1, padding)
        self.Wzr= nn.Conv2d(hidden_size, hidden_size*2, kernel_size, 1, padding)
        self.Woh = nn.Conv2d(hidden_size, hidden_size, kernel_size, 1, padding)

        self.reset_parameters()

    def forward(self, input, H=None):
        self.check_forward_input(input)
        if H is None:
            H = torch.zeros(input.size(0), self.hidden_size, input.size(2), input.size(3),
                            dtype=input.dtype, device=input.device)
            
        zx, rx, ox = torch.split(self.Wx(input), self.hidden_size, dim=1)
        zh, rh = torch.split(self.Wzr(H), self.hidden_size, dim=1)
        
        
        z = torch.sigmoid(zx + zh)
        r = torch.sigmoid(rx + rh)
        
        o = torch.tanh(ox + self.Woh(r*H))
        H = z*H+(1-z)*o

        return H

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_input(self, input: Tensor) -> None:
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))
            
            
class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=(3, 3), padding=(1, 1), name='CLSTM-Cell'):

        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.conv = nn.Conv2d(input_size + hidden_size, hidden_size*4, kernel_size, 1, padding)
        self.reset_parameters()

    def forward(self, input, H=None, C=None):
        self.check_forward_input(input)
        if H is None:
            H = torch.zeros(input.size(0), self.hidden_size, input.size(2), input.size(3),
                            dtype=input.dtype, device=input.device)
        if C is None:
            C = torch.zeros(input.size(0), self.hidden_size, input.size(2), input.size(3),
                            dtype=input.dtype, device=input.device)

        conv_out = self.conv(torch.cat([input, H], dim=1))  # concatenate the features, [b, f_X+f_h, :, :] # [b, f_h,:,:]
        _f, _i, _c, _o = torch.split(conv_out, self.hidden_size, dim=1)

        f = torch.sigmoid(_f)
        i = torch.sigmoid(_i)
        c = torch.sigmoid(_c)
        o = torch.tanh(_o)

        C = C * f + i * c
        H = o * torch.tanh(c)
        return H, C

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def check_forward_input(self, input: Tensor) -> None:
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))


class CLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers):
        super(CLSTMEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        cell_0 = [ConvLSTMCell(input_size, hidden_size)]
        self.cells = nn.ModuleList(cell_0 + [ConvLSTMCell(hidden_size, hidden_size) for i in range(1, num_layers)])

    def getShapedTensor(self, input):
            return torch.zeros(input.size(0), self.hidden_size, input.size(3), input.size(4), dtype=input.dtype, device=input.device, requires_grad=False)

    def forward(self, input, H=None, C=None):
        Hs = [self.getShapedTensor(input) for _ in range(input.size(1))]

        if not isinstance(H, type(None)):
            Hs[0] = H
            
        if isinstance(C, type(None)):
            C = self.getShapedTensor(input)

        for t in range(input.size(1)):  # numer of time-steps
            H, C = self.cells[0](input[:, t], H=H, C=C)
            Hs[t] = H

        for l in range(1, len(self.cells)):
            for t in range(input.size(1)):
                if t == 0:
                    C = self.getShapedTensor(input)
                H, C = self.cells[l](Hs[t], H=H, C=C)
                Hs[t] = H

        return Hs[-1], C  # [b,hd,:,:], [b,hd,:,:], [b,hd,:,:]
    
    
if __name__=='__main__':
    device = torch.device("cuda")
    c = CLSTMEncoder(1,64,(3,3), 3).to(device)

    with torch.no_grad():
        x = torch.rand(2,32,1,120,120).to(device)
        y = c(x)    
    
    
    
    
    
    
    