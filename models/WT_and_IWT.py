import torch
import torch.nn as nn

def dwt_init(x):
    x_1 = x/4
    x1 = x_1[..., 0::2, 0::2]
    x2 = x_1[..., 1::2, 0::2]
    x3 = x_1[..., 0::2, 1::2]
    x4 = x_1[..., 1::2, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x, device):
    r = 2
    in_batch, in_channel, in_color, in_height, in_width = x.size()
    out_batch, out_channel, out_color, out_height, out_width = in_batch, int(in_channel / (r ** 2)), in_color,  r * in_height, r * in_width
    x1 = x[:, 0:out_channel]
    x2 = x[:, out_channel:out_channel * 2]
    x3 = x[:, out_channel * 2:out_channel * 3]
    x4 = x[:, out_channel * 3:out_channel * 4]

    h = torch.zeros(1).float().to(device)
    h = h.resize_(out_batch, out_channel, out_color, out_height, out_width)

    h[..., 0::2, 0::2] = (x1 - x2 - x3 + x4).clone()
    h[..., 1::2, 0::2] = (x1 - x2 + x3 - x4).clone()
    h[..., 0::2, 1::2] = (x1 + x2 - x3 - x4).clone()
    h[..., 1::2, 1::2] = (x1 + x2 + x3 + x4).clone()

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x, device):
        return iwt_init(x, device)