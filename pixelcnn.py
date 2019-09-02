import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision

import dataloader
import utils


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


# Hout = ⌊(Hin + 2 ×  padding[0] −  dilation[0] × ( kernel_size[0] − 1) − 1)/( stride[0]) + 1⌋
fm = 64
net = nn.Sequential(
    MaskedConv2d('A', 1, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),
    nn.Conv2d(fm, 256, 1))

net.cuda()

optimizer = optim.Adam(net.parameters())
last_epoch = 0

import os

ckp = "results/pixelcnn.pt"
if os.path.isfile(ckp):
    checkpoint = torch.load(ckp)
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    last_epoch = checkpoint['epoch']
    print('Load checkpoint! Last Epoch: %d' % last_epoch)

sample = torch.empty(144, 1, 28, 28).cuda()


def mk_sample():
    sample.fill_(0)
    net.eval()
    for i in range(28):
        for j in range(28):
            out = net(sample)
            probs = F.softmax(out[:, :, i, j], dim=1)
            sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.


is_infer = False

if is_infer:
    mk_sample()
    utils.show_pics(sample)

else:
    tr, te = dataloader.load_mnist(select=8)
    epochs = 30
    for epoch in range(last_epoch + 1, epochs + last_epoch + 1):

        # train
        err_tr = []
        time_tr = time.time()
        net.train()
        for i, (input, _) in enumerate(tr):
            input = input.cuda()
            target = (input[:, 0] * 255).long()
            out = net(input)
            loss = F.cross_entropy(out, target)
            err_tr.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 9:
                print('[%d][%d/%d] loss=%.3f' % (epoch, i, len(tr), loss.item()))

        time_tr = time.time() - time_tr

        torch.save({
            'epoch': epoch,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, ckp)

        # compute error on test set
        err_te = []
        time_te = time.time()
        net.eval()
        for input, _ in te:
            input = input.cuda()
            target = (input.data[:, 0] * 255).long()
            loss = F.cross_entropy(net(input), target)
            err_te.append(loss.item())
        time_te = time.time() - time_te

        print('epoch={}; nll_tr={:.7f}; nll_te={:.7f}; time_tr={:.1f}s; time_te={:.1f}s'.format(
            epoch, np.mean(err_tr), np.mean(err_te), time_tr, time_te))

        # sample
        if epoch % 10 == 9:
            time_sm = time.time()
            mk_sample()
            fn = 'results/pixelcnn_{:03d}.png'.format(epoch)
            torchvision.utils.save_image(sample.cpu(), fn, nrow=12, padding=0)
            time_sm = time.time() - time_sm
            print('gen {}, time_sm={:1f}s'.format(fn, time_sm))