# 参考 https://github.com/ritheshkumar95/pytorch-vqvae
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import dataloader
import utils


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'GatedMaskedConv2d':
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return x.tanh() * y.sigmoid()


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=0):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        if n_classes > 0:
            self.class_cond_embedding = nn.Embedding(
                n_classes, 2 * dim
            )
        else:
            self.class_cond_embedding = None

        # Hout = ⌊(Hin + 2 ×  padding[0] −  dilation[0] × ( kernel_size[0] − 1) − 1)/( stride[0]) + 1⌋
        kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, classes=None):
        if self.mask_type == 'A':
            self.make_causal()

        h = None
        if self.class_cond_embedding:
            h = self.class_cond_embedding(classes)

        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        if h:
            out_v = self.gate(h_vert + h[:, :, None, None])
        else:
            out_v = self.gate(h_vert)

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        if h:
            out = self.gate(v2h + h_horiz + h[:, :, None, None])
        else:
            out = self.gate(v2h + h_horiz)

        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=0):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)

    def forward(self, x, label=0):
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C) 假设输入是整数
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def generate(self, x, classes=0):
        x.fill_(0)
        shape = x.size()
        for i in range(shape[1]):
            for j in range(shape[2]):
                out = self.forward(x, classes)
                probs = F.softmax(out[:, :, i, j], -1)
                x[:, i, j] = probs.multinomial(1)[:, 0]
                out = None
                probs = None

        return x


class Solver(object):
    def __init__(self, cfg, try_load_best=False):
        self.cfg = cfg
        self.net = GatedPixelCNN(input_dim=cfg.input_dim, dim=cfg.dim, n_layers=cfg.n_layers,
                                 n_classes=cfg.n_classes).cuda()

        utils.print_params(self.net)
        self.optimizer = optim.Adam(self.net.parameters())
        self.last_epoch = 0
        self.best_loss = None

        self.writer = SummaryWriter(cfg.tbp)
        ckp = cfg.ckp
        if try_load_best and os.path.isfile(cfg.ckp_best):
            ckp = cfg.ckp_best
        if os.path.isfile(ckp):
            checkpoint = torch.load(ckp)
            self.net.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.last_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            print('Load {}! Epoch: {} Best loss: {:.4f}'.format(ckp, self.last_epoch, self.best_loss))

    def sample(self, size=(32, 28, 28)):
        x = torch.empty(size).long().cuda()
        self.net.generate(x)
        x = x.cpu()
        return x

    def train(self, tr, te, epochs=50, batch_interval=10, sample_interval=10):

        fixed_images = utils.first_batch(te)
        self.writer.add_graph(self.net, fixed_images.cuda())
        img_grid = torchvision.utils.make_grid(fixed_images[:, None, :, :].float() / self.cfg.input_dim)
        self.writer.add_image('original', img_grid)

        for epoch in range(self.last_epoch + 1, epochs + self.last_epoch + 1):
            # train
            err_tr = []
            time_tr = time.time()
            self.net.train()
            for i, x in enumerate(tr):
                x = x[0].cuda()

                out = self.net(x)
                loss = F.cross_entropy(out, x)
                err_tr.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                steps = (epoch - 1) * len(tr) + i
                self.writer.add_scalar('loss/train_loss', loss.item(), steps)
                if i > 0 and i % batch_interval == 0:
                    print('[%d][%d/%d] loss=%.3f' % (epoch, i, len(tr), loss.item()))

                out = None
                loss = None

            time_tr = time.time() - time_tr
            train_loss = np.mean(err_tr)

            # test
            err_te = []
            time_te = time.time()
            self.net.eval()
            size = None
            for x in te:
                if size is None:
                    size = x[0].size()
                x = x[0].cuda()
                out = self.net(x)
                loss = F.cross_entropy(out, x)
                err_te.append(loss.item())

                out = None
                loss = None

            time_te = time.time() - time_te
            test_loss = np.mean(err_te)

            self.writer.add_scalar('epoch_loss/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch_loss/test_loss', test_loss, epoch)
            print('epoch={}; nll_tr={:.7f}; nll_te={:.7f}; time_tr={:.1f}s; time_te={:.1f}s'.format(
                epoch, train_loss, test_loss, time_tr, time_te))

            best = False
            if self.best_loss is None or test_loss < self.best_loss:
                self.best_loss = test_loss
                best = True
            savedict = {
                'epoch': epoch,
                'net': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
            }
            torch.save(savedict, self.cfg.ckp)
            if best:
                torch.save(savedict, self.cfg.ckp_best)

            # sample
            if sample_interval > 0 and epoch % sample_interval == 0:
                time_sm = time.time()
                x = self.sample(size).float() / self.cfg.input_dim
                x = x[:, None, :, :]
                fn = self.cfg.imp.format(epoch)
                img_grid = torchvision.utils.make_grid(x)
                self.writer.add_image(fn, img_grid)

                torchvision.utils.save_image(x, fn)
                time_sm = time.time() - time_sm
                print('gen {}, time_sm={:1f}s'.format(fn, time_sm))


class Config:
    input_dim = 256
    dim = 64
    n_layers = 15
    n_classes = 0

    def __init__(self, id='gatedpixelcnn'):
        self.ckp = 'results/' + id + '.pt'
        self.ckp_best = 'results/' + id + '_best.pt'
        self.imp = 'results/' + id + '_{:03d}.png'
        self.tbp = 'runs/' + id


def mnist_train():
    cfg = Config()
    solver = Solver(cfg)
    trans = transforms.Compose([transforms.ToTensor(), lambda x: (x[0] * 255).long()])
    tr, te = dataloader.load_mnist(select=8, batch=32, trans=trans)
    solver.train(tr, te)


def mnist_sample():
    cfg = Config()
    solver = Solver(cfg, try_load_best=True)
    x = solver.sample((32, 28, 28)).float() / cfg.input_dim
    x = x[:, None, :, :]
    utils.show_pics(x)


def facefeatures_train(real=False):
    if real:
        id = 'gatedpixelcnn_rff'
    else:
        id = 'gatedpixelcnn_ff'
    cfg = Config(id)
    cfg.input_dim = 512
    solver = Solver(cfg)

    if real:
        fn = "data/real_face_features.npz"
        batch = 32
    else:
        fn = "data/face_features.npz"
        batch = 64
    tr, te = dataloader.load_faces(fn=fn, batch=batch)
    solver.train(tr, te)


def facefeatures_sample(real=False):
    if real:
        id = 'gatedpixelcnn_rff'
        batch = 32
        w = 32
    else:
        id = 'gatedpixelcnn_ff'
        batch = 64
        w = 16
    cfg = Config(id)
    cfg.input_dim = 512
    solver = Solver(cfg, try_load_best=True)

    return solver.sample((batch, w, w))


if __name__ == "__main__":
    facefeatures_train(True)
