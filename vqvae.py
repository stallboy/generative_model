# 参考 https://github.com/ritheshkumar95/pytorch-vqvae

import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Function
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import dataloader
import utils


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                                    inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


vq = VectorQuantization.apply


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
                                           index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                   .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


vq_st = VectorQuantizationStraightThrough.apply


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

    @staticmethod
    def loss(x, x_tilde, z_e_x, z_q_x):
        loss_recons = F.mse_loss(x_tilde, x)  # Reconstruction loss
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())  # Vector quantization objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())  # Commitment objective
        beta = 0.25
        return loss_recons + loss_vq + beta * loss_commit, loss_recons, loss_vq


class Solver(object):
    def __init__(self, cfg, try_load_best=False):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VectorQuantizedVAE(cfg.input_dim, cfg.dim, K=cfg.K).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        utils.print_params(self.model)

        self.writer = SummaryWriter(cfg.tbp)

        self.last_epoch = 0
        self.best_loss = None
        ckp = cfg.ckp
        if try_load_best and os.path.isfile(cfg.ckp_best):
            ckp = cfg.ckp_best

        if os.path.isfile(ckp):
            checkpoint = torch.load(ckp)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            print('Load {}! Epoch: {} Best loss: {:.4f}'.format(ckp, self.last_epoch, self.best_loss))

    def train(self, train_loader, test_loader, epochs=50):
        fixed_images = utils.first_batch(test_loader)
        # self.writer.add_graph(self.model, fixed_images.to(self.device))
        img_grid = torchvision.utils.make_grid(fixed_images[:32], range=(-1, 1), normalize=True)
        self.writer.add_image('original', img_grid)

        # z = self.model.encode(fixed_images.to(self.device))
        # print(z.size())
        for epoch in range(self.last_epoch + 1, epochs + self.last_epoch + 1):
            # train
            self.model.train()
            train_loss = 0
            for batch_idx, d in enumerate(train_loader):
                data = d[0].to(self.device)

                self.optimizer.zero_grad()
                x_tilde, z_e_x, z_q_x = self.model(data)
                loss, loss_recons, loss_vq = VectorQuantizedVAE.loss(data, x_tilde, z_e_x, z_q_x)
                loss.backward()
                self.optimizer.step()

                loss_v = loss.item()
                loss_recons_v = loss_recons.item()
                loss_vq_v = loss_vq.item()
                train_loss += loss_v
                x_tilde = None
                z_e_x = None
                z_q_x = None
                loss = None
                loss_recons = None
                loss_vq = None

                # Logs
                steps = (epoch - 1) * len(train_loader) + batch_idx
                self.writer.add_scalar('loss/train/reconstruction', loss_recons_v, steps)
                self.writer.add_scalar('loss/train/quantization', loss_vq_v, steps)
                self.writer.add_scalar('loss/train_loss', loss_v, steps)

                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                               loss_v / len(data)))

            train_loss = train_loss / len(train_loader)
            print('====> Epoch: {} Train set loss: {:.4f}'.format(epoch, train_loss))
            self.writer.add_scalar('epoch_loss/train_loss', train_loss, epoch)

            # test
            test_loss, choose_orig, choose_recon = self.test(test_loader)
            print('====> Test set loss: {:.4f}'.format(test_loss))
            self.writer.add_scalar('epoch_loss/test_loss', test_loss, epoch)
            fn = self.cfg.imp.format(epoch)
            self.writer.add_image(fn, torchvision.utils.make_grid(choose_recon[:32], range=(-1, 1), normalize=True))
            torchvision.utils.save_image(choose_recon[:32], fn, range=(-1, 1), normalize=True)
            choose_orig = None
            choose_recon = None

            # checkpoint
            best = False
            if self.best_loss is None or test_loss < self.best_loss:
                self.best_loss = test_loss
                best = True
            savedict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
            }
            torch.save(savedict, self.cfg.ckp)
            if best:
                torch.save(savedict, self.cfg.ckp_best)

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        choose_orig = None
        choose_recon = None
        with torch.no_grad():
            for i, d in enumerate(test_loader):
                data_cpu = d[0]
                data = data_cpu.to(self.device)
                x_tilde, z_e_x, z_q_x = self.model(data)
                loss, _, _ = VectorQuantizedVAE.loss(data, x_tilde, z_e_x, z_q_x)
                test_loss += loss.item()
                if i == 0:
                    choose_orig = data_cpu
                    choose_recon = x_tilde.cpu()

        test_loss /= len(test_loader)
        return test_loss, choose_orig, choose_recon

    def reconstruct(self, x):
        self.model.eval()
        with torch.no_grad():
            x_tilde, z_e_x, z_q_x = self.model(x.to(self.device))
            return x_tilde.cpu()

    def gen_features(self, loaders):
        self.model.eval()
        latents_array = []
        with torch.no_grad():
            for loader in loaders:
                for d in loader:
                    data_cpu = d[0]
                    data = data_cpu.to(self.device)
                    latents = self.model.encode(data)  # [B, H, W]
                    latents_array.append(latents.cpu())
                    latents = None

        return torch.cat(latents_array)

    def sample(self, z):
        self.model.eval()
        with torch.no_grad():
            x = self.model.decode(z.to(self.device))
            return x.cpu()


class Config:
    input_dim = 3
    dim = 64
    K = 512

    def __init__(self, id='vqvae'):
        self.ckp = 'results/' + id + '.pt'
        self.ckp_best = 'results/' + id + '_best.pt'
        self.imp = 'results/' + id + '_{:03d}.png'
        self.tbp = 'runs/' + id


def _cfg(real):
    if real:
        return Config('vqvaer')
    else:
        return Config()


def _load(real):
    if real:
        return dataloader.load_real_faces(batch=64)
    else:
        return dataloader.load_faces(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))


def face_train(real=False):
    solver = Solver(_cfg(real))
    tr, te = _load(real)
    solver.train(tr, te, 50)


def face_reconstruct(real=False):
    solver = Solver(_cfg(real), try_load_best=True)
    tr, te = _load(real)
    x = utils.random_choices(te.dataset, k=32)
    x_tilde = solver.reconstruct(x)
    utils.show_pics(torch.cat([x, x_tilde]), normalize=True, range=(-1, 1))


def face_gen_features(real=False):
    solver = Solver(_cfg(real), try_load_best=True)
    tr, te = _load(real)
    features = solver.gen_features([te, tr])
    print(features.size())
    if real:
        fn = "data/real_face_features.npz"
    else:
        fn = "data/face_features.npz"
    np.savez_compressed(fn, images=features.numpy())


def face_sample(real=False):
    import gatedpixelcnn
    z = gatedpixelcnn.facefeatures_sample(real)
    solver = Solver(_cfg(real), try_load_best=True)
    x = solver.sample(z)
    if real:
        fn = "results/vqvaer_face_sample_1.png"
    else:
        fn = "results/vqvae_face_sample_1.png"
    torchvision.utils.save_image(x, fn, range=(-1, 1), normalize=True)
    utils.show_pics(x, normalize=True, range=(-1, 1))


if __name__ == '__main__':
    face_sample(True)
