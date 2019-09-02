import os
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F

import dataloader
import utils

from torch.utils.tensorboard import SummaryWriter


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        h_dim = 1000
        z_dim = 100

        self.encoder = nn.Sequential(
            # 32*32*64 -》 16*16*128 -》8*8*256 -》 4*4*512
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            utils.Flatten(),
            nn.Linear(512 * 4 * 4, h_dim),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),

            nn.Linear(h_dim, 512 * 4 * 4),
            nn.ReLU(),
            utils.Unflatten(512, 4, 4),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = self.fc1(x), self.fc2(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = self.decode(z)
        return x, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


class Solver(object):
    def __init__(self):
        self.ckp = 'results/vae.pt'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        utils.print_params(self.model)

        self.writer = SummaryWriter('runs/vae')

        self.last_epoch = 0
        if os.path.isfile(self.ckp):
            checkpoint = torch.load(self.ckp)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print('Load checkpoint! Last Epoch: {} Average loss: {:.4f}'.format(self.last_epoch, loss))

    def train(self, epochs):
        train_loader, test_loader = dataloader.load_faces()
        images = utils.first_batch(test_loader)
        self.writer.add_graph(self.model, images.to(self.device))
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image('original', img_grid)

        for epoch in range(self.last_epoch + 1, epochs + self.last_epoch + 1):
            # train
            self.model.train()
            train_loss = 0
            for batch_idx, d in enumerate(train_loader):
                data = d[0].to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                loss = loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                step = (epoch-1) * len(train_loader) + batch_idx
                self.writer.add_scalar('loss/train/batch', loss.item(), step)

                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                               loss.item() / len(data)))

            train_loss = train_loss / len(train_loader)
            self.writer.add_scalar('loss/train/epoch', train_loss, epoch)
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))


            # test
            test_loss, choose_orig, choose_recon, choose_z = self.test(test_loader)
            print('====> Test set loss: {:.4f}'.format(test_loss))
            self.writer.add_scalar('test loss', test_loss, epoch)

            sample_data, sample_z = self.sample()
            img = torch.cat([choose_orig[:8], choose_recon[:8], sample_data])
            self.writer.add_image('vae_%03d' % epoch, torchvision.utils.make_grid(img))
            torchvision.utils.save_image(img, 'results/vae_%03d.png' % epoch, nrow=8)

            # projector
            if epoch % 10 == 0:
                features = torch.cat([choose_z, sample_z])
                imgs = torch.cat([choose_orig, sample_data])
                self.writer.add_embedding(features, label_img=imgs, tag='vae_%03d' % epoch)

            # checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': test_loss,
            }, self.ckp)

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        choose_orig = None
        choose_recon = None
        choose_z = None
        with torch.no_grad():
            for i, d in enumerate(test_loader):
                data_cpu = d[0]
                data = data_cpu.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += loss_function(recon_batch, data, mu, logvar).item()
                if i == 0:
                    choose_orig = data_cpu
                    choose_recon = recon_batch.view(-1, 3, 64, 64).cpu()
                    choose_z = mu.cpu()

        test_loss /= len(test_loader)
        return test_loss, choose_orig, choose_recon, choose_z

    def sample(self):
        self.model.eval()
        with torch.no_grad():
            sample_z = torch.randn(64, 100)
            z = sample_z.to(self.device)
            sample_data = self.model.decode(z).cpu().view(64, 3, 64, 64)
            return sample_data, sample_z


solver = Solver()

is_infer = False
if is_infer:
    sample_data, _ = solver.sample()
    utils.show_pics(sample_data)
else:
    solver.train(30)
