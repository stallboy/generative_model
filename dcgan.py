import os

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision

from torch.utils.tensorboard import SummaryWriter
import utils
import dataloader

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Hout = (Hin − 1) ×  stride[0] − 2 ×  padding[0] +  dilation[0] × ( kernel_size[0] − 1) +  output_padding[0] + 1
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


writer = SummaryWriter('runs/dcgan')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netG.apply(weights_init)

netD = Discriminator().to(device)
netD.apply(weights_init)

utils.print_params(netG)
utils.print_params(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
lr = 0.0002
beta1 = 0.5
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

last_epoch = 0
is_read_checkpoint = True
is_infer = False

checkpoint_path = "results/dcgan.pt"
if is_read_checkpoint and os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    netG.load_state_dict(checkpoint['netG'])
    netD.load_state_dict(checkpoint['netD'])
    optimizerG.load_state_dict(checkpoint['optG'])
    optimizerD.load_state_dict(checkpoint['optD'])
    last_epoch = checkpoint['epoch']
    print('Load checkpoint! Last Epoch: %d' % last_epoch)

if is_infer:
    fake = netG(fixed_noise)
    fake = fake.detach().cpu()
    utils.show_pics(fake, normalize=True, range=(-1, 1))
else:
    normalizer = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_loader, test_loader = dataloader.load_faces(normalizer)

    # graph,现在tensorboard似乎好像不能支持2个graph
    noise = torch.randn(64, nz, 1, 1, device=device)
    writer.add_graph(netG, noise)
    fixed_images = utils.first_batch(test_loader)
    writer.add_graph(netD, fixed_images.to(device))

    # original img
    img_grid = torchvision.utils.make_grid(fixed_images, range=(-1, 1), normalize=True)
    writer.add_image('original', img_grid)

    epochs = 30
    for epoch in range(last_epoch + 1, epochs + last_epoch + 1):
        for i, data in enumerate(train_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0]
            real = real_cpu.to(device)
            batch_size = real.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())  # detach 断开graph
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # scalar
            global_step = (epoch - 1) * len(train_loader) + i
            writer.add_scalar('errD', errD.item(), global_step)
            writer.add_scalar('errG', errG.item(), global_step)
            writer.add_scalar('D_x', D_x, global_step)
            writer.add_scalar('D_G_z1', D_G_z1, global_step)
            writer.add_scalar('D_G_z2', D_G_z2, global_step)
            print('[%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        fake = netG(fixed_noise)
        fake = fake.detach().cpu()
        img = torch.cat([utils.random_choices(train_loader.dataset, k=8), fake])
        writer.add_image('dcgan_%03d' % epoch, torchvision.utils.make_grid(img, range=(-1, 1), normalize=True))
        torchvision.utils.save_image(img, 'results/dcgan_%03d.png' % epoch, range=(-1, 1), normalize=True)

        torch.save({
            'epoch': epoch,
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'optG': optimizerG.state_dict(),
            'optD': optimizerD.state_dict(),
        }, checkpoint_path)
