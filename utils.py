import torch
import torch.nn as nn

import torchvision
import matplotlib.pyplot as plt


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channel, height, width):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


def print_params(net, verbose=False):
    numel = 0
    for n, p in net.named_parameters():
        if verbose:
            print(n.ljust(20), "\t", p.size())
        numel = numel + p.numel()
    print("Parameter Count: ", numel)


def show_pics(pics, normalize=False, range=None):
    grid = torchvision.utils.make_grid(pics, normalize=normalize, range=range)
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    plt.imshow(ndarr)
    plt.show()


def random_choices(dataset, k=8):
    import random
    idxs = random.choices(range(len(dataset)), k=k)
    return torch.stack([dataset[i][0] for i in idxs])


def first_batch(dataloader):
    return next(iter(dataloader))[0]


if __name__ == "__main__":
    import dataloader

    face = dataloader.load_faces()[0]
    show_pics(random_choices(face.dataset, 64))
