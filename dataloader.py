import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as tdata


class SingleTensorDataset(tdata.Dataset):
    def __init__(self, tensor, trans=None):
        self.tensor = tensor
        self.trans = trans

    def __getitem__(self, index):
        t = self.tensor[index]
        if self.trans:
            t = self.trans(t)
        return (t,)

    def __len__(self):
        return self.tensor.size(0)


def load_faces(trans=None, batch=128, fn="data/faces64_full.npz"):
    result = np.load(fn)
    x_all = result["images"]
    x_all = torch.from_numpy(x_all)

    split = int(x_all.shape[0] * 19 / 20)
    x_train = SingleTensorDataset(x_all[:split], trans)
    x_test = SingleTensorDataset(x_all[split:], trans)

    train_loader = tdata.DataLoader(x_train, batch_size=batch, shuffle=True)
    test_loader = tdata.DataLoader(x_test, batch_size=batch)

    return train_loader, test_loader


def load_mnist(select=8, batch=128, trans=None):
    if trans is None:
        trans = transforms.ToTensor()
    d = torchvision.datasets.MNIST('data', train=True, download=True, transform=trans)
    if select > -1:
        idx = d.targets == select
        d.data = d.data[idx]
        d.targets = d.targets[idx]
    tr = tdata.DataLoader(d, batch_size=batch, shuffle=True)

    d = torchvision.datasets.MNIST('data', train=False, download=True, transform=trans)
    if select > -1:
        idx = d.targets == select
        d.data = d.data[idx]
        d.targets = d.targets[idx]
    te = tdata.DataLoader(d, batch_size=batch, shuffle=False)
    return tr, te


def load_real_faces(batch=32, root="D:\\MyDocuments\\GitHub\\download\\realfaces_ffhq128\\"):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_ds = torchvision.datasets.ImageFolder(root=root + "train", transform=trans)
    test_ds = torchvision.datasets.ImageFolder(root=root + "test", transform=trans)

    tr = tdata.DataLoader(train_ds, batch_size=batch, shuffle=True)
    te = tdata.DataLoader(test_ds, batch_size=batch)
    return tr, te
