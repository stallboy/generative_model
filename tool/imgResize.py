import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def save():
    sz = 64
    dataset = torchvision.datasets.ImageFolder(root="D:\\BaiduNetdiskDownload\\faces",
                                               transform=transforms.Compose([
                                                   transforms.Resize(sz),
                                                   transforms.ToTensor(),
                                                   # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ]))

    print(len(dataset))
    print(dataset[0][0].size())

    cnt = len(dataset)
    res = torch.empty(cnt, 3, sz, sz)
    for i, d in enumerate(dataset):
        res[i] = d[0]
        # if i == cnt - 1:
        #     break

    print(res.size())

    np.savez_compressed("../data/faces64_full.npz", images=res.numpy())


import os
import PIL.Image as Image


def resize():
    inputPath = "D:/BaiduNetdiskDownload/faces/anime/"
    imgList = [f for f in os.listdir( inputPath) if os.path.splitext(f)[1] == ".jpg"]
    nImgs = len(imgList)
    print(nImgs)

    outputPath = "D:/BaiduNetdiskDownload/faces64/"
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)

    for index, item in enumerate(imgList):
        if index % 200 == 0:
            print(index)
        path = os.path.join(inputPath, item)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB').resize((64,64), resample = Image.BILINEAR)
            path = os.path.join(outputPath, item)
            img.save(path)


def load():
    l = np.load("../data/faces64_full.npz")
    imgs = l["images"]
    print(imgs.shape)


resize()
