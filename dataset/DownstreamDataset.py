from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
import torch
import numpy as np
from torchvision import transforms
import random
import torch_geometric
from sklearn.preprocessing import OneHotEncoder



class DownstreamDataset(torch_geometric.data.Dataset):
    def __init__(self, root, file_names):
        super(DownstreamDataset, self).__init__()
        self.root=root
        self.file_names = file_names

        self.resize = transforms.Resize(224)
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    def get(self, item):
        file_name=self.file_names[item]

        img = Image.open(os.path.join(self.root, 'img', file_name[:-2] + 'png'))
        img = self.random_aug(img)
        graph = torch.load(os.path.join(self.root, 'graph', file_name), weights_only=False)

        # add to setting labels
        # label = Image.open(os.path.join(self.root, 'label1', file_name[:-2] + 'png'))
        # label = torch.from_numpy(np.array(label)).long()

        return img, graph

    def len(self):
        return len(self.file_names)

    def random_aug(self, img):
        img = self.resize(img)
        img = self.toTensor(img)
        img = self.normalize(img)

        return img




class DownstreamDataset_UFZ(DownstreamDataset):

    def get(self, item):
        file_name=self.file_names[item]

        img = Image.open(os.path.join(self.root, 'img', file_name[:-2] + 'png'))
        img = self.random_aug(img)
        graph = torch.load(os.path.join(self.root, 'graph', file_name), weights_only=False)

        # add to setting labels
        label = Image.open(os.path.join(self.root, 'label', file_name[:-2] + 'png'))
        label = torch.from_numpy(np.array(label)).long()

        return img, graph, label

    def classnames(self):
        return ['water','green', 'farmland', 'undev', 'resdi', 'commercial', 'institution', 'ind', 'trans']
