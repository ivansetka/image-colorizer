import random

import torchvision
from torch.utils.data import Dataset

from utils import rgb2lab


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transformation=None, use_labels=False, train=True):
        split = "train" if train else "val"
        self.dataset = torchvision.datasets.ImageNet(
            root=root_dir,
            split=split
        )
        self.transformation = transformation
        self.use_labels = use_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if self.transformation:
            image = self.transformation(image)

        image = rgb2lab(image).permute(2, 0, 1)
        if self.use_labels:
            return image, label

        return image

    def get_n_random(self, n):
        if n > len(self):
            raise RuntimeError("Parameter n is larger than number of elements!")

        indices = random.sample(range(len(self)), n)
        return [self.__getitem__(idx) for idx in indices]
