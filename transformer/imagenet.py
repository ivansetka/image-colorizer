import random

import torchvision
from torch.utils.data import Dataset

from utils import rgb2lab


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transformation=None, train=True):
        split = "train" if train else "val"
        self.dataset = torchvision.datasets.ImageNet(
            root=root_dir,
            split=split
        )
        self.transformation = transformation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        if self.transformation:
            image = self.transformation(image)

        return rgb2lab(image).permute(2, 0, 1)

    def get_n_random(self, n):
        if n > len(self):
            raise RuntimeError("Parameter n is larger than number of elements!")

        indices = random.sample(range(len(self)), n)
        return [self.__getitem__(idx) for idx in indices]
