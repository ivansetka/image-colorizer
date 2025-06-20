import os

import torch
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from torch import nn
from torchvision.models.resnet import resnet18, ResNet18_Weights


class _GeneratorContractingBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, padding, stride, negative_slope=0.2, normalize=True, dropout=0):
        super(_GeneratorContractingBlock, self).__init__()
        self.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=padding))
        self.append(nn.LeakyReLU(negative_slope, inplace=True))

        if normalize:
            self.insert(1, nn.BatchNorm2d(out_channels))

        if dropout:
            self.append(nn.Dropout(dropout))


class _GeneratorExpansiveBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0):
        super(_GeneratorExpansiveBlock, self).__init__()
        self.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)),
        self.append(nn.ReLU(inplace=True)),

        if normalize:
            self.insert(1, nn.BatchNorm2d(out_channels))

        if dropout:
            self.append(nn.Dropout(dropout))


class Generator(nn.Module):
    def __init__(self, contracting_layers, expansive_layers):
        super(Generator, self).__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        in_channels = 1
        for index, (out_channels, dropout) in enumerate(contracting_layers):
            block = _GeneratorContractingBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                padding='same' if index == 0 else 1,
                stride=1 if index == 0 else 2,
                dropout=dropout
            )
            self.down.append(block)
            in_channels = out_channels

        for out_channels, dropout in expansive_layers:
            block = _GeneratorExpansiveBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout
            )
            self.up.append(block)
            in_channels = out_channels * 2

        self.final = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding="same"),
            nn.Tanh()
        )

    def forward(self, x):
        outputs = []
        for down in self.down:
            x = down(x)
            outputs.append(x)

        for index, up in enumerate(self.up):
            x = up(x)
            x_mirror = outputs[len(outputs) - index - 2]
            x = torch.cat([x, x_mirror], dim=1)

        return self.final(x)


class PretrainedGeneratorWrapper:
    def __init__(self, image_size, weights_path=None, n_input=1, n_output=2, device='cuda'):
        super(PretrainedGeneratorWrapper, self).__init__()
        self.device = device
        self.generator = DynamicUnet(
            encoder=create_body(resnet18(weights=ResNet18_Weights.DEFAULT), n_in=n_input, cut=-2),
            n_out=n_output,
            img_size=image_size
        ).to(device)

        if weights_path is not None:
            self.load_weights(weights_path)

    def load_weights(self, weights_path):
        if not os.path.exists(weights_path):
            raise RuntimeError("Failed to load weights: invalid file path!")

        self.generator.load_state_dict(torch.load(weights_path, map_location=self.device))
