import torch.nn as nn


class _DiscriminatorBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride, negative_slope=0.2, normalize=True, dropout=0):
        super(_DiscriminatorBlock, self).__init__()
        self.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1))
        self.append(nn.LeakyReLU(negative_slope, inplace=True))

        if normalize:
            self.insert(1, nn.BatchNorm2d(out_channels))

        if dropout:
            self.append(nn.Dropout(dropout))


class Discriminator(nn.Module):
    def __init__(self, layers):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList()

        in_channels = 3
        for index, (out_channels, dropout) in enumerate(layers):
            block = _DiscriminatorBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1 if index == len(layers) - 1 else 2,
                normalize=index != 0,
                dropout=dropout
            )
            self.layers.append(block)
            in_channels = out_channels

        self.final = nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.final(x)
