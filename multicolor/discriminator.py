from torch import nn


class PatchDiscriminator(nn.Module):
    def __init__(self, layers_dim=(64, 128, 256, 512)):
        super(PatchDiscriminator, self).__init__()
        self.layers = nn.ModuleList()

        in_channels = 3
        for index, out_channels in enumerate(layers_dim):
            stride = 1 if index == len(layers_dim) - 1 else 2
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
                    nn.BatchNorm2d(out_channels) if index != 0 else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = out_channels

        self.final = nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return self.final(x)
