from torch import nn

from cnn.networks import (
    LowLevelFeaturesNetwork,
    MidLevelFeaturesNetwork,
    GlobalFeaturesNetwork,
    FusionBlock,
    ColorizationNetwork
)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.low_net = LowLevelFeaturesNetwork()
        self.mid_net = MidLevelFeaturesNetwork()
        self.global_net = GlobalFeaturesNetwork()
        self.fusion_block = FusionBlock()
        self.colorization_net = ColorizationNetwork()

    def forward(self, x):
        x_low = self.low_net(x)
        x_mid = self.mid_net(x_low)
        x_global, logits = self.global_net(x_low)
        x_fused = self.fusion_block(x_mid, x_global)

        return self.colorization_net(x_fused), logits
