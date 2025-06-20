import torch
import kornia
from torch import nn
from torchvision import models
from torchvision.models import VGG16_Weights


class PixelLoss(nn.Module):
    def __init__(self, module_weights, loss_weight=1., reduction='mean'):
        super(PixelLoss, self).__init__()
        self.loss_weight = loss_weight
        self.module_weights = module_weights
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, predictions, targets):
        pixel_loss = 0.
        for i, (prediction, target) in enumerate(zip(predictions, targets)):
            pixel_loss += self.module_weights[i] * self.loss(prediction, target)

        return self.loss_weight * pixel_loss


class PerceptualLoss(nn.Module):
    def __init__(self, loss_weight=5., layer_weights=(0.0625, 0.125, 0.25, 0.5, 1.)):
        super(PerceptualLoss, self).__init__()
        self.loss_weight = loss_weight
        self.layer_weights = layer_weights
        self.loss = nn.L1Loss()

        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features

        self.block_layers = (1, 6, 11, 18, 25)
        self.blocks = nn.ModuleList()
        prev_index = 0

        for index in self.block_layers:
            self.blocks.append(
                nn.Sequential(*vgg[prev_index:index + 1])
            )
            prev_index = index + 1

        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, prediction, target):
        prediction = (prediction - self.mean) / self.std
        target = (target - self.mean) / self.std

        perceptual_loss = 0.0
        for i, block in enumerate(self.blocks):
            prediction, target = block(prediction), block(target)
            perceptual_loss += self.layer_weights[i] * self.loss(prediction, target)

        return self.loss_weight * perceptual_loss


class AdversarialLoss(nn.Module):
    def __init__(self, loss_weight=1.):
        super(AdversarialLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, real_output, fake_output, is_discriminator=True):
        if is_discriminator:
            dis_ce_original = self.loss(real_output, 0.9 * torch.ones_like(real_output))
            dis_ce_generated = self.loss(fake_output, torch.zeros_like(fake_output))

            return dis_ce_original + dis_ce_generated

        return self.loss_weight * self.loss(fake_output, torch.ones_like(fake_output))


class ColorfulnessLoss(nn.Module):
    def __init__(self, loss_weight=0.5):
        super(ColorfulnessLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, prediction):
        L = (prediction[:, :1, :, :] + 1.0) * 50.0
        ab = prediction[:, 1:, :, :] * 128.0

        lab = torch.cat([L, ab], dim=1)
        rgb = kornia.color.lab_to_rgb(lab)

        rg = torch.abs(rgb[:, 0] - rgb[:, 1])
        yb = torch.abs(0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2])

        rg_std, rg_mean = torch.std(rg, dim=(1, 2)), torch.mean(rg, dim=(1, 2))
        yb_std, yb_mean = torch.std(yb, dim=(1, 2)), torch.mean(yb, dim=(1, 2))

        std_root = torch.sqrt(rg_std ** 2 + yb_std ** 2)
        mean_root = torch.sqrt(rg_mean ** 2 + yb_mean ** 2)
        colorfulness_loss = 1.0 - (std_root + 0.3 * mean_root).mean()

        return self.loss_weight * colorfulness_loss


class MulticolorLoss(nn.Module):
    def __init__(self, pixel_loss, perceptual_loss, adversarial_loss, colorfulness_loss):
        super(MulticolorLoss, self).__init__()
        self.pixel_loss = pixel_loss
        self.perceptual_loss = perceptual_loss
        self.adversarial_loss = adversarial_loss
        self.colorfulness_loss = colorfulness_loss

    def loss(self, prediction, target, fake_output, predicted_channels, target_channels):
        return (
            self.pixel_loss(predicted_channels, target_channels) +
            self.perceptual_loss(prediction, target) +
            self.adversarial_loss(None, fake_output, is_discriminator=False) +
            self.colorfulness_loss(prediction)
        )

    def discriminator_loss(self, real_output, fake_output):
        return self.adversarial_loss(real_output, fake_output, is_discriminator=True)
