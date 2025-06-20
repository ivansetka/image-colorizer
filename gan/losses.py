import torch
import torch.nn.functional as F

from gan import config


def generator_loss(fake_output, generated_img, original_img):
    gen_ce = F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))
    l1_loss = F.l1_loss(generated_img, original_img)

    return gen_ce + config.L1_LAMBDA * l1_loss


def discriminator_loss(real_output, fake_output):
    dis_ce_original = F.binary_cross_entropy_with_logits(real_output, 0.9 * torch.ones_like(real_output))
    dis_ce_generated = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))

    return dis_ce_original + dis_ce_generated
