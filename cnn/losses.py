import torch.nn.functional as F

from project.cnn import config


def colorization_loss(generated_img, original_img):
    return F.mse_loss(generated_img, original_img, reduction='sum')


def classification_loss(pred_class_logits, true_class):
    return config.ALPHA * F.cross_entropy(pred_class_logits, true_class)
