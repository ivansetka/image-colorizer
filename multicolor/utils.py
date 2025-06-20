import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage import color
from torchvision.transforms import transforms

from multicolor import config


def rgb2lab(image):
    image = color.rgb2lab(image)

    L = torch.tensor(image[:, :, :1] / 50.0 - 1.0, dtype=torch.float32)
    ab = torch.tensor(image[:, :, 1:] / 128.0, dtype=torch.float32)

    return torch.cat((L, ab), dim=2)


def lab2rgb(image):
    image[:, :, :1] = (image[:, :, :1] + 1.0) * 50.0
    image[:, :, 1:] = image[:, :, 1:] * 128.0
    image = image.astype(np.float64)

    return color.lab2rgb(image)


def rgb2hsv(image):
    image = image.copy()
    image = np.asarray(image)
    image = color.rgb2hsv(image)

    return torch.tensor(image, dtype=torch.float32)


def hsv2rgb(image):
    image = image.copy()
    return color.hsv2rgb(image)


def rgb2yuv(image):
    image = image.copy()
    image = color.rgb2yuv(image)

    y = torch.tensor(image[:, :, :1], dtype=torch.float32)
    u = torch.tensor(image[:, :, 1:2] / 0.436, dtype=torch.float32)
    v = torch.tensor(image[:, :, 2:] / 0.615, dtype=torch.float32)

    return torch.cat((y, u, v), dim=2)


def yuv2rgb(image):
    image = image.copy()
    image[:, :, 1:2] = image[:, :, 1:2] * 0.436
    image[:, :, 2:] = image[:, :, 2:] * 0.615
    image = image.astype(np.float64)

    return color.yuv2rgb(image)


def make_subset(dataset, subset_ratio):
    nb_samples = len(dataset)
    indices = torch.randperm(nb_samples)[:int(subset_ratio * nb_samples)]

    return torch.utils.data.Subset(dataset, indices=indices)


def transform(image):
    transformation = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    ])

    return transformation(image)


def save_random_examples(model, dataset, epoch, n=7):
    original_images = dataset.get_n_random(n)

    model.eval()
    fig, axs = plt.subplots(n, 2, figsize=(18, 9 * n))

    for i, original_img in enumerate(original_images):
        lab, _, _ = (img for img in original_img)
        lab = lab.to(config.DEVICE)
        L = lab[:1, :, :].unsqueeze(0)

        with torch.no_grad():
            generated_img, _ = model(L)
            generated_img = generated_img.squeeze(0)

        original_rgb = lab2rgb(lab.cpu().permute(1, 2, 0).numpy())

        ax = axs[i, 0]
        ax.imshow(original_rgb)
        ax.set_title("Original")
        ax.axis("off")

        generated_rgb = generated_img.cpu().permute(1, 2, 0).numpy()

        ax = axs[i, 1]
        ax.imshow(generated_rgb)
        ax.set_title("Multicolor Model")
        ax.axis("off")

    plt.savefig(fname=f'{config.EXAMPLES_DIR}/epoch_{epoch}.png')


def save_checkpoint(epoch, model, discriminator, optimizer, optimizer_D):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_d_state_dict": optimizer_D.state_dict()
    }

    torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'))
    print(f"Checkpoint saved at epoch {epoch}.")


def load_checkpoint(model, discriminator, optimizer, optimizer_D):
    checkpoint_files = glob.glob(os.path.join(config.CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
    start_epoch = 0

    if not checkpoint_files:
        print("No checkpoint found.")
        return start_epoch

    latest_checkpoint_filepath = max(checkpoint_files, key=os.path.getctime)
    checkpoint = torch.load(latest_checkpoint_filepath)

    model.load_state_dict(checkpoint["model_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer_D.load_state_dict(checkpoint["optimizer_d_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")

    return start_epoch
