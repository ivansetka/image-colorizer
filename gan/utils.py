import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage import color
from torchvision.transforms import transforms

from project.gan import config


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
        original_img = original_img.to(config.DEVICE)
        L = original_img[:1, :, :].unsqueeze(0)

        with torch.no_grad():
            generated_ab = model(L)
            generated_img = torch.cat((L, generated_ab), dim=1)
            generated_img = generated_img.squeeze(0)

        original_rgb = lab2rgb(original_img.cpu().permute(1, 2, 0).numpy())

        ax = axs[i, 0]
        ax.imshow(original_rgb)
        ax.set_title("Original")
        ax.axis("off")

        generated_rgb = lab2rgb(generated_img.cpu().permute(1, 2, 0).numpy())

        ax = axs[i, 1]
        ax.imshow(generated_rgb)
        ax.set_title("GAN Model")
        ax.axis("off")

    plt.savefig(fname=f'{config.EXAMPLES_DIR}/epoch_{epoch}.png')


def save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D, scheduler_G, scheduler_D):
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_G.state_dict(),
        "optimizer_d_state_dict": optimizer_D.state_dict(),
        "scheduler_g_state_dict": scheduler_G.state_dict(),
        "scheduler_d_state_dict": scheduler_D.state_dict()
    }

    torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'))
    print(f"Checkpoint saved at epoch {epoch}.")


def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, scheduler_G, scheduler_D):
    checkpoint_files = glob.glob(os.path.join(config.CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
    start_epoch = 0

    if not checkpoint_files:
        print("No checkpoint found.")
        return start_epoch

    latest_checkpoint_filepath = max(checkpoint_files, key=os.path.getctime)
    checkpoint = torch.load(latest_checkpoint_filepath)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    optimizer_G.load_state_dict(checkpoint["optimizer_g_state_dict"])
    optimizer_D.load_state_dict(checkpoint["optimizer_d_state_dict"])
    scheduler_G.load_state_dict(checkpoint["scheduler_g_state_dict"])
    scheduler_D.load_state_dict(checkpoint["scheduler_d_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")

    return start_epoch
