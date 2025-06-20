import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage import color
from torchvision.transforms import transforms

from project.cnn import config


def rgb2lab(image):
    image = color.rgb2lab(image)

    L = torch.tensor(image[:, :, :1] / 100.0, dtype=torch.float32)
    ab = torch.tensor((image[:, :, 1:] + 128.0) / 255.0, dtype=torch.float32)

    return torch.cat((L, ab), dim=2)


def lab2rgb(image):
    image[:, :, :1] = image[:, :, :1] * 100.0
    image[:, :, 1:] = image[:, :, 1:] * 255.0 - 128.0
    image = image.astype(np.float64)

    return color.lab2rgb(image)


def make_subset(dataset, subset_ratio):
    nb_samples = len(dataset)
    indices = torch.randperm(nb_samples)[:int(subset_ratio * nb_samples)]

    return torch.utils.data.Subset(dataset, indices=indices)


def transform(image):
    transformation = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5)
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
            generated_ab, _ = model(L)
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
        ax.set_title("CNN Model")
        ax.axis("off")

    plt.savefig(fname=f'{config.EXAMPLES_DIR}/epoch_{epoch}.png')


def save_checkpoint(epoch, model, optimizer):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'))
    print(f"Checkpoint saved at epoch {epoch}.")


def load_checkpoint(model, optimizer):
    checkpoint_files = glob.glob(os.path.join(config.CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
    start_epoch = 0

    if not checkpoint_files:
        print("No checkpoint found.")
        return start_epoch

    latest_checkpoint_filepath = max(checkpoint_files, key=os.path.getctime)
    checkpoint = torch.load(latest_checkpoint_filepath)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")

    return start_epoch
