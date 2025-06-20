import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage import color
from torchvision.transforms import transforms
from tqdm import tqdm

from project.diffusion import config


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


def save_random_examples(model, grayscale_encoder, diffusion, dataset, epoch, n=7):
    original_images = dataset.get_n_random(n)

    model.eval()
    fig, axs = plt.subplots(n, 2, figsize=(18, 9 * n))

    original_images = torch.stack(original_images, dim=0)
    L = original_images[:, :1, :, :].to(config.DEVICE)
    sampled_images = sample(model, grayscale_encoder, diffusion, L)

    for i in range(n):
        original_img = original_images[i]
        sampled_image = sampled_images[i]

        original_rgb = lab2rgb(original_img.cpu().permute(1, 2, 0).numpy())

        ax = axs[i, 0]
        ax.imshow(original_rgb)
        ax.set_title("Original")
        ax.axis("off")

        generated_rgb = lab2rgb(sampled_image.cpu().permute(1, 2, 0).numpy())

        ax = axs[i, 1]
        ax.imshow(generated_rgb)
        ax.set_title("Diffusion Model")
        ax.axis("off")

    plt.savefig(fname=f'{config.EXAMPLES_DIR}/epoch_{epoch}.png')


def sample(model, grayscale_encoder, diffusion, L):
    model.eval()
    B, _, H, W = L.shape

    with torch.no_grad():
        ab = torch.randn(B, 2, H, W).to(config.DEVICE)

        for i in tqdm(range(diffusion.noise_steps - 1, 0, -1)):
            t = torch.full(size=(B,), fill_value=i, dtype=torch.long).to(config.DEVICE)
            noisy_image = torch.cat((L, ab), dim=1)
            grayscale_features = grayscale_encoder(L)

            predicted_noise = model(noisy_image, t, grayscale_features)

            beta_t = diffusion.beta[t].view(-1, 1, 1, 1)
            alpha_t = diffusion.alpha[t].view(-1, 1, 1, 1)
            alpha_hat_t = diffusion.alpha_hat[t].view(-1, 1, 1, 1)

            if i > 1:
                noise = torch.randn_like(ab)
            else:
                noise = torch.zeros_like(ab)

            ab = (1 / torch.sqrt(alpha_t)) * (
                    ab - (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t) * predicted_noise
            ) + torch.sqrt(beta_t) * noise

    return torch.cat((L, ab), dim=1)


def save_checkpoint(epoch, model, grayscale_encoder, optimizer):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": grayscale_encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'))
    print(f"Checkpoint saved at epoch {epoch}.")


def load_checkpoint(model, grayscale_encoder, optimizer):
    checkpoint_files = glob.glob(os.path.join(config.CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
    start_epoch = 0

    if not checkpoint_files:
        print("No checkpoint found.")
        return start_epoch

    latest_checkpoint_filepath = max(checkpoint_files, key=os.path.getctime)
    checkpoint = torch.load(latest_checkpoint_filepath)

    model.load_state_dict(checkpoint["model_state_dict"])
    grayscale_encoder.load_state_dict(checkpoint["encoder_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")

    return start_epoch
