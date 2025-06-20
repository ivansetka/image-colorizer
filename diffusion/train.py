import torch.optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion import Diffusion
from imagenet import ImageNetDataset
from model import DiffusionUNetModel, GrayscaleEncoder
from project.diffusion import config
from utils import transform, make_subset, load_checkpoint, save_checkpoint, save_random_examples


def train_epoch(model, grayscale_encoder, diffusion, dataloader, optimizer):
    model.train()

    epoch_loss = 0.0
    running_loss = 0.0

    accum_steps = 3
    criterion = nn.MSELoss()
    optimizer.zero_grad()

    for i, original_img in enumerate(tqdm(dataloader)):
        L = original_img[:, :1, :, :].to(config.DEVICE)
        ab = original_img[:, 1:, :, :].to(config.DEVICE)

        t = diffusion.sample_timesteps(ab.shape[0]).to(config.DEVICE)
        noisy_ab, noise = diffusion.noise_images(ab, t)
        noisy_image = torch.cat((L, noisy_ab), dim=1)

        grayscale_features = grayscale_encoder(L)
        predicted_noise = model(noisy_image, t, grayscale_features)

        loss = criterion(noise, predicted_noise) / accum_steps
        loss.backward()
        running_loss += loss.item()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += running_loss
            running_loss = 0.0

    avg_epoch_loss = epoch_loss / (len(dataloader) / accum_steps)
    print(f"Average Loss: {avg_epoch_loss:.4f}")


if __name__ == "__main__":
    diffusion = Diffusion(noise_steps=500)

    model = DiffusionUNetModel(config.layers, dropout=config.DROPOUT).to(config.DEVICE)
    grayscale_encoder = GrayscaleEncoder(config.layers, dropout=config.DROPOUT).to(config.DEVICE)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(grayscale_encoder.parameters()),
        lr=config.LEARNING_RATE
    )

    train_dataset = ImageNetDataset(config.TRAIN_DIR, transformation=transform, train=True)
    test_dataset = ImageNetDataset(config.TEST_DIR, transformation=transform, train=False)

    train_dataset = make_subset(train_dataset, subset_ratio=0.075)
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )

    start_epoch = 0
    if config.LOAD_MODEL:
        start_epoch = load_checkpoint(model, grayscale_encoder, optimizer)

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        train_epoch(model, grayscale_encoder, diffusion, dataloader, optimizer)

        if config.SAVE_MODEL:
            save_checkpoint(epoch, model, grayscale_encoder, optimizer)

        save_random_examples(model, grayscale_encoder, diffusion, test_dataset, epoch)
