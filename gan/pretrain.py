import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from generator import PretrainedGeneratorWrapper
from imagenet import ImageNetDataset
from project.gan import config
from utils import transform, make_subset


def _pretrain_generator(generator, dataloader, optimizer, criterion):
    generator.train()
    epoch_loss = 0.

    for i, original_img in enumerate(tqdm(dataloader)):
        L = original_img[:, :1, :, :].to(config.DEVICE)
        ab = original_img[:, 1:, :, :].to(config.DEVICE)

        generated_ab = generator(L)
        loss = criterion(generated_ab, ab)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Average Loss: {avg_epoch_loss:.4f}")


if __name__ == "__main__":
    generator_wrapper = PretrainedGeneratorWrapper(
        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        device=config.DEVICE
    )
    generator = generator_wrapper.generator

    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    train_dataset = ImageNetDataset(config.TRAIN_DIR, transformation=transform, train=True)
    train_dataset = make_subset(train_dataset, subset_ratio=0.05)

    dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )

    for epoch in range(config.NUM_EPOCHS):
        _pretrain_generator(generator, dataloader, optimizer, criterion)
        torch.save(generator.state_dict(), config.PRETRAINED_GENERATOR_PATH)
