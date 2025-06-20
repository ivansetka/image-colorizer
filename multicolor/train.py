import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from discriminator import PatchDiscriminator
from imagenet import ImageNetDataset
from losses import MulticolorLoss, ColorfulnessLoss, AdversarialLoss, PerceptualLoss, PixelLoss
from model import MulticolorModel
from project.multicolor import config
from utils import transform, make_subset, load_checkpoint, save_checkpoint, save_random_examples


def train_epoch(model, discriminator, dataloader, optimizer, optimizer_D):
    model.train()
    discriminator.train()

    epoch_loss = 0.0
    criterion = MulticolorLoss(
        PixelLoss(module_weights=config.module_weights),
        PerceptualLoss(),
        AdversarialLoss(),
        ColorfulnessLoss()
    ).to(config.DEVICE)

    for i, original_img in enumerate(tqdm(dataloader)):
        lab, hsv, yuv = (img.to(config.DEVICE) for img in original_img)
        target_channels = [
            lab[:, 1:, :, :], hsv[:, :2, :, :], yuv[:, 1:, :, :]
        ]

        L = lab[:, :1, :, :].to(config.DEVICE)
        generated_img, predicted_channels = model(L)

        real_output = discriminator(lab)
        fake_output = discriminator(generated_img.detach())
        dis_loss = criterion.discriminator_loss(real_output, fake_output)

        optimizer_D.zero_grad()
        dis_loss.backward()
        optimizer_D.step()

        fake_output = discriminator(generated_img)
        loss = criterion.loss(
            generated_img,
            lab,
            fake_output,
            predicted_channels,
            target_channels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Average Loss: {avg_epoch_loss:.4f}")


if __name__ == "__main__":
    model = MulticolorModel(
        features_dim=config.features_dim,
        module_names=config.module_names,
        cscnet_dim=config.cscnet_dim,
    ).to(config.DEVICE)
    discriminator = PatchDiscriminator().to(config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)

    train_dataset = ImageNetDataset(config.TRAIN_DIR, transformation=transform, train=True)
    test_dataset = ImageNetDataset(config.TEST_DIR, transformation=transform, train=False)

    train_dataset = make_subset(train_dataset, subset_ratio=0.035)
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )

    start_epoch = 0
    if config.LOAD_MODEL:
        start_epoch = load_checkpoint(model, discriminator, optimizer, optimizer_D)

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        train_epoch(model, discriminator, dataloader, optimizer, optimizer_D)

        if config.SAVE_MODEL:
            save_checkpoint(epoch, model, discriminator, optimizer, optimizer_D)

        save_random_examples(model, test_dataset, epoch)
