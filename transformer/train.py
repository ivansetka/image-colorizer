import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from discriminator import PatchDiscriminator
from imagenet import ImageNetDataset
from losses import TransformerLoss, ColorfulnessLoss, AdversarialLoss, PerceptualLoss, PixelLoss
from model import TransformerModel
from project.transformer import config
from utils import transform, make_subset, load_checkpoint, save_checkpoint, save_random_examples


def train_epoch(model, discriminator, dataloader, optimizer, optimizer_D):
    model.train()
    discriminator.train()

    epoch_loss = 0.0
    criterion = TransformerLoss(
        PixelLoss(),
        PerceptualLoss(),
        AdversarialLoss(),
        ColorfulnessLoss()
    ).to(config.DEVICE)

    for i, original_img in enumerate(tqdm(dataloader)):
        original_img = original_img.to(config.DEVICE)
        L = original_img[:, :1, :, :].to(config.DEVICE)

        generated_ab = model(L)
        generated_img = torch.cat((L, generated_ab), dim=1)

        real_output = discriminator(original_img)
        fake_output = discriminator(generated_img.detach())
        dis_loss = criterion.discriminator_loss(real_output, fake_output)

        optimizer_D.zero_grad()
        dis_loss.backward()
        optimizer_D.step()

        fake_output = discriminator(generated_img)
        loss = criterion.loss(generated_img, original_img, fake_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Average Loss: {avg_epoch_loss:.4f}")


if __name__ == "__main__":
    model = TransformerModel(features_dim=config.features_dim).to(config.DEVICE)
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
