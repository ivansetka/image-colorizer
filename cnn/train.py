import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from imagenet import ImageNetDataset
from losses import colorization_loss, classification_loss
from model import CNNModel
from cnn import config
from utils import transform, make_subset, load_checkpoint, save_checkpoint, save_random_examples


def train_epoch(model, dataloader, optimizer):
    model.train()
    epoch_loss = 0.0

    for i, (original_img, labels) in enumerate(tqdm(dataloader)):
        L = original_img[:, :1, :, :].to(config.DEVICE)
        ab = original_img[:, 1:, :, :].to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        generated_ab, logits = model(L)
        loss = colorization_loss(generated_ab, ab) + classification_loss(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Average Loss: {avg_epoch_loss:.4f}")


if __name__ == "__main__":
    model = CNNModel().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_dataset = ImageNetDataset(config.TRAIN_DIR, transformation=transform, use_labels=True, train=True)
    test_dataset = ImageNetDataset(config.TEST_DIR, transformation=transform, use_labels=False, train=False)

    train_dataset = make_subset(train_dataset, subset_ratio=0.4)
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )

    start_epoch = 0
    if config.LOAD_MODEL:
        start_epoch = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        train_epoch(model, dataloader, optimizer)

        if config.SAVE_MODEL:
            save_checkpoint(epoch, model, optimizer)

        save_random_examples(model, test_dataset, epoch)
