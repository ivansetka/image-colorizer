import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from discriminator import Discriminator
from generator import PretrainedGeneratorWrapper
from imagenet import ImageNetDataset
from losses import discriminator_loss, generator_loss
from project.gan import config
from utils import transform, make_subset, load_checkpoint, save_checkpoint, save_random_examples


def train_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D):
    generator.train()
    discriminator.train()

    epoch_gen_loss = 0.0
    epoch_dis_loss = 0.0

    for i, original_img in enumerate(tqdm(dataloader)):
        original_img = original_img.to(config.DEVICE)
        L = original_img[:, :1, :, :]

        generated_ab = generator(L)
        generated_img = torch.cat((L, generated_ab), dim=1)

        real_output = discriminator(original_img)
        fake_output = discriminator(generated_img.detach())
        dis_loss = discriminator_loss(real_output, fake_output)

        optimizer_D.zero_grad()
        dis_loss.backward()
        optimizer_D.step()

        fake_output = discriminator(generated_img)
        gen_loss = generator_loss(fake_output, generated_img, original_img)

        optimizer_G.zero_grad()
        gen_loss.backward()
        optimizer_G.step()

        epoch_gen_loss += gen_loss.item()
        epoch_dis_loss += dis_loss.item()

    avg_epoch_gen_loss = epoch_gen_loss / len(dataloader)
    avg_epoch_dis_loss = epoch_dis_loss / len(dataloader)
    print(f"Average G Loss: {avg_epoch_gen_loss:.4f} | Average D Loss: {avg_epoch_dis_loss:.4f}")


if __name__ == "__main__":
    generator_wrapper = PretrainedGeneratorWrapper(
        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        weights_path=config.PRETRAINED_GENERATOR_PATH,
        device=config.DEVICE
    )
    generator = generator_wrapper.generator
    discriminator = Discriminator(config.discriminator_layers).to(config.DEVICE)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=config.BETAS)

    def lambda_rule(epoch):
        decay_start = config.NUM_EPOCHS // 2
        if epoch < decay_start:
            return 1.0

        return 1.0 - (epoch - decay_start) / (config.NUM_EPOCHS - decay_start)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    train_dataset = ImageNetDataset(config.TRAIN_DIR, transformation=transform, train=True)
    test_dataset = ImageNetDataset(config.TEST_DIR, transformation=transform, train=False)

    train_dataset = make_subset(train_dataset, subset_ratio=0.1)
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )

    start_epoch = 0
    if config.LOAD_MODEL:
        start_epoch = load_checkpoint(
            generator,
            discriminator,
            optimizer_G,
            optimizer_D,
            scheduler_G,
            scheduler_D
        )

    for epoch in range(start_epoch, config.NUM_EPOCHS):
        train_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D)
        scheduler_G.step(epoch)
        scheduler_D.step(epoch)

        if config.SAVE_MODEL:
            save_checkpoint(
                epoch,
                generator,
                discriminator,
                optimizer_G,
                optimizer_D,
                scheduler_G,
                scheduler_D
            )

        save_random_examples(generator, test_dataset, epoch)
