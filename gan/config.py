import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_DIR = 'E:/ImageNet'
TEST_DIR = 'E:/ImageNet'

PRETRAINED_GENERATOR_PATH = './pretrained_generator.pt'
CHECKPOINT_DIR = './checkpoints'
EXAMPLES_DIR = './examples'

NUM_EPOCHS = 20
NUM_WORKERS = 6
LEARNING_RATE = 2e-4
BETAS = (0.5, 0.999)
BATCH_SIZE = 16
L1_LAMBDA = 100

LOAD_MODEL = True
SAVE_MODEL = True


IMAGE_SIZE = 256

generator_contracting_layers = [
    (64, 0),
    (64, 0),
    (128, 0),
    (256, 0),
    (512, 0),
    (512, 0),
    (512, 0),
    (512, 0)
]

generator_expansive_layers = [
    (512, 0),
    (512, 0),
    (512, 0),
    (256, 0),
    (128, 0),
    (64, 0),
    (64, 0)
]

discriminator_layers = [
    (64, 0),
    (128, 0),
    (256, 0),
    (512, 0)
]
