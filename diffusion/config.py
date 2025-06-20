import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_DIR = 'E:/ImageNet'
TEST_DIR = 'E:/ImageNet'

CHECKPOINT_DIR = './checkpoints'
EXAMPLES_DIR = './examples'

NUM_EPOCHS = 20
NUM_WORKERS = 6
DROPOUT = 0.3
LEARNING_RATE = 1e-6
BATCH_SIZE = 4

LOAD_MODEL = True
SAVE_MODEL = True


IMAGE_SIZE = 128

layers = (128, 256, 384, 512, 512)
