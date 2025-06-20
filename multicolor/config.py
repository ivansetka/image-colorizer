import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_DIR = 'E:/ImageNet'
TEST_DIR = 'E:/ImageNet'

CHECKPOINT_DIR = './checkpoints'
EXAMPLES_DIR = './examples'

NUM_EPOCHS = 20
NUM_WORKERS = 6
LEARNING_RATE = 1e-4
BETAS = (0.9, 0.999)
BATCH_SIZE = 16

LOAD_MODEL = True
SAVE_MODEL = True


IMAGE_SIZE = 256

features_dim = [512, 512, 256, 256]
cscnet_dim = [32, 64, 128, 64, 32]
module_names = ['lab', 'hsv', 'yuv']
module_weights = [0.1, 10, 10]
