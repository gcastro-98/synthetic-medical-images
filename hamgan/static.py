"""
Namelist-like module to control all the global variables needed during the
execution.
"""

from torch import device
from torch.cuda import is_available

# ---------------------
# General parameters
# ---------------------
SEED: int = 42
NUM_WORKERS: int = 2
INPUT_PATH: str = 'data'
OUTPUT_PATH: str = 'output'
MODELS_PATH: str = 'models'
DEVICE = device("cuda" if is_available() else "cpu")  # (GPU if available)

# ---------------------
# Data parameters
# ---------------------
IMAGE_SIZE: int = 64  # 128
assert IMAGE_SIZE in (64, 128), "Other IMAGE_SIZE require custom architectures"
LABEL_TO_CLASS: dict = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6,
}
CLASS_TO_LABEL: dict = {_v: _k for _k, _v in LABEL_TO_CLASS.items()}
NUM_CLASSES: int = len(LABEL_TO_CLASS)

# ---------------------
# Training parameters
# ---------------------
BATCH_SIZE: int = 64
NUM_EPOCHS: int = 50

# ---------------------
# Hyperparameters
# ---------------------
LATENT_DIM: int = 100
# kept same hyperparameters as https://arxiv.org/pdf/1511.06434.pdf
LEARNING_RATE: float = 0.0002
BETA_1: float = 0.5
# same nomenclature as https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
nz: int = 100  # length of latent vector
ngf: int = 64  # depth of feature maps carried through the generator.
ndf: int = 64  # depth of feature maps propagated through the discriminator
# ngpu: int = 1
nc: int = 3  # number of color channels (for color images = 3)
# niter: int = NUM_EPOCHS BATCH_SIZE  # 300
n_dnn: int = 1000  # number of output features of the label's linear
