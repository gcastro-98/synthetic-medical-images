"""
Namelist-like module to control all the global variables needed during the
execution.
"""

# General parameters
SEED: int = 42
INPUT_PATH: str = 'data'
OUTPUT_PATH: str = 'output'
MODELS_PATH: str = 'models'

# Data parameters
IMAGE_SIZE: int = 128
LABEL_TO_CLASS: dict = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6,
}
NUM_CLASSES: int = len(LABEL_TO_CLASS)

# Training parameters
BATCH_SIZE: int = 64
NUM_EPOCHS: int = 50

# Hyperparameters
LATENT_DIM: int = 100
LEARNING_RATE: float = 0.0002
BETA_1: float = 0.5
