import os.path

import numpy as np
import matplotlib.pyplot as plt
import torch

from hamgan.static import nz, DEVICE, NUM_CLASSES, CLASS_TO_LABEL
from hamgan.data import HAM10000Dataset

labels_as_titles = {
        'akiec': 'Actinic keratoses',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis-like',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevi',
        'vasc': 'Vascular lesions',
    }


def plot_real_images(_show: bool = False) -> None:
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    _dataset = HAM10000Dataset()

    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(_dataset), size=(1,)).item()
        img, label = _dataset[sample_idx]
        label = np.argmax(label)

        figure.add_subplot(rows, cols, i)
        plt.title(labels_as_titles[CLASS_TO_LABEL[label]])
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))
    plt.tight_layout(pad=1.02)
    plt.savefig(os.path.join('.img', 'original_samples.png'), dpi=200)
    if _show:
        plt.show()
    plt.close()


def plot_fake_images(
        generator, n_images: int = 10, _show: bool = False) -> None:
    figure = plt.figure(figsize=(10, 10))
    cols, rows = 3, 3

    gen_z = torch.randn(n_images, nz, device=DEVICE)
    label = torch.zeros(n_images, NUM_CLASSES, device=DEVICE)
    _label_names = []
    for i in range(n_images):
        x = np.random.randint(0, NUM_CLASSES)
        label[i][x] = 1
        _label_names.append(labels_as_titles[CLASS_TO_LABEL[x]])

    gen_images = generator(gen_z, label)
    images = gen_images.to("cpu").clone().detach()
    images = images.numpy().transpose(0, 2, 3, 1)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        # plt.subplot(3, 3, i+1)
        figure.add_subplot(rows, cols, i + 1)
        plt.title(_label_names[i])
        plt.axis("off")
        plt.imshow(images[i])
    plt.tight_layout(pad=1.02)
    plt.savefig(os.path.join('.img', 'fake_samples.png'), dpi=200)
    if _show:
        plt.show()
    plt.close()
