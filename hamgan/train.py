import random
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn as nn, optim as optim
from torch import Tensor, device
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from typing import List, Tuple

from hamgan.logger import Logger
from hamgan.static import SEED, LEARNING_RATE, BETA_1, NUM_EPOCHS, \
    LABEL_TO_CLASS, OUTPUT_PATH, MODELS_PATH, IMAGE_SIZE, nz, \
    BATCH_SIZE, NUM_CLASSES
from hamgan.static import DEVICE as _DEVICE

if IMAGE_SIZE == 64:
    from hamgan.gan import Generator64 as Generator
    from hamgan.gan import Discriminator64 as Discriminator
elif IMAGE_SIZE == 128:
    raise ModuleNotFoundError("Not implemented yet!")
else:
    raise ModuleNotFoundError(
        f"IMAGE_SIZE = {IMAGE_SIZE} does not have any architecture")


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def __generate_random_noise() -> Tensor:
    return torch.randn(BATCH_SIZE, nz, device=_DEVICE)


def __generate_random_labels() -> Tensor:
    label = torch.zeros(BATCH_SIZE, NUM_CLASSES, device=_DEVICE)
    for i in range(BATCH_SIZE):
        x = np.random.randint(0, NUM_CLASSES)
        label[i][x] = 1
    return label


# random seeds for reproducibility
seed_everything()
# noise and labels from which images will be generated at checkpoint
_checkpoint_noise = __generate_random_noise()
_checkpoint_labels = __generate_random_labels()

# we create directory to save the generated images and the trained models
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
logger = Logger()


def train_gan(
        data_loader: DataLoader, use_cpu: bool = False,
        save_best_model: bool = True, save_generated_images: bool = True,
        verbose: bool = False, _freq: int = 5) -> Tuple[Generator, Discriminator]:
    DEVICE = device('cpu') if use_cpu else _DEVICE
    # initialize (with weights) generator and discriminator
    net_g = Generator().to(DEVICE)
    net_g.apply(weights_init)
    net_d = Discriminator().to(DEVICE)
    net_d.apply(weights_init)

    # loss function and optimizers
    criterion = nn.BCELoss()  # we are simply detecting whether it's real/fake

    real_label = float(1)
    fake_label = float(0)

    # setup optimizer
    optimizer_d = optim.Adam(
        net_d.parameters(), lr=LEARNING_RATE, betas=(BETA_1, 0.999))
    optimizer_g = optim.Adam(
        net_g.parameters(), lr=LEARNING_RATE, betas=(BETA_1, 0.999))
    d_error_epoch = []
    g_error_epoch = []

    for epoch in range(NUM_EPOCHS):
        # we will start iterating each batch element
        d_error_iter = 0
        g_error_iter = 0
        for i, data in enumerate(data_loader, 0):
            # DISCRIMINATOR
            # train with real
            net_d.zero_grad()
            real_cpu = data[0].to(DEVICE)
            batch_size = real_cpu.size(0)
            pathology_one_hot = data[1].to(DEVICE)
            label = torch.full((batch_size, ), real_label, device=DEVICE)

            output = net_d(real_cpu, pathology_one_hot)
            err_d_real = criterion(output, label)
            err_d_real.backward()
            # D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, device=DEVICE)
            fake = net_g(noise, pathology_one_hot)
            label.fill_(fake_label)
            output = net_d(fake.detach(), pathology_one_hot)
            err_d_fake = criterion(output, label)
            err_d_fake.backward()
            # D_G_z1 = output.mean().item()
            err_d = err_d_real + err_d_fake
            d_error_iter += err_d.item()
            optimizer_d.step()

            # GENERATOR
            net_g.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = net_d(fake, pathology_one_hot)
            err_g = criterion(output, label)
            g_error_iter += err_g.item()
            err_g.backward()
            # D_G_z2 = output.mean().item()
            optimizer_g.step()

            if (i + 1) % (BATCH_SIZE // 4) == 0 and verbose:
                # we print the losses
                _counter = f"Epoch [{epoch}/{NUM_EPOCHS}]" \
                           f"[{i}/{len(data_loader)}]"
                logger.debug(f"{_counter} --- Loss G: {err_g.item()}")
                logger.debug(f"{_counter} --- Loss D: {err_d.item()}")

        if (epoch + 1) % _freq == 0:
            # we save generated images
            with torch.no_grad():
                if save_generated_images:
                    logger.debug(
                        "CHECKPOINT: saving some generated images "
                        f"at '{OUTPUT_PATH}' directory")
                    checkpoint_images = net_g(
                        _checkpoint_noise, _checkpoint_labels)
                    # we re-scale generated images to [0, 1] and save them
                    save_image((checkpoint_images + 1) / 2,
                               os.path.join(OUTPUT_PATH,
                                            f"epoch_{epoch + 1}.png"),
                               nrow=8, normalize=True)

            # save models as checkpoint
            if save_best_model:
                logger.debug("CHECKPOINT: saving the trained"
                             f" models at '{MODELS_PATH}' directory")
                torch.save(net_g.state_dict(),
                           os.path.join(MODELS_PATH, "generator.pth"))
                torch.save(net_d.state_dict(),
                           os.path.join(MODELS_PATH, "discriminator.pth"))

        # accumulate error for each epoch
        d_error_epoch.append(d_error_iter)
        g_error_epoch.append(g_error_iter)

    _plot_losses(g_error_epoch, d_error_epoch)

    # save the trained generator
    torch.save(net_g.state_dict(),
               os.path.join(MODELS_PATH, "generator.pth"))
    # as well as the trained discriminator
    torch.save(net_d.state_dict(),
               os.path.join(MODELS_PATH, "discriminator.pth"))

    return net_g, net_d


def weights_init(m) -> callable:
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def _plot_losses(g_losses: List[float], d_losses: List[float],
                 _show: bool = False) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join('.img', 'losses.png'), dpi=200)
    if _show:
        plt.show()
    plt.close()


def _labels_to_tensor(labels: Tuple[str]) -> torch.Tensor:
    return torch.tensor(
        tuple(LABEL_TO_CLASS[_l] for _l in labels))
