import os
import matplotlib.pyplot as plt
import torch
from torch import nn as nn, optim as optim
from torchvision.utils import save_image
from typing import List, Tuple

from hamgan.data import get_data_loaders
from hamgan.gan import Generator, Discriminator
from hamgan.static import SEED, LEARNING_RATE, BETA_1, NUM_EPOCHS, \
    LATENT_DIM, LABEL_TO_CLASS, OUTPUT_PATH, MODELS_PATH


# random seed for reproducibility
torch.manual_seed(SEED)
# we set the device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# we create directory to save the generated images and the trained models
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)


def train_gan(checkpoint: bool = False) -> None:
    # we obtain dataloader (`HAM10000Dataset` initialized under the hood)
    train_loader, _, _, _ = get_data_loaders()

    # initialize generator and discriminator
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # loss function and optimizers
    adversarial_loss = nn.BCELoss()
    generator_optimizer = optim.Adam(
        generator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, 0.999))
    discriminator_optimizer = optim.Adam(
        discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, 0.999))

    # losses lists to plot them later
    generator_losses = []
    discriminator_losses = []

    # epochs loop
    for epoch in range(NUM_EPOCHS):
        # batch loop
        for i, (real_images, labels) in enumerate(train_loader):
            # adversarial ground truth
            valid = torch.ones(real_images.size(0), 1).to(device)
            fake = torch.zeros(real_images.size(0), 1).to(device)

            # we move images to device
            real_images = real_images.to(device)
            # same for labels, but first converting them into int-tensor
            labels = _labels_to_tensor(labels)
            labels = labels.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            discriminator_optimizer.zero_grad()

            # Generate fake images
            noise = torch.randn(real_images.size(0), LATENT_DIM).to(device)
            fake_images = generator(noise, labels)

            # Discriminator loss on real images
            # TODO: debug
            print(discriminator(real_images, labels).size())
            real_loss = adversarial_loss(discriminator(real_images, labels),
                                         valid)
            # Discriminator loss on fake images
            fake_loss = adversarial_loss(
                discriminator(fake_images.detach(), labels), fake)
            # Total discriminator loss
            discriminator_loss = (real_loss + fake_loss) / 2

            # Update discriminator weights
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # -----------------
            #  Train Generator
            # -----------------
            generator_optimizer.zero_grad()

            # Generator loss
            generator_loss = adversarial_loss(
                discriminator(fake_images, labels), valid)

            # Update generator weights
            generator_loss.backward()
            generator_optimizer.step()

            # #################################################################

            # keep losses for plotting later
            generator_losses.append(generator_loss.item())
            discriminator_losses.append(discriminator_loss.item())

            # Print training progress
            if (i + 1) % 200 == 0:
                print(
                    f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch [{i + 1}/{len(train_loader)}] "
                    f"Discriminator Loss: {discriminator_loss.item():.4f} "
                    f"Generator Loss: {generator_loss.item():.4f}"
                )

        # #####################################################################

        # save generated images
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                fake_images = generator(torch.randn(25, LATENT_DIM).to(device),
                                        torch.arange(0, 5).repeat(5).to(
                                            device))
                # we re-scale generated images to [0, 1]
                fake_images = (fake_images + 1) / 2
                save_image(
                    fake_images,
                    os.path.join(OUTPUT_PATH, f"epoch_{epoch + 1}.png"),
                    nrow=5, normalize=True)

            # save models as checkpoint
            if checkpoint:
                torch.save(generator.state_dict(),
                           os.path.join(MODELS_PATH, "generator.pth"))
                torch.save(discriminator.state_dict(),
                           os.path.join(MODELS_PATH, "discriminator.pth"))

    # save the trained generator
    torch.save(generator.state_dict(),
               os.path.join(MODELS_PATH, "generator.pth"))
    # as well as the trained discriminator
    torch.save(discriminator.state_dict(),
               os.path.join(MODELS_PATH, "discriminator.pth"))

    _plot_losses(generator_losses, discriminator_losses)


def _plot_losses(g_losses: List[float], d_losses: List[float]) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_PATH, 'losses.png'), dpi=200)
    plt.show()


def _labels_to_tensor(labels: Tuple[str]) -> torch.Tensor:
    return torch.tensor(
        tuple(LABEL_TO_CLASS[_l] for _l in labels))
