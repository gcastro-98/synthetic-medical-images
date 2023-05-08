import os
import matplotlib.pyplot as plt
import torch
from torch import nn as nn, optim as optim
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from typing import List, Tuple

from hamgan.static import SEED, LEARNING_RATE, BETA_1, NUM_EPOCHS, \
    LATENT_DIM, LABEL_TO_CLASS, OUTPUT_PATH, MODELS_PATH, BATCH_SIZE, \
    IMAGE_SIZE, nz, DEVICE

if IMAGE_SIZE == 64:
    from hamgan.gan import Generator64 as Generator
    from hamgan.gan import Discriminator64 as Discriminator
elif IMAGE_SIZE == 128:
    raise ModuleNotFoundError("Not implemented yet!")
else:
    raise ModuleNotFoundError(
        f"IMAGE_SIZE = {IMAGE_SIZE} does not have any architecture")


def seed_everything():
    # random.seed(SEED)
    # np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


# random seeds for reproducibility
seed_everything()
# we create directory to save the generated images and the trained models
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)


def train_gan(data_loader: DataLoader, checkpoint: bool = False)\
        -> Tuple[Generator, Discriminator]:
    # initialize (with weights) generator and discriminator
    netG = Generator().to(DEVICE)
    netG.apply(weights_init)

    netD = Discriminator().to(DEVICE)
    netD.apply(weights_init)

    # loss function and optimizers
    criterion = nn.BCELoss()

    # noise shape - (batch_size,100)
    fixed_noise = torch.randn(BATCH_SIZE, nz, device=DEVICE)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE,
                            betas=(BETA_1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE,
                            betas=(BETA_1, 0.999))
    derror_epoch = []
    gerror_epoch = []
    for epoch in range(NUM_EPOCHS):
        # for epoch in range(1):
        derror_iter = 0
        gerror_iter = 0
        for i, data in enumerate(data_loader, 0):
            # DISCRIMINATOR
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(DEVICE)
            batch_size = real_cpu.size(0)
            breed = data[1].to(DEVICE)
            label = torch.full((batch_size,), real_label, device=DEVICE)

            output = netD(real_cpu, breed)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, device=DEVICE)
            fake = netG(noise, breed)
            label.fill_(fake_label)
            output = netD(fake.detach(), breed)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            derror_iter += errD.item()
            optimizerD.step()

            # GENERATOR
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake, breed)
            errG = criterion(output, label)
            gerror_iter += errG.item()
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

        #         if(i%70==0 and i!=0):
        #             print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
        #                   % (epoch, opt.niter, i, len(dataloader),
        #                      errD.item(), errG.item()))

        # accumulate error for each epoch
        derror_epoch.append(derror_iter)
        gerror_epoch.append(gerror_iter)

    return netG, netD


# def _depr_train(train_loader) -> None:
#     generator = Generator().to(DEVICE)
#     discriminator = Discriminator().to(DEVICE)
#     adversarial_loss = nn.BCELoss()
#     generator_optimizer = optim.Adam(
#         generator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, 0.999))
#     discriminator_optimizer = optim.Adam(
#         discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, 0.999))
#
#     # losses lists to plot them later
#     generator_losses = []
#     discriminator_losses = []
#
#     # epochs loop
#     for epoch in range(NUM_EPOCHS):
#         # batch loop
#         for i, (real_images, labels) in enumerate(train_loader):
#             # adversarial ground truth
#             valid = torch.ones(real_images.size(0), 1).to(DEVICE)
#             fake = torch.zeros(real_images.size(0), 1).to(DEVICE)
#
#             # we move images to DEVICE
#             real_images = real_images.to(DEVICE)
#             # same for labels, but first converting them into int-tensor
#             labels = _labels_to_tensor(labels)
#             labels = labels.to(DEVICE)
#
#             # ---------------------
#             #  Train UnconditionalDiscriminator64
#             # ---------------------
#             discriminator_optimizer.zero_grad()
#
#             # Generate fake images
#             noise = torch.randn(real_images.size(0), LATENT_DIM).to(DEVICE)
#             fake_images = generator(noise, labels)
#
#             # UnconditionalDiscriminator64 loss on real images
#             # TODO: debug
#             print(discriminator(real_images, labels).size())
#             real_loss = adversarial_loss(discriminator(real_images, labels),
#                                          valid)
#             # UnconditionalDiscriminator64 loss on fake images
#             fake_loss = adversarial_loss(
#                 discriminator(fake_images.detach(), labels), fake)
#             # Total discriminator loss
#             discriminator_loss = (real_loss + fake_loss) / 2
#
#             # Update discriminator weights
#             discriminator_loss.backward()
#             discriminator_optimizer.step()
#
#             # -----------------
#             #  Train UnconditionalGenerator64
#             # -----------------
#             generator_optimizer.zero_grad()
#
#             # UnconditionalGenerator64 loss
#             generator_loss = adversarial_loss(
#                 discriminator(fake_images, labels), valid)
#
#             # Update generator weights
#             generator_loss.backward()
#             generator_optimizer.step()
#
#             # #################################################################
#
#             # keep losses for plotting later
#             generator_losses.append(generator_loss.item())
#             discriminator_losses.append(discriminator_loss.item())
#
#             # Print training progress
#             if (i + 1) % 200 == 0:
#                 print(
#                     f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch [{i + 1}/{len(train_loader)}] "
#                     f"UnconditionalDiscriminator64 Loss: {discriminator_loss.item():.4f} "
#                     f"UnconditionalGenerator64 Loss: {generator_loss.item():.4f}"
#                 )
#
#         # #####################################################################
#
#         # save generated images
#         if (epoch + 1) % 5 == 0:
#             with torch.no_grad():
#                 fake_images = generator(torch.randn(25, LATENT_DIM).to(DEVICE),
#                                         torch.arange(0, 5).repeat(5).to(
#                                             DEVICE))
#                 # we re-scale generated images to [0, 1]
#                 fake_images = (fake_images + 1) / 2
#                 save_image(
#                     fake_images,
#                     os.path.join(OUTPUT_PATH, f"epoch_{epoch + 1}.png"),
#                     nrow=5, normalize=True)
#
#             # save models as checkpoint
#             if checkpoint:
#                 torch.save(generator.state_dict(),
#                            os.path.join(MODELS_PATH, "generator.pth"))
#                 torch.save(discriminator.state_dict(),
#                            os.path.join(MODELS_PATH, "discriminator.pth"))
#
#     # save the trained generator
#     torch.save(generator.state_dict(),
#                os.path.join(MODELS_PATH, "generator.pth"))
#     # as well as the trained discriminator
#     torch.save(discriminator.state_dict(),
#                os.path.join(MODELS_PATH, "discriminator.pth"))
#
#     _plot_losses(generator_losses, discriminator_losses)


def weights_init(m) -> callable:
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def _plot_losses(g_losses: List[float], d_losses: List[float]) -> None:
    plt.figure(figsize=(10, 5))
    plt.title("UnconditionalGenerator64 and UnconditionalDiscriminator64 Loss During Training")
    plt.plot(g_losses, label="UnconditionalGenerator64")
    plt.plot(d_losses, label="UnconditionalDiscriminator64")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_PATH, 'losses.png'), dpi=200)
    plt.show()


def _labels_to_tensor(labels: Tuple[str]) -> torch.Tensor:
    return torch.tensor(
        tuple(LABEL_TO_CLASS[_l] for _l in labels))
