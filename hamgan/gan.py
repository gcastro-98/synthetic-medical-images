import torch
import torch.nn as nn

from hamgan.static import LATENT_DIM, NUM_CLASSES

# TODO: implement the same DCGAN as https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#  then convert it into conditional: https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/


# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM + NUM_CLASSES, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), dim=1).unsqueeze(2).unsqueeze(3)
        img = self.model(gen_input)
        return img


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        self.model = nn.Sequential(
            nn.Conv2d(NUM_CLASSES + 3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        disc_input = torch.cat((self.label_emb(labels), img), dim=1)
        validity = self.model(disc_input)
        return validity.view(-1, 1).squeeze(1)
