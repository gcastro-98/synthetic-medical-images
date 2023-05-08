import torch
import torch.nn as nn

from hamgan.static import ngf, ndf, nz, nc, n_dnn, IMAGE_SIZE, NUM_CLASSES

# While the cDCGAN implementation for IMAGE_SIZE = 64 is defined at:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# and so we have to simply update things so that the GAN is conditional:
# https://github.com/ashukid/Conditional-GAN-pytorch/blob/master/Conditional%20DCGAN.ipynb
# The inspiration for the architecture when IMAGE_SIZE = 128 was found here:
# https://github.com/pytorch/examples/issues/70


#######################################################################
# (Conditional) DCGAN: cDCGAN
#######################################################################

# IMAGE_SIZE = 64 # ###################################################

class Generator64(nn.Module):
    def __init__(self):
        assert IMAGE_SIZE == 64, \
            f"This architecture is not suitable for IMAGE_SIZE = {IMAGE_SIZE}"
        super(Generator64, self).__init__()
        # self.ngpu = _ngpu

        self.y_label = nn.Sequential(
            nn.Linear(NUM_CLASSES, n_dnn),  # 120, 1000
            nn.ReLU(True)
        )

        self.yz = nn.Sequential(
            nn.Linear(nz, 2 * nz),  # 100, 200
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(n_dnn + 2 * nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z, y):
        # mapping noise and label
        z = self.yz(z)
        y = self.y_label(y)

        # mapping concatenated input to the main generator network
        inp = torch.cat([z, y], 1)
        inp = inp.view(-1, n_dnn + 2 * nz, 1, 1)  # 1000 + 200
        output = self.main(inp)

        return output


class Discriminator64(nn.Module):
    def __init__(self):
        assert IMAGE_SIZE == 64, \
            f"This architecture is not suitable for IMAGE_SIZE = {IMAGE_SIZE}"
        super(Discriminator64, self).__init__()
        # self.ngpu = _ngpu
        self.y_label = nn.Sequential(
            nn.Linear(NUM_CLASSES, 64 * 64 * 1),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        y = self.y_label(y)
        y = y.view(-1, 1, 64, 64)
        inp = torch.cat([x, y], 1)
        output = self.main(inp)

        return output.view(-1, 1).squeeze(1)


# IMAGE_SIZE = 128 # ###################################################

# TODO: implement

#######################################################################
# (Unconditional) DCGAN
#######################################################################

# IMAGE_SIZE = 64 # ###################################################

class UnconditionalGenerator64(nn.Module):
    def __init__(self):
        assert IMAGE_SIZE == 64, \
            f"This architecture is not suitable for IMAGE_SIZE = {IMAGE_SIZE}"
        super(UnconditionalGenerator64, self).__init__()
        # self.ngpu = _ngpu
        self.main = nn.Sequential(
            # _input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, _input):
        return self.main(_input)


class UnconditionalDiscriminator64(nn.Module):
    def __init__(self):
        assert IMAGE_SIZE == 64, \
            f"This architecture is not suitable for IMAGE_SIZE = {IMAGE_SIZE}"
        super(UnconditionalDiscriminator64, self).__init__()
        # self.ngpu = _ngpu
        self.main = nn.Sequential(
            # _input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, _input):
        return self.main(_input)


# IMAGE_SIZE = 128 # ###################################################

class UnconditionalGenerator128(nn.Module):
    def __init__(self):
        assert IMAGE_SIZE == 128, \
            f"This architecture is not suitable for IMAGE_SIZE = {IMAGE_SIZE}"
        super(UnconditionalGenerator128, self).__init__()
        # self.ngpu = _ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, _input):
        return self.main(_input)


class UnconditionalDiscriminator128(nn.Module):
    def __init__(self):
        assert IMAGE_SIZE == 128, \
            f"This architecture is not suitable for IMAGE_SIZE = {IMAGE_SIZE}"
        super(UnconditionalDiscriminator128, self).__init__()
        # self.ngpu = _ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, _input):
        return self.main(_input)
