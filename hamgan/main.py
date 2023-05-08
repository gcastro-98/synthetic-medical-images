from hamgan.train import train_gan
from hamgan.data import get_data_loaders
from hamgan.validation import plot_fake_images
from hamgan.logger import Logger

logger = Logger()


def main() -> None:
    # we obtain dataloader (`HAM10000Dataset` initialized under the hood)
    logger.info("Getting data loader")
    train_loader, _, _, _ = get_data_loaders()
    logger.info("Initializing generator & discriminator")
    generator, discriminator = train_gan(train_loader)
    logger.info("Visualizing some generated images")
    plot_fake_images(generator)


if __name__ == '__main__':
    main()
