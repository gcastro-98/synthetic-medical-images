import os
from torch import load

from hamgan.train import train_gan
from hamgan.data import get_data_loaders
from hamgan.validation import plot_fake_images
from hamgan.static import MODELS_PATH, IMAGE_SIZE, DEVICE
from hamgan.logger import Logger

if IMAGE_SIZE == 64:
    from hamgan.gan import Generator64 as Generator
elif IMAGE_SIZE == 128:
    raise ModuleNotFoundError("Not implemented yet!")
else:
    raise ModuleNotFoundError(
        f"IMAGE_SIZE = {IMAGE_SIZE} does not have any architecture")

logger = Logger()


def main(_load_serialized_model: bool = False) -> None:
    if _load_serialized_model:
        logger.info("Loading already trained and serialized generator model")
        generator = Generator().to(DEVICE)
        generator.load_state_dict(load(
            os.path.join(MODELS_PATH, "generator.pth")))
        plot_fake_images(generator, _show=True)
        return

    # we obtain dataloader (`HAM10000Dataset` initialized under the hood)
    logger.info("Getting data loader")
    train_loader, _, _, _ = get_data_loaders()
    logger.info("Initializing generator & discriminator")
    generator, discriminator = train_gan(train_loader)
    logger.info("Visualizing some generated images")
    plot_fake_images(generator)


if __name__ == '__main__':
    main()
    # main(_load_serialized_model=True)
