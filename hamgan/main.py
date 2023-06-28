import os
import argparse
from torch import load, device

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


def main(use_cpu: bool = False, load_model: bool = False,
         save_best_model: bool = False, save_generated_images: bool = False,
         verbose: bool = False) -> None:
    if load_model:
        logger.info("Loading already trained and serialized generator model")
        generator = Generator().to(device('cpu') if use_cpu else DEVICE)
        generator.load_state_dict(load(
            os.path.join(MODELS_PATH, "generator.pth")))
        plot_fake_images(generator, _show=False)
        return

    # we obtain dataloader (`HAM10000Dataset` initialized under the hood)
    logger.info("Getting data loader")
    train_loader, _, _, _ = get_data_loaders()
    logger.info("Initializing generator & discriminator")

    generator, discriminator = train_gan(
        train_loader, use_cpu=use_cpu, save_best_model=save_best_model,
        save_generated_images=save_generated_images, verbose=verbose)
    logger.info("Visualizing some generated images")
    plot_fake_images(generator, _show=False)


if __name__ == '__main__':
    # 1. Instantiate the parser
    parser = argparse.ArgumentParser(
        description='Generate fake clinical images using our '
                    '(pre-trained or not) GAN model.')
    # 2. Add switches
    parser.add_argument(
        '--use_cpu', action='use_cpu',
        help='If True, GPU is not used even though it is available')
    parser.add_argument(
        '--load_model', action='load_model',
        help='If True, the GAN is not trained, but the generator is loaded '
             'from disk (it needs to be serialized from before)')
    parser.add_argument(
        '--save_best_model', action='save_best_model',
        help='If True, the generator is serialized at '
             'the end of each epoch as checkpoint')
    parser.add_argument(
        '--save_generated_images', action='save_generated_images',
        help='If True, the images generated with the'
             'generator trained at the end of each '
             'epoch are saved at disk (to later '
             'preview their evolution over time)')
    parser.add_argument(
        '--verbose', action='verbose',
        help='If True, the losses are displayed at the end of each epoch')
    # 3. Parse the arguments
    args = parser.parse_args()
    # 4. Finally, we access them
    main(use_cpu=args.use_cpu, load_model=args.load_model,
         save_best_model=args.save_best_model,
         save_generated_images=args.save_generated_images,
         verbose=args.verbose)

    # # OTHER ARGUMENT PARSING POSSIBILITIES
    # # Required positional argument
    # parser.add_argument('pos_arg', type=int,
    #                     help='A required integer positional argument')
    # # Optional positional argument
    # parser.add_argument('opt_pos_arg', type=int, nargs='?',
    #                     help='An optional integer positional argument')
    # # Optional argument
    # parser.add_argument('--opt_arg', type=int,
    #                     help='An optional integer argument')
