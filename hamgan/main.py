from hamgan.train import train_gan
from hamgan.data import get_data_loaders
from hamgan.validation import plot_fake_images


def main() -> None:
    # we obtain dataloader (`HAM10000Dataset` initialized under the hood)
    print("Getting data loader")
    train_loader, _, _, _ = get_data_loaders()
    print("Initializing generator & discriminator")
    generator, discriminator = train_gan(train_loader)
    print("Visualizing the results")
    plot_fake_images(generator)


if __name__ == '__main__':
    main()
