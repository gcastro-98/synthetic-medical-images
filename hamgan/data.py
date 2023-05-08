import os
import pandas as pd
from PIL import Image
from torch import zeros, Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from typing import Tuple
from torch import Generator as _DataGenerator

from hamgan.static import INPUT_PATH, BATCH_SIZE, IMAGE_SIZE, SEED, \
    NUM_CLASSES, LABEL_TO_CLASS, NUM_WORKERS


class HAM10000Dataset(Dataset):
    def __init__(self, root_dir: str = INPUT_PATH,
                 metadata_file: str = 'HAM10000_metadata.csv',
                 data_transforms: transforms = None) -> None:
        self.root_dir = root_dir
        self.metadata: pd.DataFrame = self._load_metadata(metadata_file)
        self.transforms = _get_data_transforms() \
            if data_transforms is None else data_transforms

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        _image_id: str = self.metadata.iloc[idx]['image_id']
        _image_path: str = self._find_image_path(_image_id)

        _image: Image = Image.open(_image_path).convert('RGB')
        _image = self.transforms(_image)

        _label_ind: int = LABEL_TO_CLASS[self.metadata.iloc[idx]['dx']]
        # one-hot encoding
        _label: Tensor = zeros(NUM_CLASSES)
        _label[_label_ind] = 1

        return _image, _label

    def _find_image_path(self, image_id) -> str:
        for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
            image_path = os.path.join(self.root_dir, folder, f"{image_id}.jpg")
            if os.path.exists(image_path):
                return image_path
        raise FileNotFoundError(f"Image {image_id} not found in any folder!")

    def _load_metadata(self, metadata_file: str) -> pd.DataFrame:
        _df = pd.read_csv(os.path.join(self.root_dir, metadata_file))
        _df['age'] = _df['age'].astype(int, errors='ignore')
        return _df


def _get_data_transforms() -> transforms.Compose:
    _mean, _std = 0, 1
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((_mean, _mean, _mean), (_std, _std, _std)),
        ]
    )


def get_data_loaders() -> Tuple[DataLoader, ...]:
    # load and split the data
    dataset = HAM10000Dataset()
    _train, _val, _test = random_split(
        dataset, [.6, .2, .2],
        generator=_DataGenerator().manual_seed(SEED))

    # ingest torch datasets data into torch dataloader
    train_loader = DataLoader(
        dataset=_train, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS)
    train_loader_at_eval = DataLoader(
        dataset=_train, batch_size=2 * BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS)
    val_loader = DataLoader(
        dataset=_val, batch_size=2 * BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(
        dataset=_test, batch_size=2 * BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, train_loader_at_eval, val_loader, test_loader
