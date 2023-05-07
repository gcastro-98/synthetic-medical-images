import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from typing import Tuple

from hamgan.static import BATCH_SIZE


class HAM10000Dataset(Dataset):
    def __init__(self, root_dir: str = 'data',
                 metadata_file: str = 'HAM10000_metadata.csv',
                 data_transforms: transforms = None) -> None:
        self.root_dir = root_dir
        self.metadata: pd.DataFrame = self._load_metadata(metadata_file)
        self.transforms = _get_data_transforms() \
            if data_transforms is None else data_transforms

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx):
        _image_id: str = self.metadata.iloc[idx]['image_id']
        _image_path: str = self._find_image_path(_image_id)

        _image: Image = Image.open(_image_path).convert('RGB')
        _image = self.transforms(_image)
        _label: str = self.metadata.iloc[idx]['dx']

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


def _get_data_transforms():
    return transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
        ]
    )


def get_data_loaders() \
        -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    # load and split the data
    dataset = HAM10000Dataset()
    _train, _val, _test = random_split(dataset, [.6, .2, .2])

    # ingest torch datasets data into torch dataloader
    train_loader = DataLoader(
        dataset=_train, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = DataLoader(
        dataset=_train, batch_size=2 * BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(
        dataset=_val, batch_size=2 * BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        dataset=_test, batch_size=2 * BATCH_SIZE, shuffle=False)

    return train_loader, train_loader_at_eval, val_loader, test_loader
