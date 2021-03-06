
import numpy as np
import albumentations as A
import albumentations.pytorch.transforms
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class LazyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            image_np = np.array(x)
            augmented = self.transform(image=image_np)
            x = augmented['image'].float()
        return x, y

    def __len__(self):
        return len(self.dataset)


def make_transform(transforms_params):
    modules = []
    for transform, params in transforms_params.items():
        modules.append(getattr(A, transform)(**params))
    modules.append(A.pytorch.transforms.ToTensorV2())

    return A.Compose(modules)


def build_data(config, batch_size, transforms):
    if config.type == 'ImageFolder':
        from torchvision.datasets import ImageFolder

        train_ds = ImageFolder(config.path.train)
        valid_ds = None

        if "valid" in config.path:
            valid_ds = ImageFolder(config.path.valid)
        elif "split_ratio" in config:
            # Calculate the length of the dataset
            len_ds = len(train_ds)
            len_train_ds = int(len_ds * config.ratio)
            len_valid_ds = len_ds - len_train_ds
            # Randomly split from all datasets
            train_ds, valid_ds = random_split(train_ds, [len_train_ds, len_valid_ds],
                                              generator=torch.Generator().manual_seed(config.random_split_seed))

        train_ds = LazyDataset(train_ds, transform=make_transform(transforms.train))
        if valid_ds:
            valid_ds = LazyDataset(valid_ds, transform=make_transform(transforms.valid))
    else:
        raise Exception('Invalid data type.')

    train_dl = DataLoader(train_ds, batch_size=batch_size.train, shuffle=True)
    if valid_ds:
        valid_dl = DataLoader(valid_ds, batch_size=batch_size.valid)
    else:
        valid_dl = None

    return train_dl, valid_dl
