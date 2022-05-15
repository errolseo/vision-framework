
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


def build_data(data):
    if data.type == 'ImageFolder':
        from torchvision.datasets import ImageFolder
        if "valid" in data.path:
            train_ds = LazyDataset(ImageFolder(data.path.train), transform=make_transform(data.transform.train))
            valid_ds = LazyDataset(ImageFolder(data.path.valid), transform=make_transform(data.transform.valid))
        else:
            ds = ImageFolder(data.path.train)
            if "ratio" in data:
                # Calculate the length of the dataset
                len_ds = len(ds)
                len_train_ds = int(len_ds * data.ratio)
                len_valid_ds = len_ds - len_train_ds
                # Randomly split from all datasets
                train_ds, valid_ds = random_split(ds, [len_train_ds, len_valid_ds],
                                                  generator=torch.Generator().manual_seed(data.random_split_seed))
                train_ds = LazyDataset(train_ds, transform=make_transform(data.transform.train))
                valid_ds = LazyDataset(valid_ds, transform=make_transform(data.transform.valid))
            else:
                train_ds = ds
                valid_ds = None
    else:
        raise Exception('Invalid data type.')

    train_dl = DataLoader(train_ds, batch_size=data.batch_size.train, shuffle=True)
    if valid_ds:
        valid_dl = DataLoader(valid_ds, batch_size=data.batch_size.train)
    else:
        valid_dl = None

    return train_dl, valid_dl
