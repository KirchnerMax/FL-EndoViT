import torch
from omegaconf import DictConfig
from torch.utils.data import Subset
from torchvision import transforms, datasets
from tqdm import tqdm


def prepare_datasets(cfg: DictConfig):
    """Return datasets for each client."""
    trainloaders = []
    validationloaders = []

    means = cfg.dataloader.mean.strip('[]').split(',')
    float_means = list(map(float, means))
    means = torch.FloatTensor(float_means)

    stds = cfg.dataloader.std.strip('[]').split(',')
    float_stds = list(map(float, stds))
    stds = torch.FloatTensor(float_stds)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(cfg.dataloader.input_size, scale=(0.6, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])

    # validation dataset
    transform_val = transforms.Compose([
        transforms.Resize([cfg.dataloader.input_size, cfg.dataloader.input_size],
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])

    print(f'Loading training datasets from: {cfg.dataloader.data_path_train}')
    dataset_train = datasets.ImageFolder(cfg.dataloader.data_path_train, transform=transform_train)
    print(f'Loading testing datasets from: {cfg.dataloader.data_path_val}')
    dataset_val = datasets.ImageFolder(cfg.dataloader.data_path_val, transform=transform_val)

    print(f'prepare datasets on: {cfg.dataloader.client_names}')
    for client in tqdm(cfg.dataloader.client_names):
        ############################ DATA ############################
        train_ds = trim_dataset(dataset_train, cfg[client], is_train=True)
        val_ds = trim_dataset(dataset_val, cfg[client], is_train=False)

        print("-" * 50)
        print(f"Total train #images for client {client}: {len(train_ds)}")
        print(f"Total val #images for client {client}: {len(val_ds)}")
        print("-" * 50)

        # Debug limit: restrict number of samples if cfg.dataloader.debug_limit is set
        N = cfg.dataloader.N
        if N is not None:
            train_ds = Subset(train_ds, range(min(N, len(train_ds))))
            val_ds = Subset(val_ds, range(min(N, len(val_ds))))

        dl_train = torch.utils.data.DataLoader(
            train_ds,
            batch_size=cfg.dataloader.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            drop_last=True
        )

        dl_val = torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg.dataloader.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            drop_last=False
        )
        ############################ DATA ############################
        trainloaders.append(dl_train)
        validationloaders.append(dl_val)
    return trainloaders, validationloaders

def trim_dataset(dataset, args, is_train=True):
    possible_subfolders = dataset.class_to_idx.keys()

    valid_subfolders = get_valid_folders(args, possible_subfolders, is_train)

    train_val_text = "Train" if is_train else "Validation"

    # select the indices of valid folders
    if (valid_subfolders != possible_subfolders):
        indices = []
        for folder in valid_subfolders:
            idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] == dataset.class_to_idx[folder]]
            indices.extend(idx)

        indices.sort()
        print(f"{train_val_text} subdatasets {valid_subfolders} will be used.")
        return Subset(dataset, indices)

    else:
        print(f"All {train_val_text} subdatasets will be used.")
        return dataset


def get_valid_folders(args, possible_folders, is_train):
    valid_folders = []
    invalid_folders = []

    train_val_text = "Train" if is_train else "Validation"

    if (is_train):
        if (
        not args.train_datasets_to_take):  # if args.train_datasets_to_take is an empty list, then take all possible folders
            return possible_folders
    else:
        if (
                not args.val_datasets_to_take):  # if args.val_datasets_to_take is an empty list, then take all possible folders
            return possible_folders

    for folder in (args.train_datasets_to_take if is_train else args.val_datasets_to_take):
        if (folder in possible_folders):
            valid_folders.append(folder)
        else:
            invalid_folders.append(folder)

    if (invalid_folders):
        print(f"WARNING: Following {train_val_text} folders were not found: {invalid_folders}")

    return valid_folders
