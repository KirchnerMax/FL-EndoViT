import os
import uuid
from datetime import datetime

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Subset
from torchvision import transforms
import torchvision.datasets as datasets
from tqdm import tqdm

from client import EndoViT_client
from set_wandb import setup_wandb


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


@hydra.main(config_path="/conf", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    group_name = datetime.now().strftime("%d%m%H%M")
    # reset clinet training rounds to 0

    for c in os.listdir('/conf/'):
        if c == 'base.yaml':
            continue
        client_cfg = OmegaConf.load(f'/conf/{c}')
        client_cfg.training_rounds = 0
        client_cfg.wandb_id = str(uuid.uuid4())
        client_cfg.group_name = group_name
        OmegaConf.save(client_cfg, f'/conf/{c}')

    # cfg.training.lr = 0.01
    # OmegaConf.save(cfg, f'../src/config/config.yaml')

    means = cfg.dataloader.mean.strip('[]').split(',')
    float_means = list(map(float, means))
    means = torch.FloatTensor(float_means)

    stds = cfg.dataloader.std.strip('[]').split(',')
    float_stds = list(map(float, stds))
    stds = torch.FloatTensor(float_stds)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])

    transform_val = transforms.Compose([
        transforms.Resize([224, 224], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])

    print(f'#### Data path: {cfg.dataloader.data_path_train} #####')
    # before executing change base config dsad to take all ds
    dataset_train = datasets.ImageFolder(cfg.dataloader.data_path_train, transform=transform_train)
    dataset_train = trim_dataset(dataset_train, cfg.DSAD_all, is_train=True)

    dataset_val = datasets.ImageFolder(cfg.dataloader.data_path_val, transform=transform_val)
    dataset_val = trim_dataset(dataset_val, cfg.DSAD_all, is_train=False)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    client = EndoViT_client(data_loader_train, data_loader_val, 0, cfg)

    model = client.get_parameters(dict(cfg))

    for i in tqdm(range(cfg.training.num_rounds)):
        model, steps, loss = client.fit(model, dict(cfg))
        print(f"Round {i}: {steps} steps, {loss} loss")
        r = client.evaluate(model, dict(cfg))
        print(f"Round {i}: {r}")


if __name__ == "__main__":
    main()