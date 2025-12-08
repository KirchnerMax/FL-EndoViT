import ast
import csv
import os

import torch
from scipy.stats import ttest_rel, ttest_ind
from torch import amp
from pathlib import Path
import json

from tqdm import tqdm
from src.trainer import Trainer

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns


def parse_iou_list(iou_list):
    """
    Parses the list of IoU string tensors into a single numpy array.

    Args:
        iou_list (list): A list of string representations of [1, 9] tensors.

    Returns:
        np.ndarray: A numpy array of shape (1040, 9).
    """
    parsed_data = []
    for item in iou_list:
        # Convert string to numerical format, remove `tensor()` syntax
        tensor_data = eval(item.replace("tensor(", "").replace(")", ""))
        parsed_data.append(tensor_data)

    # Stack the tensors into a single NumPy array [1040, 9]
    parsed_array = np.stack(parsed_data, axis=0)
    return parsed_array


def get_metrics_for_models(csv_file, models, metric_name):
    """
    Extracts a list of metric scores for a given list of models from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.
        models (list): List of model names to extract metrics for.
        metric_column (str): The column name in the CSV containing the desired metric.

    Returns:
        list: A list of metric values corresponding to the given models.
    """
    # Load CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_file)

    # Initialize an empty list to hold the metric values
    metric_values = []

    # Iterate over each model and extract its metric values
    for model in models:
        for _, row in df.iterrows():  # Iterate over DataFrame rows
            if row["model"] == model:  # Match the model name
                metric_values.append(row[metric_name])  # Collect the metric value

    return metric_values


def add_to_csv(file_path, data, header=None):
    if header is None:
        header = ['seed', 'model', 'image', 'preds', 'targets', 'iou', 'dice', 'acc_per_pixel', 'observed_classes']
    # check if csv file exists
    file_exists = os.path.isfile(file_path)

    # open csv file
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # write header if file does not exist
        if not file_exists:
            writer.writerow(header)

        # write data
        writer.writerow(data)


def load_and_evaluate(trainer, checkpoint, dataloader, config, csv_name):
    print(
        f'##############################################\nLoad and Evaluate: {checkpoint}\n##############################################\nConfig: {config}\n##############################################')
    config["General Hyperparams"]["resume_training"] = checkpoint
    trainer.load_pretrained_checkpoint(config)
    trainer._init_model(config, state_dict=trainer.checkpoint.get("model"))
    trainer.optimizer = trainer._init_optimizer(config, trainer.model, state_dict=trainer.checkpoint.get("optimizer"))
    trainer.scheduler = trainer._init_scheduler(config, trainer.optimizer,
                                                state_dict=trainer.checkpoint.get("scheduler"))
    trainer.scaler = trainer._init_scaler(config, state_dict=trainer.checkpoint.get(
        "scaler")) if trainer.do_optimizations else None
    trainer.model.half().to(trainer.device).eval()

    # metrics
    metrics = trainer.test_metrics

    logits_list, preds_list = [], []
    with torch.no_grad():
        for i, sample in tqdm(enumerate(dataloader)):
            print(sample, flush=True)
            if i % 1 == 0:
                print(f"Batch {i}", flush=True)
                inputs, targets = sample["image"].to(trainer.device, non_blocking=True), sample["mask"][:, 0, :, :].to(
                    trainer.device, non_blocking=True)
                # print(inputs.shape, targets.shape, i)
                with amp.autocast(device_type='cuda'):
                    logits = trainer.model(inputs.half())
                    # add softmax to get probabilities
                    logits = torch.nn.functional.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                logits_list.append(logits.to("cpu").detach().numpy())
                preds_list.append(preds.to("cpu").detach().numpy())

                batch_metrics = metrics(preds, targets)
                print(f"batch metrics: {batch_metrics}", flush=True)

                # add to csv
                data = [config["General Hyperparams"]["seed"],
                        checkpoint,
                        sample["img_path"],
                        preds,
                        targets,
                        batch_metrics["IoU_scores"].to('cpu'),
                        batch_metrics["Dice_scores"].to('cpu'),
                        batch_metrics["Acc_PerPixel"].to('cpu'),
                        batch_metrics["Observed_classes"].to('cpu')]
                add_to_csv(csv_name, data)

                del inputs, targets, logits, preds
                torch.cuda.empty_cache()
    return logits_list, preds_list


def get_config(conf_path):
    conf_path = Path(conf_path)
    assert conf_path.is_file(), f"Config file not found at {conf_path}, check if it is a valid json file."
    with open(conf_path, "r") as read_file:
        return json.load(read_file)


def fill_experiment_csv():
    root = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning_frozen_backbone/semantic_segmentation/output_dir"
    for resolution in os.listdir(root):
        if "." in resolution:
            continue
        for ds_portion in os.listdir(f"{root}/{resolution}"):
            if "." in ds_portion:
                continue
            for model in ["EndoViT", "EndoViT_ASAM"]:
                n_videos = ['']
                if "less_training_data" == ds_portion:
                    n_videos = os.listdir(f"{root}/{resolution}/{ds_portion}/{model}")
                for vid in range(len(n_videos)):
                    # print(resolution, ds_portion, model, n_videos[vid], flush=True)
                    experiments = [e for e in os.listdir(f"{root}/{resolution}/{ds_portion}/{model}/{n_videos[vid]}") if
                                   os.path.isdir(f"{root}/{resolution}/{ds_portion}/{model}/{n_videos[vid]}/{e}")]
                    # get three highest runs of experiments
                    if experiments != []:
                        experiments = sorted(
                            experiments,
                            key=lambda x: int(x.split("_")[1]) if "_" in x and len(x.split("_")) > 1 else -1,
                            reverse=True
                        )[:3]

                    else:
                        print(f"____ No experiments found for {resolution}/{ds_portion}/{model}/{n_videos[vid]}____",
                              flush=True)
                        continue

                    for experiment in experiments:
                        print(f"################ experiment {experiment}", flush=True)
                        run = experiment.split("_")
                        run = [s for s in run if s.startswith("Run")]

                        # load config
                        config_path = [f for f in
                                       os.listdir(f"{root}/{resolution}/{ds_portion}/{model}/{n_videos[vid]}/") if
                                       f"{run[0]}" in f and ".json" in f]
                        print(config_path, flush=True)
                        config_path = f"{root}/{resolution}/{ds_portion}/{model}/{n_videos[vid]}/{config_path[0]}"
                        print(f"config {config_path}", flush=True)

                        config = get_config(config_path)

                        config["General Hyperparams"]["batch_size"] = 1
                        config["Loggers"]["WandB"]["enable"] = False
                        config["General Hyperparams"]["test_endovit"] = True

                        # load checkpoint
                        checkpoint = f"{root}/{resolution}/{ds_portion}/{model}/{n_videos[vid]}/{experiment}/"
                        file = [f for f in os.listdir(checkpoint) if ".pth" in f]
                        checkpoint += file[0]

                        # load trainer
                        trainer = Trainer(config)
                        dataloader = trainer.test_dataloader

                        print(f"experiment {experiment}", flush=True)
                        print(f"checkpoint {checkpoint}", flush=True)

                        log_csv_name = f"{n_videos[vid]}_experiments_log_sss_frozen.csv"
                        load_and_evaluate(trainer, checkpoint, dataloader, config, log_csv_name)


def find_corresponding_pairs(models_fl, models_cen):
    pairs = []
    for fl_model in models_fl:
        # print(f'######################### Iteration #################, fl_model: {fl_model}', flush=True)
        # Extract relevant parts from the federated learning path
        fl_parts = fl_model.split('/')
        fl_resolution = fl_parts[0]  # 'high_res' or 'low_res'
        fl_train_portion = fl_parts[1]  # 'less_training_data' or 'full_dataset'
        if 'less_training_data' == fl_train_portion:
            fl_n_videos = fl_parts[3]  # 1_vid_only, 2_vids_only, 3_vids_only
        fl_train_data = fl_parts[-2]  # run ...
        # print(f'#####fl_train_data: {fl_train_data}', flush=True)
        fl_train_data_parts = fl_train_data.split('_')[10:]

        for cen_model in models_cen:
            # Extract relevant parts from the centralized learning path
            cen_parts = cen_model.split('/')
            cen_resolution = cen_parts[0]
            # print(f'cen_resolution: {cen_resolution}, fl_resolution: {fl_resolution}', flush=True)
            if cen_resolution != fl_resolution:
                # print('### resolution', flush=True)
                continue

            cen_train_portion = cen_parts[1]
            # print(f'cen_train_portion: {cen_train_portion}, fl_train_portion: {fl_train_portion}', flush=True)
            if cen_train_portion != fl_train_portion:
                # print('### proportion', flush=True)
                continue

            match = None
            if 'less_training_data' == cen_train_portion:
                cen_n_videos = cen_parts[3]  # 1_vid_only, 2_vids_only, 3_vids_only
                match = cen_n_videos == fl_n_videos

            # seed
            cen_train_data = cen_parts[-2]  # run ...
            # print(f'#####cen_train_data: {cen_train_data}', flush=True)
            cen_train_data_parts = cen_train_data.split('_')[9:]
            # print(f'fl_train_data_parts: {fl_train_data_parts}, cen_train_data_parts: {cen_train_data_parts}',
            #       flush=True)
            if cen_train_data_parts != fl_train_data_parts:
                # print('### seed', flush=True)
                continue

            # Match based on resolution, dataset portion, and run_id

            # print(f'match', flush=True)
            # print(f'fl_resolution: {fl_resolution}, cen_resolution: {cen_resolution}', flush=True)
            # print(f'fl_train_portion: {fl_train_portion}, cen_train_portion: {cen_train_portion}', flush=True)
            # print(f'fl_train_data: {fl_train_data}, cen_train_data: {cen_train_data}', flush=True)

            pairs.append((fl_model, cen_model))

    return pairs



if __name__ == "__main__":
    fill_experiment_csv()
