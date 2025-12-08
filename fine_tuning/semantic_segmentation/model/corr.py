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
    root = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/semantic_segmentation/output_dir"
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

                        log_csv_name = f"{n_videos[vid]}_experiments_log_sss.csv"
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


def generate_violinplot_seeds_of_metric_score(csv, metric, res_fl, res_cen, resolution, n_vid):
    print(f'res_fl: {res_fl}', flush=True)
    print(f'res_cen: {res_cen}', flush=True)

    # generate plot with 30 subplots 3 rows 10 columns
    fig, axes = plt.subplots(3, 10, figsize=(30, 10))
    axes = axes.flatten()  # Flatten the 2D subplots array into 1D for easier iteration

    for i, (fl, cen) in enumerate(zip(res_fl, res_cen)):

        fl_metrics = get_metrics_for_models(csv, [fl], metric)
        cen_metrics = get_metrics_for_models(csv, [cen], metric)

        fl_data = parse_iou_list(fl_metrics)
        cen_data = parse_iou_list(cen_metrics)

        num_classes = fl_data.shape[1]  # Number of classes (should be 9)

        class_datasets1 = [fl_data[:, i] for i in range(num_classes)]  # Extract Model 1 data by class
        class_datasets2 = [cen_data[:, i] for i in range(num_classes)]  # Extract Model 2 data by class

        overall_dataset1 = fl_data.flatten()
        overall_dataset2 = cen_data.flatten()

        class_datasets1.append(overall_dataset1)
        class_datasets2.append(overall_dataset2)

        for idx in range(num_classes + 1):
            axis_id = i * 10 + idx
            data1 = class_datasets1[idx]
            print(f'len data1: {len(data1)}', flush=True)
            data2 = class_datasets2[idx]
            print(f'len data2: {len(data2)}', flush=True)

            # do statistical test
            p_value, significant = compare_models(data1, data2, paired=False, alpha=0.05)
            print(f'i, idx: {i}, {idx} \n p_value: {p_value}, significant: {significant}', flush=True)
            # print mean metric
            print(f'fl mean: {np.mean(data1)}, cen mean: {np.mean(data2)}', flush=True)

            pos1 = 1  # X-position for Model 1
            pos2 = 2  # X-position for Model 2

            # Create violin plots for each model
            axes[axis_id].violinplot(data1, positions=[pos1], showmeans=False, showmedians=True, widths=0.6)
            axes[axis_id].violinplot(data2, positions=[pos2], showmeans=False, showmedians=True, widths=0.6)

            # Overlay scatter points for Model 1
            jitter1 = np.random.uniform(-0.1, 0.1, size=len(data1))
            axes[axis_id].scatter(np.full(len(data1), pos1) + jitter1, data1, color='blue', alpha=0.6, s=8,
                                  label="Model 1")

            # Overlay scatter points for Model 2
            jitter2 = np.random.uniform(-0.1, 0.1, size=len(data2))
            axes[axis_id].scatter(np.full(len(data2), pos2) + jitter2, data2, color='orange', alpha=0.6, s=8,
                                  label="Model 2")

            # Set titles and formatting
            if idx < num_classes:
                axes[axis_id].set_title(f"Class {idx + 1}", fontsize=10)
            else:
                axes[axis_id].set_title("Overall", fontsize=10)
            axes[axis_id].set_ylabel(f"{metric}")
            # Set y-axis limits for this subplot
            axes[axis_id].set_ylim(-0.05, 1.05)  # Ensure y-axis range is consistent across all subplots
            axes[axis_id].set_xticks([pos1, pos2])
            axes[axis_id].set_xticklabels(["w/FL-B ", "w/Cen-B"], fontsize=8)

            # Add p-value and significance to the subplot as text
            axes[axis_id].text(0.75, 0.95, f"p={p_value:.4f}",
                               fontsize=9, color="black", ha="center", transform=axes[axis_id].transAxes)
            axes[axis_id].text(0.25, 0.95, f"Sig: {'Yes' if significant else 'No'}",
                               fontsize=9, color="red" if significant else "green", ha="center",
                               transform=axes[axis_id].transAxes)
            axes[axis_id].text(0.5, 0.9, f"FL Mean: {np.mean(data1):.4f}",
                               fontsize=9, color="blue", ha="center", transform=axes[axis_id].transAxes)
            axes[axis_id].text(0.5, 0.85, f"FL Mean: {np.mean(data2):.4f}",
                               fontsize=9, color="orange", ha="center", transform=axes[axis_id].transAxes)

    # Layout adjustments
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout leaving room for the title
    fig.suptitle(
        f"Model Comparison of Model with Federated and Centralized Backbone\nMetric {metric} over Classes and Overall",
        y=0.98, fontsize=16)
    plt.savefig(f"images/seeds/{n_vid}_{resolution}_{metric}_model_comparison_violin.png")  # Save the figure
    plt.show()


def extract_run_name(model_string):
    # Extracts Run01, Run02, etc., from the model string
    return next((part for part in model_string.split('_') if part.startswith('Run')), "RunUnknown")


def generate_violinplot_with_hue(csv_file, metric, res_fl, res_cen, resolution, n_vid):
    """
    Generate violin plots comparing Federated Learning (FL) and Centralized Learning (CEN) results
    for a specific metric using Seaborn with hue for differentiation.

    Args:
        csv_file (str): Path to the CSV file containing the data.
        metric (str): Metric to use for comparison ('iou' or 'dice').
        res_fl (list): List of model paths for FL.
        res_cen (list): List of model paths for CEN, corresponding to res_fl.
        resolution (str): The resolution type to save the plot.
        n_vid (str): The description for the number of videos, used in saving the plot.
    """
    sns.set_theme(style="darkgrid")
    # Load the data from CSV
    df = pd.read_csv(csv_file)

    # List to collect rows for the new DataFrame
    consolidated_data = []

    # Iterate over FL and CEN model paths
    for fl_model, cen_model in zip(res_fl, res_cen):
        # Filter data for each FL and CEN model
        fl_rows = df[df['model'] == fl_model]
        cen_rows = df[df['model'] == cen_model]

        # Parse metric values from strings into tensors
        def parse_tensor_column(values):
            parsed_values = []
            for val in values:
                try:
                    # Remove 'tensor(' and ')' if they exist
                    if val.startswith("tensor(") and val.endswith(")"):
                        val = val[7:-1]
                    # Safely evaluate and convert
                    parsed = ast.literal_eval(val)  # Converts to a Python list or number
                    parsed_values.append(torch.tensor(parsed))  # Convert to Tensor
                except (ValueError, SyntaxError, TypeError) as e:
                    print(f"Warning: Skipping invalid value: {val} (Error: {e})")
            return parsed_values

        fl_values = parse_tensor_column(fl_rows[metric])
        cen_values = parse_tensor_column(cen_rows[metric])

        # Flatten the data for violin plot preparation
        for class_idx in range(len(fl_values[0])):  # Assuming all tensors are the same size
            class_fl = [tensor[class_idx].item() for tensor in fl_values]
            class_cen = [tensor[class_idx].item() for tensor in cen_values]

            # Add FL and CEN data to the consolidated list
            consolidated_data.extend([
                {"model": extract_run_name(fl_model), "category": "FL", "values": val, "metric": metric,
                 "class": class_idx + 1}
                for val in class_fl
            ])
            consolidated_data.extend([
                {"model": extract_run_name(cen_model), "category": "CEN", "values": val, "metric": metric,
                 "class": class_idx + 1}
                for val in class_cen
            ])

    # Create a DataFrame for visualization
    plot_df = pd.DataFrame(consolidated_data)

    # Set up the plot
    fig, axes = plt.subplots(4, 9, figsize=(33, 24))
    axes = axes.flatten()  # Flatten the 2D subplots array into 1D for easier iteration

    for idx in range(9):
        sns.violinplot(data=plot_df[plot_df["class"] == idx + 1], x="model", y="values", hue="category", split=True,
                       fill=False, inner="quart", cut=0, linewidth=1.5, ax=axes[idx])

        # calculate p-value
        data1 = plot_df[(plot_df["class"] == idx + 1) & (plot_df["category"] == "FL") & (plot_df["model"] == "Run01")][
            "values"]
        data2 = plot_df[(plot_df["class"] == idx + 1) & (plot_df["category"] == "CEN") & (plot_df["model"] == "Run01")][
            "values"]
        # calculate p-value
        data3 = plot_df[(plot_df["class"] == idx + 1) & (plot_df["category"] == "FL") & (plot_df["model"] == "Run02")][
            "values"]
        data4 = plot_df[(plot_df["class"] == idx + 1) & (plot_df["category"] == "CEN") & (plot_df["model"] == "Run02")][
            "values"]
        # calculate p-value
        data5 = plot_df[(plot_df["class"] == idx + 1) & (plot_df["category"] == "FL") & (plot_df["model"] == "Run03")][
            "values"]
        data6 = plot_df[(plot_df["class"] == idx + 1) & (plot_df["category"] == "CEN") & (plot_df["model"] == "Run03")][
            "values"]

        p_value1, significant1 = compare_models(data1, data2, paired=True, alpha=0.05)
        print(f'idx: {idx} \n p_value: {p_value1}, significant: {significant1}', flush=True)
        print(f'fl mean: {np.mean(data1)}, cen mean: {np.mean(data2)}', flush=True)
        p_value2, significant2 = compare_models(data3, data4, paired=True, alpha=0.05)
        print(f'idx: {idx} \n p_value: {p_value2}, significant: {significant2}', flush=True)
        print(f'fl mean: {np.mean(data3)}, cen mean: {np.mean(data4)}', flush=True)
        p_value3, significant3 = compare_models(data5, data6, paired=True, alpha=0.05)
        print(f'idx: {idx} \n p_value: {p_value3}, significant: {significant3}', flush=True)
        print(f'fl mean: {np.mean(data5)}, cen mean: {np.mean(data6)}', flush=True)

        # Customize the plot
        axes[idx].set_title(f"Class {idx + 1}")
        axes[idx].set_ylabel(f"{metric.capitalize()}")
        axes[idx].set_ylim(-0.05, 1.35)
        axes[idx].set_xticklabels(["1", "2", "3"])
        axes[idx].text(0.15, 0.95, f"p={p_value1:.4f}",
                       fontsize=9, color="black", ha="center", transform=axes[idx].transAxes)
        axes[idx].text(0.45, 0.95, f"p={p_value2:.4f}",
                       fontsize=9, color="black", ha="center", transform=axes[idx].transAxes)
        axes[idx].text(0.75, 0.95, f"p={p_value3:.4f}",
                       fontsize=9, color="black", ha="center", transform=axes[idx].transAxes)
        axes[idx].text(0.15, 0.9, f"Sig: {'Yes' if significant1 else 'No'}",
                       fontsize=9, color="red" if significant1 else "green", ha="center",
                       transform=axes[idx].transAxes)
        axes[idx].text(0.45, 0.9, f"Sig: {'Yes' if significant2 else 'No'}",
                       fontsize=9, color="red" if significant2 else "green", ha="center",
                       transform=axes[idx].transAxes)
        axes[idx].text(0.75, 0.9, f"Sig: {'Yes' if significant3 else 'No'}",
                       fontsize=9, color="red" if significant3 else "green", ha="center",
                       transform=axes[idx].transAxes)
        # Remove legend
        if axes[idx].get_legend() is not None:
            axes[idx].get_legend().remove()

    # Add a shared legend outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()  # Retrieve the legend handles/labels from the first subplot
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, fontsize=12, bbox_to_anchor=(0.5, 1.03))

    # Save the plot
    save_path = f"images/seeds/{n_vid}_{resolution}_{metric}_model_comparison_violin_hue.png"
    plt.savefig(save_path)
    plt.show()

    print(f"Plot saved at: {save_path}")


def compare_models(list1, list2, paired=True, alpha=0.05):
    """
    Compare two models using a statistical test.

    Parameters:
    - list1 (list or array): Data points of Model 1.
    - list2 (list or array): Data points of Model 2.
    - paired (bool): Whether to use a paired t-test or independent t-test.
    - alpha (float): Significance level for the test (default 0.05).

    Returns:
    - p_value (float): The p-value of the test.
    - significant (bool): True if there is a significant difference, False otherwise.
    """
    list1 = np.array(list1)
    list2 = np.array(list2)

    if paired:
        # Paired t-test
        stat, p_value = ttest_rel(list1, list2)
    else:
        # Independent two-sample t-test
        stat, p_value = ttest_ind(list1, list2, equal_var=False)  # Welch's t-test

    significant = p_value < alpha

    return p_value, significant


def generate_plot_for_csv(csv, vid):
    df = pd.read_csv(csv)
    models = df["model"].unique()

    # shorter the model path
    base_path = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/semantic_segmentation/output_dir/"
    models = [model.replace(base_path, "") for model in models if base_path in model]

    # compare endovit with endovit_asam
    models_fl = [model for model in models if "ASAM" in model]
    models_cen = [model for model in models if not "ASAM" in model]

    # check corresponding models
    pairs = find_corresponding_pairs(models_fl, models_cen)

    # print(f'length of fl models: {len(models_fl)}')
    # print(f'length of cen models: {len(models_cen)}')
    # print(f'length of pairs: {len(pairs)}')

    # for i, pair in enumerate(pairs):
    #     print(f"Pair {i}:", flush=True)
    #     generate_violinplot_of_metric_score(csv, 'iou', [base_path + pair[0]], [base_path + pair[1]], i)
    #     generate_violinplot_of_metric_score(csv, 'dice', [base_path + pair[0]], [base_path + pair[1]], i)

    # combine seeds
    low_res_pairs = []
    high_res_pairs = []
    for i, pair in enumerate(pairs):
        # print(f"Pair {i}:", flush=True)
        if pair[0].split('/')[0] == 'low_res':
            low_res_pairs.append(pair)
        else:
            high_res_pairs.append(pair)

    high_res_fl = [base_path + pair[0] for pair in high_res_pairs]
    low_res_fl = [base_path + pair[0] for pair in low_res_pairs]
    high_res_cen = [base_path + pair[1] for pair in high_res_pairs]
    low_res_cen = [base_path + pair[1] for pair in low_res_pairs]

    # generate_violinplot_of_metric_score(csv, 'iou', high_res_fl, high_res_cen, 'no_seeds')
    # generate_violinplot_of_metric_score(csv, 'dice', high_res_fl, high_res_cen, 'no_seeds')
    # generate_violinplot_of_metric_score(csv, 'iou', low_res_fl, low_res_cen, 'no_seeds')
    # generate_violinplot_of_metric_score(csv, 'dice', low_res_fl, low_res_cen, 'no_seeds')

    # generate_violinplot_seeds_of_metric_score(csv, 'iou', high_res_fl, high_res_cen, 'high_res', vid)
    # generate_violinplot_seeds_of_metric_score(csv, 'iou', low_res_fl, low_res_cen, 'low_res', vid)

    generate_violinplot_with_hue(csv, 'iou', high_res_fl, high_res_cen, 'high_res', vid)


def generate_plot_all(csvs, vids, metric='iou', res='high_res'):
    fig, axes = plt.subplots(4, 9, figsize=(33, 24))
    axes = axes.flatten()  # Flatten the 2D subplots array into 1D for easier iteration

    for cvs_i, map in enumerate(zip(csvs, vids)):
        csv, vid = map
        df = pd.read_csv(csv)
        models = df["model"].unique()

        # shorter the model path
        base_path = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/semantic_segmentation/output_dir/"
        models = [model.replace(base_path, "") for model in models if base_path in model]

        # compare endovit with endovit_asam
        models_fl = [model for model in models if "ASAM" in model]
        models_cen = [model for model in models if not "ASAM" in model]

        # check corresponding models
        pairs = find_corresponding_pairs(models_fl, models_cen)

        # combine seeds
        res_pairs = []
        for i, pair in enumerate(pairs):
            # print(f"Pair {i}:", flush=True)
            if pair[0].split('/')[0] == res:
                res_pairs.append(pair)

        res_fl = [base_path + pair[0] for pair in res_pairs]
        res_cen = [base_path + pair[1] for pair in res_pairs]

        consolidated_data = []

        # Iterate over FL and CEN model paths
        for fl_model, cen_model in zip(res_fl, res_cen):
            # Filter data for each FL and CEN model
            fl_rows = df[df['model'] == fl_model]
            cen_rows = df[df['model'] == cen_model]

            # Parse metric values from strings into tensors
            def parse_tensor_column(values):
                parsed_values = []
                for val in values:
                    try:
                        # Remove 'tensor(' and ')' if they exist
                        if val.startswith("tensor(") and val.endswith(")"):
                            val = val[7:-1]
                        # Safely evaluate and convert
                        parsed = ast.literal_eval(val)  # Converts to a Python list or number
                        parsed_values.append(torch.tensor(parsed))  # Convert to Tensor
                    except (ValueError, SyntaxError, TypeError) as e:
                        print(f"Warning: Skipping invalid value: {val} (Error: {e})")
                return parsed_values

            fl_values = parse_tensor_column(fl_rows[metric])
            cen_values = parse_tensor_column(cen_rows[metric])

            # Flatten the data for violin plot preparation
            for class_idx in range(len(fl_values[0])):  # Assuming all tensors are the same size
                class_fl = [tensor[class_idx].item() for tensor in fl_values]
                class_cen = [tensor[class_idx].item() for tensor in cen_values]

                # Add FL and CEN data to the consolidated list
                consolidated_data.extend([
                    {"model": extract_run_name(fl_model), "category": "FL", "values": val, "metric": metric,
                     "class": class_idx + 1}
                    for val in class_fl
                ])
                consolidated_data.extend([
                    {"model": extract_run_name(cen_model), "category": "CEN", "values": val, "metric": metric,
                     "class": class_idx + 1}
                    for val in class_cen
                ])

        # Create a DataFrame for visualization
        plot_df = pd.DataFrame(consolidated_data)

        for idx in range(9):
            axis_idx = cvs_i * 9 + idx
            sns.violinplot(
                data=plot_df[plot_df["class"] == idx + 1],
                x="model",
                y="values",
                hue="category",
                split=True,
                fill=False,
                inner="quart",
                cut=0,
                linewidth=1.5,
                ax=axes[axis_idx],
                order=["Run01", "Run02", "Run03"]  # Specify the desired order of models
            )

            # calculate p-value
            data1 = \
                plot_df[(plot_df["class"] == idx + 1) & (plot_df["category"] == "FL") & (plot_df["model"] == "Run01")][
                    "values"]
            data2 = \
                plot_df[(plot_df["class"] == idx + 1) & (plot_df["category"] == "CEN") & (plot_df["model"] == "Run01")][
                    "values"]
            # calculate p-value
            data3 = \
                plot_df[(plot_df["class"] == idx + 1) & (plot_df["category"] == "FL") & (plot_df["model"] == "Run02")][
                    "values"]
            data4 = \
                plot_df[(plot_df["class"] == idx + 1) & (plot_df["category"] == "CEN") & (plot_df["model"] == "Run02")][
                    "values"]
            # calculate p-value
            data5 = \
                plot_df[(plot_df["class"] == idx + 1) & (plot_df["category"] == "FL") & (plot_df["model"] == "Run03")][
                    "values"]
            data6 = \
                plot_df[(plot_df["class"] == idx + 1) & (plot_df["category"] == "CEN") & (plot_df["model"] == "Run03")][
                    "values"]

            p_value1, significant1 = compare_models(data1, data2, paired=False, alpha=0.05)
            print(f'idx: {idx} \n p_value: {p_value1}, significant: {significant1}', flush=True)
            print(f'fl mean: {np.mean(data1)}, cen mean: {np.mean(data2)}', flush=True)
            p_value2, significant2 = compare_models(data3, data4, paired=False, alpha=0.05)
            print(f'idx: {idx} \n p_value: {p_value2}, significant: {significant2}', flush=True)
            print(f'fl mean: {np.mean(data3)}, cen mean: {np.mean(data4)}', flush=True)
            p_value3, significant3 = compare_models(data5, data6, paired=False, alpha=0.05)
            print(f'idx: {idx} \n p_value: {p_value3}, significant: {significant3}', flush=True)
            print(f'fl mean: {np.mean(data5)}, cen mean: {np.mean(data6)}', flush=True)

            # Customize the plot
            axes[axis_idx].set_title(f"Class {idx + 1}")
            axes[axis_idx].set_ylabel(f"{metric.capitalize()}")
            axes[axis_idx].set_ylim(-0.05, 1.3)
            axes[axis_idx].text(0.15, 0.95, f"p={p_value1:.4f}",
                                fontsize=9, color="black", ha="center", transform=axes[axis_idx].transAxes)
            axes[axis_idx].text(0.45, 0.95, f"p={p_value2:.4f}",
                                fontsize=9, color="black", ha="center", transform=axes[axis_idx].transAxes)
            axes[axis_idx].text(0.75, 0.95, f"p={p_value3:.4f}",
                                fontsize=9, color="black", ha="center", transform=axes[axis_idx].transAxes)
            axes[axis_idx].text(0.15, 0.9, f"Sig: {'Yes' if significant1 else 'No'}",
                                fontsize=9, color="red" if significant1 else "green", ha="center",
                                transform=axes[axis_idx].transAxes)
            axes[axis_idx].text(0.45, 0.9, f"Sig: {'Yes' if significant2 else 'No'}",
                                fontsize=9, color="red" if significant2 else "green", ha="center",
                                transform=axes[axis_idx].transAxes)
            axes[axis_idx].text(0.75, 0.9, f"Sig: {'Yes' if significant3 else 'No'}",
                                fontsize=9, color="red" if significant3 else "green", ha="center",
                                transform=axes[axis_idx].transAxes)
            # Add the text entries with a condition to include a star (*) if one value is better
            axes[axis_idx].text(0.15, 0.85,
                                f"{'* ' if np.median(data1) > np.median(data2) else ''}m: {np.median(data1):.3f}",
                                fontsize=9, color="blue", ha="center", transform=axes[axis_idx].transAxes)

            axes[axis_idx].text(0.15, 0.8,
                                f"{'* ' if np.median(data2) > np.median(data1) else ''}m: {np.median(data2):.3f}",
                                fontsize=9, color="darkorange", ha="center", transform=axes[axis_idx].transAxes)
            axes[axis_idx].text(0.45, 0.85,
                                f"{'* ' if np.median(data3) > np.median(data4) else ''}m: {np.median(data3):.3f}",
                                fontsize=9, color="blue", ha="center", transform=axes[axis_idx].transAxes)
            axes[axis_idx].text(0.45, 0.8,
                                f"{'* ' if np.median(data4) > np.median(data3) else ''}m: {np.median(data4):.3f}",
                                fontsize=9, color="darkorange", ha="center", transform=axes[axis_idx].transAxes)
            axes[axis_idx].text(0.75, 0.85,
                                f"{'* ' if np.median(data5) > np.median(data6) else ''}m: {np.median(data5):.3f}",
                                fontsize=9, color="blue", ha="center", transform=axes[axis_idx].transAxes)
            axes[axis_idx].text(0.75, 0.8,
                                f"{'* ' if np.median(data6) > np.median(data5) else ''}m: {np.median(data6):.3f}",
                                fontsize=9, color="darkorange", ha="center", transform=axes[axis_idx].transAxes)
            # Remove legend
            if axes[axis_idx].get_legend() is not None:
                axes[axis_idx].get_legend().remove()

    # Add a shared legend outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()  # Retrieve the legend handles/labels from the first subplot
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, fontsize=12, bbox_to_anchor=(0.5, 1.03))

    # save plot
    fig.show()
    # save_path = f"images/seeds/all_{metric}_{res}_model_comparison_violin_hue.png"
    # plt.savefig(save_path)

    # print(f"Plot saved at: {save_path}")


if __name__ == "__main__":
    # fill_experiment_csv()
    csv_1 = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/1_vid_only_experiments_log_sss.csv"
    csv_2 = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/2_vids_only_experiments_log_sss.csv"
    csv_4 = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/4_vids_only_experiments_log_sss.csv"
    csv_total = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/_experiments_log_sss.csv"
    csvs = [csv_1, csv_2, csv_4, csv_total]
    vids = ['1_vid_only', '2_vids_only', '4_vids_only', 'total']
    # for csv, vid in zip(csvs, vids):
    #     generate_plot_for_csv(csv, vid)
    generate_plot_all(csvs, vids, 'iou', 'high_res')
    generate_plot_all(csvs, vids, 'iou', 'low_res')
