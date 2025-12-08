import os.path
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection
from scipy.stats import wilcoxon


def fill_csv(log, seed):
    header = ['backbone', 'n_videos', 'seed', 'video', 'map']
    log_parts = log.split('/')
    if 'full_dataset' in log_parts:
        n_videos = 'total'
    else:
        pattern = r"(\d+)_videos_only"
        match = re.search(pattern, log)
        n_videos = match.group(1)
    if 'EndoViT_FedASAM' in log:
        backbone = 'FL-B'
    else:
        backbone = 'CEN-B'

    # Regular expression pattern to match desired lines
    pattern = r"mAP_ivt_video\s+(\d+):\s+([0-9.]+)"
    # Store extracted lines
    results = {}

    # Read the log file and extract matching lines
    items = []
    with open(log, "r") as file:
        for line in file:
            match = re.search(pattern, line.strip())
            if match:
                x = int(match.group(1))  # Extract the x-value as integer
                y = float(match.group(2))  # Extract the y-value as float
                items.append([backbone, n_videos, seed, x, y])

    # Store the extracted values in atr.csv
    # If new with header otherwise just attach
    if not os.path.exists("atr.csv"):
        with open("atr.csv", "w") as file:
            file.write(",".join(header) + "\n")
    with open("atr.csv", "a") as file:
        for item in items:
            file.write(",".join(map(str, item)) + "\n")


def get_colors(statistical_relevant, means):
    seed_palette = {}
    for i in range(3):
        if not statistical_relevant[i]:
            # Fail to reject H0: (the distributions of the two samples are equal)
            seed_palette["Run0" + str(i + 1) + "FL"] = "#E2E2E2"  # gray
            seed_palette["Run0" + str(i + 1) + "CEN"] = "#E2E2E2"
        else:
            # Reject H0: (the distributions of the two samples are not equal)
            if means[2 * i] > means[2 * i + 1]:
                seed_palette["Run0" + str(i + 1) + "FL"] = "#998ec3"  # purple
                seed_palette["Run0" + str(i + 1) + "CEN"] = "#E2E2E2"
            else:
                seed_palette["Run0" + str(i + 1) + "FL"] = "#E2E2E2"  # orange
                seed_palette["Run0" + str(i + 1) + "CEN"] = "#f1a340"
    return seed_palette


def get_plot():
    header = ['backbone', 'n_videos', 'seed', 'video', 'map']
    df = pd.read_csv(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/atr.csv",
        header=0, names=header)

    # Plot the data
    unique_n_videos = df['n_videos'].nunique()
    fig, ax = plt.subplots(unique_n_videos, 1, figsize=(8.27, 11.69))
    fig.suptitle(f'Action Triplet Recognition (Fully Fine-Tuned)', fontsize=20, weight='bold', y=0.95)  # Add overall title

    for n, n_video in enumerate(df['n_videos'].unique()):
        print(f"Number of videos: {n_video}")
        sns.violinplot(data=df[df['n_videos'] == n_video], x='seed', y='map', hue='backbone',
                       hue_order=['FL-B', 'CEN-B'],
                       split=True, inner='quarts', linewidth=1.5, fill=True, ax=ax[n])
        sns.stripplot(data=df[df['n_videos'] == n_video], x='seed', y='map', hue='backbone',
                      hue_order=['FL-B', 'CEN-B'], palette=['#9D02D7', '#D77A02'],
                      dodge=True, jitter=True, size=4, ax=ax[n], alpha=0.7)

        ax[n].set_title(f'{n_video} Videos', fontsize=16)
        ax[n].set_xlabel('Seed', fontsize=15)
        ax[n].set_ylabel('mAP', fontsize=15)
        ax[n].set_ylim([-0.05, 1.05])
        ax[n].tick_params(axis='x', labelsize=12)
        ax[n].tick_params(axis='y', labelsize=12)
        ax[n].get_legend().remove()

        seed_palette = {}

        for i, seed in enumerate(df['seed'].unique()):
            data_fl = df[(df['backbone'] == 'FL-B') & (df['n_videos'] == n_video) & (df['seed'] == seed)][
                "map"].to_numpy()
            data_cen = df[(df['backbone'] == 'CEN-B') & (df['n_videos'] == n_video) & (df['seed'] == seed)][
                "map"].to_numpy()

            # print(data_fl)
            # print(data_cen)

            # mean fl
            mean_fl = np.mean(data_fl)
            mean_cen = np.mean(data_cen)
            # standard deviation
            std_fl = np.std(data_fl)
            std_cen = np.std(data_cen)

            print(f"Seed: {seed}")
            print(f"FL-B: {mean_fl:.4f} ± {std_fl:.4f}")
            print(f"CEN-B: {mean_cen:.4f} ± {std_cen:.4f}")

            delta = (data_fl - data_cen) * 100
            # print(delta)

            try:
                stat, p = wilcoxon(x=data_fl, y=data_cen)
            except ValueError:
                p = np.nan
                stat = np.nan

            print(f"Statistic={stat}, p-value={p}")
            if p < 0.01:
                print(
                    f"[{seed}] Statistical relevant: p-value={p} < 0.01 --> Reject H0 --> (the distributions of the two samples are not equal)")
            else:
                print(
                    f"[{seed}] Statistical not relevant: p-value={p} >= 0.01 --> Fail to reject H0 --> (the distributions of the two samples are equal)")

            # print p-value on the plot
            ax[n].annotate(f"p={p:.4f}", xy=(0.35 * (0.5 + i), 0.9), xycoords='axes fraction', ha='center', fontsize=11,
                           color='black')
            annotation_text = "Reject H0" if p < 0.01 else "Fail to reject H0"
            ax[n].annotate(annotation_text, xy=(0.35 * (0.5 + i), 0.8), xycoords='axes fraction', ha='center',
                           fontsize=11, color='black')
            # plot means and say which one is better if Reject H0
            if p < 0.01:
                if mean_fl > mean_cen:
                    delta = mean_fl - mean_cen
                    ax[n].annotate(f"FL-BB better by {delta * 100:.2f}%", xy=(0.35 * (0.5 + i), 0.7),
                                   xycoords='axes fraction', ha='center', fontsize=11,
                                   color='black')
                    seed_palette["Run0" + str(i + 1) + "FL"] = "#998ec3"  # purple
                    seed_palette["Run0" + str(i + 1) + "CEN"] = "#E2E2E2"
                else:
                    delta = mean_cen - mean_fl
                    ax[n].annotate(f"CEN-BB better by {delta * 100:.2f}%", xy=(0.35 * (0.5 + i), 0.7),
                                   xycoords='axes fraction', ha='center', fontsize=11,
                                   color='black')
                    seed_palette["Run0" + str(i + 1) + "FL"] = "#E2E2E2"  # orange
                    seed_palette["Run0" + str(i + 1) + "CEN"] = "#f1a340"
            else:
                # Fail to reject H0: (the distributions of the two samples are equal)
                seed_palette["Run0" + str(i + 1) + "FL"] = "#E2E2E2"  # gray
                seed_palette["Run0" + str(i + 1) + "CEN"] = "#E2E2E2"

        # Identify all PolyCollection objects (which are the violin shapes)
        violin_polygons = [c for c in ax[n].collections if isinstance(c, PolyCollection)]
        print(len(violin_polygons))
        seed_list = ['Run01FL', 'Run01CEN', 'Run02FL', 'Run02CEN', 'Run03FL', 'Run03CEN']
        print(seed_palette)

        for seed, violin in zip(seed_list, violin_polygons):
            if seed in seed_palette:
                violin.set_facecolor(seed_palette[seed])  # Apply correct color

    plt.tight_layout()
    # handles, labels = ax[-1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12, frameon=False)
    plt.subplots_adjust(hspace=0.5, top=0.9)
    plt.show()

    # save plot
    fig.savefig(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/images/atd/1.png")


if __name__ == "__main__":
    # base = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2"
    # # Full dataset
    # # seed 1665
    # fill_csv(base + "/finetuning/action_triplet_detection/output_dir/full_dataset/ViT_backbone/EndoViT/run_0004_10:50-23.01.25__FullDataset_EndoViT_Seed_1665/mae_cholect45-crossval_k5.log", 1665)
    # fill_csv(base + "/finetuning/action_triplet_detection/output_dir/full_dataset/ViT_backbone/EndoViT_FedASAM/run_0007_11:39-23.01.25__FullDataset_EndoViT_Seed_1665/mae_cholect45-crossval_k5.log", 1665)
    # # seed 8914
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/full_dataset/ViT_backbone/EndoViT/run_0005_11:01-23.01.25__FullDataset_EndoViT_Seed_8914/mae_cholect45-crossval_k5.log", 8914)
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/full_dataset/ViT_backbone/EndoViT_FedASAM/run_0008_11:52-23.01.25__FullDataset_EndoViT_Seed_8914/mae_cholect45-crossval_k5.log", 8914)
    # # seed 37
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/full_dataset/ViT_backbone/EndoViT/run_0006_11:13-23.01.25__FullDataset_EndoViT_Seed_37/mae_cholect45-crossval_k5.log", 37)
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/full_dataset/ViT_backbone/EndoViT_FedASAM/run_0009_12:03-23.01.25__FullDataset_EndoViT_Seed_37/mae_cholect45-crossval_k5.log", 37)
    # # 2 videos only
    # # seed 1665
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT/2_videos_only/run_0001_14:58-24.01.25__2VidsOnly_EndoViT_Videos_27_5/mae_cholect45-crossval_k5.log", 1665)
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/2_videos_only/run_0007_08:51-24.01.25__2VidsOnly_EndoViT_Videos_27_5/mae_cholect45-crossval_k5.log", 1665)
    # # seed 8914
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT/2_videos_only/run_0002_15:10-24.01.25__2VidsOnly_EndoViT_Videos_70_47/mae_cholect45-crossval_k5.log", 8914)
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/2_videos_only/run_0008_09:05-24.01.25__2VidsOnly_EndoViT_Videos_70_47/mae_cholect45-crossval_k5.log", 8914)
    # # seed 37
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT/2_videos_only/run_0003_15:22-24.01.25__2VidsOnly_EndoViT_Videos_51_50/mae_cholect45-crossval_k5.log", 37)
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/2_videos_only/run_0009_09:19-24.01.25__2VidsOnly_EndoViT_Videos_51_50/mae_cholect45-crossval_k5.log", 37)
    # # 4 videos only
    # # seed 1665
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT/4_videos_only/run_0001_15:34-24.01.25__4VidsOnly_EndoViT_Videos_36_14_57_15/mae_cholect45-crossval_k5.log", 1665)
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/4_videos_only/run_0007_09:33-24.01.25__4VidsOnly_EndoViT_Videos_36_14_57_15/mae_cholect45-crossval_k5.log", 1665)
    # # seed 8914
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT/4_videos_only/run_0002_15:46-24.01.25__4VidsOnly_EndoViT_Videos_23_66_31_27/mae_cholect45-crossval_k5.log", 8914)
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/4_videos_only/run_0008_09:47-24.01.25__4VidsOnly_EndoViT_Videos_23_66_31_27/mae_cholect45-crossval_k5.log", 8914)
    # # seed 37
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT/4_videos_only/run_0003_15:58-24.01.25__4VidsOnly_EndoViT_Videos_42_31_6_52/mae_cholect45-crossval_k5.log", 37)
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/4_videos_only/run_0009_10:01-24.01.25__4VidsOnly_EndoViT_Videos_42_31_6_52/mae_cholect45-crossval_k5.log", 37)
    # # 8 videos only
    # # seed 1665
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT/8_videos_only/run_0001_16:10-24.01.25__8VidsOnly_EndoViT_Videos_18_57_51_60_27_68_36_48/mae_cholect45-crossval_k5.log", 1665)
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/8_videos_only/run_0007_10:15-24.01.25__8VidsOnly_EndoViT_Videos_18_57_51_60_27_68_36_48/mae_cholect45-crossval_k5.log", 1665)
    # # seed 8914
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT/8_videos_only/run_0002_16:22-24.01.25__8VidsOnly_EndoViT_Videos_2_36_5_47_48_8_6_18/mae_cholect45-crossval_k5.log", 8914)
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/8_videos_only/run_0008_10:29-24.01.25__8VidsOnly_EndoViT_Videos_2_36_5_47_48_8_6_18/mae_cholect45-crossval_k5.log", 8914)
    # # seed 37
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT/8_videos_only/run_0003_16:34-24.01.25__8VidsOnly_EndoViT_Videos_47_57_8_80_15_68_40_27/mae_cholect45-crossval_k5.log", 37)
    # fill_csv("/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/8_videos_only/run_0009_10:42-24.01.25__8VidsOnly_EndoViT_Videos_47_57_8_80_15_68_40_27/mae_cholect45-crossval_k5.log", 37)

    get_plot()
