import glob
import os
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection
from scipy.stats import wilcoxon


def fill_csv(path, seed):
    header = ['stage', 'seed', 'n_video', 'video', 'backbone', 'accuracy', 'f1']

    # extract seed from path search for seed_x where x is the seed
    path_split = path.split('/')
    stage = 'FE'
    n_videos = 'full' if 'full_dataset' in path_split else '2' if '2_videos_only' in path_split else '4' if '4_videos_only' in path_split else '8'
    backbone = 'FL-BB' if 'EndoViT_FedASAM' in path_split else 'CEN-BB'

    for file in glob.iglob(f"{path}/*.txt", recursive=False):
        with open(file, 'r') as f:
            # extract video id, accuracy and f1
            lines = f.readlines()
            line_content = lines[0].split(';')
            vid = line_content[0].split(' ')[-1]
            if int(vid) < 49:
                # print(int(vid))
                continue
            acc = line_content[1].split(' ')[-1]
            f1 = line_content[-1].split(' ')[-1]

            print(f'Writing to csv: {stage}, {seed}, {n_videos}, {int(vid)}, {backbone}, {acc}, {f1}')
            # Store the extracted values in spr.csv
            # If new with header otherwise just attach
            if not os.path.exists("spr.csv"):
                with open("spr.csv", "w") as file:
                    file.write(",".join(header) + "\n")
            with open("spr.csv", "a") as file:
                file.write(f"{stage},{seed},{n_videos},{int(vid)},{backbone},{acc},{f1}\n")


def get_plot(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)

    # Dynamically determine the number of subplots needed based on unique 'n_video' values
    unique_n_videos = df['n_video'].unique()
    num_subplots = len(unique_n_videos)

    # Calculate rows and columns to optimize space in a grid layout
    num_cols = 1  # Set number of columns
    num_rows = math.ceil(num_subplots / num_cols)  # Determine required rows for the grid

    # Adjust figure size dynamically based on number of subplots
    fig, ax = plt.subplots(
        num_rows, num_cols, figsize=(22, 16),
        squeeze=False  # Ensures consistency even with one row or column
    )
    ax = ax.ravel()  # Flatten the array for easier indexing

    # Plot each `n_video` group
    for j, n in enumerate(unique_n_videos):
        for stage in df['stage'].unique():
            print(f"Plotting: Stage: {stage}, n_videos: {n}")
            data = df[(df['stage'] == stage) & (df['n_video'] == n)]

            # Violin plot for the current `n_video` and stage
            sns.violinplot(
                data=data, x='seed', y='accuracy', hue='backbone',
                hue_order=['FL-BB', 'CEN-BB'], split=True, inner='quartile',
                linewidth=1.5, fill=False, ax=ax[j]
            )

            # Add strip plot for additional details
            sns.stripplot(
                data=data, x='seed', y='accuracy', hue='backbone',
                hue_order=['FL-BB', 'CEN-BB'], dodge=True, jitter=True, size=4,
                alpha=0.7, palette='dark', ax=ax[j]
            )

        # Set labels and titles for the subplot
        ax[j].set_title(f"n_videos: {n}", fontsize=12)
        ax[j].set_ylim(-0.05, 1.3)
        ax[j].set_ylabel('Accuracy')
        ax[j].set_xlabel('Seed')
        ax[j].legend(title="Backbone", loc='best', fontsize=8, frameon=False)

        for i, seed in enumerate(df['seed'].unique()):
            data_fl = data[(data['n_video'] == n) & (data['seed'] == seed) & (data['backbone'] == 'FL-BB')]['accuracy']
            data_cen = data[(data['n_video'] == n) & (data['seed'] == seed) & (data['backbone'] == 'CEN-BB')][
                'accuracy']

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
            print(delta)

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
            ax[j].annotate(f"p={p:.4f}", xy=(0.35 * (0.5 + i), 0.9), xycoords='axes fraction', ha='center', fontsize=10,
                           color='black', weight='bold')
            annotation_text = "Reject H0" if p < 0.01 else "Fail to reject H0"
            ax[j].annotate(annotation_text, xy=(0.35 * (0.5 + i), 0.85), xycoords='axes fraction', ha='center',
                           fontsize=10, color='black', weight='bold')

    # Turn off unused subplots if any
    for j in range(len(ax)):
        if j >= num_subplots:
            ax[j].axis('off')

    # Adjust plot layout for better visibility
    plt.tight_layout(pad=2.0, w_pad=3.0, h_pad=2.5)
    plt.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)
    plt.savefig('spr_optimized.png')  # Save as an optimized plot
    plt.show()


def fill_csv_stage_one():
    paths = []
    txts = []
    # full ds
    ## FL
    ### 37
    #### Stage 1
    paths.append(
        '/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/surgical_phase_recognition/output_dir/full_dataset/ViT_backbone/EndoViT_FedASAM/seed_37/21:47-25.10.24_FeatureExtraction_FullDataset_EndoViT_FedASAM_Seed_37/cholec80_pickle_export/1.0fps')
    #### Stage 2
    ### 1665
    #### Stage 1
    paths.append(
        '/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/surgical_phase_recognition/output_dir/full_dataset/ViT_backbone/EndoViT_FedASAM/seed_1665/14:17-25.10.24_FeatureExtraction_FullDataset_EndoViT_FedASAM_Seed_1665/cholec80_pickle_export/1.0fps')
    #### Stage 2
    ### 8914
    #### Stage 1
    paths.append(
        '/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/surgical_phase_recognition/output_dir/full_dataset/ViT_backbone/EndoViT_FedASAM/seed_8914/18:23-25.10.24_FeatureExtraction_FullDataset_EndoViT_FedASAM_Seed_8914/cholec80_pickle_export/1.0fps')
    #### Stage 2

    ## CEN
    ### 37
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/00_original_code/EndoViT/finetuning/surgical_phase_recognition/output_dir/full_dataset/ViT_backbone/EndoViT/seed_37/21:10-15.10.24_FeatureExtraction_FullDataset_EndoViT_Seed_37/cholec80_pickle_export/1.0fps")
    #### Stage 2
    ### 1665
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/00_original_code/EndoViT/finetuning/surgical_phase_recognition/output_dir/full_dataset/ViT_backbone/EndoViT/seed_1665/10:07-15.10.24_FeatureExtraction_FullDataset_EndoViT_Seed_1665/cholec80_pickle_export/1.0fps")
    #### Stage 2
    ### 8914
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/00_original_code/EndoViT/finetuning/surgical_phase_recognition/output_dir/full_dataset/ViT_backbone/EndoViT/seed_8914/16:44-15.10.24_FeatureExtraction_FullDataset_EndoViT_Seed_8914/cholec80_pickle_export/1.0fps")
    #### Stage 2

    # 2 videos
    ## FL
    ### 37
    #### Stage 1
    paths.append(
        '/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/2_videos_only/videos_22_40/21:22-25.10.24_FeatureExtraction_2VidsOnly_EndoViT_FedASAM_Videos_22_40/cholec80_pickle_export/1.0fps')
    #### Stage 2
    ### 1665
    #### Stage 1
    paths.append(
        '/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/2_videos_only/videos_25_1/14:43-25.10.24_FeatureExtraction_2VidsOnly_EndoViT_FedASAM_Videos_25_1/cholec80_pickle_export/1.0fps')
    #### Stage 2
    ### 8914
    #### Stage 1
    paths.append(
        '/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/2_videos_only/videos_33_11/18:24-25.10.24_FeatureExtraction_2VidsOnly_EndoViT_FedASAM_Videos_33_11/cholec80_pickle_export/1.0fps')
    #### Stage 2

    ## CEN
    ### 37
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/00_original_code/EndoViT/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT/2_videos_only/videos_22_40/17:42-15.10.24_FeatureExtraction_2VidsOnly_EndoViT_Videos_22_40/cholec80_pickle_export/1.0fps")
    #### Stage 2
    ### 1665
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/00_original_code/EndoViT/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT/2_videos_only/videos_25_1/10:10-15.10.24_FeatureExtraction_2VidsOnly_EndoViT_Videos_25_1/cholec80_pickle_export/1.0fps")
    #### Stage 2
    ### 8914
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/00_original_code/EndoViT/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT/2_videos_only/videos_33_11/14:09-15.10.24_FeatureExtraction_2VidsOnly_EndoViT_Videos_33_11/cholec80_pickle_export/1.0fps")
    #### Stage 2

    # 4 videos
    ## FL
    ### 37
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/4_videos_only/videos_5_17_25_31/03:11-26.10.24_FeatureExtraction_4VidsOnly_EndoViT_FedASAM_Videos_5_17_25_31/cholec80_pickle_export/1.0fps")
    #### Stage 2
    ### 1665
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/4_videos_only/videos_8_28_13_4/00:15-26.10.24_FeatureExtraction_4VidsOnly_EndoViT_FedASAM_Videos_8_28_13_4/cholec80_pickle_export/1.0fps")
    #### Stage 2
    ### 8914
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/4_videos_only/videos_24_28_38_15/06:27-26.10.24_FeatureExtraction_4VidsOnly_EndoViT_FedASAM_Videos_24_28_38_15/cholec80_pickle_export/1.0fps")
    #### Stage 2

    ## CEN
    ### 37
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/00_original_code/EndoViT/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT/4_videos_only/videos_5_17_25_31/23:34-15.10.24_FeatureExtraction_4VidsOnly_EndoViT_Videos_5_17_25_31/cholec80_pickle_export/1.0fps")
    #### Stage 2
    ### 1665
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/00_original_code/EndoViT/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT/4_videos_only/videos_8_28_13_4/20:56-15.10.24_FeatureExtraction_4VidsOnly_EndoViT_Videos_8_28_13_4/cholec80_pickle_export/1.0fps")
    #### Stage 2
    ### 8914
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/00_original_code/EndoViT/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT/4_videos_only/videos_24_28_38_15/02:25-16.10.24_FeatureExtraction_4VidsOnly_EndoViT_Videos_24_28_38_15/cholec80_pickle_export/1.0fps")
    #### Stage 2

    # 8 videos
    ## FL
    ### 37
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/8_videos_only/videos_1_22_12_8_6_33_31_29/09:30-26.10.24_FeatureExtraction_8VidsOnly_EndoViT_FedASAM_Videos_1_22_12_8_6_33_31_29/cholec80_pickle_export/1.0fps")
    #### Stage 2
    ### 1665
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/8_videos_only/videos_18_28_17_2_40_36_7_34/16:00-26.10.24_FeatureExtraction_8VidsOnly_EndoViT_FedASAM_Videos_18_28_17_2_40_36_7_34/cholec80_pickle_export/1.0fps")
    #### Stage 2
    ### 8914
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT_FedASAM/8_videos_only/videos_20_38_25_7_29_4_2_24/12:50-26.10.24_FeatureExtraction_8VidsOnly_EndoViT_FedASAM_Videos_20_38_25_7_29_4_2_24/cholec80_pickle_export/1.0fps")
    #### Stage 2

    ## CEN
    ### 37
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/00_original_code/EndoViT/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT/8_videos_only/videos_1_22_12_8_6_33_31_29/05:40-16.10.24_FeatureExtraction_8VidsOnly_EndoViT_Videos_1_22_12_8_6_33_31_29/cholec80_pickle_export/1.0fps")
    #### Stage 2
    ### 1665
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/00_original_code/EndoViT/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT/8_videos_only/videos_18_28_17_2_40_36_7_34/12:01-16.10.24_FeatureExtraction_8VidsOnly_EndoViT_Videos_18_28_17_2_40_36_7_34/cholec80_pickle_export/1.0fps")
    #### Stage 2
    ### 8914
    #### Stage 1
    paths.append(
        "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/00_original_code/EndoViT/finetuning/surgical_phase_recognition/output_dir/less_training_data/ViT_backbone/EndoViT/8_videos_only/videos_20_38_25_7_29_4_2_24/08:37-16.10.24_FeatureExtraction_8VidsOnly_EndoViT_Videos_20_38_25_7_29_4_2_24/cholec80_pickle_export/1.0fps")
    #### Stage 2

    for i, path in enumerate(paths):
        if (i + 1) % 3 == 1:
            fill_csv(path, 37)
        elif (i + 1) % 3 == 2:
            fill_csv(path, 1665)
        else:
            fill_csv(path, 8914)


def fill_csv_two(log, seed):
    header = ['stage', 'seed', 'n_video', 'video', 'backbone', 'accuracy', 'f1']

    # Dictionary to store results
    acc = []
    cur_acc = {}
    s1 = False
    s2 = False
    f1 = False

    # Read the log file
    with open(log, "r") as file:
        for line in file:
            # Check if the line contains "Stage" and "Accuracy"
            if "Stage" in line and "Accuracy" in line:
                parts = line.split()
                stage = parts[1].lower()  # Extract stage (e.g., "S1", "S2") and convert to lowercase
                accuracy = float(parts[3])  # Extract accuracy value as a float
                cur_acc[f"{stage}_acc"] = accuracy  # Add to dictionary
                if stage == 's1':
                    s1 = True
                elif stage == 's2':
                    s2 = True
            if "f1_score_per_video: " in line:
                # extract from tensor
                f1 = float(line.split(' ')[2][7:-1])
                cur_acc['f1'] = f1
                f1 = True
            if s1 and s2 and f1:
                # print(cur_acc)
                acc.append(cur_acc.copy())
                s1 = False
                s2 = False
                f1 = False
    # print(acc)

    s = seed.split('_')[1:]
    if len(s) > 1:
        if s in [['1', '22', '12', '8', '6', '33', '31', '29'], ['25', '1'], ['8', '28', '13', '4']]:
            seed = 8914
        elif s in [['20', '38', '25', '7', '29', '4', '2', '24'], ['33', '11'], ['5', '17', '25', '31']]:
            seed = 1665
        elif s in [['18', '28', '17', '2', '40', '36', '7', '34'], ['22', '40'], ['24', '28', '38', '15']]:
            seed = 37
    else:
        seed = int(s[0])

    # print(seed)

    bb = 'FL-BB' if 'EndoViT_FedASAM' in log or 'EndoViT_ASAM' in log else 'CEN-BB'
    # print(bb, log)

    for i, a in enumerate(acc):
        # print(a)
        # # ['stage', 'seed', 'n_video', 'video', 'backbone', 'accuracy', 'f1']
        item = ['S2', seed, len(s) if len(s) != 1 else 'full', i, bb, a['s2_acc'], a['f1']]
        print(item)
        # Store the extracted values in spr.csv
        # If new with header otherwise just attach
        if not os.path.exists("spr_2.csv"):
            with open("spr_2.csv", "w") as file:
                file.write(",".join(header) + "\n")
        with open("spr_2.csv", "a") as file:
            file.write(f"{item[0]},{item[1]},{item[2]},{item[3]},{item[4]},{item[5]},{item[6]}\n")


def fill_csv_stage_two():
    base = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/surgical_phase_recognition/output_dir"
    for i, f in enumerate(os.listdir(base)):
        if os.path.isdir(os.path.join(base, f)):  # Check if it's a folder
            for j, b in enumerate(os.listdir(f"{base}/{f}")):
                if os.path.isdir(os.path.join(base, f, b)):  # Also check if sub-path is a folder
                    if b == "ResNet50_backbone":
                        continue
                    for k, s in enumerate(os.listdir(f"{base}/{f}/{b}")):
                        if "EndoViT" in s:
                            if "archive" in s:
                                continue
                            for l, seed in enumerate(os.listdir(f"{base}/{f}/{b}/{s}")):
                                if os.path.isdir(os.path.join(base, f, b, s, seed)):
                                    if not "Archive" in seed:
                                        # print(f"/{f}/{b}/{s}/{seed}")
                                        if f == "full_dataset":
                                            p = f"{base}/{f}/{b}/{s}/{seed}/test_tcn_out.txt"
                                            print(p)
                                            fill_csv_two(p, seed)
                                        else:
                                            for v in os.listdir(f"{base}/{f}/{b}/{s}/{seed}"):
                                                if os.path.isdir(os.path.join(base, f, b, s, seed, v)):
                                                    p = f"{base}/{f}/{b}/{s}/{seed}/{v}/test_tcn_out.txt"
                                                    print(p)
                                                    fill_csv_two(p, v)


def get_combined_plot(csv1, csv2, metric='accuracy'):
    # combine two csv files to one big one
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    df = pd.concat([df1, df2])
    print(df)

    # Dynamically determine the number of subplots needed based on unique 'n_video' values
    unique_n_videos = df['n_video'].unique()
    num_subplots = len(unique_n_videos)
    num_subplots = len(unique_n_videos)
    # Calculate rows and columns to optimize space in a grid layout
    num_cols = 2  # Set number of columns
    num_rows = math.ceil(num_subplots)  # Determine required rows for the grid

    # Adjust figure size dynamically based on number of subplots
    fig, ax = plt.subplots(
        num_rows, num_cols, figsize=(22, 16),
        squeeze=False  # Ensures consistency even with one row or column
    )
    fig.suptitle('Surgical Phase Recognition (Fully Fine-Tuned)', fontsize=28, weight='bold',
                 y=0.999)  # Adjusted font size
    for i, stage in enumerate(df['stage'].unique()):
        subtitle_x_position = i / len(df['stage'].unique()) + (1 / (2 * len(df['stage'].unique())))
        fig.text(subtitle_x_position, 0.96, f"Stage: {stage}", ha='center', fontsize=18, weight='bold')

    # Plot each `n_video` group
    for j, n in enumerate(unique_n_videos):
        for i, stage in enumerate(df['stage'].unique()):
            print(f"Plotting: Stage: {stage}, n_videos: {n}")
            data = df[(df['stage'] == stage) & (df['n_video'] == n)]

            # Violin plot for the current `n_video` and stage
            sns.violinplot(
                data=data, x='seed', y=metric, hue='backbone',
                hue_order=['FL-BB', 'CEN-BB'], split=True, inner='quartile',
                linewidth=1.5, fill=True, ax=ax[j, i], cut=0
            )

            # Add strip plot for additional details
            sns.stripplot(
                data=data, x='seed', y=metric, hue='backbone',
                hue_order=['FL-BB', 'CEN-BB'], dodge=True, jitter=True, size=4,
                alpha=0.7, palette=['#9D02D7', '#D77A02'], ax=ax[j, i]
            )

            # Set labels and titles for the subplot
            ax[j, i].set_title(f"n_videos: {n}", fontsize=20, pad=12)  # Increased font size
            ax[j, i].set_ylabel("F1" if metric == "f1" else "Accuracy", fontsize=20)  # Adjust label based on metric
            ax[j, i].set_xlabel('Seed', fontsize=20, labelpad=12)  # Increased font size
            ax[j, i].set_ylim(-0.05, 1.25)
            ax[j, i].legend(title="Backbone", loc='best', fontsize=10, frameon=False)

            seed_palette = {}
            for k, seed in enumerate(df['seed'].unique()):
                data_fl = data[(data['n_video'] == n) & (data['seed'] == seed) & (data['backbone'] == 'FL-BB')][metric]
                data_cen = data[(data['n_video'] == n) & (data['seed'] == seed) & (data['backbone'] == 'CEN-BB')][
                    metric]

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

                # print(f"Statistic={stat}, p-value={p}")
                # if p < 0.01:
                #     print(
                #         f"[{seed}] Statistical relevant: p-value={p} < 0.01 --> Reject H0 --> (the distributions of the two samples are not equal)")
                # else:
                #     print(
                #         f"[{seed}] Statistical not relevant: p-value={p} >= 0.01 --> Fail to reject H0 --> (the distributions of the two samples are equal)")

                # print p-value on the plot
                ax[j, i].annotate(f"p={p:.4f}", xy=(0.35 * (0.5 + k), 0.9), xycoords='axes fraction', ha='center', fontsize=10,
                               color='black', weight='bold')
                annotation_text = "Reject H0" if p < 0.01 else "Fail to reject H0"
                ax[j, i].annotate(annotation_text, xy=(0.35 * (0.5 + k), 0.85), xycoords='axes fraction', ha='center',
                               fontsize=10, color='black', weight='bold')
                # plot means and say which one is better if Reject H0
                if p < 0.01:
                    if mean_fl > mean_cen:
                        delta = mean_fl - mean_cen
                        ax[j, i].annotate(f"FL-BB better by {delta*100:.2f}%", xy=(0.35 * (0.5 + k), 0.8), xycoords='axes fraction', ha='center', fontsize=10,
                                   color='black', weight='bold')
                        seed_palette["Run0" + str(k + 1) + "FL"] = "#998ec3"  # purple
                        seed_palette["Run0" + str(k + 1) + "CEN"] = "#E2E2E2"
                    else:
                        delta = mean_cen - mean_fl
                        ax[j, i].annotate(f"CEN-BB better by {delta*100:.2f}%", xy=(0.35 * (0.5 + k), 0.8), xycoords='axes fraction', ha='center', fontsize=10,
                                   color='black', weight='bold')
                        seed_palette["Run0" + str(k + 1) + "FL"] = "#E2E2E2"  # orange
                        seed_palette["Run0" + str(k + 1) + "CEN"] = "#f1a340"
                else:
                    # Fail to reject H0: (the distributions of the two samples are equal)
                    seed_palette["Run0" + str(k + 1) + "FL"] = "#E2E2E2"  # gray
                    seed_palette["Run0" + str(k + 1) + "CEN"] = "#E2E2E2"

            # Identify all PolyCollection objects (which are the violin shapes)
            violin_polygons = [c for c in ax[j, i].collections if isinstance(c, PolyCollection)]
            # print(len(violin_polygons))
            seed_list = ['Run01FL', 'Run01CEN', 'Run02FL', 'Run02CEN', 'Run03FL', 'Run03CEN']
            # print(seed_palette)

            for seed, violin in zip(seed_list, violin_polygons):
                if seed in seed_palette:
                    violin.set_facecolor(seed_palette[seed])  # Apply correct color

            # remove legend
            ax[j, i].get_legend().remove()


    # fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.99, hspace=0.4, wspace=0.3)

    # Adjust plot layout for better visibility
    plt.tight_layout()
    plt.savefig(f'/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/images/spr/spr_optimized_{metric}.png')  # Save as an optimized plot
    plt.show()


if __name__ == '__main__':
    # fill_csv_stage_one()
    # fill_csv_stage_two()
    # get_plot('spr.csv')
    get_combined_plot('spr.csv', 'spr_2.csv', 'f1')
    get_combined_plot('spr.csv', 'spr_2.csv', 'accuracy')
