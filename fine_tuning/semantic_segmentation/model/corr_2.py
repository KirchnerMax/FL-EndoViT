import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.stats import wilcoxon


def calculate_p_values(data_raw, metric):
    p_values = []
    statistical_relevant = []
    means = []
    data = data_raw.fillna(1)
    for i, seed in enumerate(sorted(data["seed"].unique())):
        data_fl = data[(data["seed"] == seed) & (data["backbone"] == "FL-Backbone")]
        data_cen = data[(data["seed"] == seed) & (data["backbone"] == "CEN-Backbone")]

        # sort data_fl and data_cen by image so that they match
        data_fl = data_fl.sort_values(by=["image"])
        data_cen = data_cen.sort_values(by=["image"])

        # convert data_fl and data_cen to np.array
        data_fl = data_fl[metric].to_numpy()
        data_cen = data_cen[metric].to_numpy()
        mean_fl = \
        data_raw[(data["seed"] == seed) & (data["backbone"] == "FL-Backbone") & (data_raw["class_observed"] == 1)][
            metric].mean()
        mean_cen = \
        data_raw[(data["seed"] == seed) & (data["backbone"] == "CEN-Backbone") & (data_raw["class_observed"] == 1)][
            metric].mean()

        # calculate p-value
        try:
            stat, p = wilcoxon(x=data_fl, y=data_cen)
        except ValueError:
            p = np.nan
            stat = np.nan
        # print(f"Statistic={stat}, p-value={p}")
        # if p < 0.01:
        #     print(f"Statistical relevant: p-value={p} < 0.01 --> Reject H0 (the distributions of the two samples are not equal)")
        # else:
        #     print(f"Statistical not relevant: p-value={p} >= 0.01 --> Fail to reject H0: (the distributions of the two samples are equal)")
        p_values.append(p)
        statistical_relevant.append(p < 0.01)
        means.append(mean_fl)
        means.append(mean_cen)
    return p_values, statistical_relevant, means


def get_colors(statistical_relevant, means):
    seed_palette = {}
    for i in range(3):
        if not statistical_relevant[i]:
            # Fail to reject H0: (the distributions of the two samples are equal)
            seed_palette["Run0" + str(i + 1) + "FL"] = "#E2E2E2" # gray
            seed_palette["Run0" + str(i + 1) + "CEN"] = "#E2E2E2"
        else:
            # Reject H0: (the distributions of the two samples are not equal)
            if means[2 * i] > means[2 * i + 1]:
                seed_palette["Run0" + str(i + 1) + "FL"] = "#998ec3" # purple
                seed_palette["Run0" + str(i + 1) + "CEN"] = "#E2E2E2"
            else:
                seed_palette["Run0" + str(i + 1) + "FL"] = "#E2E2E2" # orange
                seed_palette["Run0" + str(i + 1) + "CEN"] = "#f1a340"
    return seed_palette


def create_half_violin_plots(csv, metric, resolution, safe=False):
    """
    Creates half-violin plots for the given metric and resolution.

    Args:
        csv (str): Path to the CSV file.
        metric (str): Metric to plot (e.g., "iou", "dice", "acc_per_pixel").
        resolution (str): Resolution to plot (e.g., "low_res", "high_res").

    Returns:
        None
    """
    store_means = []
    # Load the CSV file
    df = pd.read_csv(csv, header=0, dtype={"n_videos": "str"})

    # Filter the DataFrame based on the resolution
    df = df[df["res"] == resolution]

    sns.set_theme(style="whitegrid", font_scale=1.5)  # Increase font scale for better visibility
    n_classes = len(df["class"].unique())
    n_videos = len(df["n_videos"].unique())
    fig, ax = plt.subplots(n_videos, 8, figsize=(3 * 11, 3 * 8))
    fig.suptitle(f'Surgical Scene Segmentation ({resolution} | Fully Fine-Tuned)', fontsize=30, weight='bold',
                 y=0.98)  # Increase font size and adjust position
    for i, n_video in enumerate(df["n_videos"].unique()):
        for j, cls in enumerate(df["class"].unique()):
            if j == 8:
                continue
            # print(cls, n_video)
            data = df[(df["n_videos"] == n_video) & (df["class"] == cls)]
            p_values, statistical_relevant, means = calculate_p_values(data, metric)

            # print(f"n_videos: {n_video}, class: {cls}, means: {means}")
            store_means.append(means)

            seed_palette = get_colors(statistical_relevant, means)
            # print(seed_palette)

            data_plot = data[data["class_observed"] == 1]
            sns.violinplot(data=data_plot,
                           x='seed',
                           y=metric,
                           hue='backbone',
                           hue_order=["FL-Backbone", "CEN-Backbone"],
                           ax=ax[i, j],
                           inner="quartile",
                           fill=True,
                           linewidth=1.5,
                           split=True,
                           cut=0,
                           order=sorted(data_plot["seed"].unique()))
            ax[i, j].set_title(f"n_videos: {n_video}, class: {cls}",
                               fontsize=24)  # Increase font size for subplot titles
            ax[i, j].set_xlabel("Seed", fontsize=24)  # Increase font size for x-axis label
            ax[i, j].set_ylabel("IoU" if metric == "iou" else metric, fontsize=28)  # Increase font size for y-axis label
            ax[i, j].set_ylim(-0.05, 1.05)

            # Identify all PolyCollection objects (which are the violin shapes)
            violin_polygons = [c for c in ax[i, j].collections if isinstance(c, PolyCollection)]
            # print(len(violin_polygons))

            seed_list = ['Run01FL', 'Run01CEN', 'Run02FL', 'Run02CEN', 'Run03FL', 'Run03CEN']
            # print(seed_palette)
            # Assign colors based on the seed
            # Filter out means that are zero and remove corresponding seeds from the list
            for k, (mean, rrun) in enumerate(zip(means, seed_list)):
                if mean == 0.0:
                    seed_palette.pop(rrun, None)  # Remove from palette only if the key exists
                    seed_list.remove(rrun)  # Remove from list only if the element exists

            # Now, use the filtered seed_list and seed_palette to set the colors
            # print(seed_list)  # Verify if the zero mean seeds were removed
            # print(seed_palette)  # Verify if the corresponding colors were removed

            for seed, violin in zip(seed_list, violin_polygons):
                if seed in seed_palette:
                    violin.set_facecolor(seed_palette[seed])  # Apply correct color

            # # display p-values and statistical relevance for each seed
            # ax[i, j].text(0.15, 0.95, f"p={p_values[0]:.4f}",
            #               fontsize=9, color="black", ha="center", transform=ax[i, j].transAxes)
            # ax[i, j].text(0.45, 0.95, f"p={p_values[1]:.4f}",
            #               fontsize=9, color="black", ha="center", transform=ax[i, j].transAxes)
            # ax[i, j].text(0.75, 0.95, f"p={p_values[2]:.4f}",
            #               fontsize=9, color="black", ha="center", transform=ax[i, j].transAxes)
            # ax[i, j].text(0.15, 0.9, f"Sig: {'Yes' if statistical_relevant[0] else 'No'}",
            #               fontsize=9, color="red" if statistical_relevant[0] else "green", ha="center",
            #               transform=ax[i, j].transAxes)
            # ax[i, j].text(0.45, 0.9, f"Sig: {'Yes' if statistical_relevant[1] else 'No'}",
            #               fontsize=9, color="red" if statistical_relevant[1] else "green", ha="center",
            #               transform=ax[i, j].transAxes)
            # ax[i, j].text(0.75, 0.9, f"Sig: {'Yes' if statistical_relevant[2] else 'No'}",
            #               fontsize=9, color="red" if statistical_relevant[2] else "green", ha="center",
            #               transform=ax[i, j].transAxes)
            # # Add the text entries with a condition to include a star (*) if one value is better
            # ax[i, j].text(0.15, 0.85,
            #               f"{'* ' if means[0] > means[1] else ''}m: {means[0]:.3f}",
            #               fontsize=9, color="blue", ha="center", transform=ax[i, j].transAxes)
            #
            # ax[i, j].text(0.15, 0.8,
            #               f"{'* ' if means[1] > means[0] else ''}m: {means[1]:.3f}",
            #               fontsize=9, color="darkorange", ha="center", transform=ax[i, j].transAxes)
            # ax[i, j].text(0.45, 0.85,
            #               f"{'* ' if means[2] > means[3] else ''}m: {means[2]:.3f}",
            #               fontsize=9, color="blue", ha="center", transform=ax[i, j].transAxes)
            # ax[i, j].text(0.45, 0.8,
            #               f"{'* ' if means[3] > means[2] else ''}m: {means[3]:.3f}",
            #               fontsize=9, color="darkorange", ha="center", transform=ax[i, j].transAxes)
            # ax[i, j].text(0.75, 0.85,
            #               f"{'* ' if means[4] > means[5] else ''}m: {means[4]:.3f}",
            #               fontsize=9, color="blue", ha="center", transform=ax[i, j].transAxes)
            # ax[i, j].text(0.75, 0.8,
            #               f"{'* ' if means[5] > means[4] else ''}m: {means[5]:.3f}",
            #               fontsize=9, color="darkorange", ha="center", transform=ax[i, j].transAxes)

            # Remove legend
            if ax[i, j].get_legend() is not None:
                ax[i, j].get_legend().remove()
    # Add a shared legend outside the subplots
    handles, labels = ax[0, 0].get_legend_handles_labels()  # Retrieve the legend handles/labels from the first subplot
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, fontsize=16,  # Increase font size for legend
               bbox_to_anchor=(0.5, 1.05))

    # Save the plot
    plt.tight_layout()
    if safe:
        base = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/images"
        plt.savefig(f"{base}/sss/{metric}_{resolution}.png")
    plt.show()

    # get average of means for each n_video

    print("Average of means for each n_video:")
    for i in range(4):
        csv_means = []
        # get first 8 elements of store_means
        m = store_means[i*8:(i+1)*8]
        m = np.array(m)
        print(m)
        # get average of means
        for j in range(6):
            x = 0
            for k in range(8):
                # print(m[k][j])
                x += m[k][j]
            x /= 8
            csv_means.append(x)


def create_half_violin_plots_small(csv, metric, resolution, safe=False):
    """
    Creates half-violin plots for the given metric and resolution.

    Args:
        csv (str): Path to the CSV file.
        metric (str): Metric to plot (e.g., "iou", "dice", "acc_per_pixel").
        resolution (str): Resolution to plot (e.g., "low_res", "high_res").

    Returns:
        None
    """
    store_means = []
    # Load the CSV file
    df = pd.read_csv(csv, header=0, dtype={"n_videos": "str"})

    # Filter the DataFrame based on the resolution
    df = df[df["res"] == resolution]

    # Create the plot
    sns.set_theme(style="whitegrid")
    n_classes = len(df["class"].unique())
    n_videos = len(df["n_videos"].unique())
    fig, ax = plt.subplots(n_videos, 1, figsize=(11, 8))
    fig.suptitle(f'Surgical Scene Segmentation ({resolution} | Fully Fine-Tuned)', fontsize=25, weight='bold', y=1.0)  # Add overall title

    for i, n_video in enumerate(df["n_videos"].unique()):
        # print(cls, n_video)
        data = df[(df["n_videos"] == n_video)]
        p_values, statistical_relevant, means = calculate_p_values(data, metric)

        print(means)

        seed_palette = get_colors(statistical_relevant, means)
        # print(seed_palette)

        data_plot = data[data["class_observed"] == 1]
        sns.violinplot(data=data_plot,
                       x='seed',
                       y=metric,
                       hue='backbone',
                       hue_order=["FL-Backbone", "CEN-Backbone"],
                       ax=ax[i],
                       inner="quartile",
                       fill=True,
                       linewidth=1.5,
                       split=True,
                       cut=0,
                       order=sorted(data_plot["seed"].unique()))
        ax[i].set_title(f"n_videos: {n_video}")
        ax[i].set_xlabel("Seed")
        ax[i].set_ylabel(metric)
        ax[i].set_ylim(-0.05, 1.05)

        # Identify all PolyCollection objects (which are the violin shapes)
        violin_polygons = [c for c in ax[i].collections if isinstance(c, PolyCollection)]
        # print(len(violin_polygons))

        seed_list = ['Run01FL', 'Run01CEN', 'Run02FL', 'Run02CEN', 'Run03FL', 'Run03CEN']
        # print(seed_palette)
        # Assign colors based on the seed
        # Filter out means that are zero and remove corresponding seeds from the list
        for k, (mean, rrun) in enumerate(zip(means, seed_list)):
            if mean == 0.0:
                seed_palette.pop(rrun, None)  # Remove from palette only if the key exists
                seed_list.remove(rrun)  # Remove from list only if the element exists

        # Now, use the filtered seed_list and seed_palette to set the colors
        # print(seed_list)  # Verify if the zero mean seeds were removed
        # print(seed_palette)  # Verify if the corresponding colors were removed

        for seed, violin in zip(seed_list, violin_polygons):
            if seed in seed_palette:
                violin.set_facecolor(seed_palette[seed])  # Apply correct color

        # Remove legend
        if ax[i].get_legend() is not None:
            ax[i].get_legend().remove()
    # Add a shared legend outside the subplots
    handles, labels = ax[0].get_legend_handles_labels()  # Retrieve the legend handles/labels from the first subplot
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, fontsize=12,
               bbox_to_anchor=(0.5, 1.03))

    # Save the plot
    plt.tight_layout()
    # if safe:
    #     base = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/images"
    #     plt.savefig(f"{base}/sss/{metric}_{resolution}.png")
    plt.show()



if __name__ == '__main__':
    create_half_violin_plots("sss.csv", "iou", "low_res", safe=True)
    print('********************************************************************************')
    create_half_violin_plots("sss.csv", "iou", "high_res", safe=True)
    #
    # create_half_violin_plots_small("sss.csv", "iou", "low_res")
    # print('********************************************************************************')
    # create_half_violin_plots_small("sss.csv", "iou", "high_res")
