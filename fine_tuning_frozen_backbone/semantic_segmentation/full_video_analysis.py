import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def main(csv, metric='iou', n_videos='total'):
    if not os.path.exists(f'sss_full_video_{n_videos}.csv'):
        df = pd.read_csv(csv)
        df = df[df['n_videos'] == n_videos]
        results = [
            {
                'resolution': resolution,
                'backbone': backbone,
                'run': run,
                'image': image,
                metric: df[(df['image'] == image) & (df['seed'] == run) & (df['backbone'] == backbone) & (
                            df['res'] == resolution)][metric].mean()
            }
            for resolution in df['res'].unique()
            for backbone in df['backbone'].unique()
            for run in df['seed'].unique()
            for image in df['image'].unique()
        ]

        # new df
        df = pd.DataFrame(results)
        print(df)
        # safe df
        df.to_csv(f'sss_full_video_{n_videos}.csv', index=False)
    else:
        df = pd.read_csv(f'sss_full_video_{n_videos}.csv')

    print(df)

    metrics = {
        resolution: {
            backbone: [
                df[(df['resolution'] == resolution) & (df['backbone'] == backbone) & (df['image'] == image)][metric].mean()
                for image in df['image'].unique()
            ]
            for backbone in df['backbone'].unique()
        }
        for resolution in df['resolution'].unique()
    }
    print(metrics)

    for resolution in metrics.keys():
        print(resolution)
        try:
            stat, p = wilcoxon(x=metrics[resolution]['FL-Backbone'], y=metrics[resolution]['CEN-Backbone'])
        except ValueError:
            p, stat = np.nan, np.nan

        print(f"Wilcoxon test: p-value = {p}, stat = {stat}")
        print(f"FL-B mean mAP: {np.mean(metrics[resolution]['FL-Backbone'])}")
        print(f"CEN-B mean mAP: {np.mean(metrics[resolution]['CEN-Backbone'])}")
        print(f"FL-B mean mAP: {np.std(metrics[resolution]['FL-Backbone'])}")
        print(f"CEN-B mean mAP: {np.std(metrics[resolution]['CEN-Backbone'])}")
        print("Interpretation: p-value < 0.01, Reject H0 --> There is a difference between the variables A and B"
              if p < 0.01 else "Interpretation: p-value >= 0.01, Fail to Reject H0 --> There is no difference between the variables A and B")

        w_s = stat
        delta = [a - b for a, b in zip(metrics[resolution]['FL-Backbone'], metrics[resolution]['CEN-Backbone'])]
        # how many values are 0
        delta = [d for d in delta if d == 0]
        n = len(metrics[resolution]['FL-Backbone']) - len(delta)
        z = (w_s - ((n*(n+1))/4))/ np.sqrt((n*(n+1)*(2*n+1))/24)

        print(z)


if __name__ == "__main__":
    main(csv='sss_frozen.csv', metric='iou')
    # main(csv='sss_frozen.csv', metric='iou', n_videos=1)
