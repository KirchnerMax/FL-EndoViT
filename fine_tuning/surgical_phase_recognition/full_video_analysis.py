import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def main(csv, metric='f1', n_videos='full'):
    df = pd.read_csv(csv)
    df = df[df['n_video'] == n_videos]

    metrics = {backbone: [df[(df['backbone'] == backbone) & (df['video'] == video)][metric].mean()
                          for video in df['video'].unique()]
               for backbone in df['backbone'].unique()}
    print(metrics)

    try:
        stat, p = wilcoxon(x=metrics['FL-BB'], y=metrics['CEN-BB'])
    except ValueError:
        p, stat = np.nan, np.nan

    print(f"Wilcoxon test: p-value = {p}, stat = {stat}")
    print(f"FL-B mean mAP: {np.mean(metrics['FL-BB'])}")
    print(f"CEN-B mean mAP: {np.mean(metrics['CEN-BB'])}")
    print(f"FL-B std mAP: {np.std(metrics['FL-BB'])}")
    print(f"CEN-B std mAP: {np.std(metrics['CEN-BB'])}")
    print("Interpretation: p-value < 0.01, Reject H0 --> There is a difference between the variables A and B"
          if p < 0.01 else "Interpretation: p-value >= 0.01, Fail to Reject H0 --> There is no difference between the variables A and B")


if __name__ == "__main__":
    # main(csv='spr.csv', metric='f1')
    # main(csv='spr_2.csv', metric='f1')
    # main(csv='spr.csv', metric='accuracy')
    # main(csv='spr_2.csv', metric='accuracy')

    # main(csv='spr.csv', metric='f1', n_videos='8')
    # main(csv='spr_2.csv', metric='f1', n_videos='8')
    # main(csv='spr.csv', metric='f1', n_videos='4')
    # main(csv='spr_2.csv', metric='f1', n_videos='4')
    # main(csv='spr.csv', metric='f1', n_videos='2')
    # main(csv='spr_2.csv', metric='f1', n_videos='2')

    main(csv='spr.csv', metric='accuracy', n_videos='8')
    main(csv='spr_2.csv', metric='accuracy', n_videos='8')
    main(csv='spr.csv', metric='accuracy', n_videos='4')
    main(csv='spr_2.csv', metric='accuracy', n_videos='4')
    main(csv='spr.csv', metric='accuracy', n_videos='2')
    main(csv='spr_2.csv', metric='accuracy', n_videos='2')