import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def main(n_videos='total'):
    df = pd.read_csv('atr.csv')
    df = df[df['n_videos'] == n_videos]

    mAPs = {backbone: [df[(df['backbone'] == backbone) & (df['video'] == video)]['map'].mean()
                       for video in df['video'].unique()]
            for backbone in df['backbone'].unique()}
    print(mAPs)

    try:
        stat, p = wilcoxon(x=mAPs['FL-B'], y=mAPs['CEN-B'])
    except ValueError:
        p, stat = np.nan, np.nan

    print(f"Wilcoxon test: p-value = {p}, stat = {stat}")

    print(f"FL-B mean mAP: {np.mean(mAPs['FL-B'])}")
    print(f"CEN-B mean mAP: {np.mean(mAPs['CEN-B'])}")
    print(f"FL-B std mAP: {np.std(mAPs['FL-B'])}")
    print(f"CEN-B std mAP: {np.std(mAPs['CEN-B'])}")

    print("Interpretation: p-value < 0.01, Reject H0 --> There is a difference between the variables A and B"
          if p < 0.01 else "Interpretation: p-value >= 0.01, Fail to Reject H0 --> There is no difference between the variables A and B")


if __name__ == "__main__":
    main()
    main(n_videos='2')
    main(n_videos='4')
    main(n_videos='8')