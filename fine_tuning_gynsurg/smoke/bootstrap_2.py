import pandas as pd
import numpy as np
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import accuracy_score as sk_accuracy_score

def f1_score(y_true, y_pred, average='macro'):
    return sk_f1_score(y_true, y_pred, average=average)


def compute_metric(y_true, y_pred, metric='f1'):
    """Compute selected metric. Supported: 'f1' (macro F1), 'acc' (accuracy)."""
    if metric == 'f1':
        return f1_score(y_true, y_pred, average='macro')
    elif metric == 'acc':
        return sk_accuracy_score(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def main(cen=None, fl=None, metric='f1'):
    """Evaluate one pair of prediction files (cen, fl). Paths must point to CSV files.
    metric: 'f1' or 'acc'"""
    df_cen = pd.read_csv(cen)
    df_fl = pd.read_csv(fl)
    if df_cen is None or df_fl is None:
        raise ValueError("DataFrames df_cen and df_fl must be provided.")
    # match rows by position: dataframes must be same length
    if len(df_cen) != len(df_fl):
        raise ValueError(f"Prediction files must have the same number of rows. Got {len(df_cen)} and {len(df_fl)}")
    # raise error if 'gt' column not in both dataframes
    if 'gt' not in df_cen.columns or 'gt' not in df_fl.columns:
        raise ValueError("'gt' column must be present in both prediction files.")

    # ensure that the 'gt' columns match exactly (same values in the same order)
    gt_cen = df_cen['gt'].values
    gt_fl = df_fl['gt'].values
    if not np.array_equal(gt_cen, gt_fl):
        # find first mismatch index (if any)
        mismatch_idx = np.where(gt_cen != gt_fl)[0]
        if mismatch_idx.size > 0:
            i = int(mismatch_idx[0])
            raise ValueError(
                f"'gt' columns differ at row {i}: cen={gt_cen[i]!r} vs fl={gt_fl[i]!r}. "
                "Ensure prediction files have identical ground-truth rows in the same order."
            )
        else:
            # This can happen if arrays are not elementwise-equal due to dtype/representation issues
            raise ValueError(
                "'gt' columns are not equal between files (possible dtype/representation mismatch). "
                "Consider casting types or checking ordering."
            )

    # assume that rows correspond directly
    df = pd.DataFrame({
        'gt': gt_cen,
        'pred_cen': df_cen['pred'].values,
        'pred_fl': df_fl['pred'].values,
    })

    # bootstrapping 
    bootstrap_repetitions = 1000  # number of bootstrap samples
    diffs = []
    n = len(df)

    for _ in range(bootstrap_repetitions):
        idx = np.random.randint(0, n, n)
        sample = df.iloc[idx]
        # print(sample)
        m1 = compute_metric(sample['gt'].values, sample['pred_cen'].values, metric=metric)
        m2 = compute_metric(sample['gt'].values, sample['pred_fl'].values, metric=metric)
        diffs.append(m1 - m2)
    diffs = np.array(diffs)
    # print(diffs)

    # perform wilcoxon signed-rank test on diffs
    from scipy.stats import wilcoxon

    stat, p = wilcoxon(diffs)
    print(f"Wilcoxon test statistic: {stat}, p-value: {p}")
    if p < 0.01:
        print("The difference between CEN and FL predictions is statistically significant (p < 0.01).")
    else:
        print("The difference between CEN and FL predictions is not statistically significant (p >= 0.01).")

    # which is better?
    mean_diff = np.mean(diffs)
    if mean_diff > 0:
        print("CEN predictions are better on average.")
    elif mean_diff < 0:
        print("FL predictions are better on average.")
    else:
        print("CEN and FL predictions are equally good on average.")


def load_and_validate_pair(cen_path, fl_path):
    """Load two CSVs and validate their 'gt' columns match exactly. Returns a concatenated DataFrame with columns ['gt','pred_cen','pred_fl']."""
    df_cen = pd.read_csv(cen_path)
    df_fl = pd.read_csv(fl_path)

    if 'gt' not in df_cen.columns or 'gt' not in df_fl.columns:
        raise ValueError(f"'gt' column missing in {cen_path} or {fl_path}")

    gt_cen = df_cen['gt'].values
    gt_fl = df_fl['gt'].values
    if not np.array_equal(gt_cen, gt_fl):
        mismatch_idx = np.where(gt_cen != gt_fl)[0]
        if mismatch_idx.size > 0:
            i = int(mismatch_idx[0])
            raise ValueError(
                f"'gt' columns differ at row {i} between {cen_path} and {fl_path}: "
                f"{gt_cen[i]!r} vs {gt_fl[i]!r}"
            )
        else:
            raise ValueError(f"'gt' columns are not equal between {cen_path} and {fl_path}")

    return pd.DataFrame({
        'gt': gt_cen,
        'pred_cen': df_cen['pred'].values,
        'pred_fl': df_fl['pred'].values,
    })


def combine_and_run(cen_paths, fl_paths, metric='f1'):
    """Given lists of CEN and FL CSV paths (equal-length), combine them by concatenation and run main() logic on the combined set."""
    if len(cen_paths) != len(fl_paths):
        raise ValueError('cen_paths and fl_paths must have the same number of files')

    parts = []
    for c, f in zip(cen_paths, fl_paths):
        part = load_and_validate_pair(c, f)
        parts.append(part)

    combined = pd.concat(parts, ignore_index=True)

    # Reuse main's evaluation logic by temporarily writing to CSVs or calling an internal runner.
    # We'll implement an internal runner here to avoid IO.
    # bootstrap and wilcoxon
    bootstrap_repetitions = 1000
    diffs = []
    n = len(combined)
    for _ in range(bootstrap_repetitions):
        idx = np.random.randint(0, n, n)
        sample = combined.iloc[idx]
        m1 = compute_metric(sample['gt'].values, sample['pred_cen'].values, metric=metric)
        m2 = compute_metric(sample['gt'].values, sample['pred_fl'].values, metric=metric)
        diffs.append(m1 - m2)
    diffs = np.array(diffs)

    from scipy.stats import wilcoxon
    stat, p = wilcoxon(diffs)
    print(f"(Combined) Wilcoxon test statistic: {stat}, p-value: {p}")
    if p < 0.01:
        print("(Combined) The difference between CEN and FL predictions is statistically significant (p < 0.01).")
    else:
        print("(Combined) The difference between CEN and FL predictions is not statistically significant (p >= 0.01).")

    mean_diff = np.mean(diffs)
    if mean_diff > 0:
        print("(Combined) CEN predictions are better on average.")
    elif mean_diff < 0:
        print("(Combined) FL predictions are better on average.")
    else:
        print("(Combined) CEN and FL predictions are equally good on average.")


def _parse_file_list(s):
    """Parse comma-separated file list into list of paths (strip whitespace)."""
    return [x.strip() for x in s.split(',') if x.strip()]


def cli():
    import argparse
    parser = argparse.ArgumentParser(description='Compare CEN vs FL prediction CSVs, optionally combining multiple folds.')
    parser.add_argument('--cen', help='Path to CEN CSV or comma-separated list of CEN CSVs')
    parser.add_argument('--fl', help='Path to FL CSV or comma-separated list of FL CSVs')
    parser.add_argument('--metric', choices=['f1','acc'], default='f1', help="Metric to compute: 'f1' (macro) or 'acc' (accuracy). Default: f1")
    args = parser.parse_args()

    if not args.cen or not args.fl:
        parser.error('Both --cen and --fl must be provided')

    cen_list = _parse_file_list(args.cen)
    fl_list = _parse_file_list(args.fl)

    if len(cen_list) == 1 and len(fl_list) == 1:
        main(cen_list[0], fl_list[0], metric=args.metric)
    else:
        combine_and_run(cen_list, fl_list, metric=args.metric)



if __name__ == "__main__":
    cli()

