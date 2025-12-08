import pandas as pd
import numpy as np
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import accuracy_score as sk_accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

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

    # class-wise analysis (supports both 'f1' and 'acc')
    y_true = df['gt'].values
    y_pred_cen = df['pred_cen'].values
    y_pred_fl = df['pred_fl'].values

    labels = unique_labels(y_true, np.concatenate([y_pred_cen, y_pred_fl]))
    class_rows = []
    print('\nClass-wise results:')
    if metric == 'f1':
        p_cen, r_cen, f_cen, s_cen = precision_recall_fscore_support(y_true, y_pred_cen, labels=labels, zero_division=0)
        p_fl, r_fl, f_fl, s_fl = precision_recall_fscore_support(y_true, y_pred_fl, labels=labels, zero_division=0)
        print('class\tsupport\tcen_f1\tfl_f1\tdelta_f1\tcen_prec\tfl_prec\tcen_rec\tfl_rec')
        for lab, sup, fc, ff, pc, pf, rc, rf in zip(labels, s_cen, f_cen, f_fl, p_cen, p_fl, r_cen, r_fl):
            delta = fc - ff
            print(f"{lab}\t{sup}\t{fc:.3f}\t{ff:.3f}\t{delta:.3f}\t{pc:.3f}\t{pf:.3f}\t{rc:.3f}\t{rf:.3f}")
            class_rows.append({
                'class': lab,
                'support': int(sup),
                'cen_metric': float(fc),
                'fl_metric': float(ff),
                'delta_metric': float(delta),
                'cen_prec': float(pc),
                'fl_prec': float(pf),
                'cen_rec': float(rc),
                'fl_rec': float(rf),
            })
    elif metric == 'acc':
        print('class\tsupport\tcen_acc\tfl_acc\tdelta_acc')
        for lab in labels:
            mask = (y_true == lab)
            sup = int(np.sum(mask))
            if sup == 0:
                cen_acc = 0.0
                fl_acc = 0.0
            else:
                cen_acc = float(np.mean(y_pred_cen[mask] == y_true[mask]))
                fl_acc = float(np.mean(y_pred_fl[mask] == y_true[mask]))
            delta = cen_acc - fl_acc
            print(f"{lab}\t{sup}\t{cen_acc:.3f}\t{fl_acc:.3f}\t{delta:.3f}")
            class_rows.append({
                'class': lab,
                'support': sup,
                'cen_metric': cen_acc,
                'fl_metric': fl_acc,
                'delta_metric': delta,
            })
    else:
        raise ValueError(f"Unsupported metric for class-wise analysis: {metric}")

    return {
        'diffs': diffs,
        'mean_diff': mean_diff,
        'wilcoxon_stat': float(stat),
        'wilcoxon_p': float(p),
        'class_rows': class_rows,
    }


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

    # class-wise analysis for combined (supports both 'f1' and 'acc')
    y_true = combined['gt'].values
    y_pred_cen = combined['pred_cen'].values
    y_pred_fl = combined['pred_fl'].values
    labels = unique_labels(y_true, np.concatenate([y_pred_cen, y_pred_fl]))
    class_rows = []
    print('\n(Combined) Class-wise results:')
    if metric == 'f1':
        p_cen, r_cen, f_cen, s_cen = precision_recall_fscore_support(y_true, y_pred_cen, labels=labels, zero_division=0)
        p_fl, r_fl, f_fl, s_fl = precision_recall_fscore_support(y_true, y_pred_fl, labels=labels, zero_division=0)
        print('class\tsupport\tcen_f1\tfl_f1\tdelta_f1\tcen_prec\tfl_prec\tcen_rec\tfl_rec')
        for lab, sup, fc, ff, pc, pf, rc, rf in zip(labels, s_cen, f_cen, f_fl, p_cen, p_fl, r_cen, r_fl):
            delta = fc - ff
            print(f"{lab}\t{sup}\t{fc:.3f}\t{ff:.3f}\t{delta:.3f}\t{pc:.3f}\t{pf:.3f}\t{rc:.3f}\t{rf:.3f}")
            class_rows.append({
                'class': lab,
                'support': int(sup),
                'cen_metric': float(fc),
                'fl_metric': float(ff),
                'delta_metric': float(delta),
                'cen_prec': float(pc),
                'fl_prec': float(pf),
                'cen_rec': float(rc),
                'fl_rec': float(rf),
            })
    elif metric == 'acc':
        print('class\tsupport\tcen_acc\tfl_acc\tdelta_acc')
        for lab in labels:
            mask = (y_true == lab)
            sup = int(np.sum(mask))
            if sup == 0:
                cen_acc = 0.0
                fl_acc = 0.0
            else:
                cen_acc = float(np.mean(y_pred_cen[mask] == y_true[mask]))
                fl_acc = float(np.mean(y_pred_fl[mask] == y_true[mask]))
            delta = cen_acc - fl_acc
            print(f"{lab}\t{sup}\t{cen_acc:.3f}\t{fl_acc:.3f}\t{delta:.3f}")
            class_rows.append({
                'class': lab,
                'support': sup,
                'cen_metric': cen_acc,
                'fl_metric': fl_acc,
                'delta_metric': delta,
            })
    else:
        raise ValueError(f"Unsupported metric for class-wise analysis: {metric}")

    return {
        'diffs': diffs,
        'mean_diff': mean_diff,
        'wilcoxon_stat': float(stat),
        'wilcoxon_p': float(p),
        'class_rows': class_rows,
    }


def _parse_file_list(s):
    """Parse comma-separated file list into list of paths (strip whitespace)."""
    return [x.strip() for x in s.split(',') if x.strip()]


def cli():
    import argparse
    parser = argparse.ArgumentParser(description='Compare CEN vs FL prediction CSVs, optionally combining multiple folds.')
    parser.add_argument('--cen', help='Path to CEN CSV or comma-separated list of CEN CSVs')
    parser.add_argument('--fl', help='Path to FL CSV or comma-separated list of FL CSVs')
    parser.add_argument('--metric', choices=['f1','acc'], default='f1', help="Metric to compute: 'f1' (macro) or 'acc' (accuracy). Default: f1")
    parser.add_argument('--out', help='Optional path to save class-wise results CSV. If combining, a single CSV will be written; if single pair, that pair\'s class results will be written.')
    args = parser.parse_args()

    if not args.cen or not args.fl:
        parser.error('Both --cen and --fl must be provided')

    cen_list = _parse_file_list(args.cen)
    fl_list = _parse_file_list(args.fl)

    if len(cen_list) == 1 and len(fl_list) == 1:
        res = main(cen_list[0], fl_list[0], metric=args.metric)
    else:
        res = combine_and_run(cen_list, fl_list, metric=args.metric)

    if args.out and res is not None:
        # save class_rows to CSV
        out_df = pd.DataFrame(res['class_rows'])
        out_df.to_csv(args.out, index=False)
        print(f"Saved class-wise results to {args.out}")



if __name__ == "__main__":
    cli()

