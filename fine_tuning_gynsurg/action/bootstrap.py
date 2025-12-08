#!/usr/bin/env python3
"""
Simple bootstrap script to compare two prediction CSVs (CEN vs FL).

Assumes CSVs have columns: id, gt, pred

Outputs:
 - prints observed accuracy and macro-F1 for each method and the difference
 - runs N bootstrap resamples (default 1000) sampling rows with replacement
 - computes 95% CI for metric differences and two-sided p-value
 - writes bootstrap samples differences to `bootstrap_diffs_{n_reps}.csv`

Usage:
 python bootstrap.py --cen cen_predictions.csv --fl fl_predictions.csv --reps 1000
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt



def load_preds(path):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # require at least 'gt' and 'pred' columns; we will match rows by position
    expected = {'gt', 'pred'}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"CSV {path} must contain columns gt and pred. Found: {list(df.columns)}")
    # include id if present for hierarchical bootstrap
    if 'id' in df.columns:
        return df[['id', 'gt', 'pred']].copy()
    return df[['gt', 'pred']].copy()


# note: we no longer merge by id; matching is by dataframe position (index)


def compute_metrics(y_true, y_pred, average='macro'):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, f1


def bootstrap_compare(df, metric_fn, n_reps=1000, random_state=0, hierarchical=False):
    rng = np.random.default_rng(random_state)
    n = len(df)
    diffs = np.empty(n_reps)

    if hierarchical:
        # build mapping from group id -> row indices
        if 'id' not in df.columns:
            raise ValueError('Hierarchical bootstrap requested but no "id" column found in dataframe')
        group_ids = pd.unique(df['id'].values)
        G = len(group_ids)
        group_to_indices = {g: np.flatnonzero(df['id'].values == g) for g in group_ids}

        for i in range(n_reps):
            # sample groups with replacement (one draw per original group)
            sampled_groups = rng.choice(group_ids, size=G, replace=True)
            selected_indices = []
            for g in sampled_groups:
                inds = group_to_indices[g]
                # sample within-group rows with replacement to preserve group size
                k = len(inds)
                if k == 0:
                    continue
                picks = rng.integers(0, k, size=k)
                selected_indices.append(inds[picks])
            if len(selected_indices) == 0:
                sample_idx = np.array([], dtype=int)
            else:
                sample_idx = np.concatenate(selected_indices)
            sample = df.iloc[sample_idx]
            m1 = metric_fn(sample['gt'].values, sample['pred_cen'].values)
            m2 = metric_fn(sample['gt'].values, sample['pred_fl'].values)
            diffs[i] = m1 - m2
        return diffs

    # non-hierarchical (original) bootstrap: sample rows with replacement
    for i in range(n_reps):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx]
        m1 = metric_fn(sample['gt'].values, sample['pred_cen'].values)
        m2 = metric_fn(sample['gt'].values, sample['pred_fl'].values)
        diffs[i] = m1 - m2
    return diffs


def two_sided_pvalue(diffs, observed_diff):
    # p-value = fraction of bootstrap diffs as or more extreme than observed
    extreme = np.sum(np.abs(diffs) >= abs(observed_diff))
    return extreme / len(diffs)


def ci_from_diffs(diffs, alpha=0.05):
    lo = np.percentile(diffs, 100 * (alpha / 2.0))
    hi = np.percentile(diffs, 100 * (1 - alpha / 2.0))
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cen', required=True, help='Path to CEN predictions CSV')
    parser.add_argument('--fl', required=True, help='Path to FL predictions CSV')
    parser.add_argument('--reps', type=int, default=1000, help='Bootstrap repetitions')
    parser.add_argument('--out', default=None, help='Output CSV for bootstrap diffs')
    parser.add_argument('--hierarchical', action='store_true', help='Use hierarchical bootstrap (sample groups by id first)')
    args = parser.parse_args()

    df_cen = load_preds(args.cen)
    df_fl = load_preds(args.fl)
    # match rows by position: dataframes must be same length
    if len(df_cen) != len(df_fl):
        raise ValueError(f"Prediction files must have the same number of rows. Got {len(df_cen)} and {len(df_fl)}")
    # include id if both files provided it
    if 'id' in df_cen.columns and 'id' in df_fl.columns:
        df = pd.DataFrame({
            'id': df_cen['id'].values,
            'gt': df_cen['gt'].values,
            'pred_cen': df_cen['pred'].values,
            'pred_fl': df_fl['pred'].values,
        })
    else:
        df = pd.DataFrame({
            'gt': df_cen['gt'].values,
            'pred_cen': df_cen['pred'].values,
            'pred_fl': df_fl['pred'].values,
        })

    # ensure integer labels
    df['gt'] = df['gt'].astype(int)
    df['pred_cen'] = df['pred_cen'].astype(int)
    df['pred_fl'] = df['pred_fl'].astype(int)

    # observed metrics
    acc_cen, f1_cen = compute_metrics(df['gt'].values, df['pred_cen'].values)
    acc_fl, f1_fl = compute_metrics(df['gt'].values, df['pred_fl'].values)

    print(f"Observed: CEN acc={acc_cen:.4f}, macro-f1={f1_cen:.4f} | FL acc={acc_fl:.4f}, macro-f1={f1_fl:.4f}")

    # detailed pre-bootstrap reports
    print('\nClassification report for CEN:')
    print(classification_report(df['gt'].values, df['pred_cen'].values, digits=4))
    print('\nClassification report for FL:')
    print(classification_report(df['gt'].values, df['pred_fl'].values, digits=4))

    # confusion matrices (and save to PNG)
    cm_cen = confusion_matrix(df['gt'].values, df['pred_cen'].values)
    cm_fl = confusion_matrix(df['gt'].values, df['pred_fl'].values)

    try:
        # row-normalize (per true class) for better visualization
        def row_normalize(cm):
            cm = cm.astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            # avoid division by zero
            row_sums[row_sums == 0] = 1.0
            return cm / row_sums

        cm_cen_norm = row_normalize(cm_cen)
        cm_fl_norm = row_normalize(cm_fl)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        im0 = ax[0].imshow(cm_cen_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
        ax[0].set_title('Confusion matrix - CEN (row-normalized)')
        fig.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(cm_fl_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
        ax[1].set_title('Confusion matrix - FL (row-normalized)')
        fig.colorbar(im1, ax=ax[1])
        plt.suptitle('Confusion matrices (pre-bootstrap)')
        plt.tight_layout()
        out_fig = 'confusion_matrices_pre_bootstrap.png'
        fig.savefig(out_fig)
        plt.close(fig)
        print(f"Saved confusion matrices to {out_fig}")
    except Exception as e:
        print(f"Could not save confusion matrices: {e}")

    # bootstrapping for accuracy difference
    def acc_fn(y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    def f1_fn(y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro')

    print(f"Running {args.reps} bootstrap repetitions (this may take a while)...")
    hierarchical_flag = args.hierarchical or ('id' in df.columns)
    acc_diffs = bootstrap_compare(df, acc_fn, n_reps=args.reps, random_state=42, hierarchical=hierarchical_flag)
    f1_diffs = bootstrap_compare(df, f1_fn, n_reps=args.reps, random_state=43, hierarchical=hierarchical_flag)

    obs_acc_diff = acc_cen - acc_fl
    obs_f1_diff = f1_cen - f1_fl

    acc_lo, acc_hi = ci_from_diffs(acc_diffs)
    f1_lo, f1_hi = ci_from_diffs(f1_diffs)

    p_acc = two_sided_pvalue(acc_diffs, obs_acc_diff)
    p_f1 = two_sided_pvalue(f1_diffs, obs_f1_diff)

    print('\nBootstrap results:')
    print(f"Accuracy difference (CEN - FL): observed={obs_acc_diff:.4f} | 95% CI=({acc_lo:.4f}, {acc_hi:.4f}) | p_two-sided={p_acc:.4f}")
    print(f"Macro-F1 difference (CEN - FL): observed={obs_f1_diff:.4f} | 95% CI=({f1_lo:.4f}, {f1_hi:.4f}) | p_two-sided={p_f1:.4f}")

    # save diffs
    out_csv = args.out if args.out is not None else f"bootstrap_diffs_{args.reps}.csv"
    out_df = pd.DataFrame({'acc_diff': acc_diffs, 'f1_diff': f1_diffs})
    out_df.to_csv(out_csv, index=False)
    print(f"Saved bootstrap diffs to {out_csv}")

if __name__ == '__main__':
    main()
