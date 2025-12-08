import ast
import os

import numpy as np
import pandas as pd


def extract_run_name(model_string):
    # Extracts Run01, Run02, etc., from the model string
    return next((part for part in model_string.split('_') if part.startswith('Run')), "RunUnknown")


def safe_parse_observed_classes(value):
    """
    Safely parses the observed_classes column to extract a list.
    Strips any non-literal elements.

    Args:
        value (str): The string representation of the observed_classes column.

    Returns:
        list: A parsed list of observed classes (True/False values).
    """
    try:
        # Try to use ast.literal_eval for safer parsing
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Handle cases where the value includes non-literal elements
        # Example cleanup for 'tensor([...])' patterns
        if "tensor" in value:
            value = value.replace("tensor", "").replace("(", "").replace(")", "")
            return ast.literal_eval(value)  # Retry parsing after cleanup
        raise ValueError(f"Unable to parse observed_classes: {value}")


def extract_backbone(param):
    # Extracts the backbone from the model string
    if 'ASAM' in param:
        return 'FL-Backbone'
    else:
        return 'CEN-Backbone'


def convert_old_to_new_format(old_csv, n_videos, output_csv=None):
    """
    Converts a CSV file with the old format to the new format.

    Args:
        old_csv (str): Path to the CSV file with the old format.
        n_videos (str): Number of videos (to populate the "n_videos" column in the new format).
        output_csv (str, optional): If not None, the transformed data is saved to this file.

    Returns:
        new_df (pd.DataFrame): DataFrame with the new format.
    """
    # Define the headers
    header_old = ["seed", "model", "image", "preds", "targets", "iou", "dice", "acc_per_pixel", "observed_classes"]
    header_new = ["n_videos", "seed", "backbone", "res", "class", "image", "class_observed", "iou", "dice",
                  "acc_per_pixel"]

    # Load the CSV file with the old format
    old_df = pd.read_csv(old_csv, names=header_old,
                         skiprows=1)  # Assumes the first row is the header; adjust `skiprows` if needed

    # Prepare the new DataFrame
    new_data = []
    for _, row in old_df.iterrows():
        # For each observed class, create a new row
        observed_classes = safe_parse_observed_classes(row["observed_classes"])
        for cls_idx, is_observed in enumerate(observed_classes):
            # Map old data into new format
            new_data.append({
                "n_videos": str(n_videos),
                "seed": extract_run_name(row["model"]),
                "backbone": extract_backbone(row["model"]),
                "res": 'high_res' if 'high_res' in row["model"] else 'low_res',
                "class": cls_idx + 1,  # Assuming observed_classes indices match classes (class numbers start from 1)
                "image": row["image"],
                "class_observed": is_observed,  # True/False for class observed
                "iou": safe_parse_observed_classes(row["iou"])[cls_idx] if is_observed else np.nan,
                "dice": safe_parse_observed_classes(row["dice"])[cls_idx] if is_observed else np.nan,
                "acc_per_pixel": row["acc_per_pixel"]
            })

    # Create a DataFrame from the transformed data
    new_df = pd.DataFrame(new_data, columns=header_new)

    # Save to CSV if output_csv is provided
    if output_csv:
        if os.path.exists(output_csv):  # Check if the file already exists
            new_df.to_csv(output_csv, mode='a', header=False, index=False)  # Append without writing the headers
            print(f"Appended new data to: {output_csv}")
        else:
            new_df.to_csv(output_csv, index=False)  # Write with headers if file does not exist
            print(f"New format saved to: {output_csv}")

    return new_df


if __name__ == "__main__":
    csv_1 = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/1_vid_only_experiments_log_sss.csv"
    csv_2 = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/2_vids_only_experiments_log_sss.csv"
    csv_4 = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/4_vids_only_experiments_log_sss.csv"
    csv_total = "/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/_experiments_log_sss.csv"

    # load csv and merge into new format
    header_old = ["seed", "model", "image", "preds", "targets", "iou", "dice", "acc_per_pixel", "observed_classes"]
    header_new = ["n_videos", "seed", "backbone", "res", "class", "image", "class_observed", "iou", "dice",
                  "acc_per_pixel"]

    # Convert and save new CSVs
    new_data_1 = convert_old_to_new_format(csv_1, n_videos="1", output_csv="sss.csv")
    new_data_2 = convert_old_to_new_format(csv_2, n_videos="2", output_csv="sss.csv")
    new_data_4 = convert_old_to_new_format(csv_4, n_videos="4", output_csv="sss.csv")
    new_data_t = convert_old_to_new_format(csv_total, n_videos="total", output_csv="sss.csv")
