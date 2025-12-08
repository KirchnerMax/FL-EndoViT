import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd
import random
import os
import numpy as np
import cv2


def sample_random_frames_sequential(video_path, n_frames=10, seed=None, resize=None, as_rgb=True):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Some files on disk have ':' replaced by '_' in the filename.
        # Try alternate paths before giving up.
        dirname = os.path.dirname(video_path)
        basename = os.path.basename(video_path)
        alt_basename = basename.replace(':', '_')
        alt_path = os.path.join(dirname, alt_basename)
        opened = False
        if alt_path != video_path and os.path.exists(alt_path):
            cap = cv2.VideoCapture(alt_path)
            if cap.isOpened():
                video_path = alt_path
                opened = True

        if not opened:
            alt_path2 = video_path.replace(':', '_')
            if alt_path2 != video_path and os.path.exists(alt_path2):
                cap = cv2.VideoCapture(alt_path2)
                if cap.isOpened():
                    video_path = alt_path2
                    opened = True

        if not opened and not cap.isOpened():
            print(f"Warning: could not open video file {video_path}")
            # As a last resort, don't raise inside DataLoader worker; return placeholder frames
            h, w = (resize[1], resize[0]) if resize is not None else (224, 224)
            black = np.zeros((h, w, 3), dtype=np.uint8)
            frames = [black.copy() for _ in range(n_frames)]
            indices = list(range(n_frames))
            return frames, indices

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # fallback: count frames (rare)
        total = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            total += 1
        cap.release()
        cap = cv2.VideoCapture(video_path)

    n = min(n_frames, total)
    if n == 0:
        cap.release()
        return [], []

    indices = sorted(random.sample(range(total), n))  # unique samples
    frames = []
    target_idx_ptr = 0
    current_idx = 0

    while target_idx_ptr < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx == indices[target_idx_ptr]:
            if resize is not None:
                frame = cv2.resize(frame, (resize[0], resize[1]), interpolation=cv2.INTER_AREA)
            if as_rgb:
                # Make a copy after channel-reversal to ensure positive strides
                frame = frame[:, :, ::-1].copy()
            frames.append(frame)
            target_idx_ptr += 1
        current_idx += 1

    # If some indices not found (corrupt/short), we can pad with last frame
    if len(frames) < len(indices) and len(frames) > 0:
        last = frames[-1]
        frames += [last] * (len(indices) - len(frames))

    cap.release()
    return frames, indices


class GynSurgAction(Dataset):
    def __init__(self, root, df_path, k_fold, split, transform=None):
        self.transform = transform if transform else ToTensor()
        self.root = root
        self.df_path = df_path
        self.k_fold = k_fold
        self.split = split

        df = pd.read_csv(df_path)
        # Filter the DataFrame based on k_fold and split
        df = df[(df['split'] == self.k_fold) & (df['set_type'] == self.split)]
        samples = []
        for _, row in df.iterrows():
            samples.append({
                'case_id': row['case_id'],
                'video_path': row['vids'],
                'label': row['labels']
            })
        self.data = samples
        self.label2idx = {label: idx for idx, label in enumerate(sorted(df['labels'].unique()))}
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get item by index. Returns 10 random frames from the video, the label, and case_id."""
        video_path = self.data[idx]['video_path']
        video_path = video_path.split("/")[1:]
        video_path = "/".join(video_path)
        video_path = f"{self.root}/{video_path}"

        # load randomly 10 frames from mp4
        frames, frame_indices = sample_random_frames_sequential(video_path, n_frames=10, seed=None, resize=(224, 224), as_rgb=True)

        # get label and case_id
        label = self.data[idx]['label']
        label_idx = self.label2idx[label]
        case_id = self.data[idx]['case_id']

        # Stack frames into a single numpy array (N, H, W, C)
        frames_np = np.stack(frames)

        # Convert to torch tensor: (N, C, H, W), float in [0,1]
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
        # Normalize to match transforms.Normalize(mean=0.5, std=0.5)
        frames_tensor = (frames_tensor - 0.5) / 0.5

        # Return as tuple (frames, label, case_id) so DataLoader batches into
        # (batch_frames, batch_labels, batch_case_ids). case_id is a string.
        return frames_tensor, torch.tensor(label_idx, dtype=torch.long), case_id


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = GynSurgAction(
        root="/mnt/cluster/datasets/GynSurg_Action_3sec/GynSurg_bleeding_dataset/",
        df_path='./df_bleeding.csv',
        k_fold=0,
        split='train'
    )

    # print length of dataset
    print(f"Dataset length: {len(ds)}")
    # get first item and unpack (frames_tensor, label_idx, case_id)
    item = ds[0]
    print(f"First item: (frames_tensor shape, label_idx, case_id) = ({item[0].shape}, {item[1]}, {item[2]})")
    print(f"Label to ID mapping: {ds.label2idx}")

    idx2label = {v: k for k, v in ds.label2idx.items()}

    frames_tensor, label_idx_tensor, case_id = item
    # convert frames tensor back to HWC float images in [0,1] for plotting
    # frames_tensor shape: (N, C, H, W) and normalized as (x - 0.5)/0.5
    frames_vis = (frames_tensor * 0.5 + 0.5).permute(0, 2, 3, 1).cpu().numpy()
    num_frames = frames_vis.shape[0]
    print(f"Number of frames: {num_frames}")
    if num_frames == 0:
        print("No frames to display")
    else:
        for i in range(num_frames):
            plt.subplot(2, 5, i+1)
            plt.imshow(frames_vis[i])
            plt.axis('off')
        # add title with label and case_id
        label_idx = int(label_idx_tensor)
        plt.suptitle(f"Label idx: {label_idx}, Label: {idx2label.get(label_idx, 'N/A')}, Case ID: {case_id}")
        plt.show()


