import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model import ResNetLSTMClassifier
from data import GynSurgAction
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np


def evaluate_on_test(model, dl_test, device, num_classes=2, max_batches=10):
    """Run a short evaluation on dl_test for up to max_batches and return (acc, macro_f1).

    This function is intentionally lightweight (limits number of batches) so it can be
    called frequently during training without spending too much time.
    """
    # preserve original training mode and restore it after evaluation
    was_training = model.training
    model.eval()
    all_preds = []
    all_labels = []
    try:
        with torch.no_grad():
            for b, batch in enumerate(dl_test):
                if b >= max_batches:
                    break
                # dataset may return (frames, label, case_id) or (frames, label)
                if len(batch) == 3:
                    input_data, targets, _case_ids = batch
                else:
                    input_data, targets = batch
                input_data = input_data.to(device)
                targets = targets.to(device)
                output = model(input_data)
                # model may return logits or (logits, probs, preds)
                if isinstance(output, (tuple, list)):
                    logits = output[0]
                else:
                    logits = output

                if num_classes == 2:
                    logits = logits.view(-1)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).long().cpu().numpy()
                    labels = targets.view(-1).long().cpu().numpy()
                else:
                    preds = logits.argmax(dim=1).cpu().numpy()
                    labels = targets.view(-1).long().cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
    finally:
        # restore previous training/eval mode so training continues correctly
        if was_training:
            model.train()
        else:
            model.eval()

    if len(all_labels) > 0:
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
    else:
        acc, f1 = None, None
    return acc, f1


def train(kfold=0):
    # load training dataset with augmentations
    # Resize to 256x256 and apply augmentations: rotation, color jitter, gaussian blur
    train_transform = T.Compose([
        T.Resize((256, 256)),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # Gaussian blur: torchvision.transforms.GaussianBlur added in newer versions; fallback if missing
    ])
    # Add GaussianBlur if available
    try:
        from torchvision.transforms import GaussianBlur
        train_transform.transforms.insert(-1, GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)))
    except Exception:
        pass

    # For validation/testing, only resize and normalize
    test_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    ds_train = GynSurgAction(
        root="/mnt/cluster/datasets/GynSurg_Action_3sec/GynSurg_action_dataset/",
        df_path='./df_action.csv',
        split='train',
        k_fold=kfold,
        transform=train_transform
    )
    dl_train = DataLoader(ds_train, batch_size=16, shuffle=True, num_workers=4)

    # load test dataset
    ds_test = GynSurgAction(
        root="/mnt/cluster/datasets/GynSurg_Action_3sec/GynSurg_action_dataset/",
        df_path='./df_action.csv',
        split='test',
        k_fold=kfold,
        transform=test_transform
    )
    dl_test = DataLoader(ds_test, batch_size=16, shuffle=False, num_workers=4)

    print(f"Train dataset size: {len(ds_train)}")
    print(f"Test dataset size: {len(ds_test)}")

    # Determine number of classes from dataset
    num_classes = len(ds_train.label2idx) if hasattr(ds_train, 'label2idx') else 2

    # device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetLSTMClassifier(lstm_hidden_size=256, lstm_layers=2, num_classes=num_classes).to(device)
    print(model)

    # loss
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer: SGD with momentum; backbone lr = 0.1 * main lr
    lr_init = 0.005
    momentum = 0.9
    # Separate params: backbone and head
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if 'feature_extractor' in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.SGD([
        {'params': head_params, 'lr': lr_init},
        {'params': backbone_params, 'lr': lr_init * 0.1}
    ], momentum=momentum, weight_decay=1e-4)

    # Polynomial LR schedule (per-iteration)
    total_epochs = 40
    iters_per_epoch = len(dl_train)
    total_iters = total_epochs * iters_per_epoch
    power = 0.9

    # training loop
    epochs = total_epochs
    first_batch = True
    # inter-batch evaluation config: evaluate every `eval_interval` training iterations
    eval_interval = 200  # run a short evaluation every 200 training minibatches (tune as needed)
    max_eval_batches = 10  # limit number of test batches used per inter-batch eval
    last_eval = None
    best_acc = -1.0
    best_f1 = -1.0
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(dl_train, desc=f"Train Epoch {epoch+1}/{epochs}", leave=False)):
            # dataset may return (frames, label, case_id) or (frames, label)
            if len(batch) == 3:
                input_data, targets, _case_ids = batch
            else:
                input_data, targets = batch
            input_data = input_data.to(device)
            targets = targets.to(device)

            if first_batch:
                print(f"[DEBUG] input_data.shape={input_data.shape}, targets.shape={targets.shape}, num_classes={num_classes}")
                first_batch = False

            # prepare targets
            if num_classes == 2:
                targets_proc = targets.float()
            else:
                targets_proc = targets.long()

            # forward
            output = model(input_data)
            if num_classes == 2:
                output = output.view(-1)  # (batch,)
                loss = criterion(output, targets_proc)
            else:
                # output shape: (batch, num_classes)
                loss = criterion(output, targets_proc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update poly LR per iteration
            current_iter = epoch * iters_per_epoch + i
            lr = lr_init * ((1 - float(current_iter) / float(max(1, total_iters))) ** power)
            # set lr for param groups (head first, backbone second)
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 0.1

            # periodic inter-batch evaluation (lightweight)
            if (current_iter + 1) % eval_interval == 0:
                acc, f1 = evaluate_on_test(model, dl_test, device, num_classes=num_classes, max_batches=max_eval_batches)
                last_eval = (acc, f1)
                # update best metrics
                if acc is not None and f1 is not None:
                    if acc > best_acc:
                        best_acc = acc
                    if f1 > best_f1:
                        best_f1 = f1
                # update tqdm postfix for quick visibility
                try:
                    tqdm.write(f"[Inter-eval] iter={current_iter+1} | acc={acc if acc is not None else 'n/a'} | macro-f1={f1 if f1 is not None else 'n/a'}")
                except Exception:
                    pass

    # evaluation
    model.eval()
    all_preds = []
    all_labels = []
    test_records = []
    with torch.no_grad():
        for batch in tqdm(dl_test, desc="Test", leave=False):
            # dataset may return (frames, label, case_id) or (frames, label)
            if len(batch) == 3:
                input_data, targets, case_ids = batch
            else:
                input_data, targets = batch
                case_ids = [None] * input_data.size(0)
            input_data = input_data.to(device)
            targets = targets.to(device)
            output = model(input_data)
            if num_classes == 2:
                output = output.view(-1)
                probs = torch.sigmoid(output)
                preds = (probs > 0.5).long().cpu().numpy()
                labels = targets.view(-1).long().cpu().numpy()
            else:
                preds = output.argmax(dim=1).cpu().numpy()
                labels = targets.view(-1).long().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            for cid, gt, pr in zip(case_ids, labels.tolist(), preds.tolist()):
                test_records.append((cid, int(gt), int(pr)))

    # compute metrics
    if len(all_labels) > 0:
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Test Accuracy: {acc:.4f} | Test Macro-F1: {f1:.4f}")
        # update best metrics with final test
        if acc is not None and f1 is not None:
            if acc > best_acc:
                best_acc = acc
            if f1 > best_f1:
                best_f1 = f1
    else:
        print("No test samples found to evaluate.")

    # write test predictions CSV for this fold
    out_csv = f"test_RESNET50_predictions_fold_{kfold}.csv"
    try:
        import csv
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'gt', 'pred'])
            for rec in test_records:
                writer.writerow(rec)
        print(f"Wrote test predictions to {out_csv}")
    except Exception as e:
        print(f"Failed to write test CSV: {e}")

    # plot confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for gt, pr in zip(all_labels, all_preds):
        cm[gt, pr] += 1

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(num_classes)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix Fold {kfold}')
    plt.savefig(f'confusion_matrix_RESNET50_fold_{kfold}.png')
    plt.close()
    print(f"Saved confusion matrix to confusion_matrix_RESNET50_fold_{kfold}.png")

    return {"best_acc": best_acc, "best_f1": best_f1}


def main():
    parser = argparse.ArgumentParser(description='Train ResNet-LSTM classifier (single fold)')
    parser.add_argument('--kfold', type=int, default=None, help='Which fold to run (0-based). If not set, runs folds 0..4 sequentially')
    parser.add_argument('--cuda-device', type=int, default=None, help='CUDA device index to use for this process')
    args = parser.parse_args()

    # If cuda-device specified, set CUDA_VISIBLE_DEVICES so torch picks that GPU as device 0
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    all_fold_results = []
    if args.kfold is not None:
        print(f"\n=== Training fold {args.kfold+1} ===")
        res = train(kfold=args.kfold)
        all_fold_results.append(res)
    else:
        for kfold in range(5):
            print(f"\n=== Training fold {kfold+1}/5 ===")
            res = train(kfold=kfold)
            all_fold_results.append(res)

    print("\n=== Per-fold best metrics summary ===")
    for i, r in enumerate(all_fold_results):
        print(f"Fold {i+1}: best_acc={r['best_acc']:.4f} | best_macro_f1={r['best_f1']:.4f}")

if __name__ == "__main__":
    main()