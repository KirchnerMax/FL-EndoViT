import torch
import torch.nn as nn
import math
import sys
import os
from torchvision.models import resnet18, ResNet18_Weights

# Ensure MAE prep utility is importable (pretraining/MAE folder)
try:
    # path: .../FL_EndoViT_rebuttal/fine_tuning_gynsurg/action/code -> go up 4 levels to repo root
    candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../pretraining/MAE'))
    if os.path.isdir(candidate):
        sys.path.insert(0, candidate)
    else:
        # also try repo-root relative path
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        candidate2 = os.path.join(repo_root, 'pretraining', 'MAE')
        if os.path.isdir(candidate2):
            sys.path.insert(0, candidate2)
    from prepare_mae_model import prepare_mae_model
except Exception:
    # If import fails, prepare_mae_model will be required at runtime when using MAE class
    prepare_mae_model = None

class ResNetLSTMClassifier(nn.Module):
    """
    ResNet-LSTM model for binary video classification.
    Input: (batch_size, sequence_length, channels, height, width) 
           E.g., (16, 10, 3, 224, 224) for 10 frames.
    Output: (batch_size, 1) probability of bleeding (0 to 1).
    """
    def __init__(self, lstm_hidden_size=512, lstm_layers=1, dropout_rate=0.5, num_classes=2, freeze_backbone=False):
        super(ResNetLSTMClassifier, self).__init__()
        
        # 1. Spatial Feature Extractor (ResNet)
        # Use ResNet-18 pre-trained on ImageNet for powerful feature extraction
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Remove the final fully-connected layer (the classification head)
        # to use the network as a feature extractor.
        # The output of the preceding Global Average Pooling is 512 features for ResNet-18.
        self.feature_extractor = nn.Sequential(*(list(resnet.children())[:-1]))
        
        # Optionally freeze ResNet parameters for feature extraction (Transfer Learning)
        # If freeze_backbone is True, backbone weights will not be updated.
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            
        # The size of the feature vector output by the ResNet (after flattening)
        # ResNet-18 output is 512 features (before the original FC layer)
        self.cnn_feature_dim = 512  

        # 2. Temporal Sequence Model (LSTM)
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True  # Input shape will be (batch, sequence, features)
        )
        
        # 3. Classification Head
        # For binary classification we output a single logit (use BCEWithLogitsLoss).
        # For multi-class classification we output `num_classes` logits (use CrossEntropyLoss).
        self.num_classes = num_classes
        if self.num_classes == 2:
            out_features = 1
        else:
            out_features = self.num_classes

        layers = [nn.Dropout(dropout_rate), nn.Linear(lstm_hidden_size, out_features)]
        # Do NOT apply activation here. Leave logits raw; the loss will apply the correct function.
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len, C, H, W) e.g. (16, 10, 3, 224, 224)
        batch_size, seq_len, C, H, W = x.size()
        
        # --- 1. ResNet Feature Extraction (Frame by Frame) ---
        
        # Reshape to treat all frames in the batch/sequence equally for CNN processing
        # New shape: (batch_size * seq_len, C, H, W) e.g. (160, 3, 224, 224)
        cnn_input = x.view(batch_size * seq_len, C, H, W)
        
        # Pass through the ResNet feature extractor
        cnn_output = self.feature_extractor(cnn_input)
        
        # Flatten the output to get the feature vector for each frame
        # New shape: (batch_size * seq_len, self.cnn_feature_dim) e.g. (160, 512)
        cnn_output = cnn_output.view(cnn_output.size(0), -1) 
        
        # Reshape back to sequence format for the LSTM
        # New shape: (batch_size, seq_len, self.cnn_feature_dim) e.g. (16, 10, 512)
        lstm_input = cnn_output.view(batch_size, seq_len, self.cnn_feature_dim)
        
        # --- 2. LSTM Temporal Sequence Modeling ---
        
        # Pass the sequence of features to the LSTM
        # output shape: (batch_size, seq_len, hidden_size)
        # hidden_state is a tuple (h_n, c_n), where h_n is the final hidden state
        lstm_output, (hidden_state, cell_state) = self.lstm(lstm_input)
        
        # We only need the final hidden state (h_n) for classification.
        # h_n shape: (num_layers, batch_size, hidden_size)
        
        # Select the output of the last LSTM layer (index -1)
        final_lstm_output = hidden_state[-1] # shape: (batch_size, hidden_size)
        
        # --- 3. Classification Head ---
        # Pass the sequence summary (final hidden state) through the classifier
        prediction = self.classifier(final_lstm_output) # shape: (batch_size, out_features)

        return prediction


class MAEViTLSTMClassifier(nn.Module):
    """ViT (MAE) backbone + LSTM classifier.
    Expects input shape (batch, seq_len, C, H, W). Uses MAE-prepared ViT to extract
    patch embeddings, projects high-level feature to a 512-dim map, pools and feeds
    to LSTM then classifier. Parameters for prepare_mae_model are passed via mae_kwargs.
    """
    def __init__(self, lstm_hidden_size=512, lstm_layers=1, dropout_rate=0.5, num_classes=2, mae_kwargs=None):
        super(MAEViTLSTMClassifier, self).__init__()
        self.cnn_feature_dim = 512
        self.num_classes = num_classes
        self.mae_kwargs = mae_kwargs or {}

        if prepare_mae_model is None:
            raise RuntimeError("prepare_mae_model not available. Ensure pretraining/MAE is on PYTHONPATH.")

        # instantiate MAE backbone without letting the helper load the checkpoint
        # (we will load checkpoints manually to support multiple checkpoint layouts)
        self.basemodel = prepare_mae_model(
                            mae_kwargs.get("mae_model", 'vit_base_patch16'),
                            0,  # keep nb_classes=0 to get per-patch embeddings
                            mae_kwargs.get("drop_path", 0.1),
                            mae_kwargs.get("pool_type", None),
                            '',  # do NOT pass mae_ckpt here; load manually below
                            mae_kwargs.get("freeze_weights", 3),
                            mae_kwargs.get("reinit_n_layers", -1),
                            True, # return_optimizer_groups=True
                            mae_kwargs.get("weight_decay", 0.0),
                            mae_kwargs.get("layer_decay", 1.0),
                            mae_kwargs.get("verbose", False),
                            mae_kwargs.get("debug", False)
                        )

        # prepare_mae_model may return a dict/tuple containing the model plus extra
        # optimizer/group info depending on its `return_optimizer_groups` flag.
        # Normalize to ensure self.basemodel is the actual nn.Module before using it
        # and capture any returned param_groups so the training code can reuse them.
        self.param_groups = None
        if isinstance(self.basemodel, dict):
            # capture param_groups if provided
            if 'param_groups' in self.basemodel:
                self.param_groups = self.basemodel.get('param_groups')
            # common keys that may hold the model
            for candidate_key in ('model', 'backbone', 'net', 'base_model'):
                if candidate_key in self.basemodel and hasattr(self.basemodel[candidate_key], 'load_state_dict'):
                    self.basemodel = self.basemodel[candidate_key]
                    break
        elif isinstance(self.basemodel, (list, tuple)):
            # sometimes helper returns (model, opt_groups, ...)
            if len(self.basemodel) > 0 and hasattr(self.basemodel[0], 'load_state_dict'):
                self.basemodel = self.basemodel[0]

        # If a checkpoint path is provided, load it robustly here (supports different layouts)
        mae_ckpt_path = mae_kwargs.get('mae_ckpt', '')
        if mae_ckpt_path:
            if not os.path.exists(mae_ckpt_path):
                raise FileNotFoundError(f"MAE checkpoint not found: {mae_ckpt_path}")
            ck = torch.load(mae_ckpt_path, map_location='cpu')
            # Determine state dict inside checkpoint
            checkpoint_model = None
            if isinstance(ck, dict):
                if 'model' in ck and isinstance(ck['model'], dict):
                    checkpoint_model = ck['model']
                    which = "model"
                elif 'state_dict' in ck and isinstance(ck['state_dict'], dict):
                    checkpoint_model = ck['state_dict']
                    which = "state_dict"
                elif 'model_state_dict' in ck and isinstance(ck['model_state_dict'], dict):
                    checkpoint_model = ck['model_state_dict']
                    which = "model_state_dict"
                elif 'net' in ck and isinstance(ck['net'], dict):
                    checkpoint_model = ck['net']
                    which = "net"
                else:
                    # Heuristic: top-level dict looks like a raw state_dict if keys contain '.'
                    sample_keys = list(ck.keys())[:5]
                    if sample_keys and all(isinstance(k, str) and '.' in k for k in sample_keys):
                        checkpoint_model = ck
                        which = 'raw_state_dict'
            if checkpoint_model is None:
                raise KeyError(f"Could not find model weights in checkpoint: {mae_ckpt_path}. Top keys: {list(ck.keys())}")

            # Remove possible module prefixes (e.g., 'module.') and load
            sanitized = {}
            for k, v in checkpoint_model.items():
                new_k = k
                if new_k.startswith('module.'):
                    new_k = new_k.split('module.', 1)[1]
                sanitized[new_k] = v

            missing, unexpected = self.basemodel.load_state_dict(sanitized, strict=False)
            print(f"Loaded MAE weights from '{mae_ckpt_path}' using key '{which}'. missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")

        # project MAE embedding (per-frame) to desired channel sizes using a linear layer
        # `in_ch` is the embedding dimension produced by the ViT backbone (commonly 768)
        in_ch = getattr(self.basemodel, 'num_features', None)
        if in_ch is None:
            # fallback: assume 768
            in_ch = 768
        # We'll obtain one embedding vector per frame (by pooling per-patch embeddings)
        # and then project it to `self.cnn_feature_dim` before feeding to the LSTM.
        self.frame_proj = nn.Linear(in_ch, self.cnn_feature_dim)

        # LSTM and classifier
        self.lstm = nn.LSTM(
            input_size=self.cnn_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )

        if self.num_classes == 2:
            out_features = 1
        else:
            out_features = self.num_classes
        layers = [nn.Dropout(dropout_rate), nn.Linear(lstm_hidden_size, out_features)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq, C, H, W)
        batch_size, seq_len, C, H, W = x.size()
        cnn_input = x.view(batch_size * seq_len, C, H, W)

        # MAE backbone: handle several possible output formats and normalize to (B*seq, E, p, p)
        # Common formats: (Bseq, N, E) [per-patch embeddings], (Bseq, E, H, W) [feature map],
        # (Bseq, H, W, E), or (Bseq, E) (global-pooled). We call basemodel with Bseq = batch_size*seq_len.
        bseq = batch_size * seq_len
        # Prefer using forward_features to obtain patch embeddings / feature maps
        if hasattr(self.basemodel, 'forward_features'):
            out = self.basemodel.forward_features(cnn_input)
        else:
            out = self.basemodel(cnn_input)

        # If backbone returns a tuple/list, pick the first tensor-like entry
        if isinstance(out, (tuple, list)):
            found = None
            for elem in out:
                if isinstance(elem, torch.Tensor):
                    found = elem
                    break
            if found is None:
                raise RuntimeError(f"MAE backbone returned a tuple/list but no tensor found: {type(out)}")
            out = found

        if not isinstance(out, torch.Tensor):
            raise RuntimeError(f"MAE backbone returned unexpected type: {type(out)}")

        # Reduce per-patch or per-spatial maps to a single embedding vector per frame
        # out may be:
        #  - (Bseq, N, E): per-patch embeddings -> mean over patches -> (Bseq, E)
        #  - (Bseq, E, H, W): feature maps -> global avg pool -> (Bseq, E)
        #  - (Bseq, E): already per-frame embedding
        if out.dim() == 3:
            # (Bseq, N, E)
            frame_emb = out.mean(dim=1)  # (Bseq, E)
        elif out.dim() == 4:
            # (Bseq, E, H, W) or (Bseq, H, W, E)
            if out.shape[1] >= out.shape[-1]:
                # assume (Bseq, E, H, W)
                frame_emb = out.mean(dim=(2, 3))  # (Bseq, E)
            else:
                # assume (Bseq, H, W, E)
                frame_emb = out.permute(0, 3, 1, 2).mean(dim=(2, 3))
        elif out.dim() == 2:
            # (Bseq, E)
            frame_emb = out
        else:
            raise RuntimeError(f"Unsupported MAE backbone output with ndim={out.dim()} and shape={tuple(out.shape)}")

        # project per-frame embedding to cnn_feature_dim
        projected = self.frame_proj(frame_emb)  # (Bseq, cnn_feature_dim)

        # reshape to (batch, seq, feature)
        lstm_input = projected.view(batch_size, seq_len, self.cnn_feature_dim)

        lstm_output, (hidden_state, cell_state) = self.lstm(lstm_input)
        final_lstm_output = hidden_state[-1]
        logits = self.classifier(final_lstm_output)

        probs = torch.sigmoid(logits) if self.num_classes == 2 else torch.softmax(logits, dim=1)
        pred_classes = torch.argmax(probs, dim=1) if self.num_classes > 2 else (probs > 0.5).int()
        return logits, probs, pred_classes




if __name__ == "__main__":
    # Example usage and sanity check
    model = ResNetLSTMClassifier(lstm_hidden_size=256, lstm_layers=2)
    print(model)
    
    # Create a dummy input tensor with shape (batch_size, seq_len, C, H, W)
    dummy_input = torch.randn(2, 10, 3, 224, 224)  # e.g., batch_size=2, seq_len=10
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")  # Should be (2, 1)
    print(f"Output values: {output}")        # Probabilities between 0 and 1

    # example MAE model
    mae_model = MAEViTLSTMClassifier(lstm_hidden_size=256, lstm_layers=2, mae_kwargs={
        'mae_model': 'vit_base_patch16',
        'mae_ckpt': '/mnt/ceph/tco/TCO-Staff/Homes/kirchnema/04_pycharm/05_endovit_fl/03_EndoViT_FL_2/EndoViT_2/finetuning/action_triplet_detection/output_dir/full_dataset/ViT_backbone/EndoViT/run_0003_23:05-07.11.24__FullDataset_EndoViT_Seed_37/mae_cholect45-crossval_k5_best_acc.pth',  # path to checkpoint if available
        'freeze_weights': 3,  # freeze all except classification head,
        'drop_path': 0.1
    }, num_classes=5)
    print(mae_model)
    dummy_input = torch.randn(2, 10, 3, 224, 224)  # e.g., batch_size=2, seq_len=10
    logits, probs, pred_classes = mae_model(dummy_input)
    print(f"Output predicted classes: {pred_classes}")  # Should be (2, 1)
    print(f"Output prediction probabilities: {probs}")        # Probabilities between 0 and 1



