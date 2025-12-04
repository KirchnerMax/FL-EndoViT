# FL-EndoViT: Pretraining Vision Transformers via Federated Learning on Endoscopic Image Collections

Official implementation of the paper: **FL-EndoViT: Pretraining Vision Transformers via Federated Learning on Endoscopic Image Collections** (Submitted to MIDL 2026)

![Method, overview](/images/methods.png)

[Paper arXiv](https://arxiv.org/abs/2504.16612) | [Endo700k Dataset Collection](https://github.com/DominikBatic/EndoViT#download-endo700k) | [EndoViT Baseline](https://github.com/DominikBatic/EndoViT)

## Abstract 

**Purpose:** Data privacy regulations hinder the creation of generalizable foundation models (FMs) for surgery by preventing multi-institutional data aggregation. This study investigates federated learning (FL) as a privacy-preserving solution to collaboratively train robust surgical FMs.

**Methods:** We introduce Federated EndoViT (FL-EndoViT), which adapts the Masked Autoencoder (MAE) pretraining strategy for FL, enhanced with adaptive Sharpness-Aware Minimization (FedSAM) to manage surgical data heterogeneity. Pretrained on the large-scale Endo700k dataset, FL-EndoViT is evaluated against a centralized baseline on different tasks including scene segmentation, action recognition, and phase recognition.

**Results:** FedSAM is critical for successful pretraining, overcoming the convergence failures of standard federated methods. The resulting FL-EndoViT performs comparably to its centralized counterpart, with significant advantages in data-scarce, high-resolution segmentation and generalization to new surgical events. We also establish that full, end-to-end fine-tuning is necessary for optimal performance.

**Conclusion:** This work establishes FL with adaptive optimization as a viable paradigm for creating robust, privacy-preserving surgical FMs. Our findings provide a scalable framework for collaborative Surgical Data Science and underscore the optimizer's critical role in handling data heterogeneity. Future work should explore video-based models to incorporate spatiotemporal dynamics.

## Framework Overview

The framework consists of two distinct phases:

1. **Federated Pretraining:** Self-supervised training on decentralized data (Endo700k dataset collection) using Masked Autoencoder (MAE) approach, client-side Adaptive FedSAM, and server-side Stoachastic Weight Averaging (SWA).
2. **Downstream Task Fine-Tuning:** The pretrained encoder is fine-tuned (end-to-end) on specific surgical tasks like Surgical Phase Recognition, Action Triplet Recognition, Surgical Scene Segmentation, and Classification tasks of the GynSurg dataset.

## Installation 

## Data Preparation 

## Usage

## Results 

## Citation 

## Acknowledgements

## Contact

For questions or feedback, please contact: 
Max Kirchner - max.kirchner@nct-dresden.de
