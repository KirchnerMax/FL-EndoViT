#!/bin/bash
#SBATCH --job-name=Pretrain_FL_EndoViT
#SBATCH --output=outputs/%j/train-%j.out
#SBATCH --error=outputs/%j/train-error%j.out
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:v100:2
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=max.kirchner@ukdd.de
#SBATCH --time=24:00:00


ROOT_DIR='./outputs'

# Experiment 2B - FL-EndoViT FedProx Segmentation w/out ASAM
python simulation_main_fedprox.py log_dir=$ROOT_DIR/%j/experiment_2B/ \
Server.output_dir=$ROOT_DIR/experiment_2B/Server \
Cholec80.train_datasets_to_take=[Cholec80_for_ActionTripletDetection] \
Aggregator=FedProx \
config_path=/mnt/cluster/workspaces/kirchnema/FL_EndoViT_rebuttal/pretraining/conf_fedprox/
