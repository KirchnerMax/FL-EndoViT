#!/bin/bash
#SBATCH --job-name=Pretrain_FL_EndoViT
#SBATCH --output=outputs/%j/train-%j.out
#SBATCH --error=outputs/%j/train-error%j.out
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=max.kirchner@nct-dresden.de
#SBATCH --time=24:00:00


ROOT_DIR='./outputs'

# Experiment 2C - FL-EndoViT QFedAvg Segmentation w/out ASAM
python simulation_main.py log_dir=$ROOT_DIR/%j/experiment_2C/ \
Server.output_dir=$ROOT_DIR/experiment_2C/Server \
Cholec80.train_datasets_to_take=[Cholec80_for_ActionTripletDetection] \
Aggregator=QFedAvg \
q=0.2 \
dataloader.N=1024 \
swa.swa_start=3 \
training.num_rounds=5 \
num_clients=3

