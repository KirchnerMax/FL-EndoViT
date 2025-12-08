#!/bin/bash
#SBATCH --job-name=Pretrain_FL_EndoViT
#SBATCH --output=outputs/%j/train-%j.out
#SBATCH --error=outputs/%j/train-error%j.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:rtx5000:1
#SBATCH --mem-per-cpu=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH  --mail-user=max.kirchner@nct-dresden.de
#SBATCH --time=24:00:00


ROOT_DIR='./outputs'

## Experiment 1A - Segmentation
#python ASAM_simulation_main.py log_dir=$ROOT_DIR/experiment_1A/ Server.output_dir=$ROOT_DIR/experiment_1A/Server Cholec80.train_datasets_to_take=[Cholec80_for_Segmentation]
#
## Experiment 1B - ATR
#python ASAM_simulation_main.py log_dir=$ROOT_DIR/experiment_1B/ Server.output_dir=$ROOT_DIR/experiment_1B/Server Cholec80.train_datasets_to_take=[Cholec80_for_ActionTripletDetection]
#
## Experiment 1C - PR
#python ASAM_simulation_main.py log_dir=$ROOT_DIR/experiment_1C/ Server.output_dir=$ROOT_DIR/experiment_1C/Server Cholec80.train_datasets_to_take=[Cholec80_for_SurgicalPhaseRecognition]
#
## Experiment 1D - FL-EndoViT-Segmentation w/out ASAM
#python simulation_main.py log_dir=$ROOT_DIR/experiment_1D/ Server.output_dir=$ROOT_DIR/experiment_1D/Server Cholec80.train_datasets_to_take=[Cholec80_for_Segmentation]
#
## Experiment 1E - FL-EndoViT-ATR w/out ASAM
#python simulation_main.py log_dir=$ROOT_DIR/experiment_1E/ Server.output_dir=$ROOT_DIR/experiment_1E/Server Cholec80.train_datasets_to_take=[Cholec80_for_ActionTripletDetection]
#
## Experiment 1F - FL-EndoViT-PR w/out ASAM
#python simulation_main.py log_dir=$ROOT_DIR/experiment_1F/ Server.output_dir=$ROOT_DIR/experiment_1F/Server Cholec80.train_datasets_to_take=[Cholec80_for_SurgicalPhaseRecognition]

# Experiment 2A - FL-EndoViT FedMedian Segmentation w/out ASAM
python simulation_main.py log_dir=$ROOT_DIR/%j/experiment_2A/ Server.output_dir=$ROOT_DIR/experiment_2A/Server Cholec80.train_datasets_to_take=[Cholec80_for_ActionTripletDetection] Aggregator=FedMedian

# Experiment 2B - FL-EndoViT FedProx Segmentation w/out ASAM
python simulation_main.py log_dir=$ROOT_DIR/%j/experiment_2B/ Server.output_dir=$ROOT_DIR/experiment_2B/Server Cholec80.train_datasets_to_take=[Cholec80_for_ActionTripletDetection] Aggregator=FedProx
