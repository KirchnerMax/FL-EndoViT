import copy
import logging
import math
import os
import shutil
from collections import OrderedDict
from logging import INFO
from typing import List, Tuple, Dict, Union, Optional

import flwr.server
import torch
import wandb
from flwr.common import Context, Parameters, Scalar
from flwr.common.logger import log

import hydra
import numpy as np
from flwr.server import ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from omegaconf import DictConfig, OmegaConf
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.tensorboard import SummaryWriter

import MAE.mae_model as models_mae
import client
from MAE.engine_pretrain import evaluate
from MAE.util import misc
from set_wandb import setup_wandb


class FedAvgSWAServer(FedAvg):
    def __init__(self, cfg, valloader, logger):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg

        self.validationloader = valloader

        self.csv_logger = logger

        self.swa_model = None
        self.model = models_mae.__dict__[cfg.model_specific.model](norm_pix_loss=cfg.model_specific.norm_pix_loss,
                                                                   loss_type=cfg.model_specific.loss,
                                                                   hpf_loss=cfg.model_specific.high_pass_filter_loss)
        super().__init__(on_fit_config_fn=self.fit_config, min_available_clients=cfg.num_clients,
                         min_fit_clients=cfg.num_clients,
                         min_evaluate_clients=cfg.num_clients,
                         fit_metrics_aggregation_fn=aggregate_metrics,
                         evaluate_metrics_aggregation_fn=aggregate_metrics,
                         evaluate_fn=self.get_evaluate_fn())
        self.round = 1
        self.swa_n = 0
        self.lr = cfg.training.lr
        self.best_val_loss = math.inf

        dir_path = './conf/optimizer_cache'
        if os.path.exists(dir_path):
            # Delete the directory and its contents
            shutil.rmtree(dir_path)
        os.makedirs(f'./conf/optimizer_cache')

        self.log_dir = cfg.log_dir + '/Server'
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=self.log_dir)

    def __del__(self):
        print("************************************  Server deleted  ************************************")

    def get_evaluate_fn(self):
        def evaluate_fn(server_round, parameters_ndarrays, config):
            self.tr_cfg = OmegaConf.load(f'./conf/config_server.yaml')

            print("************************************  Server EVAL  ************************************")
            if server_round <= self.cfg.swa.swa_start:
                return np.inf, {}
            model = self.swa_model.float()
            device = self.device

            # Ensure the bias tensor is of the same type as the input tensor
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    if module.weight.dtype != module.bias.dtype:
                        module.bias = module.bias.to(module.weight.dtype)

            val_stats = evaluate(self.validationloader, model, device, self.log_writer,
                                 server_round, self.cfg, epoch_end=True, is_swa=True, c_name='server', cid=10,
                                 tr_cfg=self.tr_cfg)
            log_stats = {**{f'swa_val_evaluate_fn_{k}': v for k, v in val_stats.items()},
                         'epoch': server_round}
            print(f"[EVAL] Log_stats: {log_stats}")

            self.csv_logger.log_metrics(log_stats, server_round, 'server', 10)

            if self.cfg.use_wandb:
                setup_wandb(self.tr_cfg, 10)
                wandb.log(log_stats)
                wandb.finish()
            if log_stats['swa_val_evaluate_fn_loss_swa'] is None:
                return_loss = {'swa_val_loss': float('inf')}
            else:
                return_loss = {'swa_val_loss': log_stats['swa_val_evaluate_fn_loss_swa']}

            return return_loss, log_stats

        return evaluate_fn


    def fit_config(self, server_round: int):
        """ Retrun a config to the clients"""
        config = {
            "round": server_round,
            "lr": self.lr,
        }
        return config


    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, flwr.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, flwr.common.FitRes], BaseException]],
    ) -> Tuple[Optional[flwr.common.Parameters], Dict[str, flwr.common.Scalar]]:
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        ####### SWA ########
        if self.cfg.swa.swa:
            self.do_swa = self.cfg.swa.swa_start <= self.round
        else:
            self.do_swa = False

        if (self.do_swa and self.swa_model is None):
            print("-" * 50)
            print("-" * 50)
            print("Creating SWA model!")
            print("-" * 50)
            print("-" * 50)
            self.swa_model = AveragedModel(self.model).to(self.device)

        if (self.cfg.swa.swa and self.round > self.cfg.swa.swa_start and
                (self.round - self.cfg.swa.swa_start) % self.cfg.swa.swa_c == 0):
            # end of cycle
            print("Number of cycles:", self.swa_n)
            # update based on FedSam
            # Update SWA model
            if self.cfg.swa.swa_c > 1:
                self.lr = self.schedule_cycling_lr(self.round, self.cfg.swa.swa_c, self.cfg.training.lr,
                                                   self.cfg.swa.swa_lr)
                # Update LR for Clients
                self.cfg.training.lr = self.lr

            if self.cfg.swa.swa and self.round > self.cfg.swa.swa_start and (
                    self.round - self.cfg.swa.swa_start) % self.cfg.swa.swa_c == 0:  # end of cycle
                print("Number of models:", self.swa_n)
                alpha = 1.0 / (self.swa_n + 1.0)
                self.swa_model.to('cpu')
                self.model.to('cpu')
                for param1, param2 in zip(self.swa_model.parameters(), self.model.parameters()):
                    param1.data *= (1.0 - alpha)
                    param1.data += param2.data * alpha
                self.swa_n += 1.0

        self.round += 1
        ####### SWA ########

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = flwr.common.parameters_to_ndarrays(aggregated_parameters)

            print(f"Aggregated metrics: {aggregated_metrics}")
            print(f"Aggregated metrics keys: {aggregated_metrics.keys()}")

            if aggregated_metrics is not None:
                if aggregated_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = aggregated_metrics['val_loss']
                    log(INFO, f"Saving round {server_round} aggregated_ndarrays...")
                    # np.savez(f"{self.cfg.Server.output_dir}/best-weights.npz", *aggregated_ndarrays)
                    self.save_model(self.model, server_round, 'best_model_server')
                    if self.cfg.swa.swa and self.swa_model is not None:
                        self.save_model(self.swa_model, server_round, 'best_swa_model_server')

            # Save aggregated_ndarrays
            log(INFO, f"Saving round {server_round} aggregated_ndarrays...")
            # np.savez(f"{self.cfg.Server.output_dir}/round-{server_round}-weights.npz", *aggregated_ndarrays)
            self.save_model(self.model, server_round, f'best_model_round{server_round}_server')
            if self.cfg.swa.swa and self.swa_model is not None:
                self.save_model(self.swa_model, server_round, f'best_model_round{server_round}_server')

            self.model = self.set_parameters(self.model, aggregated_ndarrays)
        return aggregated_parameters, aggregated_metrics


    def schedule_cycling_lr(self, round, c, lr1, lr2):
        t = 1 / c * (round % c + 1)
        lr = (1 - t) * lr1 + t * lr2
        return lr


    def set_parameters(self, model, parameters):
        """Loads model and replaces it parameters with the ones given."""
        # Convert parameters to a list if it is not already iterable
        if not isinstance(parameters, list):
            parameters = list(parameters)

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model


    def save_model(self, model, epoch, model_name):
        to_save = {
            'model': model.state_dict(),
            'epoch': epoch,
            'args': self.cfg,
        }
        torch.save(to_save, f"{self.cfg.Server.output_dir}/{model_name}.pth")


def aggregate_metrics(metrics_list):
    """Aggregate metrics"""
    if len(metrics_list) == 0:
        return {}
    # Sum all metrics
    print(f"Metrics list: {metrics_list}")
    aggregated_metrics = {}
    for metrics in metrics_list:
        for key, value in metrics[1].items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = value
            else:
                aggregated_metrics[key] += value
    return aggregated_metrics


@hydra.main(version_base="1.3", config_path="../../config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    # main()
    tr_cfg = OmegaConf.load(f'./conf/config_server.yaml')
    print(tr_cfg)