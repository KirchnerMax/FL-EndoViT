import copy
import logging
import math
import os
import shutil
from collections import OrderedDict
from logging import INFO
from typing import List, Tuple, Dict, Union, Optional

import flwr.server
import numpy as np
import torch
import wandb
from flwr.common import Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, FedMedian, QFedAvg, FedAdam, FedAvgM, Krum
from omegaconf import DictConfig, OmegaConf
from torch.optim.swa_utils import AveragedModel
from torch.utils.tensorboard import SummaryWriter

import MAE.mae_model as models_mae
from MAE.engine_pretrain import evaluate
from set_wandb import setup_wandb


def aggregate_metrics(metrics_list):
    """Aggregate metrics across all clients"""
    if not metrics_list:
        return {}

    aggregated_metrics = {}
    for metrics in metrics_list:
        for key, value in metrics[1].items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = value
            else:
                aggregated_metrics[key] += value
    return aggregated_metrics


class BaseSWAServer:
    """Base class implementing common SWA functionality"""

    def __init__(self, cfg, valloader, logger):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Server runs on: {self.device}")
        self.cfg = cfg
        self.validationloader = valloader
        self.csv_logger = logger

        self.swa_model = None
        self.model = models_mae.__dict__[cfg.model_specific.model](
            norm_pix_loss=cfg.model_specific.norm_pix_loss,
            loss_type=cfg.model_specific.loss,
            hpf_loss=cfg.model_specific.high_pass_filter_loss
        )

        self.round = 1
        self.swa_n = 0
        self.lr = cfg.training.lr
        self.best_val_loss = math.inf

        self._setup_directories(cfg)
        self._setup_logging()

    def _setup_directories(self, cfg):
        """Setup required directories"""
        # Setup optimizer cache
        dir_path = f"{cfg.config_path}" + '/optimizer_cache/'
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

        # Setup logging directory
        self.log_dir = self.cfg.log_dir + '/Server'
        os.makedirs(self.log_dir, exist_ok=True)

    def _setup_logging(self):
        """Setup logging"""
        self.log_writer = SummaryWriter(log_dir=self.log_dir)

    def __del__(self):
        print("************************************  Server deleted  ************************************")

    def get_evaluate_fn(self):
        """Returns evaluation function for SWA model"""

        def evaluate_fn(server_round, parameters_ndarrays, cfg):
            self.tr_cfg = OmegaConf.load(f'{self.cfg["config_path"]}config_server.yaml')

            # Determine if SWA is active for this round
            do_swa = self.cfg.swa.swa and (server_round >= self.cfg.swa.swa_start)
            if do_swa:
                print(f"************************************  Server EVAL round{server_round} [SWA MODEL] ************************************")
                # Prepare and evaluate SWA model only
                model = self._prepare_model_for_evaluation(server_round)
            else:
                print(f"************************************  Server EVAL round{server_round} [BASE MODEL] ************************************")
                # Evaluate base model only
                model = self.model

            val_stats = self._evaluate_model(model, server_round)
            return self._process_evaluation_results(val_stats, server_round)

        return evaluate_fn

    def _prepare_model_for_evaluation(self, server_round):
        """Prepare SWA model for evaluation"""
        self.handle_swa_update(server_round)
        model = self.swa_model.float()

        # Ensure consistent tensor types
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                if module.weight.dtype != module.bias.dtype:
                    module.bias = module.bias.to(module.weight.dtype)

        return model

    def _evaluate_model(self, model, server_round):
        """Evaluate the model"""
        is_swa = server_round >= self.cfg.swa.swa_start
        return evaluate(
            self.validationloader, model, self.device, self.log_writer,
            server_round, self.cfg, epoch_end=True, is_swa=is_swa,
            c_name='server', cid=10, tr_cfg=self.tr_cfg
        )

    def _process_evaluation_results(self, val_stats, server_round):
        """Process evaluation results and handle logging"""
        log_stats = {
            **{f'swa_val_evaluate_fn_{k}': v for k, v in val_stats.items()},
            'epoch': server_round
        }

        print(f"[EVAL] Log_stats: {log_stats}")
        self.csv_logger.log_metrics(log_stats, server_round, 'server', 10)

        if self.cfg.use_wandb:
            setup_wandb(self.tr_cfg, 10)
            wandb.log(log_stats)
            wandb.finish()

        # Robustly handle missing loss key
        loss_key = 'swa_val_evaluate_fn_loss_swa'
        if loss_key in log_stats:
            swa_val_loss = float('inf') if log_stats[loss_key] is None else log_stats[loss_key]
        else:
            # Try fallback to base model loss if SWA not active
            base_loss_key = 'swa_val_evaluate_fn_loss'
            swa_val_loss = log_stats.get(base_loss_key, float('inf'))
        return swa_val_loss, log_stats

    def fit_config(self, server_round: int):
        """Return training configuration for clients"""
        return {
            "round": server_round,
            "lr": self.lr,
        }

    def handle_swa_update(self, server_round):
        """Handle SWA model updates"""
        print("************************************  Server SWA Update  ************************************")
        
        if self.cfg.swa.swa:
            self.do_swa = self.cfg.swa.swa_start <= server_round
        else:
            self.do_swa = False
            
        if self.do_swa and self.swa_model is None:
            print("-" * 50, "\nCreating SWA model!\n", "-" * 50)
            self.swa_model = AveragedModel(self.model).to(self.device)

        if (self.do_swa and (self.round - self.cfg.swa.swa_start) % self.cfg.swa.swa_c == 0):
            self._update_swa_model()
        print(f"Current round: {server_round}")
        print(f"Do SWA: {self.do_swa}")
        print(f"CFG SWA Active: {self.cfg.swa.swa}")
        # print(f"Round: {self.swa_model}")
        print("*********************************************************************************************")

    def _update_swa_model(self):
        """Update SWA model parameters"""
        print("Number of cycles:", self.swa_n)

        if self.cfg.swa.swa_c > 1:
            self.lr = self.schedule_cycling_lr(
                self.round, self.cfg.swa.swa_c,
                self.cfg.training.lr, self.cfg.swa.swa_lr
            )
            self.cfg.training.lr = self.lr

        print("Number of models:", self.swa_n)
        alpha = 1.0 / (self.swa_n + 1.0)
        self.swa_model.to('cpu')
        self.model.to('cpu')

        for param1, param2 in zip(self.swa_model.parameters(), self.model.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha

        self.swa_n += 1.0

    @staticmethod
    def schedule_cycling_lr(round, c, lr1, lr2):
        """Calculate cycling learning rate"""
        t = 1 / c * (round % c + 1)
        return (1 - t) * lr1 + t * lr2

    def set_parameters(self, model, parameters):
        """Load parameters into model"""
        if not isinstance(parameters, list):
            parameters = list(parameters)

        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model

    def save_model(self, model, epoch, model_name):
        """Save model checkpoint"""
        to_save = {
            'model': model.state_dict(),
            'epoch': epoch,
            'args': self.cfg,
        }
        # Only create output directory if it doesn't exist
        output_dir = self.cfg.Server.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(to_save, f"{output_dir}/{model_name}.pth")

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, flwr.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, flwr.common.FitRes], BaseException]],
    ) -> Tuple[Optional[flwr.common.Parameters], Dict[str, flwr.common.Scalar]]:
        # Call aggregate_fit from FedAvg or FedMedian
        print(f"************************************  Server AGGREGATE FIT Round {server_round} ************************************")
        print(f"losses: {[res.metrics.keys() for _, res in results]}")
        
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            aggregated_ndarrays: List[np.ndarray] = flwr.common.parameters_to_ndarrays(aggregated_parameters)

            if aggregated_metrics is not None and 'val_loss' in aggregated_metrics:
                if aggregated_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = aggregated_metrics['val_loss']
                    log(INFO, f"Saving round {server_round} aggregated_ndarrays...")
                    self.save_model(self.model, server_round, 'best_model_server')
                    if self.cfg.swa.swa and self.swa_model is not None:
                        self.save_model(self.swa_model, server_round, 'best_swa_model_server')

            log(INFO, f"Saving round {server_round} aggregated_ndarrays...")
            self.save_model(self.model, server_round, f'best_model_round{server_round}_server')
            if self.cfg.swa.swa and self.swa_model is not None:
                self.save_model(self.swa_model, server_round, f'best_model_round{server_round}_server')

            self.model = self.set_parameters(self.model, aggregated_ndarrays)

        self.round += 1
        print(f"********************************************************************************************************************")
        return aggregated_parameters, aggregated_metrics


class FedAvgSWAServer(BaseSWAServer, FedAvg):
    """FedAvg with SWA implementation"""

    def __init__(self, cfg, valloader, logger):
        BaseSWAServer.__init__(self, cfg, valloader, logger)
        FedAvg.__init__(
            self,
            on_fit_config_fn=self.fit_config,
            min_available_clients=cfg.num_clients,
            min_fit_clients=cfg.num_clients,
            min_evaluate_clients=cfg.num_clients,
            fit_metrics_aggregation_fn=aggregate_metrics,
            evaluate_metrics_aggregation_fn=aggregate_metrics,
            evaluate_fn=self.get_evaluate_fn()
        )


class FedMedianSWAServer(BaseSWAServer, FedMedian):
    """FedMedian with SWA implementation"""

    def __init__(self, cfg, valloader, logger):
        BaseSWAServer.__init__(self, cfg, valloader, logger)
        FedMedian.__init__(
            self,
            on_fit_config_fn=self.fit_config,
            min_available_clients=cfg.num_clients,
            min_fit_clients=cfg.num_clients,
            min_evaluate_clients=cfg.num_clients,
            fit_metrics_aggregation_fn=aggregate_metrics,
            evaluate_metrics_aggregation_fn=aggregate_metrics,
            evaluate_fn=self.get_evaluate_fn()
        )

class QFedAvgSWAServer(BaseSWAServer, QFedAvg):
    """QFedAvg with SWA implementation"""

    def __init__(self, cfg, valloader, logger):
        BaseSWAServer.__init__(self, cfg, valloader, logger)
        QFedAvg.__init__(
            self,
            q_param=cfg.q,
            on_fit_config_fn=self.fit_config,
            min_available_clients=cfg.num_clients,
            min_fit_clients=cfg.num_clients,
            min_evaluate_clients=cfg.num_clients,
            fit_metrics_aggregation_fn=aggregate_metrics,
            evaluate_metrics_aggregation_fn=aggregate_metrics,
            evaluate_fn=self.get_evaluate_fn()
        )

class KRUMSWAServer(BaseSWAServer, Krum):
    """KRUM with SWA implementation"""

    def __init__(self, cfg, valloader, logger):
        BaseSWAServer.__init__(self, cfg, valloader, logger)
        Krum.__init__(
            self,
            on_fit_config_fn=self.fit_config,
            min_available_clients=cfg.num_clients,
            min_fit_clients=cfg.num_clients,
            min_evaluate_clients=cfg.num_clients,
            fit_metrics_aggregation_fn=aggregate_metrics,
            evaluate_metrics_aggregation_fn=aggregate_metrics,
            evaluate_fn=self.get_evaluate_fn(),
            num_clients_to_keep=int(cfg.num_clients * 2/3) # 9 clients -> 6 clients
        )

class FedAdamSWAServer(BaseSWAServer, FedAdam):
    """FedAdam with SWA implementation"""

    def __init__(self, cfg, valloader, logger):
        BaseSWAServer.__init__(self, cfg, valloader, logger)
        FedAdam.__init__(
            self,
            on_fit_config_fn=self.fit_config,
            min_available_clients=cfg.num_clients,
            min_fit_clients=cfg.num_clients,
            min_evaluate_clients=cfg.num_clients,
            fit_metrics_aggregation_fn=aggregate_metrics,
            evaluate_metrics_aggregation_fn=aggregate_metrics,
            evaluate_fn=self.get_evaluate_fn()
        )

class FedAvgMSWAServer(BaseSWAServer, FedAvgM):
    """FedAvgM with SWA implementation"""

    def __init__(self, cfg, valloader, logger):
        BaseSWAServer.__init__(self, cfg, valloader, logger)
        FedAvgM.__init__(
            self,
            on_fit_config_fn=self.fit_config,
            min_available_clients=cfg.num_clients,
            min_fit_clients=cfg.num_clients,
            min_evaluate_clients=cfg.num_clients,
            fit_metrics_aggregation_fn=aggregate_metrics,
            evaluate_metrics_aggregation_fn=aggregate_metrics,
            evaluate_fn=self.get_evaluate_fn(),
            server_learning_rate=0.5,
            server_momentum=0.9
        )