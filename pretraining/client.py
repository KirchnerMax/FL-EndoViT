import datetime
import math
import os
import subprocess
import sys
from collections import OrderedDict
import time
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional, Dict
import wandb

import flwr
import torch
import timm.optim.optim_factory as optim_factory
import set_wandb
from flwr.common import FitIns, FitRes, NDArrays, Scalar
from omegaconf import OmegaConf
from timm.data import Mixup
from torch.utils.tensorboard import SummaryWriter

from MAE import mae_model
# LLRD - Layer wise learning rate decay
import MAE.util.lr_decay as lrd
from MAE.engine_pretrain import train_and_validate_one_epoch, evaluate, save_model_based_on_validation_results
from MAE.util import misc, lr_sched
from MAE.util.misc import NativeScalerWithGradNormCount as NativeScaler

from torch.utils.data import Subset, DataLoader


class DS_Name(Enum):
    DSAD = 0
    ESAD = 1
    GLENDA = 2
    HeiCo = 3
    hsdb_instrument = 4
    LapGyn4 = 5
    PSI_AVA = 6
    SurgicalActions160 = 7
    Cholec80 = 8


class EndoViT_client(flwr.client.NumPyClient):
    def __init__(self, trainloader, validationloader, cid, cfg, csv_logger):
        # --------------- general ---------------------------------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.cid = cid
        self.client_name = DS_Name(cid).name
        self.cfg = cfg
        self.csv_logger = csv_logger

        # --------------- model ----------------------------------------------------------------------------------------
        self.model = mae_model.__dict__[cfg.model_specific.model](norm_pix_loss=cfg.model_specific.norm_pix_loss,
                                                                  loss_type=cfg.model_specific.loss,
                                                                  hpf_loss=cfg.model_specific.high_pass_filter_loss)
        self.model.to(self.device)
        # print("Model = %s" % str(self.model))

        self.n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print('number of params (M): %.2f' % (self.n_parameters / 1.e6))

        # --------------- training settings ----------------------------------------------------------------------------
        self.lr = cfg.training.lr

        # following timm: set wd as 0 for bias and norm layers
        if (cfg.model_specific.layer_decay == 1.):
            param_groups = optim_factory.add_weight_decay(self.model, cfg.model_specific.weight_decay)
        else:
            # build optimizer with layer-wise lr decay (lrd)
            param_groups = lrd.param_groups_lrd(self.model, cfg.model_specific.weight_decay,
                                                no_weight_decay_list={'pos_embed', 'cls_token'} | {"decoder_pos_embed",
                                                                                                   "mask_token"},
                                                layer_decay=cfg.model_specific.layer_decay)

        if cfg.Aggregator == "FedProx" or cfg.Aggregator == "FedMedian":
            self.optimizer = torch.optim.SGD(param_groups, lr=cfg.training.lr, momentum=0.9,)
        else:
            self.optimizer = torch.optim.AdamW(param_groups, lr=cfg.training.lr, betas=(0.9, 0.95))
        if os.path.isfile(f'{cfg["config_path"]}/optimizer_cache/optimizer_{self.client_name}.pth'):
            self.optimizer.load_state_dict(torch.load(f'{cfg["config_path"]}/optimizer_cache/optimizer_{self.client_name}.pth'))
        self.loss_scaler = NativeScaler()

        misc.load_model(args=cfg.model_specific, model_without_ddp=self.model, optimizer=self.optimizer,
                        loss_scaler=self.loss_scaler)

        # --------------- tracking settings ----------------------------------------------------------------------------
        os.makedirs(cfg.log_dir, exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=cfg.log_dir)

        self.best_loss = float("inf")
        self.best_epoch = -1

        self.best_result = {
            "loss": self.best_loss,
            "epoch": self.best_epoch,
        }

        self.tr_cfg = OmegaConf.load(f'{cfg["config_path"]}/config_{self.client_name.upper()}.yaml')


    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Loads model and replaces it parameters with the ones given."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, cfg):
        tr_config = OmegaConf.load(f'{self.cfg["config_path"]}/config_{self.client_name.upper()}.yaml')
        epoch = tr_config.training_rounds
        print(f"************************************  Client FIT {self.cid} - {self.client_name.upper()} - Round: {epoch}  ************************************")
        print(f"Start Training + Validation: Client {self.cid} - {self.client_name.upper()} - Round: {epoch}")
        self.set_parameters(parameters)
        # training
        start_time = time.time()
        log_stats, val_stats = train_and_validate_one_epoch(
            self.model, self.model, self.trainloader, self.validationloader,
            self.optimizer, self.device, epoch, self.loss_scaler,
            self.cfg,
            None, None, -1,
            log_writer=self.log_writer,
            best_result_dict=self.best_result,
            best_result_dict_swa={},
            mixup_fn=None,
            cid = self.cid,
            c_name=self.client_name,
            tr_cfg=self.tr_cfg,
            csv_logger=self.csv_logger,
        )
        train_log_stats = {**{f'train_{k}': v for k, v in log_stats.items()},
                           **{f'val_{k}': v for k, v in val_stats.items()},
                           # CHANGED 16.02.2023. f'test_{k}' -> f'val_{k}'
                           'epoch': epoch,
                           'n_parameters': self.n_parameters}

        self.csv_logger.log_metrics(train_log_stats, epoch, center=self.client_name, cid=self.cid)

        if self.cfg.use_wandb:
            set_wandb.setup_wandb(self.tr_cfg, self.cid)
            wandb.log(train_log_stats)
            wandb.finish()

        if self.cfg.log_dir:
            if self.log_writer is not None:
                self.log_writer.flush()
            with open(os.path.join(self.cfg.log_dir, f"log_{self.client_name}.txt"), mode="a", encoding="utf-8") as f:
                f.write("{")
                for i, (k, v) in enumerate(train_log_stats.items()):
                    if ("patches_below" not in k):
                        if ("_lr" in k):
                            f.write(f"\"{k}\": {v:.2e}")
                        elif ("_loss" in k):
                            f.write(f"\"{k}\": {v:.4f}")
                        elif (k == "epoch"):
                            f.write(f"\"{k}\": {v:2d}")
                        elif (k == "n_parameters"):
                            f.write(f"\"{k}\": {v / 1e+6:3.2f} (M)")
                        else:
                            f.write(f"\"{k}\": {v:.4f}")
                        f.write(", " if i + 1 != len(train_log_stats.items()) else "}\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training - Round time {}'.format(total_time_str))

        # save the model and stats
        tr_config.training_rounds += 1
        OmegaConf.save(tr_config, f'{self.cfg["config_path"]}config_{self.client_name.upper()}.yaml')


        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_result["loss"],
        }
        # torch.save(state, f'./outputs/latest_{self.client_name}.pth')
        torch.save(state, f'{self.cfg.log_dir}/latest_{self.client_name}.pth')

        # Moved this from before the train_and_validate_one_epoch to after. Now epoch % 5 == 0 instead of previously epoch % 5 == 1.
        # By resetting the loss every 5 epochs the best model will be saved in each of the intervals [epoch 0, epoch 1>, [epoch 1, epoch 5], [epoch 6, epoch 10] and so on
        if epoch % 5 == 0 or epoch + 1 == self.cfg.training.num_rounds:
            src = f'{self.cfg.log_dir}/latest_{self.client_name}.pth'
            out_name = f'best_ckpt_{self.best_result["epoch"]}_loss_{self.best_result["loss"]:.4f}_{self.client_name}.pth'
            dest = self.cfg.log_dir + '/' + out_name

            subprocess.call(['mv', src, dest])

            self.best_result["loss"] = float("inf")
            self.best_result["epoch"] = -1

            # save final model
            if (epoch + 1 == self.cfg.training.num_rounds):
                if (self.cfg.log_dir):
                    # NOTE: If swa is on this will save the swa model regardless if the normal model was better.
                    src = dest
                    dest = f'{self.cfg.log_dir}/final_{self.client_name}.pth'

                    # make the outputs dir if it doesn't exist
                    Path(self.cfg.log_dir).parent.mkdir(parents=True, exist_ok=True)
                    subprocess.call(['cp', src, dest])

        if self.optimizer:
            torch.save(self.optimizer.state_dict(), f'{self.cfg["config_path"]}/optimizer_cache/optimizer_{self.client_name}.pth')

        print(f"{50 * '-'}")

        return self.get_parameters(self.cfg), len(self.trainloader) * self.cfg.dataloader.batch_size, train_log_stats

    def evaluate(self, parameters, cfg):
        """
        :param parameters:
        :param cfg:
        :return:
        """
        self.set_parameters(parameters)
        tr_config = OmegaConf.load(f'{self.cfg["config_path"]}/config_{self.client_name.upper()}.yaml')
        epoch = tr_config.training_rounds - 1

        epoch_end = True
        val_stats = evaluate(self.validationloader, self.model, self.device, self.log_writer, epoch,
                             self.cfg,
                             epoch_end=epoch_end,
                             is_swa=False,
                             cid=self.cid, c_name=self.client_name, tr_cfg=self.tr_cfg)

        log_stats = {**{f'val_evaluate_fn_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_parameters': self.n_parameters}
        print(f"[EVAL] Log_stats: {log_stats}")

        self.csv_logger.log_metrics(log_stats, epoch, center=self.client_name, cid=self.cid)

        if self.cfg.use_wandb:
            set_wandb.setup_wandb(self.tr_cfg, self.cid)
            wandb.log(log_stats)
            wandb.finish()
        if log_stats['val_evaluate_fn_loss'] is None:
            return_loss = {'val_loss': float('inf')}
        else:
            return_loss = {'val_loss': log_stats['val_evaluate_fn_loss']}

        return return_loss['val_loss'], len(self.validationloader) * self.cfg.dataloader.batch_size, log_stats


class FedProxClient(EndoViT_client):
    """
    A client class that extends EndoViT_client to implement FedProx functionality.
    This class can be used to implement FedProx-specific logic if needed.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional FedProx-specific initialization can go here if needed.
        self.mu = self.cfg.fed_prox.mu

    def fit(self, parameters, cfg):
        tr_config = OmegaConf.load(f'{self.cfg["config_path"]}config_{self.client_name.upper()}.yaml')
        epoch = tr_config.training_rounds

        print(f"Start Training + Validation: Client {self.cid} - {self.client_name.upper()} - Round: {epoch}")
        self.set_parameters(parameters)

        # training
        start_time = time.time()
        log_stats, val_stats = train_and_validate_one_epoch(
            self.model, self.model, self.trainloader, self.validationloader,
            self.optimizer, self.device, epoch, self.loss_scaler,
            self.cfg,
            None, None, -1,
            log_writer=self.log_writer,
            best_result_dict=self.best_result,
            best_result_dict_swa={},
            mixup_fn=None,
            cid=self.cid,
            c_name=self.client_name,
            tr_cfg=self.tr_cfg,
            csv_logger=self.csv_logger,
            fedprox=True
        )
        train_log_stats = {**{f'train_{k}': v for k, v in log_stats.items()},
                           **{f'val_{k}': v for k, v in val_stats.items()},
                           # CHANGED 16.02.2023. f'test_{k}' -> f'val_{k}'
                           'epoch': epoch,
                           'n_parameters': self.n_parameters}

        self.csv_logger.log_metrics(train_log_stats, epoch, center=self.client_name, cid=self.cid)

        if self.cfg.use_wandb:
            set_wandb.setup_wandb(self.tr_cfg, self.cid)
            wandb.log(train_log_stats)
            wandb.finish()

        if self.cfg.log_dir:
            if self.log_writer is not None:
                self.log_writer.flush()
            with open(os.path.join(self.cfg.log_dir, f"log_{self.client_name}.txt"), mode="a", encoding="utf-8") as f:
                f.write("{")
                for i, (k, v) in enumerate(train_log_stats.items()):
                    if ("patches_below" not in k):
                        if ("_lr" in k):
                            f.write(f"\"{k}\": {v:.2e}")
                        elif ("_loss" in k):
                            f.write(f"\"{k}\": {v:.4f}")
                        elif (k == "epoch"):
                            f.write(f"\"{k}\": {v:2d}")
                        elif (k == "n_parameters"):
                            f.write(f"\"{k}\": {v / 1e+6:3.2f} (M)")
                        else:
                            f.write(f"\"{k}\": {v:.4f}")
                        f.write(", " if i + 1 != len(train_log_stats.items()) else "}\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training - Round time {}'.format(total_time_str))

        # save the model and stats
        tr_config.training_rounds += 1
        OmegaConf.save(tr_config, f'{self.cfg["config_path"]}config_{self.client_name.upper()}.yaml')

        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_result["loss"],
        }
        # torch.save(state, f'./outputs/latest_{self.client_name}.pth')
        torch.save(state, f'{self.cfg.log_dir}/latest_{self.client_name}.pth')

        # Moved this from before the train_and_validate_one_epoch to after. Now epoch % 5 == 0 instead of previously epoch % 5 == 1.
        # By resetting the loss every 5 epochs the best model will be saved in each of the intervals [epoch 0, epoch 1>, [epoch 1, epoch 5], [epoch 6, epoch 10] and so on
        if epoch % 5 == 0 or epoch + 1 == self.cfg.training.num_rounds:
            src = f'{self.cfg.log_dir}/latest_{self.client_name}.pth'
            out_name = f'best_ckpt_{self.best_result["epoch"]}_loss_{self.best_result["loss"]:.4f}_{self.client_name}.pth'
            dest = self.cfg.log_dir + '/' + out_name

            subprocess.call(['mv', src, dest])

            self.best_result["loss"] = float("inf")
            self.best_result["epoch"] = -1

            # save final model
            if (epoch + 1 == self.cfg.training.num_rounds):
                if (self.cfg.log_dir):
                    # NOTE: If swa is on this will save the swa model regardless if the normal model was better.
                    src = dest
                    dest = f'{self.cfg.log_dir}/final_{self.client_name}.pth'

                    # make the outputs dir if it doesn't exist
                    Path(self.cfg.log_dir).parent.mkdir(parents=True, exist_ok=True)
                    subprocess.call(['cp', src, dest])

        if self.optimizer:
            torch.save(self.optimizer.state_dict(), f'{self.cfg["config_path"]}/optimizer_cache/optimizer_{self.client_name}.pth')

        if self.get_parameters(self.cfg) == None:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Parameters of model is none !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        return self.get_parameters(self.cfg), len(self.trainloader) * self.cfg.dataloader.batch_size, train_log_stats


def generate_client_fn(trainloaders, validationloaders, csv_logger, config):
    """Return a new instance of the client."""

    def client_fn(cid: str):
        if config.Aggregator == "FedProx":
            print(f"Creating FedProxClient for client {cid}")
            client = FedProxClient(
                trainloader=trainloaders[int(cid)],
                validationloader=validationloaders[int(cid)],
                cid=int(cid),
                cfg=config,
                csv_logger=csv_logger,
            )
        else:
            client = EndoViT_client(
                trainloader=trainloaders[int(cid)],
                validationloader=validationloaders[int(cid)],
                cid=int(cid),
                cfg=config,
                csv_logger=csv_logger,
                )
        return client.to_client()

    return client_fn
