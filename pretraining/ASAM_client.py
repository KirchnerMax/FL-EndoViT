import datetime
import os
import subprocess
import time
from pathlib import Path

import wandb
from omegaconf import OmegaConf

import set_wandb
from MAE.ASAM_engine_pretrain import train_and_validate_one_epoch
from MAE.util import misc
from SAM.minimizers import ASAM
from client import EndoViT_client
import torch

class ASAMClient(EndoViT_client):
    def __init__(self, trainloader, validationloader, cid, cfg, csv_logger):
        super().__init__(trainloader, validationloader, cid, cfg, csv_logger)
        self.rho = cfg.asam.rho
        self.eta = cfg.asam.eta

    def fit(self, parameters, cfg):
        tr_config = OmegaConf.load(f'/conf/config_{self.client_name.upper()}.yaml')
        epoch = tr_config.training_rounds
        print(f"Start Training + Validation: Client {self.cid} - {self.client_name.upper()} - Round: {epoch}")
        self.set_parameters(parameters)


        #### TRAINING ####
        start_time = time.time()
        log_stats, val_stats = train_and_validate_one_epoch(
            self.model,
            self.model,
            self.trainloader,
            self.validationloader,
            self.optimizer,
            self.device,
            epoch,
            self.loss_scaler,
            self.cfg,
            None, None, -1,
            log_writer=self.log_writer,
            best_result_dict=self.best_result,
            best_result_dict_swa={},
            mixup_fn=None,
            cid=self.cid,
            c_name=self.client_name,
            tr_cfg=self.tr_cfg,
            rho=self.rho,
            eta=self.eta,
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
        if self.cfg.log_dir and misc.is_main_process():
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
        OmegaConf.save(tr_config, f'{cfg["config_path"]}/config_{self.client_name.upper()}.yaml')

        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_result["loss"],
        }
        torch.save(state, f'./outputs/latest_{self.client_name}.pth')

        # Moved this from before the train_and_validate_one_epoch to after. Now epoch % 5 == 0 instead of previously epoch % 5 == 1.
        # By resetting the loss every 5 epochs the best model will be saved in each of the intervals [epoch 0, epoch 1>, [epoch 1, epoch 5], [epoch 6, epoch 10] and so on
        if epoch % 5 == 0 or epoch + 1 == self.cfg.training.num_rounds:
            src = self.cfg.log_dir + '/' + f'latest_{self.client_name}.pth'
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
            torch.save(self.optimizer.state_dict(), f'{cfg["config_path"]}/optimizer_cache/optimizer_{self.client_name}.pth')

        return self.get_parameters(self.cfg), len(self.trainloader) * self.cfg.dataloader.batch_size, train_log_stats


def generate_client_fn(trainloaders, validationloaders, csv_logger, config):
    """Return a new instance of the client."""

    def client_fn(cid: str):
        client = ASAMClient(
            trainloader=trainloaders[int(cid)],
            validationloader=validationloaders[int(cid)],
            cid=int(cid),
            cfg=config,
            csv_logger=csv_logger,
        )
        return client.to_client()

    return client_fn