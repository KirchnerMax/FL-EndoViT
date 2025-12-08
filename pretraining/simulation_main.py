import argparse
import logging
import os
import pickle
import random
import uuid
from datetime import datetime

import flwr
import hydra
import numpy as np
import torch
from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from client import generate_client_fn
from dataset import prepare_datasets
from log2csv import Log2CSV
from server import FedAvgSWAServer, FedMedianSWAServer, QFedAvgSWAServer, KRUMSWAServer


def get_strategy(strategy_name: str, cfg, valloader, logger):
    """Select server strategy based on configuration"""
    strategies = {
        "FedAvg": FedAvgSWAServer,
        "FedMedian": FedMedianSWAServer,
        "FedProx": FedAvgSWAServer, # FedProxy only changes the client_fn, so we can use FedAvgSWAServer
        "QFedAvg": QFedAvgSWAServer,
        "KRUM": KRUMSWAServer,  # KRUM is not implemented in the current codebase, using FedAvgSWAServer as a placeholder
        "SCAFFOLD": FedAvgSWAServer,  # SCAFFOLD is not implemented in the current codebase, using FedAvgSWAServer as a placeholder
        "FedAdam": FedAvgSWAServer,  # FedAdam is not implemented in the current codebase, using FedAvgSWAServer as a placeholder
        "FedAvgM": FedAvgSWAServer,  # FedAvgM is not implemented in the current codebase, using FedAvgSWAServer as a placeholder
    }

    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available strategies: {list(strategies.keys())}")

    return strategies[strategy_name](cfg, valloader, logger)


@hydra.main(config_path="./conf", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # 0 - Parse Config
    print(OmegaConf.to_yaml(cfg))
    group_name = datetime.now().strftime("%d%m%H%M")
    for c in os.listdir(f'{cfg["config_path"]}'):
        if c == 'base.yaml' or c == 'optimizer_cache':
            continue
        client_cfg = OmegaConf.load(f'{cfg["config_path"]}{c}')
        client_cfg.training_rounds = 0
        client_cfg.wandb_id = str(uuid.uuid4())
        client_cfg.group_name = group_name
        OmegaConf.save(client_cfg, f'{cfg["config_path"]}{c}')

    # Set the random seed if provided (affects client sampling and batching)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # Set logging
    csv_logger = Log2CSV(csv_file=cfg.log_dir + '/log_metrics.csv')


    # 1 - Prepare Datasets
    trainloaders, validationloaders = prepare_datasets(cfg)
    print(trainloaders, validationloaders)
    # Log the number of clients and their partitions
    print(f"Number of clients (len of trainloaders): {len(trainloaders)}")

    # 2 Define clients and server
    strategy_name = cfg.Aggregator
    print(f"stragety_name = {strategy_name}")
    # server_app = ServerApp(
    #     config=ServerConfig(num_rounds=cfg.training.num_rounds),
    #     strategy=get_strategy(strategy_name, cfg, validationloaders[0], csv_logger),
    # )
    #
    # client_app = flwr.client.ClientApp(
    #     client_fn=generate_client_fn(trainloaders, validationloaders, csv_logger, cfg)
    # )

    def server_fn(context: Context) -> ServerAppComponents:
        server_config = ServerConfig(num_rounds=cfg.training.num_rounds)
        strategy = get_strategy(strategy_name, cfg, validationloaders[0], csv_logger)
        return ServerAppComponents(config=server_config, strategy=strategy)

    def client_fn(context: Context):
        print(f"Context: {context}")
        logger.log(msg=f"context: {context}", level=logging.INFO)
        logger.log(msg=f"nocde_config: {context.node_config['partition-id']}", level=logging.INFO)
        cid = context.node_config["partition-id"]
        return generate_client_fn(
            trainloaders,
            validationloaders,
            csv_logger,
            cfg
        )(int(cid))

    logger = logging.getLogger("flwr")
    logger.setLevel(logging.INFO)

    server_app = ServerApp(server_fn=server_fn)
    client_app = ClientApp(client_fn=client_fn)


    history = flwr.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config={'client_resources': {"num_cpus": 16, "num_gpus": cfg.gpu_partition},
                        'ttl': 43200}, # 12 hours
        verbose_logging=False,
    )

    # 5. Save results
    print(history)
    save_path = HydraConfig.get().runtime.output_dir
    save_path += "/history.pkl"
    results = {"history": history}
    with open(str(save_path), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
