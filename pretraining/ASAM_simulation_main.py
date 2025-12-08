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
from flwr.server import ServerApp, ServerConfig
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from ASAM_client import generate_client_fn
from dataset import prepare_datasets
from log2csv import Log2CSV
from server import FedAvgSWAServer

@hydra.main(config_path="./conf", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # 0 - Parse Config
    print(OmegaConf.to_yaml(cfg))
    group_name = datetime.now().strftime("%d%m%H%M")
    for c in os.listdir('/conf/'):
        if c == 'base.yaml' or c == 'optimizer_cache':
            continue
        client_cfg = OmegaConf.load(f'/conf/{c}')
        client_cfg.training_rounds = 0
        client_cfg.wandb_id = str(uuid.uuid4())
        client_cfg.group_name = group_name
        OmegaConf.save(client_cfg, f'/conf/{c}')

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

    # 2 Define clients and server
    server_app = ServerApp(
        config=ServerConfig(num_rounds=cfg.training.num_rounds),
        strategy=FedAvgSWAServer(cfg,
                                 validationloaders[0],
                                 csv_logger,
                                 ),
    )

    client_app = flwr.client.ClientApp(
        client_fn=generate_client_fn(trainloaders,
                                     validationloaders,
                                     csv_logger,
                                     cfg)
    )
    logger = logging.getLogger("flwr")
    logger.setLevel(logging.INFO)
    history = flwr.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config={'client_resources': {"num_cpus": 8, "num_gpus": cfg.gpu_partition}},
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
