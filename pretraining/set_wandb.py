import os
from datetime import datetime
from enum import Enum

import wandb

class DSName(Enum):
    DSAD = 0
    ESAD = 1
    GLENDA = 2
    HeiCo = 3
    hsdb_instrument = 4
    LapGyn4 = 5
    PSI_AVA = 6
    SurgicalActions160 = 7
    Cholec80 = 8
    Server = 10

def setup_wandb(cfg, clientID):
    wandb.login()
    date_str = cfg.group_name
    group_name = cfg.group_name

    # Initialize project
    wandb.init(
        project="EndoViT_FL_Pretraining",
        name=f'{DSName(clientID)}_{date_str}_{clientID}',
        group=group_name,
        tags=["pretraining_FL"],
        id=cfg.wandb_id,
        resume="allow",
        reinit=True
    )

    # Add config to wandb
    # wandb.config.update(cfg)




