import argparse
from humanfriendly import format_timespan

from train.trainer_inv import TrainerInv
from utils.base_utils import load_cfg, safe_state
import os
import torch
import numpy as np
import time
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/shape/syn/compressor.yaml')
# parser.add_argument('--cfg', type=str, default='configs/mat/syn/compressor.yaml')
parser.add_argument('-e', '--extra_name', type=str, default=None)
flags, unknown = parser.parse_known_args()

train_time_st = time.time()
cfg = load_cfg(flags.cfg)
cfg = OmegaConf.create(cfg)
unknown_cfg = OmegaConf.from_dotlist(unknown)
cfg = OmegaConf.merge(cfg, unknown_cfg)
cfg = OmegaConf.to_container(cfg, resolve=True)

safe_state(0)

if flags.extra_name is not None:
    cfg['name'] = '_'.join([cfg['name'], flags.extra_name])
TrainerInv(cfg, config_path=flags.cfg).run()

print(f"Training done, costs {format_timespan(time.time() - train_time_st)}.")
