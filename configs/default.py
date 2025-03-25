""" Default config with base parameters for model training and inference"""
from pathlib import Path

from easydict import EasyDict

config = EasyDict()


#  ----- Runtime -----
config.device = 'cuda'
config.rank = 0
config.world_size = 1

#  ----- Dataset -----
config.dataset_path = Path('')
config.segm_num_classes = 1

#  ----- Model -----
config.model_type = 'DeeplabV3'
config.encoder_name = 'efficientnet-b5'
config.ckpt_path = Path('')

#  ----- Loss -----
config.ce_weight = None
config.ce_label_smoothing = 0

#  ----- Training -----
config.lr = 1e-3
config.momentum = 9e-1
config.weight_decay = 1e-5

config.clip_grad_value = 17

config.save_model_on = 1
