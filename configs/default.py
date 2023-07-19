""" Default config with base parameters for model training and inference"""
from pathlib import Path

from easydict import EasyDict

config = EasyDict()


#  ----- Runtime -----
config.device = 'cuda'
config.rank = 0
config.world_size = 1

#  ----- Dataset -----
config.dataset_root = Path('/home/nekrut/datasets/water_processing/v1')
config.segm_num_classes = 1
config.num_workers = 2

#  ----- Model -----
config.model_type = 'DL3+'
config.encoder_name = 'efficientnet-b2'
config.ckpt_path = None #Path('/home/nekrut/tmp/train_bubbles/18_04_23-23_34_49/epoch_500.pt')

#  ----- Loss -----
config.ce_weight = None
config.ce_label_smoothing = 0

#  ----- Training -----
config.warmup_steps = 10
config.epochs = 500
config.save_model_on = 100
config.log_step = 1

config.lr = 1e-1
config.momentum = 9e-1
config.decay = 1e-6
config.batch_size = 10
config.accumulate_batches = 1

config.clip_grad_value = 17

config.logs_dir = Path('/home/nekrut/tmp/train_bubbles')
