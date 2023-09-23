""" Default config with base parameters for model training and inference"""
from pathlib import Path

from easydict import EasyDict

config = EasyDict()


#  ----- Runtime -----
config.device = 'cuda'
config.rank = 0
config.world_size = 1

#  ----- Dataset -----
config.dataset_root = Path('/home/nekrut/dataset/water_processing/v1')
config.segm_num_classes = 1
config.num_workers = 2

#  ----- Model -----
config.model_type = 'DL3+'
config.encoder_name = 'efficientnet-b2'
config.ckpt_path = None #Path('./tmp_output/03_09_23-13_33_18/epoch_5000.pt')

#  ----- Loss -----
config.ce_weight = None
config.ce_label_smoothing = 0

#  ----- Training -----
config.warmup_steps = 100
config.epochs = 5000
config.save_model_on = 1000
config.log_step = 100

config.lr = 1e-1
# config.momentum = 1e-5
config.decay = 1e-4
config.batch_size = 10
config.accumulate_batches = 1

config.clip_grad_value = 30

config.logs_dir = Path('./tmp_output')

#  ----- Training -----
config.test_root = None
