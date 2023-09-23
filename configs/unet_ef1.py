""" Default config with base parameters for model training and inference"""
from pathlib import Path

from easydict import EasyDict

config = EasyDict()


#  ----- Runtime -----
config.device = 'cuda'
config.rank = 0
config.world_size = 1

#  ----- Dataset -----
config.dataset_root = Path('/home/nekrut/dataset/water_processing/seg_fs/')
config.segm_num_classes = 1
config.num_workers = 2

#  ----- Model -----
config.model_type = 'unet'
config.encoder_name = 'efficientnet-b0'
config.ckpt_path = None

#  ----- Loss -----
config.ce_weight = None
config.ce_label_smoothing = 0

#  ----- Training -----
config.warmup_steps = 100
config.epochs = 5000
config.save_model_on = 1000
config.log_step = 100

config.lr = 1e-2
config.decay = 1e-4
config.batch_size = 2
config.accumulate_batches = 1

config.clip_grad_value = 100

config.logs_dir = Path('/home/nekrut/tmp/seg_train_wdp/unet_fs')

#  ----- Training -----
config.test_root = None
