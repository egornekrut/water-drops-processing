""" Default config with base parameters for model training and inference"""
from pathlib import Path

from easydict import EasyDict

config = EasyDict()


#  ----- Runtime -----
config.device = 'cuda'
config.rank = 0
config.world_size = 1

#  ----- Dataset -----
config.dataset_root = Path('/mnt/dataset/center/train')
config.segm_num_classes = 1
config.num_workers = 2

#  ----- Model -----
config.model_type = 'DL3+'
config.encoder_name = 'efficientnet-b2'
config.ckpt_path = Path('/home/nekrut/python_projects/water-drops-processing/tmp_output/21_07_23-12_46_16/epoch_40.pt')

#  ----- Loss -----
config.ce_weight = None
config.ce_label_smoothing = 0

#  ----- Training -----
config.warmup_epochs = 20
config.epochs = 200
config.save_model_on = 25
config.log_step = 1

config.lr = 1e-3
# config.momentum = 1e-5
config.decay = 1e-4
config.batch_size = 11
config.accumulate_batches = 1

config.clip_grad_value = 0.1

config.logs_dir = Path('/mnt/tmp_output')

#  ----- Training -----
config.test_root = Path('/mnt/dataset/center/test')
