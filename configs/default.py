""" Default config with base parameters for model training and inference"""
from pathlib import Path

from easydict import EasyDict

config = EasyDict()


#  ----- Runtime -----
config.device = 'cpu'
config.rank = 0
config.world_size = 1

#  ----- Dataset -----
config.dataset_root = Path('/mnt/dataset/center/train')
config.segm_num_classes = 1
config.num_workers = 2

#  ----- Model -----
config.model_type = 'DL3+'
config.encoder_name = 'efficientnet-b2'
config.ckpt_path = None # Путь до .pt файла 

#  ----- Loss -----
config.ce_weight = None
config.ce_label_smoothing = 0

#  ----- Training -----
config.warmup_steps = 5
config.epochs = 100
config.save_model_on = 50
config.log_step = 5

config.lr = 1e-3
config.momentum = 1e-5
config.decay = 1e-6
config.batch_size = 33
config.accumulate_batches = 1

config.clip_grad_value = 30

config.logs_dir = Path('/mnt/tmp_output')
config.test_root = Path('/mnt/dataset/center/test')

#  ----- Inference -----
config.prob_thres = 0.5
