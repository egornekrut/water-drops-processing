""" Default config with base parameters for model training and inference"""
from pathlib import Path

from easydict import EasyDict

config = EasyDict()

#  ----- Runtime -----
config.device = 'cuda' # cuda если есть видеокарта
config.rank = 0
config.world_size = 1

#  ----- Dataset -----
config.dataset_root = Path('/home/nekrut/dataset/water_processing/seg_fs/')
config.segm_num_classes = 1
config.num_workers = 4

#  ----- Model -----
config.step_1_ckpt_path = Path('./weights/yolos-seg-beta.pt')

#  ----- Model 2 -----
config.model_type = 'unet'
config.encoder_name = 'efficientnet-b2'
config.step_2_ckpt_path = Path('./weights/epoch_2000.pt')

#  ----- Loss -----
config.ce_weight = None
config.ce_label_smoothing = 0

#  ----- Training -----
config.ckpt_path = None
config.warmup_steps = 10
config.epochs = 200
config.save_model_on = 50
config.log_step = 1

config.lr = 1e-2
config.decay = 1e-4
config.batch_size = 32
config.accumulate_batches = 1

config.clip_grad_value = 10

config.logs_dir = Path('/home/nekrut/tmp/fframe_train/')

#  ----- Test -----
config.test_root = None
config.step1_thres = 0.8
config.step2_thres = 0.99
