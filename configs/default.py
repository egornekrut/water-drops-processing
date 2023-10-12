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
config.num_workers = 1

#  ----- Model -----
config.step_1_ckpt_path = Path('./weights/yolon_step1.pt')

#  ----- Model 2 -----
config.model_type = 'unet'
config.encoder_name = 'efficientnet-b4'
config.step_2_ckpt_path = Path('/home/nekrut/tmp/seg_train_wdp/unet_fs/08_10_23-22_02_12/epoch_1000.pt') # Path('/home/nekrut/tmp/seg_train_wdp/unet_fs_ef1/09_10_23-23_20_44/epoch_2000.pt') #

#  ----- Loss -----
config.ce_weight = None
config.ce_label_smoothing = 0

#  ----- Training -----
config.warmup_steps = 100
config.epochs = 2000
config.save_model_on = 100
config.log_step = 10

config.lr = 1e-3
config.decay = 1e-6
config.batch_size = 4
config.accumulate_batches = 1

config.clip_grad_value = 10

config.logs_dir = Path('/home/nekrut/tmp/seg_train_wdp/unet_fs_ef1')

#  ----- Training -----
config.test_root = None

config.prob_thres = 0.5
