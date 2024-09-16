from easydict import EasyDict

config = EasyDict()

config.workers = 4
config.exp_name = 'yolov8n-seg_aug_data_nano_sep14_data_lr3e-4'
config.data_yml = '/home/nekrut/dataset/wdp_yolo_det_dataset_sep14/wdp_yolo_det_dataset_sep14.yaml'
config.optimizer = 'AdamW'
config.epochs = 200
config.batch = 64
config.lr0 = 3e-4
config.weight_decay = 1e-6
config.warmup_epochs = 20
config.label_smoothing = 0.1
config.dropout = 0.1
config.cos_lr = False
config.pretrained = True
config.cls_weight = 2
config.box_weight = 10
config.dfl_weight = 5
config.model = 'yolov8n-seg.pt'
config.device = 'cuda:0'

config.yolo_configuration = {
    'weights': './weights/yolon-seg_sep14_data.pt',
    'inference_config': {
        'retina_masks': True,
        'half': True,
        'agnostic_nms': True,
        'conf': 0.45,
        'verbose': False,
        'tracker': 'botsort.yaml',
        'track_high_thresh': 0.45,
    },
}
config.contact_frame_model = {
    'weights': './weights/fframe_v2.pt',
    'thres': 0.6,
}