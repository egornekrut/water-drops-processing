from ultralytics import YOLO
from ultralytics.models.yolo.segment.train import SegmentationTrainer
import pandas as pd
from src.utils.config import get_config_from_path

loss_stat = {}

def on_train_batch_end(trainer: SegmentationTrainer):
    names = ('box_loss', ' seg_loss', 'cls_loss', 'dfl_loss', 'total')
    items = trainer.loss_items.cpu().tolist()
    items.append(trainer.loss.item())
    breakpoint()
    loss_stat[0] = {key: value for key, value in zip(names, items)}

def on_train_epoch_end(trainer: SegmentationTrainer):
    pd.DataFrame.from_dict(loss_stat, orient='index').to_parquet('./hnm.parquet')

def train_yolo(config):
    model = YOLO(config.model)
    if config.batch == 1:
        model.add_callback('on_train_batch_end', on_train_batch_end)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

    model.train(
        project='WDP',
        name=config.exp_name,
        data=config.data_yml,
        optimizer=config.optimizer,
        epochs=config.epochs,
        imgsz=640,
        batch=config.batch,
        patience=100,
        save_period=-1,
        deterministic=False,
        single_cls=False,
        cos_lr=config.cos_lr,
        lr0=config.lr0,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        amp=True,
        half=True,
        workers=config.workers,
        agnostic_nms=True,
        mask_ratio=1,
        overlap_mask=False,
        cache='ram',
        pretrained=config.pretrained,
        verbose=True,
        label_smoothing=config.label_smoothing,
        plots=True,
        dropout=config.dropout,
        device='cuda:0',
        resume=False,
        box=config.box_weight,
        cls=config.cls_weight,
        dfl=config.dfl_weight,
        retina_masks=True,
        scale=0,
        mosaic=0,
        translate=0,
        degrees=90,
        save=True,
        profile=True,
        auto_augment=False,
        hsv_s=0,
        hsv_h=0,
        erasing=0.5,
        crop_fraction=0.5,
    )

if __name__ == '__main__':
    config = get_config_from_path('./configs/detection_yolov8s-seg.py')
    train_yolo(config)
