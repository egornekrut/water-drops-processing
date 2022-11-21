from datetime import datetime
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from model import setup_segmentation_model
from src.utils import get_scheduler, get_optimizer


class SegmentationTrainer:
    """
    Base class for segmentation training.
    """
    def __init__(self, config):
        self.config = config
        self.config.rank = 1

        self.config.device = torch.device(config.device)

        self.model = setup_segmentation_model(self.config, False)
        self.model.train()

        self.optimizer = get_optimizer(self.config, self.model)
        self.loss_fn = get_loss_fn(self.config).to(self.config.device)
        self.scheduler = get_scheduler(config, self.optimizer)

        # self.train_loader = get_dataloader(config, is_training=True)

    def train(self):
        """
        Start training process using parameters from config.
        """
        # Initializing in a separate cell, so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # create output folder and copy config
        if self.config.rank == 0:
            self.config.train_output = Path(self.config.train_output)
            self.config.train_output = self.config.train_output / timestamp
            self.config.train_output.mkdir(parents=True, exist_ok=True)

            save_cfg_path = self.config.train_output / 'config.py'
            save_cfg_path.write_bytes(self.config.config_path.read_bytes())

            tensorboard_path = (self.config.train_output / 'tensorboard').as_posix()
            self.writer = SummaryWriter(log_dir=tensorboard_path)
        else:
            self.writer = None

        for epoch in range(self.config.total_epochs):

            if self.config.rank == 0:
                print(f'EPOCH {epoch + 1}\n')

            self.train_loader.sampler.set_epoch(epoch)

            self._train_one_epoch_(epoch + 1)

            # save ckpt every save_model_on epoch
            if (epoch + 1) % self.config.save_model_on == 0 and self.config.rank == 0:
                ckpt_name = f'{self.config.model_type}_{epoch + 1}.pth'
                ckpt_path = self.config.train_output / ckpt_name
                torch.save(
                    {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'epoch': epoch + 1,
                    },
                    ckpt_path,
                )
            # if (epoch + 1) % self.config.verbose == 0 and self.config.verbose > 0 and self.val_dataloader:
            #     self.validate(epoch + 1)

    def _train_one_epoch_(self, epoch_number):
        running_loss = 0.

        self.model.train()

        for i, inputs in enumerate(self.train_loader):
            inputs = inputs.to(self.config.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            loss = self.loss_fn(outputs)
            loss.backward()

            clip_grad_norm_(self.model.parameters(), self.config.clip_grad_value)

            # Adjust learning weights
            self.optimizer.step()

            self.scheduler.step()

            # Gather data to report
            running_loss += loss.item()

        running_loss /= self.train_loader.__len__()

        # Log the running loss averaged per batch
        last_lr = self.scheduler.get_lr()[0]
        print(f'EPOCH {epoch_number}, Training loss: {running_loss:.3e},'
              f' LR: {last_lr}', flush=True)

        if self.writer:
            self.writer.add_scalars(
                'Training',
                {'loss': running_loss},
                epoch_number,
            )
            self.writer.add_scalars(
                'LR',
                {'lr': last_lr},
                epoch_number,
            )
            self.writer.flush()

    # def validate(self, epoch_number):
    #     self.model.eval()
    #
    #     val_running_abs_loss = 0.
    #     val_running_grad_loss = 0.
    #
    #     with torch.no_grad():
    #         for inputs_val, gt_depth_val, gt_mask_val, mean, std in self.val_dataloader:
    #             # Collapse all minibatches and send to the device
    #             inputs_val = inputs_val.view(-1, *inputs_val.shape[-3:]).to(self.device)
    #
    #             gt_depth_val = gt_depth_val.view(-1, *gt_depth_val.shape[-3:]).to(self.device).squeeze(dim=1)
    #             gt_mask_val = gt_mask_val.view(-1, *gt_mask_val.shape[-3:]).to(self.device).squeeze(dim=1)
    #
    #             mean = mean.view(-1, *mean.shape[-3:]).to(self.device)
    #             std = std.view(-1, *std.shape[-3:]).to(self.device)
    #
    #             outputs_val = self.model(inputs_val)
    #             outputs_val = outputs_val * std + mean
    #
    #             abs_loss, grad_loss = self.loss_fn(
    #                 outputs_val.squeeze(dim=1),
    #                 gt_depth_val,
    #                 gt_mask_val,
    #             )
    #             val_running_abs_loss += float(abs_loss)
    #             val_running_grad_loss += float(grad_loss)
    #
    #         val_running_abs_loss /= self.val_dataloader.__len__()
    #         val_running_grad_loss /= self.val_dataloader.__len__()
    #         val_running_loss = val_running_abs_loss + val_running_grad_loss
    #
    #         print(f'LOSS VAL: epoch {epoch_number}, rank {self.config.rank} = {val_running_loss:.3e}', flush=True)
    #
    #         if self.writer:
    #             self.writer.add_scalars(
    #                 'Validation',
    #                 {
    #                     'Abs': val_running_abs_loss,
    #                     'Grad': val_running_grad_loss,
    #                     'Total': val_running_loss,
    #                 },
    #                 epoch_number,
    #             )
    #             self.writer.flush()


def get_loss_fn(config) -> CrossEntropyLoss:
    """
    Get loss function for segmentation task.
    :param config: config
    :return: CE pytorch loss function.
    """
    if config.ce_weight:
        config.ce_weigth = torch.tensor(config.ce_weight)

    loss = CrossEntropyLoss(
        weight=config.ce_weight,
        label_smoothing=config.ce_label_smoothing,
    )
    return loss
