import math
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.cuda.amp.grad_scaler import GradScaler
import numpy as np
from torch import inference_mode, save, load
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.segmentation.losses import ComboLoss
from torchvision.ops import sigmoid_focal_loss
from src.fframe.dataset import FFDataset
from src.fframe.model import FrameClassModel
from src.utils.config import get_config
from src.utils.logger import TensorBoardLog, logger


# def worker_init_fn(worker_id):
#     np.random.seed(np.random.get_state()[1][0] + worker_id)
def get_sampler(target):
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in (0, 1)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    return WeightedRandomSampler(samples_weight, len(samples_weight))


def train(config):
    config.logs_dir = Path(config.logs_dir) / datetime.now().strftime('%d_%m_%y-%H_%M_%S')
    tensorboard_dir = config.logs_dir / 'tensorboard_logs'
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = FFDataset()
    # test_dataset = BubDataset(config, inference_mode=True)

    # Initialize summary writer for tensorboard
    tb_logger = TensorBoardLog(tensorboard_dir.as_posix())
    
    sampler = get_sampler(np.array(tuple(train_dataset.has_contact.values())))

    trainloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True,
        sampler=sampler,
        num_workers=config.num_workers,
        # worker_init_fn=worker_init_fn,
        pin_memory=True,
    )

    # testloader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=config.num_workers,
    #     pin_memory=False,
    # )
    model = FrameClassModel(config.batch_size)
    model.to(config.device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.decay,
    )
    if config.ckpt_path:
        optimizer.load_state_dict(load(config.ckpt_path, map_location='cpu')['optimizer'])

    lambda_step = lambda step: 0.5 * (
        1 + math.cos(math.pi * (step - config.warmup_steps) / (config.epochs - config.warmup_steps))
    ) if step > config.warmup_steps else step / config.warmup_steps

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda_step,
    )

    # loss_fn = ComboLoss(alpha=0.5)
    scaler = GradScaler()

    running_loss = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_start = time.time()
        for step, (img, gt) in enumerate(trainloader):
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda'):
                model_output = model(img.to(config.device))

                loss = sigmoid_focal_loss(
                    model_output.view(-1),
                    gt.view(-1).to(config.device),
                    reduction='mean',
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # if step % config.log_step == 0 and step != 0:
            # log = f'epoch {epoch + 1}/{config.epochs} ' \
            #       f'step {step + 1}/{len(dataloader)} ' \
            #       f'loss {loss.detach().cpu().numpy()}'
            # logger.info(log)

            #     running_loss = []
            # else:
            running_loss.append(loss.detach().cpu().numpy())

        # global_step = (epoch - 1) * len(trainloader) + step
        # model.eval()

        # test_loss = []
        # for img, gt in testloader:
        #     with no_grad():
        #         model_output = model(img.to(config.device))
        #         loss = sigmoid_focal_loss(
        #             model_output,
        #             gt.to(config.device),
        #             reduction='mean',
        #         )
        #         test_loss.append(loss.cpu().numpy())

        for param_lr in optimizer.param_groups:
            learning_rate = param_lr['lr']
            break

        tb_logger.update(epoch, {'train_loss': sum(running_loss) / len(running_loss),'learning_rate': learning_rate})

        log = f'{epoch}, train_loss: {sum(running_loss) / len(running_loss):.4f}, epoch time {(time.time() - epoch_start):.3f} secs'
        logger.info(log)

        if epoch % config.save_model_on == 0:
            checkpoint_name = str(Path(config.logs_dir, f'epoch_{epoch}.pt'))
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save(checkpoint, checkpoint_name)

        scheduler.step()


if __name__ == "__main__":
    config = get_config()
    train(config)
