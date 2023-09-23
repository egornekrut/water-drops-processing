import time
from pathlib import Path
from torch import no_grad
import numpy as np
from torch import save
from torch.optim import SGD, RAdam, AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.ops.focal_loss import sigmoid_focal_loss

from src.segmentation.dataset import BubDataset
from src.segmentation.model import setup_segmentation_model
from src.utils.logger import TensorBoardLog
from src.utils.logger import logger
from datetime import datetime
from src.utils.config import get_config


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def train(config):
    config.logs_dir = Path(config.logs_dir) / datetime.now().strftime('%d_%m_%y-%H_%M_%S')
    tensorboard_dir = config.logs_dir / 'tensorboard_logs'
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = BubDataset(config)
    # test_dataset = BubDataset(config, inference_mode=True)

    # Initialize summary writer for tensorboard
    tb_logger = TensorBoardLog(tensorboard_dir.as_posix())

    trainloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
        worker_init_fn=worker_init_fn,
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
    model = setup_segmentation_model(config)

    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.decay,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[int(0.5 * config.epochs), int(0.75 * config.epochs)],
        gamma=0.1,
    )

    running_loss = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_start = time.time()
        for step, (img, gt) in enumerate(trainloader):
            # Warmup for the first steps
            if epoch == 1 and step <= config.warmup_steps:
                learning_rate = config.lr * (step / config.warmup_steps) ** 4
                for param_lr in optimizer.param_groups:
                    param_lr['lr'] = learning_rate
            else:
                learning_rate = config.lr
                for param_lr in optimizer.param_groups:
                    param_lr['lr'] = learning_rate

            optimizer.zero_grad()

            model_output = model(img.to(config.device))
            loss = sigmoid_focal_loss(
                model_output,
                gt.to(config.device),
                reduction='mean',
            )
            loss.backward()
            optimizer.step()

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
