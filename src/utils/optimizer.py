from torch.optim import AdamW
from torch.optim import Optimizer
from torch.optim import RAdam
from torch.optim import SGD


def get_optimizer(config, model) -> Optimizer:
    if config.opt_type == 'SGD':
        optimizer = SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.opt_type == 'RADAM':
        optimizer = RAdam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.opt_type == 'ADAMW':
        optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError('Possible values for config.opt_type are [SGD, RADAM, ADAMW].')

    return optimizer
