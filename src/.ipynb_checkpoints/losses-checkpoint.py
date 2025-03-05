import torch.nn.functional as F
import torch
from src.utils.distributed import AllReduce

def I_JEPA(z, h):
    loss = F.smooth_l1_loss(z, h)
    loss = AllReduce.apply(loss)
    return loss


def get_loss_function(loss_name):
    loss_functions = {
        "I-JEPA": loss_fn,
        #"LIDAR": get_lidar,
    }
    if loss_name not in loss_functions:
        raise ValueError(f"Invalid loss function '{loss_name}'. Only {list(loss_functions.keys())} are defined.")
    return loss_functions[loss_name]