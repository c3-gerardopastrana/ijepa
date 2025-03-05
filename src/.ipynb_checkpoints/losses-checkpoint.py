import torch.nn.functional as F
import torch
from src.utils.distributed import AllReduce
from reptrix import alpha, rankme, lidar


class Lidar:
    def __init__(self, batch_size, num_target, num_context):
        self.batch_size = batch_size
        self.num_target = num_target
        self.num_context = num_context

    def lidar_student(self, z, h):
        return lidar.get_lidar(
            z, self.batch_size, self.num_context, del_sigma_augs=0.001
        )

    def lidar_teacher(self, z, h):
        return lidar.get_lidar(
            h, self.batch_size, self.num_target, del_sigma_augs=0.001
        )


def jepa(z, h):
    loss = F.smooth_l1_loss(z, h)
    loss = AllReduce.apply(loss)
    return loss


def get_loss_function(loss_name, *args, **kwargs):
    loss_functions = {name: obj for name, obj in globals().items() if callable(obj)}
    
    if loss_name in loss_functions:
        return loss_functions[loss_name](*args, **kwargs) if args or kwargs else loss_functions[loss_name]
    
    raise ValueError(f"Invalid loss function '{loss_name}'. Available options: {list(loss_functions.keys())}")
