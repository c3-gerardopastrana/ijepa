import torch.nn.functional as F
import torch
from src.utils.distributed import AllReduce
from reptrix import alpha, rankme, lidar

def jepa(z, h):
    loss = F.smooth_l1_loss(z, h)
    loss = AllReduce.apply(loss)
    return loss

def lidar_student(z, h, batch_size, num_context):
    return lidar.get_lidar(z, batch_size, num_context, del_sigma_augs=0.001)

def lidar_teacher(z, h, batch_size, num_target):
    return lidar.get_lidar(h, batch_size, num_target, del_sigma_augs=0.001)

def get_lidar_matrices(
    activations: torch.Tensor,
    num_samples: int,
    num_augs: int,
    del_sigma_augs: float = 1e-6
):
   
    activations_arr = activations

    if activations_arr.shape[0] == num_samples * num_augs:
        activations_arr = activations_arr.reshape(num_samples, num_augs, -1)
    else:
        d0, d1 = activations_arr.shape[:2]
        assert (
            d0 == num_samples and d1 == num_augs
        ), "Tensor activations should have shape (num_samples, num_augs,...)"
        activations_arr = activations_arr.reshape(d0, d1, -1)

    # Compute object activations (mean over augmentations)
    object_activations = activations_arr.mean(dim=1, keepdim=True)  # Mean over augmentations
    mean_activations = object_activations.mean(dim=0, keepdim=True)  # Mean over samples

    # Compute inter-object covariance (sigma_obj)
    sigma_b = (object_activations - mean_activations).squeeze().T @ (object_activations - mean_activations).squeeze()

    # Compute intra-object covariance (sigma_augs) and take the mean across objects
    sigma_w = torch.bmm(
        (activations_arr - object_activations).permute((0, 2, 1)),
        (activations_arr - object_activations),
    ).mean(dim=0)
    
    # Add small identity matrix to ensure invertibility
    sigma_w += del_sigma_augs * torch.eye(sigma_w.shape[0], device=sigma_w.device)

    return sigma_b, sigma_w


def sina(z, h, num_samples, num_augs, del_sigma_augs: float = 1e-1):
    """
    This function calculates the Frobenius norm of the lidar matrix divided by the trace.
    The Frobenius norm is equivalent to calulating the trace of B^-1AB^-1A where:
    - B is sigma_w (covariance matrix of lidar),
    - A is sigma_b (covariance matrix of the signal),
    The trace is equivalent to the trace of B^-1A.
    """
    print(z.size())
    sigma_b, sigma_w = get_lidar_matrices(z, num_samples, num_augs, del_sigma_augs)
    epsilon = 1e-8 

    # Convert to float32 to invert
    sigma_w = sigma_w.to(torch.float32)
    sigma_b = sigma_b.to(torch.float32)
    sigma_b = AllReduce.apply(sigma_b)
    sigma_w = AllReduce.apply(sigma_w)

    # loss
    sigma_w_inv_b = torch.linalg.solve(sigma_w, sigma_b)
    frobenius_norm = torch.trace(sigma_w_inv_b @ sigma_w_inv_b).sqrt()
    trace = torch.trace(sigma_w_inv_b)
    loss = (frobenius_norm / (trace + epsilon)).to(z.dtype)
    loss = AllReduce.apply(loss)
    print("rankkkkk", torch.linalg.matrix_rank(sigma_w), sigma_w.size())
    
    
    return loss

def get_loss_function(loss_name, batch_size=None, num_target=None, num_context=None):
    loss_functions = {
        "jepa": jepa,
        "lidar_student": lidar_student,
        "lidar_teacher": lidar_teacher,
        "sina": sina
    }

    if loss_name in loss_functions:
        if loss_name == "jepa":
            return lambda z, h: loss_functions[loss_name](z, h)
        elif loss_name == "lidar_student":
            return lambda z, h: loss_functions[loss_name](z, h, batch_size, num_context)
        elif loss_name == "lidar_teacher":
            return lambda z, h: loss_functions[loss_name](z, h, batch_size, num_target)
        elif loss_name == "sina":
            return lambda z, h: loss_functions[loss_name](z, h, batch_size, num_context)

    raise ValueError(f"Invalid loss function '{loss_name}'. Available options: {list(loss_functions.keys())}")
