import torch.nn.functional as F
import torch
from src.utils.distributed import AllReduce

import torch.nn.functional as F
import torch
from src.utils.distributed import AllReduce
from reptrix import lidar

class LossFunctions:
    def __init__(self, batch_size=None, num_target=None, num_context=None, del_sigma_augs=1, epsilon=0):
        self.num_samples = batch_size
        self.num_target = num_target
        self.num_context = num_context
        self.del_sigma_augs = del_sigma_augs
        self.epsilon = epsilon
    
    def jepa(self, z, h):
        loss = F.smooth_l1_loss(z, h)
        return AllReduce.apply(loss)

    def lidar_student(self, z, h):
        return lidar.get_lidar(z, self.num_samples, self.num_context, self.del_sigma_augs)

    def lidar_teacher(self, z, h):
        return lidar.get_lidar(h, self.num_samples, self.num_target, self.del_sigma_augs)
    
    @staticmethod
    def get_lidar_matrices(activations, num_samples, num_augs, del_sigma_augs=1e-6):
        # Compute object activations (mean over augmentations)
        object_activations = activations.mean(dim=1, keepdim=True)  # Mean over augmentations
        mean_activations = object_activations.mean(dim=0, keepdim=True)  # Mean over samples
    
        # Compute inter-object covariance (sigma_obj)
        diff_object_activations = object_activations - mean_activations
        sigma_b = diff_object_activations.squeeze().T @ diff_object_activations.squeeze()
    
        # Compute intra-object covariance (sigma_augs) and take the mean across objects
        diff_activations = activations - object_activations
        sigma_w = torch.bmm(
            diff_activations.permute((0, 2, 1)),
            diff_activations,
        ).mean(dim=0)
    
        # Add small identity matrix to ensure invertibility
        sigma_w += del_sigma_augs * torch.eye(sigma_w.shape[0], device=sigma_w.device)
    
        return sigma_b, sigma_w

    
    def sina(self, z, h):
        """
        This function calculates the Frobenius norm of the lidar matrix divided by the trace.
        The Frobenius norm is equivalent to calulating the trace of B^-1AB^-1A where:
        - B is sigma_w (covariance matrix of lidar),
        - A is sigma_b (covariance matrix of the signal),
        The trace is equivalent to the trace of B^-1A.
        """
        z = z.mean(dim=1)  # (batch_size, embedding)
        z = F.normalize(z, p=2, dim=1)
        z = z.reshape(self.num_samples, self.num_context, -1)
        sigma_b, sigma_w = self.get_lidar_matrices(z, self.num_samples, self.num_context, self.del_sigma_augs)
    
        # Cast to float 32 and make symmetric
        sigma_w = sigma_w.to(torch.float32)
        sigma_b = sigma_b.to(torch.float32)
        sigma_w = (sigma_w + sigma_w.T) / 2
        sigma_b = (sigma_b + sigma_b.T) / 2
    
        # loss
        sigma_w_inv_b = torch.linalg.solve(sigma_w, sigma_b) 
        sigma_w_inv_b = sigma_w_inv_b + self.epsilon  * torch.eye(sigma_w_inv_b.shape[0], device=sigma_w_inv_b.device)
        frobenius_norm = torch.trace(sigma_w_inv_b @ sigma_w_inv_b)
        frobenius_norm = torch.sqrt(frobenius_norm.abs())
        trace = torch.trace(sigma_w_inv_b)
        loss = torch.log(frobenius_norm / trace)
        loss = AllReduce.apply(loss)
        
        return loss
    
    def get_loss_function(self, loss_name):
        loss_map = {
            "jepa": self.jepa,
            "lidar_student": self.lidar_student,
            "lidar_teacher": self.lidar_teacher,
            "sina": self.sina,
        }
        if loss_name not in loss_map:
            raise ValueError(f"Invalid loss function '{loss_name}'. Available options: {list(loss_map.keys())}")
        return loss_map[loss_name]


    def get_lidar_matrices_properties(self, z):
        """
        Given the embeddings computes
        - Rank, condition of sigma_b, sigma_w, sigma_w_inv_b
        - Entropy from normalized eigenvalues
        - Sum of squared off-diagonal elements
        - Variance of diagonal elements
        - trace and frobenius norm of lidar loss
        Returns a dictionary
        """
        z = z.mean(dim=1)  # (batch_size, embedding)
        z = F.normalize(z, p=2, dim=1)
        z = z.reshape(self.num_samples, self.num_context, -1)
        sigma_b, sigma_w = self.get_lidar_matrices(z, self.num_samples, self.num_context, self.del_sigma_augs)
    
        # Cast to float 32 and make symmetric
        sigma_w = sigma_w.to(torch.float32)
        sigma_b = sigma_b.to(torch.float32)
        sigma_w = (sigma_w + sigma_w.T) / 2
        sigma_b = (sigma_b + sigma_b.T) / 2
    
        # loss
        sigma_w_inv_b = torch.linalg.solve(sigma_w, sigma_b) 
        sigma_w_inv_b += self.epsilon  * torch.eye(sigma_w_inv_b.shape[0], device=sigma_w_inv_b.device)
        frobenius_norm = torch.trace(sigma_w_inv_b @ sigma_w_inv_b)
        frobenius_norm = torch.sqrt(frobenius_norm.abs())
        trace = torch.trace(sigma_w_inv_b)
        
        # properties
        rank_sigma_w = torch.linalg.matrix_rank(sigma_w).item()
        rank_sigma_b = torch.linalg.matrix_rank(sigma_b).item()
        rank_sigma_w_inv_b = torch.linalg.matrix_rank(sigma_w_inv_b).item()
        
        condition_sigma_w = torch.linalg.cond(sigma_w).item()
        condition_sigma_b = torch.linalg.cond(sigma_b).item()
        condition_sigma_w_inv_b = torch.linalg.cond(sigma_w_inv_b).item()

        eigvals = torch.linalg.eigvalsh(sigma_w_inv_b)
        eigvals_norm = eigvals / eigvals.sum()
        eps = 1e-10  # Small constant to avoid log(0)
        max_eigval_norm = eigvals_norm.max()
        min_eigval_norm = eigvals_norm.min()
        eigvals_norm = torch.clamp(eigvals_norm, min=eps)
        entropy = -(eigvals_norm * eigvals_norm.log()).sum().item()
        off_diag = sigma_w_inv_b - torch.diag(torch.diagonal(sigma_w_inv_b))
        sum_squared_off_diag = torch.sum(off_diag ** 2).item()
        diag_var = torch.var(torch.diagonal(sigma_w_inv_b)).item()
        
        return {
            "rank simga_w": rank_sigma_w,
            "rank sigma_b": rank_sigma_b,
            "rank sigma_inv_b": rank_sigma_w_inv_b,

            "condition simga_w": condition_sigma_w,
            "condition sigma_b": condition_sigma_b,
            "condition sigma_inv_b": condition_sigma_w_inv_b,
            
            "entropy": entropy,
            "sum_squared_off_diag": sum_squared_off_diag,
            "diag_var": diag_var,
            "trace": trace,
            "frobenius norm": frobenius_norm,

            "max normalized eigenvalue": max_eigval_norm,
            "min normalized eigenvalue": min_eigval_norm
            
        }

