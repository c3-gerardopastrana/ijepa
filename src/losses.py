import torch.nn.functional as F
import torch
from src.utils.distributed import AllReduce

import torch.nn.functional as F
import torch
from src.utils.distributed import AllReduce
from reptrix import lidar
import math

class LossFunctions:
    def __init__(self, batch_size=None, num_target=None, num_context=None, del_sigma_augs=1, epsilon=0, embed_dim=192, scaler=None):
        self.num_samples = batch_size
        self.num_target = num_target
        self.num_context = num_context
        self.del_sigma_augs = del_sigma_augs
        self.epsilon = epsilon

        # Norm 1 noise
        self.diag_values_w = torch.rand(embed_dim, device="cuda")
        self.diag_values_b = torch.rand(embed_dim, device="cuda")       
        self.diag_matrix_w = torch.diag(self.diag_values_w / torch.norm(self.diag_values_w))
        self.diag_matrix_b = torch.diag(self.diag_values_b / torch.norm(self.diag_values_b))
        
        self.saved_grads = {}
        self.scaler = scaler
        self.embed_dim = embed_dim
        
        self.lambda_ = 1
        self.current_it = 0
        self.stabilization_period = 1000
        self.max_condition = 1e3
        
    def jepa(self, z, h):
        loss = F.smooth_l1_loss(z, h)
        return AllReduce.apply(loss)

    def lidar_student(self, z, h):
        return lidar.get_lidar(z, self.num_samples, self.num_context, self.del_sigma_augs)

    def lidar_teacher(self, z, h):
        return lidar.get_lidar(h, self.num_samples, self.num_target, self.del_sigma_augs)
    
    def get_lidar_matrices(self, activations, num_samples, num_augs, del_sigma_augs=1e-6, add_noise = True):
        
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
        if add_noise:
            sigma_w = (torch.eye(self.embed_dim, device="cuda") - self.diag_matrix_w) @ sigma_w + self.diag_matrix_w
            sigma_b = (torch.eye(self.embed_dim, device="cuda") - self.diag_matrix_b) @ sigma_b + self.diag_matrix_b
        
        return sigma_b, sigma_w

            
    def sina(self, z, h):
        """
        This function calculates the Frobenius norm of the lidar matrix divided by the trace.
        The Frobenius norm is equivalent to calulating the trace of B^-1AB^-1A where:
        - B is sigma_w (covariance matrix of lidar),
        - A is sigma_b (covariance matrix of the signal),
        The trace is equivalent to the trace of B^-1A.
        """
        z = z.reshape(1, -1)
        z_reshaped = z.reshape(self.num_samples, self.num_context, -1)
        
        sigma_b, sigma_w = self.get_lidar_matrices(z_reshaped, self.num_samples, self.num_context, self.del_sigma_augs)
        
        # Cast to float 32 and make symmetric
        sigma_w = sigma_w.to(torch.float32)
        sigma_b = sigma_b.to(torch.float32)
        sigma_w = (sigma_w + sigma_w.T) / 2
        sigma_b = (sigma_b + sigma_b.T) / 2

        # loss
        sigma_w_inv_b = torch.linalg.solve(sigma_w, sigma_b) 
        sigma_w_inv_b = sigma_w_inv_b + self.epsilon  * torch.eye(sigma_w_inv_b.shape[0], device=sigma_w_inv_b.device)
        
        # Hook to access gradients later
        sigma_w_inv_b.register_hook(lambda grad: self.save_matrix_grad(grad, "sigma_w_inv_b"))
        z.register_hook(lambda grad: self.save_matrix_grad(grad, "z"))
                
        frobenius_norm = torch.trace(sigma_w_inv_b @ sigma_w_inv_b)
        frobenius_norm = torch.sqrt(frobenius_norm.abs())
        trace = torch.trace(sigma_w_inv_b)
        loss = torch.log(frobenius_norm / trace)

        return loss

    
    def save_matrix_grad(self, grad, key):
        """Generic hook to save gradients dynamically in a dictionary"""
        if self.scaler is not None:
            scale = self.scaler.get_scale()
            self.saved_grads[key] = grad / scale
        else:
            self.saved_grads[key] = grad


    def gap_loss(self, z, h):
        """
        Computes the loss given by (lamnda_max-lambda_min)/trace
        """
        z = z.reshape(self.num_samples, self.num_context, -1)
        
        sigma_b, sigma_w = self.get_lidar_matrices(z, self.num_samples, self.num_context, self.del_sigma_augs)
    
        # Cast to float 32 and make symmetric
        sigma_w = sigma_w.to(torch.float64)
        sigma_b = sigma_b.to(torch.float64)
        sigma_w = (sigma_w + sigma_w.T) / 2
        sigma_b = (sigma_b + sigma_b.T) / 2
    

        # Compute eigenvalues of B and W
        # sigma_w_inv_b = torch.linalg.solve(sigma_w, sigma_b) 
        # eigvals = torch.linalg.eigvals(sigma_w_inv_b).real
        # lambda_max = torch.max(eigvals)
        # lambda_min = torch.min(eigvals)
        
        # epsilon = torch.tensor(1e-6, dtype=sigma_b.dtype, device=sigma_b.device)
        # #lambda_min = torch.max(lambda_min, epsilon)

        # epsilon=1e-3
        # threshold = lambda_min + epsilon

        # # Mask for valid eigenvalues
        # mask = eigvals < threshold  # Shape: (d,), True where valid
    
        # # Select valid eigenvalues
        # valid_eigvals = eigvals[mask]
    
        # # Compute the average (avoid division by zero)
        # if valid_eigvals.numel() > 0:
        #     loss = -valid_eigvals.mean()
        # else:
        #     loss = torch.tensor(0.0, dtype=eigvals.dtype, device=eigvals.device)
        #loss = gap / trace

        # gap = (lambda_max - lambda_min)
        # trace = torch.trace(sigma_w_inv_b)
        # loss = torch.log(gap / trace)
        
        sigma_w_inv_b = torch.linalg.solve(sigma_w, sigma_b) 
        
        # Hook to access gradients later
        sigma_w_inv_b.register_hook(lambda grad: self.save_matrix_grad(grad, "sigma_w_inv_b"))
        z.register_hook(lambda grad: self.save_matrix_grad(grad, "z"))
                
        max_frobenius_norm = torch.trace(sigma_w_inv_b @ sigma_w_inv_b)
        max_frobenius_norm = torch.sqrt(max_frobenius_norm.abs())

        sigma_b_inv_w = torch.linalg.solve(sigma_b, sigma_w ) 
        min_frobenius_norm = torch.trace(sigma_b_inv_w @ sigma_b_inv_w)
        min_frobenius_norm = 1/torch.sqrt(min_frobenius_norm.abs())
 
        trace = torch.trace(sigma_w_inv_b)
        gap = max_frobenius_norm - min_frobenius_norm
        loss = torch.log(gap / trace)

        if self.current_it > self.stabilization_period:
            current_condition = max_frobenius_norm * (1 / min_frobenius_norm)
            delta = min_frobenius_norm*(self.max_condition - current_condition)/(self.max_condition - 1)
            delta = delta/(self.embed_dim + 1)
            if delta > 0:
                self.lambda_ = self.lamnda - delta
                self.current_it = 0
        self.current_it +=1
        
        return loss

    
    def get_loss_function(self, loss_name):
        loss_map = {
            "jepa": self.jepa,
            "lidar_student": self.lidar_student,
            "lidar_teacher": self.lidar_teacher,
            "sina": self.sina,
            "gap": self.gap_loss
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
        z = z.reshape(self.num_samples, self.num_context, -1)
        sigma_b, sigma_w = self.get_lidar_matrices(z, self.num_samples, self.num_context, self.del_sigma_augs)
    
        # Cast to float 32 and make symmetric
        sigma_w = sigma_w.to(torch.float64)
        sigma_b = sigma_b.to(torch.float64)
        sigma_w = (sigma_w + sigma_w.T) / 2
        sigma_b = (sigma_b + sigma_b.T) / 2
    
        # loss components
        sigma_w_inv_b = torch.linalg.solve(sigma_w, sigma_b) 
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
        
        # Get eigenvalues and filter out complex ones
        eigvals = torch.linalg.eigvals(sigma_w_inv_b)
        is_real = torch.abs(eigvals.imag) <= 1e-10
        complex_count = torch.sum(~is_real).item()
        
        # Filter out complex eigenvalues
        real_eigvals = eigvals[is_real].real
        
        # If we have any real eigenvalues, compute statistics
        if len(real_eigvals) > 0:
            # Normalize real eigenvalues
            eigvals_norm = real_eigvals / real_eigvals.sum()
            eps = 1e-10  # Small constant to avoid log(0)
            max_eigval_norm = eigvals_norm.max().item()
            min_eigval_norm = eigvals_norm.min().item()
            quantile_25 = torch.quantile(eigvals_norm, 0.25).item()
            quantile_50 = torch.quantile(eigvals_norm, 0.5).item()
            quantile_75 = torch.quantile(eigvals_norm, 0.75).item()
            eigvals_norm = torch.clamp(eigvals_norm, min=eps)
            entropy = -(eigvals_norm * eigvals_norm.log()).sum().item()
        else:
            # No real eigenvalues
            max_eigval_norm = 0
            min_eigval_norm = 0
            quantile_25 = 0
            quantile_50 = 0
            quantile_75 = 0
            entropy = 0
        
        off_diag = sigma_w_inv_b - torch.diag(torch.diagonal(sigma_w_inv_b))
        sum_squared_off_diag = torch.sum(off_diag ** 2).item()
        diag_var = torch.var(torch.diagonal(sigma_w_inv_b)).item()

        # Original matrix stats
        sigma_b, sigma_w = self.get_lidar_matrices(z, self.num_samples, self.num_context, self.del_sigma_augs, add_noise=False)
        sigma_w = sigma_w.to(torch.float64)
        sigma_b = sigma_b.to(torch.float64)
        sigma_w = (sigma_w + sigma_w.T) / 2
        sigma_b = (sigma_b + sigma_b.T) / 2
        sigma_w_inv_b = torch.linalg.solve(sigma_w, sigma_b) 
        trace_original = torch.trace(sigma_w_inv_b)
        
        condition_sigma_w_inv_b_original = torch.linalg.cond(sigma_w_inv_b).item()
        eigvals = torch.linalg.eigvals(sigma_w_inv_b)
        is_real = torch.abs(eigvals.imag) <= 1e-10
        complex_count_original = torch.sum(~is_real).item()
        
        # Filter out complex eigenvalues
        real_eigvals = eigvals[is_real].real
        
        # If we have any real eigenvalues, compute statistics
        if len(real_eigvals) > 0:
            # Normalize real eigenvalues
            eigvals_norm = real_eigvals / real_eigvals.sum()
            eps = 1e-10  # Small constant to avoid log(0)
            max_eigval_norm_original = eigvals_norm.max().item()
            min_eigval_norm_original = eigvals_norm.min().item()
            quantile_25_original = torch.quantile(eigvals_norm, 0.25).item()
            quantile_50_original = torch.quantile(eigvals_norm, 0.5).item()
            quantile_75_original = torch.quantile(eigvals_norm, 0.75).item()
            eigvals_norm_original = torch.clamp(eigvals_norm, min=eps)
            entropy_original = -(eigvals_norm * eigvals_norm.log()).sum().item()
        else:
            # No real eigenvalues
            max_eigval_norm_original = 0
            min_eigval_norm_original = 0
            quantile_25_original = 0
            quantile_50_original = 0
            quantile_75_original = 0
            entropy_original = 0
        
        off_diag_original = sigma_w_inv_b - torch.diag(torch.diagonal(sigma_w_inv_b))
        sum_squared_off_diag_original = torch.sum(off_diag ** 2).item()
        diag_var_original = torch.var(torch.diagonal(sigma_w_inv_b)).item()
        
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
            "min normalized eigenvalue": min_eigval_norm,
            "quantile_25": quantile_25,
            "quantile_50": quantile_50,
            "quantile_75": quantile_75,
            "complex_eigenvalue_count": complex_count,

           "condition_sigma_w_inv_b_original": condition_sigma_w_inv_b_original,
            "entropy_original_matrix": entropy_original,
            "sum_squared_off_diag_original_matrix": sum_squared_off_diag_original,
            "diag_var_original_matrix": diag_var_original,
            "trace_original_matrix": trace_original,
            "max normalized eigenvalue_original_matrix": max_eigval_norm_original,
            "min normalized eigenvalue_original_matrix": min_eigval_norm_original,
            "quantile_25_original_matrix": quantile_25_original,
            "quantile_50_original_matrix": quantile_50_original,
            "quantile_75_original_matrix": quantile_75_original,
            "complex_eigenvalue_count_original_matrix": complex_count_original,
            "lambda": self.lambda_
        }
