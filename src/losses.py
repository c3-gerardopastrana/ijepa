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
        
        self.lambda_ = 1e-1
        self.current_it = 0
        self.stabilization_period = 1
        self.max_condition = 1e4
        
    def jepa(self, z, h):
        loss = F.smooth_l1_loss(z, h)
        return AllReduce.apply(loss)

    def lidar_student(self, z, h):
        return lidar.get_lidar(z, self.num_samples, self.num_context, self.del_sigma_augs)

    def lidar_teacher(self, z, h):
        return lidar.get_lidar(h, self.num_samples, self.num_target, self.del_sigma_augs)
    
    def get_lidar_matrices(self, activations, num_samples, num_augs, lambda_=None):
        if lambda_ is None:
            lambda_ = self.lambda_
        # Compute object activations (mean over augmentations)
        object_activations = activations.mean(dim=1, keepdim=True)  # Mean over augmentations
        mean_activations = object_activations.mean(dim=0, keepdim=True)  # Mean over samples
    
        # Compute inter-object covariance (sigma_obj)
        diff_object_activations = object_activations - mean_activations
        diff_object_activations = diff_object_activations.view(num_samples, -1)
        sigma_b = diff_object_activations.T @ diff_object_activations
    
        # Compute intra-object covariance (sigma_augs) and take the mean across objects
        diff_activations = activations - object_activations
        sigma_w = torch.bmm(
            diff_activations.permute((0, 2, 1)),
            diff_activations,
        ).mean(dim=0)
    
        # Add small identity matrix to ensure invertibility
        eye = torch.eye(self.embed_dim, device="cuda", dtype=sigma_w.dtype)
        sigma_w = sigma_w + lambda_ * eye
        sigma_b = sigma_b + lambda_ * eye
        sigma_w = sigma_w + sigma_w.T
        sigma_w = sigma_w.to(torch.float32) / 2
        sigma_b = sigma_b + sigma_b.T
        sigma_b = sigma_b.to(torch.float32) / 2
    
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

        # loss
        sigma_w_inv_b = torch.linalg.solve(sigma_w, sigma_b) 

        # Hook to access gradients later
        sigma_w_inv_b.register_hook(lambda grad: self.save_matrix_grad(grad, "sigma_w_inv_b"))
        z.register_hook(lambda grad: self.save_matrix_grad(grad, "z"))
                
        max_frobenius_norm = torch.trace(sigma_w_inv_b @ sigma_w_inv_b)
        max_frobenius_norm = torch.sqrt(max_frobenius_norm.abs())
        trace = torch.trace(sigma_w_inv_b).abs()
       
        lambda_target = torch.tensor(2**14, dtype=sigma_w_inv_b.dtype, device=sigma_w_inv_b.device)
    
        penalty = (trace - lambda_target).pow(2) / lambda_target.pow(2)  # scale-free, minimal tuning
        loss = torch.log(max_frobenius_norm) -   torch.log(trace) + penalty
        
        return loss

    
    def save_matrix_grad(self, grad, key):
        """Generic hook to save gradients dynamically in a dictionary"""
        if self.scaler is not None:
            scale = self.scaler.get_scale()
            self.saved_grads[key] = grad / scale
        else:
            self.saved_grads[key] = grad


    def gap_loss(self, z, h, add_noise=True, collect_grads=True):
        """
        Computes the loss given by (lamnda_max-lambda_min)/trace
        """
        z = z.reshape(self.num_samples, self.num_context, -1)
        
        sigma_b, sigma_w = self.get_lidar_matrices(z, self.num_samples, self.num_context)
    
        # Cast to float 32 and make symmetric
        sigma_w = sigma_w.to(torch.float32)
        sigma_b = sigma_b.to(torch.float32)
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
        if collect_grads:
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
        if gap / trace < 0:
            print("gap", gap)
            print('trace', trace)
            print(max_frobenius_norm)
            print(min_frobenius_norm)
            self.lambda_ -= 1e-5
            self.current_it = -1
            loss = torch.log(gap / trace.abs())
        

        if self.current_it > self.stabilization_period:
            with torch.no_grad():
                current_condition_b = torch.linalg.cond(sigma_b) 
                min_eigenvalue_b = torch.linalg.eigvalsh(sigma_b).min()
                delta_b = min_eigenvalue_b * (self.max_condition - current_condition_b)/(self.max_condition - 1)
                delta_b = delta_b / (torch.norm(sigma_b - torch.eye(self.embed_dim, device="cuda"), p='fro') + 1)
    
                current_condition_w = torch.linalg.cond(sigma_w) 
                min_eigenvalue_w = torch.linalg.eigvalsh(sigma_w).min()
                delta_w = min_eigenvalue_w * (self.max_condition - current_condition_w)/(self.max_condition - 1)
                delta_w = delta_w / (torch.norm(sigma_w - torch.eye(self.embed_dim, device="cuda"), p='fro') + 1)
    
                delta = 5 * torch.min(delta_b, delta_w)
    
                if delta > 0 and self.lambda_ - delta > 0:
                    self.lambda_ = self.lambda_ - delta
                    self.current_it = -1
                    
        self.current_it +=1
        
        return loss

    def sina_cov(self, z, h):
        
        # Compute covariance matrix
        z = z.reshape(self.num_samples * self.num_context, -1)
        sigma = sigma = z.T @ z
        sigma = sigma.to(torch.float32)
        sigma = (sigma + sigma.T) / 2


        # loss
        max_frobenius_norm = torch.linalg.norm(sigma, ord='fro')
        trace = torch.trace(sigma)
        loss = torch.log(max_frobenius_norm / trace)
        
        # Hook to access gradients later
        z.register_hook(lambda grad: self.save_matrix_grad(grad, "z"))
        
        return loss
        
    def get_loss_function(self, loss_name):
        loss_map = {
            "jepa": self.jepa,
            "lidar_student": self.lidar_student,
            "lidar_teacher": self.lidar_teacher,
            "sina": self.sina,
            "gap": self.gap_loss,
            "sina_cov": self.sina_cov
        }
        if loss_name not in loss_map:
            raise ValueError(f"Invalid loss function '{loss_name}'. Available options: {list(loss_map.keys())}")
        return loss_map[loss_name]


    def lda_from_matrices_and_accuracy(self, z):
        """
        Perform LDA-like projection and compute classification accuracy based on perturbations.
    
        Args:
            z: Tensor of shape (num_samples, num_context, feature_dim) representing perturbed samples.
    
        Returns:
            accuracy: Classification accuracy after projection.
            projections: Projected features in the LDA space.
            evals: Eigenvalues from the generalized eigenvalue decomposition.
            evecs: Eigenvectors from the generalized eigenvalue decomposition.
        """
        z = z.reshape(self.num_samples, self.num_context, -1)
        sigma_b, sigma_w = self.get_lidar_matrices(z, self.num_samples, self.num_context)
        sigma_w_inv_b = torch.linalg.solve(sigma_w, sigma_b)
    
        # Eigendecomposition
        evals_complex, evecs_complex = torch.linalg.eig(sigma_w_inv_b)
    
        # Remove complex components (if any)
        tol = 1e-6
        is_real = (evals_complex.imag.abs() < tol)
        evals = evals_complex[is_real].real
        evecs = evecs_complex[:, is_real].real
    
        # # Sort eigenvalues and take top components
        # sorted_idx = torch.argsort(evals, descending=True)
        # evals = evals[sorted_idx]
        # evecs = evecs[:, sorted_idx]
    
        # Project to LDA space
        projections = z.reshape(self.num_samples * self.num_context, -1) @ evecs  
    
        # Assign labels: Each set of num_context perturbations belongs to the same class
        labels = torch.repeat_interleave(torch.arange(self.num_samples), self.num_context).to(z.device)
    
        # Compute mean of projected features per class (this is a form of "centroid" classification)
        unique_labels = torch.unique(labels)
        centroids = torch.stack([
            projections[labels == label].mean(0)
            for label in unique_labels
        ])
    
        # Nearest centroid classification
        dists = torch.cdist(projections, centroids)
        preds = dists.argmin(dim=1)
    
        # Calculate accuracy
        accuracy = (preds == labels).float().mean().item()
    
        return accuracy, projections, evals, evecs
        
    def get_lidar_matrices_properties(self, z): 
        # try:
        z = z.reshape(self.num_samples, self.num_context, -1)
        sigma_b, sigma_w = self.get_lidar_matrices(z, self.num_samples, self.num_context, self.del_sigma_augs)
        sigma_w_inv_b = torch.linalg.solve(sigma_w, sigma_b)
        frob_norm = torch.trace(sigma_w_inv_b @ sigma_w_inv_b).abs().sqrt().item()
        trace = torch.trace(sigma_w_inv_b).item()
        rank_w = torch.linalg.matrix_rank(sigma_w).item()
        rank_b = torch.linalg.matrix_rank(sigma_b).item()
        rank_inv = torch.linalg.matrix_rank(sigma_w_inv_b).item()
        cond_w = torch.linalg.cond(sigma_w).item()
        cond_b = torch.linalg.cond(sigma_b).item()
        cond_inv = torch.linalg.cond(sigma_w_inv_b).item()
        
        eigvals = torch.linalg.eigvals(sigma_w_inv_b)
        is_real = (eigvals.imag.abs() <= 1e-10)
        complex_count = (~is_real).sum().item()
        real_eigs = eigvals[is_real].real
        
        if len(real_eigs) > 0:
            eigs_norm = real_eigs / real_eigs.sum()
            eps = 1e-10
            eigs_norm = torch.clamp(eigs_norm, min=eps)
            entropy = -(eigs_norm * eigs_norm.log()).sum().item()
            stats = {
                "entropy": entropy,
                "max_normalized_eigenvalue": eigs_norm.max().item(),
                "min_normalized_eigenvalue": eigs_norm.min().item(),
                "quantile_25": torch.quantile(eigs_norm, 0.25).item(),
                "quantile_50": torch.quantile(eigs_norm, 0.5).item(),
                "quantile_75": torch.quantile(eigs_norm, 0.75).item()
                }
        else:
            stats = {key: 0 for key in ["entropy", "max_normalized_eigenvalue", "min_normalized_eigenvalue", "quantile_25", "quantile_50", "quantile_75"]}
        
        off_diag = sigma_w_inv_b - torch.diag(torch.diagonal(sigma_w_inv_b))
        sum_squared_off_diag = (off_diag**2).sum().item()
        diag_var = torch.var(torch.diagonal(sigma_w_inv_b)).item()
        accuracy, projections, evals, evecs = self.lda_from_matrices_and_accuracy(z)

        return {
            "trace": trace,
            "frobenius_norm": frob_norm,
            "rank_sigma_w": rank_w,
            "rank_sigma_b": rank_b,
            "rank_sigma_inv_b": rank_inv,
            "condition_sigma_w": cond_w,
            "condition_sigma_b": cond_b,
            "condition_sigma_inv_b": cond_inv,
            "complex_eigenvalue_count": complex_count,
            "sum_squared_off_diag": sum_squared_off_diag,
            "diag_var": diag_var,
            "supervised_accuracy": accuracy,
            **stats
        }
        
        # except RuntimeError as e:
        #     # Print the error message and error details
        #     print(f"RuntimeError occurred: {str(e)}")
        #     print(f"Error Details: {e.__class__} - {e.args}")
        #     return {
        #         "trace": 0,
        #         "frobenius_norm": 0,
        #         "rank_sigma_w": torch.linalg.matrix_rank(sigma_w).item(),
        #         "rank_sigma_b": torch.linalg.matrix_rank(sigma_b).item(),
        #         "rank_sigma_inv_b": 0,
        #         "condition_sigma_w": torch.linalg.cond(sigma_w).item(),
        #         "condition_sigma_b": torch.linalg.cond(sigma_b).item(),
        #         "condition_sigma_inv_b": float("inf"),
        #         "complex_eigenvalue_count": 0,
        #         "sum_squared_off_diag": 0,
        #         "diag_var": 0,
        #         "entropy": 0,
        #         "max_normalized_eigenvalue": 0,
        #         "min_normalized_eigenvalue": 0,
        #         "quantile_25": 0,
        #         "quantile_50": 0,
        #         "quantile_75": 0,
        #         "supervised_accuracy": 0,
        #     }


