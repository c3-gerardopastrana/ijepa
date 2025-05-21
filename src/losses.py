import torch.nn.functional as F
import torch
from src.utils.distributed import AllReduce
from src.utils_metrics import get_lidar_properties as metrics

class LossFunctions:
    def __init__(self, batch_size=None, num_target=None, num_context=None, embed_dim=192, scaler=None, lambda_ = None):
        self.num_samples = batch_size
        self.num_target = num_target
        self.num_context = num_context

        self.saved_grads = {}
        self.scaler = scaler
        self.embed_dim = embed_dim
        
        if lambda_ is None:
            self.lambda_ = 0.1 / embed_dim
            
        
    def jepa(self, z, h):
        loss = F.smooth_l1_loss(z, h)
        return AllReduce.apply(loss)

    def lidar_student(self, z, h):
        return lidar.get_lidar(z, self.num_samples, self.num_context, self.del_sigma_augs)

    def lidar_teacher(self, z, h):
        return lidar.get_lidar(h, self.num_samples, self.num_target, self.del_sigma_augs)
    
    def get_lidar_matrices(self, activations, num_samples, num_augs):

        # Reshape input to match num_samples x num_augs
        activations = activations.view(num_augs, num_samples, -1).permute(1, 0, 2)

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

        # Normalize covariane matrix by n samples
        sigma_b /= (num_samples - 1)
        sigma_w /= (num_augs - 1)
    
        # Add small identity matrix to ensure invertibility and symmetrize
        eye = torch.eye(self.embed_dim, device="cuda", dtype=sigma_w.dtype)
        sigma_w = sigma_w + self.lambda_ * eye
        
        sigma_b = sigma_b + sigma_b.T
        sigma_b = sigma_b.to(torch.float32) / 2
        
        sigma_w = sigma_w + sigma_w.T
        sigma_w = sigma_w.to(torch.float32) / 2
        
        sigma_w_inv_b = torch.linalg.solve(sigma_w, sigma_b)
        sigma_w_inv_b = sigma_w_inv_b + sigma_w_inv_b.T
        sigma_w_inv_b = sigma_w_inv_b.to(torch.float32) / 2
    
        return sigma_b, sigma_w - self.lambda_ * eye, sigma_w_inv_b
        
            
    def sina(self, z, h):
        """
        This function calculates the Frobenius norm of the lidar matrix divided by the trace.
        The Frobenius norm is equivalent to calulating the trace of B^-1AB^-1A where:
        - B is sigma_w (covariance matrix of lidar),
        - A is sigma_b (covariance matrix of the signal),
        The trace is equivalent to the trace of B^-1A.
        """
        sigma_b, sigma_w, sigma_w_inv_b = self.get_lidar_matrices(z, self.num_samples, self.num_context)

        # Hook to access gradients later
        sigma_w_inv_b.register_hook(lambda grad: self.save_matrix_grad(grad, "sigma_w_inv_b"))
        z.register_hook(lambda grad: self.save_matrix_grad(grad, "z"))

        # loss
        max_frobenius_norm = torch.trace(sigma_w_inv_b @ sigma_w_inv_b)
        max_frobenius_norm = torch.sqrt(max_frobenius_norm.abs())
        trace = torch.trace(sigma_w_inv_b).abs()

        lambda_target = torch.tensor(self.embed_dim //2, dtype=sigma_w_inv_b.dtype, device=sigma_w_inv_b.device)
        penalty = (trace - lambda_target).pow(2) / self.embed_dim
        
        loss = torch.log(max_frobenius_norm) -  torch.log(trace) + penalty
        
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
        

    def get_lidar_matrices_properties(self, z): 
        return metrics(z, self.get_lidar_matrices, self.num_samples, self.num_context)


       


