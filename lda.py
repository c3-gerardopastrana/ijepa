import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

def lda(X, y, n_classes, lamb):
    X = X.view(X.shape[0], -1)
    N, D = X.shape
    
    # Find unique classes present in the data
    labels, counts = torch.unique(y, return_counts=True)
    
    # Calculate overall mean and centered data
    X_bar = X - torch.mean(X, 0)
    
    # Calculate total scatter matrix
    St = X_bar.t().matmul(X_bar) / (N - 1)
    
    # Initialize within-class scatter matrix with requires_grad=True
    Sw = torch.zeros((D, D), dtype=X.dtype, device=X.device, requires_grad=True)
    
    # Pre-allocate tensor for class means
    Xc_mean = torch.zeros((n_classes, D), dtype=X.dtype, device=X.device)
    
    # Calculate within-class scatter matrix
    for c, Nc in zip(labels, counts):
        Xc = X[y == c]
        # Store class mean in the pre-allocated tensor at index c
        Xc_mean[int(c), :] = torch.mean(Xc, 0)
        
        # Center the data for this class
        Xc_bar = Xc - Xc_mean[int(c), :]
        
        # Add weighted contribution to within-class scatter
        # Use max(1, Nc-1) to avoid division by zero for single-sample classes
        Sw = Sw + Xc_bar.t().matmul(Xc_bar) / max(1, Nc - 1) * (Nc / N)
    
    # Calculate between-class scatter matrix
    
    # Compute overall mean
    mu = torch.mean(X, dim=0, keepdim=True)  # (1, D)
    
    # Initialize Sb
    Sb = torch.zeros((D, D), dtype=X.dtype, device=X.device)
    
    # Compute Sb directly from class means and counts
    for c, Nc in zip(labels, counts):
        mu_c = Xc_mean[int(c), :].unsqueeze(0)  # (1, D)
        delta = mu_c - mu  # (1, D)
        Sb += (Nc / N) * delta.t().matmul(delta)  # (D, D)

    #Sb = St - Sw
    # mu = torch.trace(Sw) / D # 1.0 / D #
    # shrinkage = 0.01 # 1- torch.trace(Sw) 
    # Sw_reg = (1-shrinkage) * Sw + torch.eye(D, dtype=X.dtype, device=X.device, requires_grad=False) * shrinkage * mu
    # add mean? add something to Sw_reg?
    #lambda_ = (1.0 / D) * (1 - torch.trace(Sw))
    Sw_reg = Sw + torch.eye(D, dtype=X.dtype, device=X.device, requires_grad=False) * lamb
    temp = torch.linalg.solve(Sw_reg, Sb) #torch.linalg.pinv(Sw, hermitian=True).matmul(Sb)
    temp = (temp + temp.T) / 2
    
    return Xc_mean, temp, Sw, Sb, St


def lda_loss(evals, n_classes, n_eig=None, margin=None):
    n_components = n_classes - 1
    evals = evals[-n_components:]
    # evecs = evecs[:, -n_components:]
    # print('evals', evals.shape, evals)
    # print('evecs', evecs.shape)

    # calculate loss
    if margin is not None:
        threshold = torch.min(evals) + margin
        n_eig = torch.sum(evals < threshold)
   
    probs = evals / evals.sum()
    entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-9)))

    loss = -entropy
    #loss = torch.mean(evals[:n_eig]) # small eigen values are on left
    
    # eigvals_norm = evals / evals.sum()
    # eps = 1e-10 
    # eigvals_norm = torch.clamp(eigvals_norm, min=eps)
    # oss = (eigvals_norm * eigvals_norm.log()).sum()
    #loss = torch.log(eigvals_norm.max()-eigvals_norm.min())
    return loss
    
def sina_loss(sigma_w_inv_b, sigma_w, sigma_b, xc_mean, sigma_t):
    mu = xc_mean.mean(dim=0)       # (D,)
    mean_term = torch.sum(mu ** 2)
    # loss = (torch.log(torch.trace(sigma_t)) - torch.log(torch.trace(sigma_b))) + mean_term
    n = torch.tensor(512, dtype=sigma_w_inv_b.dtype, device=sigma_w_inv_b.device)

    max_frobenius_norm = torch.trace(sigma_w_inv_b @ sigma_w_inv_b)
    max_frobenius_norm = torch.sqrt(max_frobenius_norm.abs()) 
    trace = torch.trace(sigma_w_inv_b).abs()
    lambda_target = torch.tensor(2**7, dtype=sigma_w_inv_b.dtype, device=sigma_w_inv_b.device)
    penalty = (trace - lambda_target).pow(2) / n
    # penalty = 0.01 * (torch.log(torch.trace(sigma_w)) - torch.log(torch.trace(sigma_b)))
    loss = torch.log(max_frobenius_norm) -  torch.log(trace) + penalty

    # trace_w = torch.trace(sigma_w)
    # frob_norm_sq_w = torch.sum(sigma_w ** 2)
    # d = sigma_w.shape[0]
    
    
    # w_sphericity = frob_norm_sq_w - (trace_w ** 2) / d

    # b_sphericity = torch.log(torch.norm(sigma_b, p='fro')) - torch.log(torch.trace(sigma_b))
    # loss = triangle_loss(xc_mean, sigma_b, sigma_w, epsilon = 0.15)
    #mean_term + (1-torch.trace(sigma_b + sigma_w)) + 10 * ((sigma_w.sum() - torch.diagonal(sigma_w).sum()))
    #loss = wasserstein_loss(xc_mean, sigma_t) # + w_sphericity
    # + torch.norm(sigma_w, p='fro') ** 2 / n
    #triangle_loss(Xc_mean, Sb, Sw, epsilon = 0.15)
    #loss = mean_term - torch.trace(sigma_w + sigma_b) + w_sphericity

    
    
    #max_frobenius_norm = torch.norm(sigma_b, p='fro')
    #trace = torch.trace(sigma_w_inv_b).abs()
    # lambda_target = torch.tensor(512, dtype=sigma_w_inv_b.dtype, device=sigma_w_inv_b.device)
    # # dim = sigma_w_inv_b.shape[0]
    # # identity = torch.eye(dim, dtype=sigma_w_inv_b.dtype, device=sigma_w_inv_b.device)
    # # diff = sigma_w_inv_b - lambda_target * identity
    # # loss = torch.norm(diff, p='fro')**2
    # penalty = (trace - lambda_target).pow(2)  # scale-free, minimal tuning
    #torch.relu(lambda_target - sigma)#(trace - lambda_target).pow(2)/lambda_target # scale-free, minimal tuning
    # loss = torch.log(max_frobenius_norm) -   torch.log(trace) + penalty

   
    
    # sigma_b_inv_w =  torch.linalg.pinv(sigma_w_inv_b, hermitian=True)
    # min_frobenius_norm = torch.trace(sigma_b_inv_w @ sigma_b_inv_w)
    # min_frobenius_norm = 1/torch.sqrt(min_frobenius_norm.abs())
    
    # gap =  max_frobenius_norm - min_frobenius_norm
    #loss = torch.log(gap / trace)
    
    #trace = torch.trace(1/2*(sigma_w_inv_b + sigma_w_inv_b.T))
    #loss = torch.log(max_frobenius_norm) - torch.log(trace)
    
    # off_diag = sigma_w_inv_b - torch.diag(torch.diagonal(sigma_w_inv_b))
    # sum_squared_off_diag = torch.sum(off_diag ** 2).item()
    # loss = sum_squared_off_diag - trace
    return loss

class LDA(nn.Module):
    def __init__(self, n_classes, lamb):
        super(LDA, self).__init__()
        self.n_classes = n_classes
        self.n_components = n_classes - 1
        self.lamb = lamb
        self.lda_layer = partial(lda, n_classes=n_classes, lamb=lamb)

    def forward(self, X, y):
        # Perform batch-wise LDA (temporary, not global yet)
        Xc_mean, sigma_w_inv_b, sigma_w, sigma_b, sigma_t = self.lda_layer(X, y)

        return Xc_mean, sigma_w_inv_b, sigma_w, sigma_b, sigma_t

    def transform(self, X):
        return X.matmul(self.scalings_)[:, :self.n_components]

    def predict(self, X):
        logit = X.matmul(self.coef_.t()) + self.intercept_
        return torch.argmax(logit, dim=1)

    def predict_proba(self, X):
        logit = X.matmul(self.coef_.t()) + self.intercept_
        return nn.functional.softmax(logit, dim=1)

    def predict_log_proba(self, X):
        logit = X.matmul(self.coef_.t()) + self.intercept_
        return nn.functional.log_softmax(logit, dim=1)



class RunningLDAStats:
    def __init__(self, n_classes, n_features, device='cpu'):
        self.n_classes = n_classes
        self.n_features = n_features
        self.device = 'cpu'  # FORCE CPU

        self.class_sums = torch.zeros((n_classes, n_features), device=self.device)
        self.class_counts = torch.zeros(n_classes, device=self.device)
        self.within_class_scatter = torch.zeros((n_features, n_features), device=self.device)

    @torch.no_grad()
    def update(self, X, y):
        X = X.view(X.shape[0], -1).detach().to('cpu')
        y = y.detach().to('cpu')

        for cls in range(self.n_classes):
            mask = (y == cls)
            if mask.sum() == 0:
                continue
            Xc = X[mask]
            Nc = Xc.shape[0]
            mean_c = Xc.mean(dim=0)

            self.class_sums[cls] += Xc.sum(dim=0)
            self.class_counts[cls] += Nc

            Xc_centered = Xc - mean_c
            scatter_c = Xc_centered.t().matmul(Xc_centered)
            self.within_class_scatter += scatter_c


    def finalize(self, lamb=1e-4):
        means = self.class_sums / self.class_counts.unsqueeze(1)

        total_samples = self.class_counts.sum()
        overall_mean = self.class_sums.sum(dim=0) / total_samples

        Sb = torch.zeros((self.n_features, self.n_features), device=self.device)
        for cls in range(self.n_classes):
            Nc = self.class_counts[cls]
            if Nc > 0:
                mean_diff = (means[cls] - overall_mean).unsqueeze(1)
                Sb += Nc * mean_diff.matmul(mean_diff.t())

        Sw = self.within_class_scatter + lamb * torch.eye(self.n_features, device=self.device)

        return Sw, Sb, means


def spherical_lda(X, y, n_classes, lamb):
    N, D = X.shape
    labels, counts = torch.unique(y, return_counts=True)
    assert len(labels) == n_classes  # require all classes to be present
    
    # Compute global mean direction and normalize
    global_mean = torch.mean(X, 0)
    global_mean = F.normalize(global_mean, p=2, dim=0)
    
    # Initialize containers
    class_means_list = []
    Sw = torch.zeros((D, D), dtype=X.dtype, device=X.device)
    Sb = torch.zeros((D, D), dtype=X.dtype, device=X.device)
    
    # Calculate all class means
    for c in labels:
        class_idx = int(c)
        Xc = X[y == c]
        class_mean = F.normalize(torch.mean(Xc, dim=0), p=2, dim=0)
        class_means_list.append(class_mean)
    
    Xc_mean = torch.stack(class_means_list)
    
    # Compute scatter matrices
    for i, (c, Nc) in enumerate(zip(labels, counts)):
        Xc = X[y == c]
        class_mean = Xc_mean[i]

        # Vectorized cosine similarities
        cos_similarities = Xc @ class_mean
        cos_similarities = torch.clamp(cos_similarities, -1.0 + 1e-6, 1.0 - 1e-6)

        # Vectorized difference: samples projected away from mean direction
        diffs = Xc - cos_similarities.unsqueeze(1) * class_mean.unsqueeze(0)

        # Vectorized scatter matrix
        class_scatter = diffs.T @ diffs

        Sw = Sw + class_scatter
    
    Sw = Sw / N  # Normalize by total number of samples
    
    # Compute between-class scatter
    for i, (c, Nc) in enumerate(zip(labels, counts)):
        class_mean = Xc_mean[i]
        cos_sim = torch.dot(class_mean, global_mean)
        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)

        diff = class_mean.unsqueeze(1) - cos_sim * global_mean.unsqueeze(1)
        diff_outer = diff @ diff.T
        Sb = Sb + (Nc / N) * diff_outer

    
    # import numpy as np
    # from sklearn.covariance import LedoitWolf
    
    # X_np = Sw.cpu().detach().numpy()

    # lw = LedoitWolf()
    # lw.fit(X_np)
    # shrinkage = lw.shrinkage_
    # mu = torch.trace(Sw) / D
    # shrinkage = torch.nn.Parameter(torch.tensor(0.0, dtype=Sw.dtype, device=Sw.device))
    # shrinkage = torch.sigmoid(shrinkage)
    
    Sw_reg = Sw + torch.eye(D, dtype=X.dtype, device=X.device) * lamb
    #Sw_reg = (1-shrinkage) * Sw + torch.eye(D, dtype=X.dtype, device=X.device, requires_grad=False) * shrinkage * mu
    
    
    # Generalized eigenvalue problem
    temp = torch.linalg.pinv(Sw_reg, hermitian=True) @ Sb
    
    # Eigen decomposition
    evals_complex, evecs_complex = torch.linalg.eig(temp)
    tol = 1e-6
    is_complex = torch.abs(evals_complex.imag) > tol
    hasComplexEVal = torch.any(is_complex)
    
    if hasComplexEVal:
        print(f"Warning: Found {torch.sum(is_complex)} eigenvalues with imaginary part > {tol}. Keeping only real ones.")
    
    real_idx = ~is_complex
    evals = evals_complex[real_idx].real
    evecs = evecs_complex[:, real_idx].real
    
    if evals.numel() > 0:
        evals, inc_idx = torch.sort(evals)
        evecs = evecs[:, inc_idx]
    else:
        print("Warning: All eigenvalues were complex.")
        evals = torch.tensor([], dtype=temp.dtype, device=temp.device)
        evecs = torch.zeros((D, 0), dtype=temp.dtype, device=temp.device)
    
    return hasComplexEVal, Xc_mean, evals, evecs, temp
    
def triangle_loss(Xc_mean, Sb, Sw, epsilon = 0.15):
    """
    Wasserstein proxy loss:
    Penalizes deviation of class means from 0 and total scatter matrix from (1/n) * I.

    Args:
        Xc_mean: (n_classes, D) tensor of class means
        St: (D, D) total scatter matrix
        n_classes: int, number of classes (used to scale identity)

    Returns:
        Scalar proxy Wasserstein^2 loss
    """
    D = Sw.shape[0]
    device = Sw.device

    # 1. Mean penalty: encourage mean of class means to be near 0
    mu = Xc_mean.mean(dim=0)  # (D,)
    mean_term = torch.sum(mu ** 2)

    # 2. Frobenius norm penalty: encourage St ≈ (1/n) * I
    target = (1.0 / D) * torch.eye(D, device=device)
    frob_term_b = torch.norm(Sb - epsilon * target, p='fro') ** 0.1
    frob_term_w = torch.norm(Sw - (1-epsilon) * target, p='fro') **0.1
    

    return mean_term + frob_term_b + frob_term_w
    
def wasserstein_proxy_loss(Xc_mean, St):
    """
    Wasserstein proxy loss:
    Penalizes deviation of class means from 0 and total scatter matrix from (1/n) * I.

    Args:
        Xc_mean: (n_classes, D) tensor of class means
        St: (D, D) total scatter matrix
        n_classes: int, number of classes (used to scale identity)

    Returns:
        Scalar proxy Wasserstein^2 loss
    """
    D = St.shape[0]
    device = St.device

    # 1. Mean penalty: encourage mean of class means to be near 0
    mu = Xc_mean.mean(dim=0)  # (D,)
    mean_term = torch.sum(mu ** 2)

    # 2. Frobenius norm penalty: encourage St ≈ (1/n) * I
    target = (1.0 / D) * torch.eye(D, device=device)
    frob_term = torch.norm(St - target, p='fro') ** 2

    return mean_term
    #return mean_term + frob_term
    
def wasserstein_loss(Xc_mean, St):
    """
    Computes the squared 2-Wasserstein distance between 
    N(mu, St) and N(0, 1/d * I), where d = dim.

    Args:
        Xc_mean: (n_classes, D) tensor of class means
        St: (D, D) total scatter matrix

    Returns:
        Scalar Wasserstein^2 loss
    """
    D = St.shape[0]
    device = St.device

    # 1. Mean penalty
    mu = Xc_mean.mean(dim=0)
    mean_term = torch.sum(mu ** 2)

    # 2. Trace of covariance
    trace_term = torch.trace(St)

    # 3. Approximate trace of sqrt of covariance using partial eigendecomposition
    
    
    with torch.cuda.amp.autocast(enabled=False):
        k = 100 #D // 3  # <= D // 3 required by torch.lobpcg
        St_fp32 = St.to(dtype=torch.float32)
        eps = 1e-4
        St_fp32 = St.to(dtype=torch.float32) + eps * torch.eye(St.shape[0], device=St.device)
        init = torch.randn(St_fp32.shape[0], k, device=St.device, dtype=torch.float32)
        eigvals, _ = torch.lobpcg(St_fp32, k=k, X=init, niter=100, largest=True, method="ortho")
        sqrt_trace = torch.sum(torch.sqrt(torch.clamp(eigvals, min=1e-6)))

    # Wasserstein^2
    wasserstein_sq = mean_term + trace_term - (2 / D**0.5) * sqrt_trace

    return wasserstein_sq

    
def kl_divergence_loss(Xc_mean, St):
    """
    Computes the KL divergence (up to constants) between
    N(mu, St) and N(0, 1/D * I), where D = dim.

    Args:
        Xc_mean: (n_classes, D) tensor of class means
        St: (D, D) total scatter (covariance) matrix
    Returns:
        Scalar KL loss (up to constants)
    """
    D = St.shape[0]
    n = D  # target is N(0, 1/D * I)

    mu = Xc_mean.mean(dim=0)  # (D,)
    mean_term = n * torch.sum(mu ** 2)

    trace_term = n * torch.trace(St)

    # Cholesky-based log-determinant
    chol = torch.linalg.cholesky(St)
    log_det_term = -2 * torch.log(chol.diagonal()).sum()

    kl_loss = 0.5 * (trace_term + mean_term + log_det_term)
    return kl_loss

class SphericalLDA(nn.Module):
    def __init__(self, n_classes, lamb=1e-4):
        super(SphericalLDA, self).__init__()
        self.n_classes = n_classes
        self.n_components = n_classes - 1  # Maximum meaningful LDA dimensions
        self.lamb = lamb
        self.lda_layer = partial(spherical_lda, n_classes=n_classes, lamb=lamb)
    
    def forward(self, X, y):
        hasComplexEVal, Xc_mean, evals, evecs, sigma_w_inv_b = self.lda_layer(X, y)
        
        # Store projection matrix 
        self.scalings_ = evecs
        
        # Project class means and normalize to create prototypes
        projected_means = Xc_mean.matmul(evecs)
        
        # Project back to original space and normalize to ensure they're on the hypersphere
        self.coef_ = F.normalize(projected_means.matmul(evecs.t()), p=2, dim=1)
        
        # Intercept is not meaningful in spherical space when using cosine similarity
        self.intercept_ = torch.zeros(self.n_classes, dtype=X.dtype, device=X.device)
        
        return hasComplexEVal, evals, sigma_w_inv_b
    
    def transform(self, X):
        # Normalize input
        #X = F.normalize(X.view(X.shape[0], -1), p=2, dim=1)
        
        # Project data
        X_new = X.matmul(self.scalings_)
        
        # Return only the most discriminative components
        return X_new[:, :self.n_components]
    
    def predict(self, X):
        # Normalize input embeddings
        #X = F.normalize(X.view(X.shape[0], -1), p=2, dim=1)
        
        # Compute cosine similarities with class prototypes
        similarities = X.matmul(self.coef_.t())
        
        # Return class with highest similarity
        return torch.argmax(similarities, dim=1)
    
    def predict_proba(self, X):
        #X = F.normalize(X.view(X.shape[0], -1), p=2, dim=1)
        similarities = X.matmul(self.coef_.t())
        
        # Convert similarities to probabilities using softmax
        proba = nn.functional.softmax(similarities, dim=1)
        return proba
    
    def predict_log_proba(self, X):
        #X = F.normalize(X.view(X.shape[0], -1), p=2, dim=1)
        similarities = X.matmul(self.coef_.t())
        log_proba = nn.functional.log_softmax(similarities, dim=1)
        return log_proba

if __name__ == '__main__':
    import numpy as np
    np.set_printoptions(precision=4, suppress=True)
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score

    features, labels = load_iris(return_X_y=True)
    print(features.shape, labels.shape)

    n_classes = 3
    n_components = n_classes - 1
    N, D = features.shape  # 150, 4
    lamb = 0.0005
    n_eig = 2
    margin = 0.01

    device = torch.device('cpu:0')
    X = torch.from_numpy(features).to(device)
    y = torch.from_numpy(labels).to(device)

    lda = LDA(n_classes, lamb)
    _, evals = lda(X, y)

    # calculate lda loss
    loss = lda_loss(evals, n_classes, n_eig, margin)
    loss.backward()
    print('finished backward')

    # use LDA as classifier
    y_pred = lda.predict(X)
    print('accuracy on training data', accuracy_score(y, y_pred))