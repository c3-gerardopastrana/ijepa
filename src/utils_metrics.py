import torch

def get_lidar_properties(z, get_lidar_matrices, num_samples, num_context):
    """
    Compute LDA projections and diagnostic statistics from within-class and between-class scatter matrices.

    Args:
        z: Tensor of shape (num_samples, num_context, feature_dim)
        get_lidar_matrices: Function that returns (sigma_b, sigma_w, sigma_w_inv_b)
        num_samples: Number of unique samples/classes
        num_context: Number of augmentations per class
        del_sigma_augs: Whether to delete augmentations when computing sigma_w

    Returns:
        Dictionary of diagnostics and statistics.
    """
    sigma_b, sigma_w, sigma_w_inv_b = get_lidar_matrices(z, num_samples, num_context)

    # Eigendecomposition
    evals_complex, evecs_complex = torch.linalg.eig(sigma_w_inv_b)
    tol = 1e-6
    is_complex = evals_complex.imag.abs() > tol
    complex_count = is_complex.sum().item()

    # Get real components
    is_real = ~is_complex
    evals = evals_complex[is_real].real
    evecs = evecs_complex[:, is_real].real

    # LDA projection
    projections = z.reshape(num_samples * num_context, -1) @ evecs
    labels = torch.repeat_interleave(torch.arange(num_samples), num_context).to(z.device)

    # Centroids and classification
    unique_labels = torch.unique(labels)
    centroids = torch.stack([
        projections[labels == label].mean(0)
        for label in unique_labels
    ])
    dists = torch.cdist(projections, centroids)
    preds = dists.argmin(dim=1)
    accuracy = (preds == labels).float().mean().item()

    def eigen_stats_from_symmetric_matrix(mat: torch.Tensor, eps: float = 1e-10) -> dict:
        eigs = torch.linalg.eigvalsh(mat)
        if eigs.numel() == 0 or eigs.sum() <= eps:
            return {key: 0.0 for key in [
                "entropy", "max_normalized_eigenvalue", "min_normalized_eigenvalue",
                "quantile_25", "quantile_50", "quantile_75"
            ]}
        eigs_norm = eigs / eigs.sum()
        eigs_norm = torch.clamp(eigs_norm, min=eps)
        return {
            "entropy": -(eigs_norm * eigs_norm.log()).sum().item(),
            "max_normalized_eigenvalue": eigs_norm.max().item(),
            "min_normalized_eigenvalue": eigs_norm.min().item(),
            "quantile_25": torch.quantile(eigs_norm, 0.25).item(),
            "quantile_50": torch.quantile(eigs_norm, 0.5).item(),
            "quantile_75": torch.quantile(eigs_norm, 0.75).item(),
        }

    def compute_diag_offdiag_values(mat: torch.Tensor):
        diag = torch.diagonal(mat)
        off_diag = mat - torch.diag(diag)
        sum_squared_off_diag = (off_diag ** 2).sum().item()
        diag_var = torch.var(diag, unbiased=False).item()
        return off_diag, sum_squared_off_diag, diag_var

    # Norms of class means
    z_reshaped = z.view(num_samples, num_context, -1).permute(1, 0, 2)
    object_activations = z_reshaped.mean(dim=1, keepdim=True)
    mean_activations = object_activations.mean(dim=0, keepdim=True)
    norms = torch.norm(mean_activations, dim=1)
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()
    centered = torch.norm(mean_activations.mean(dim=0)).item()

    # Eigenvalue stats
    stats = eigen_stats_from_symmetric_matrix(sigma_w_inv_b)
    entropy_w = eigen_stats_from_symmetric_matrix(sigma_w)['entropy']
    entropy_b = eigen_stats_from_symmetric_matrix(sigma_b)['entropy']
    entropy_t = eigen_stats_from_symmetric_matrix(sigma_w + sigma_b)['entropy']

    # Off-diagonal & diag variance
    _, sum_squared_off_diag, diag_var = compute_diag_offdiag_values(sigma_w_inv_b)
    _, sum_squared_off_diag_w, diag_var_w = compute_diag_offdiag_values(sigma_w)
    _, sum_squared_off_diag_b, diag_var_b = compute_diag_offdiag_values(sigma_b)

    return {
        "trace": torch.trace(sigma_w_inv_b).item(),
        "frobenius_norm": torch.trace(sigma_w_inv_b @ sigma_w_inv_b).abs().sqrt().item(),
        "rank_sigma_w": torch.linalg.matrix_rank(sigma_w).item(),
        "rank_sigma_b": torch.linalg.matrix_rank(sigma_b).item(),
        "rank_sigma_inv_b": torch.linalg.matrix_rank(sigma_w_inv_b).item(),
        "condition_sigma_w": torch.linalg.cond(sigma_w).item(),
        "condition_sigma_b": torch.linalg.cond(sigma_b).item(),
        "condition_sigma_inv_b": torch.linalg.cond(sigma_w_inv_b).item(),
        "complex_eigenvalue_count": complex_count,
        "sum_squared_off_diag": sum_squared_off_diag,
        "diag_var": diag_var,
        "supervised_accuracy": accuracy,
        "entropy_w": entropy_w,
        "entropy_b": entropy_b,
        "entropy_t": entropy_t,
        "diag_var_b": diag_var_b,
        "diag_var_w": diag_var_w,
        "trace_b": torch.trace(sigma_b).item(),
        "trace_w": torch.trace(sigma_w).item(),
        "sum_squared_off_diag_w": sum_squared_off_diag_w,
        "sum_squared_off_diag_b": sum_squared_off_diag_b,
        "mean_of_class_means_norms": mean_norm,
        "std_of_class_mean_norms": std_norm,
        "distance_of_mean_class_means_origin": centered,
        **stats
    }
