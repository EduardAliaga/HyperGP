import torch

def rbf_kernel(X1, X2, ell, sf, jitter=1e-6):
    """
    Radial Basis Function (RBF) kernel with anisotropic lengthscales.
    
    Args:
        X1 (torch.Tensor): First set of points [n1, D]
        X2 (torch.Tensor): Second set of points [n2, D]
        ell (torch.Tensor): Vector of lengthscales, one per dimension [D]
        sf (torch.Tensor): Signal amplitude (scalar)
        jitter (float, optional): Small constant for numerical stability
        
    Returns:
        torch.Tensor: Kernel matrix [n1, n2]
    """
    # Reshape lengthscales for broadcasting
    ell = ell.view(1, 1, -1)  # [1, 1, D]
    
    # Calculate pairwise distances, scaled by lengthscales
    diff = (X1.unsqueeze(1) - X2.unsqueeze(0)) / ell  # [n1, n2, D]
    
    # Compute RBF kernel
    result = sf**2 * torch.exp(-0.5 * (diff**2).sum(-1))  # [n1, n2]
    
    # Add jitter to diagonal if X1 and X2 are the same set of points
    if X1.shape == X2.shape and torch.allclose(X1, X2):
        n = X1.shape[0]
        result = result + jitter * torch.eye(n, device=X1.device)
    
    return result

def matern_kernel(X1, X2, ell, sf, nu=1.5, jitter=1e-6):
    """
    Matérn kernel with anisotropic lengthscales.
    
    Args:
        X1 (torch.Tensor): First set of points [n1, D]
        X2 (torch.Tensor): Second set of points [n2, D]
        ell (torch.Tensor): Vector of lengthscales, one per dimension [D]
        sf (torch.Tensor): Signal amplitude (scalar)
        nu (float, optional): Smoothness parameter (1/2, 3/2, 5/2)
        jitter (float, optional): Small constant for numerical stability
        
    Returns:
        torch.Tensor: Kernel matrix [n1, n2]
    """
    # Implementation for Matérn kernel
    # This is a placeholder for potential future implementation
    # Currently returning the RBF kernel
    return rbf_kernel(X1, X2, ell, sf, jitter)

def solve_gp_system(K_ss, Y_s, K_sq, jitter=1e-6):
    """
    Solves the GP system to get posterior mean predictions.
    Tries Cholesky decomposition first, falls back to direct solve if needed.
    
    Args:
        K_ss (torch.Tensor): Kernel matrix for support points [NK, NK]
        Y_s (torch.Tensor): Support set labels (one-hot) [NK, N_way]
        K_sq (torch.Tensor): Cross-kernel matrix [NK, NQ]
        jitter (float, optional): Small constant for numerical stability
        
    Returns:
        torch.Tensor: Posterior mean at query points [NQ, N_way]
    """
    try:
        # Try Cholesky decomposition first (more stable)
        L = torch.linalg.cholesky(K_ss)
        temp = torch.triangular_solve(Y_s, L, upper=False)[0]
        alpha = torch.triangular_solve(temp, L.T, upper=True)[0]
    except Exception as e:
        # If Cholesky fails, fall back to direct solve
        print(f"Cholesky failed, using direct solve: {str(e)}")
        alpha = torch.linalg.solve(K_ss, Y_s)
    
    # Compute posterior mean
    mu_q = K_sq.T @ alpha
    
    return mu_q