#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
# Modified for diffusion-inspired sampling based on DIAL-MPC paper.
#

"""
Diffusion-Inspired Sampling for MPPI

This module implements the diffusion-inspired variance scheduling from DIAL-MPC paper.
The key insight is to use a two-level variance schedule following Equation 7:

σ_{i,h} = exp(-(N-i)/(β_1*N) - (H-h)/(β_2*H)) * I

where:
- N = total optimization iterations
- i = current iteration (0 to N-1)
- H = planning horizon
- h = timestep (0 to H-1)
- β_1 = iteration-level annealing parameter
- β_2 = horizon-level annealing parameter

This is from Equation (7) in the DIAL-MPC paper:
"Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing"

Reference: https://github.com/LeCAR-Lab/dial-mpc
"""

import torch
import numpy as np
from scipy.interpolate import BSpline
import scipy.interpolate as si
from torch.distributions.multivariate_normal import MultivariateNormal


def bspline(c_arr, t_arr=None, n=100, degree=3):
    """
    B-spline fitting (same as STORM's implementation)
    
    Args:
        c_arr: Data points to fit (torch.Tensor)
        t_arr: Parameter positions (default: uniform)
        n: Number of output samples
        degree: B-spline degree
    
    Returns:
        Fitted curve samples
    """
    sample_device = c_arr.device
    sample_dtype = c_arr.dtype
    cv = c_arr.cpu().numpy()
    count = len(cv)

    if t_arr is None:
        t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
    else:
        t_arr = t_arr.cpu().numpy()
    
    spl = si.splrep(t_arr, cv, k=degree, s=0.5)
    xx = np.linspace(0, cv.shape[0], n)
    samples = si.splev(xx, spl, ext=3)
    samples = torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)
    
    return samples


def generate_halton_samples(n_samples, n_dims, seed_val=0, device='cpu', dtype=torch.float32):
    """
    Generate Halton low-discrepancy sequence samples
    
    Args:
        n_samples: Number of samples
        n_dims: Number of dimensions
        seed_val: Random seed for Halton sequence
        device: torch device
        dtype: torch dtype
    
    Returns:
        Uniform samples in [0, 1]^n_dims
    """
    try:
        import ghalton
        sequencer = ghalton.GeneralizedHalton(n_dims, seed_val)
        samples = np.array(sequencer.get(n_samples))
    except ImportError:
        # Fallback to simple Halton sequence
        def halton_sequence(index, base):
            result = 0
            f = 1.0 / base
            i = index
            while i > 0:
                result += f * (i % base)
                i = i // base
                f = f / base
            return result
        
        # First n_dims primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        if n_dims > len(primes):
            primes = primes + list(range(73, 73 + n_dims - len(primes)))
        
        samples = np.zeros((n_samples, n_dims))
        for i in range(n_samples):
            for j in range(n_dims):
                samples[i, j] = halton_sequence(i + seed_val + 1, primes[j])
    
    return torch.tensor(samples, device=device, dtype=dtype)


def uniform_to_gaussian(uniform_samples):
    """
    Convert uniform [0, 1] samples to Gaussian N(0, 1) via inverse CDF
    
    Uses the inverse error function: x = sqrt(2) * erfinv(2*u - 1)
    
    Args:
        uniform_samples: Tensor of uniform samples in [0, 1]
    
    Returns:
        Gaussian samples ~ N(0, 1)
    """
    # Clamp to avoid numerical issues at boundaries
    uniform_samples = torch.clamp(uniform_samples, 1e-6, 1 - 1e-6)
    # Inverse CDF of standard normal
    gaussian_samples = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * uniform_samples - 1)
    return gaussian_samples


class DiffusionVarianceScheduler:
    """
    Implements the diffusion-inspired variance scheduling from DIAL-MPC.
    
    Follows Equation 7 from the DIAL-MPC paper:
    σ_{i,h} = exp(-(N-i)/(β_1*N) - (H-h)/(β_2*H)) * I
    
    where:
    - N = total number of iterations
    - i = current iteration (0 to N-1)
    - H = planning horizon
    - h = timestep (0 to H-1)
    - β_1 = iteration-level annealing parameter
    - β_2 = horizon-level annealing parameter
    
    This creates a "coarse-to-fine" optimization where:
    - Early iterations (i=0) have small variance
    - Later iterations (i=N-1) have large variance
    - Early timesteps (h=0) have small variance
    - Later timesteps (h=H-1) have large variance
    """
    
    def __init__(self, config, horizon, d_action, tensor_args):
        """
        Args:
            config: Diffusion configuration dict
            horizon: Planning horizon H
            d_action: Action dimension
            tensor_args: {'device': ..., 'dtype': ...}
        """
        self.horizon = horizon
        self.d_action = d_action
        self.tensor_args = tensor_args
        
        # Extract config parameters
        self.n_diffuse = config.get('n_diffuse', 4)
        self.n_diffuse_init = config.get('n_diffuse_init', 10)
        # β_1 and β_2 from Equation 7
        self.beta_1 = config.get('beta_1', 1.0)  # iteration-level annealing
        self.beta_2 = config.get('beta_2', 1.0)  # horizon-level annealing
        self.sigma_base = config.get('sigma_base', 1.0)
        self.sigma_min = config.get('sigma_min', 0.01)
        self.temp_sample = config.get('temp_sample', 0.05)
        
        # Pre-compute horizon indices for efficiency
        self.h_indices = torch.arange(horizon, **tensor_args)
        # Shape: [H]
        
    def get_variance_schedule(self, iteration, is_first_step=False):
        """
        Get the variance for a given diffusion iteration using Equation 7.
        
        σ_{i,h} = exp(-(N-i)/(β_1*N) - (H-h)/(β_2*H))
        
        Args:
            iteration: Current iteration index (0 to N-1)
            is_first_step: Whether this is the first control step
        
        Returns:
            sigma: Variance tensor of shape [H, d_action]
        """
        N = self.n_diffuse_init if is_first_step else self.n_diffuse
        H = self.horizon
        
        # Equation 7: exp(-(N-i)/(β_1*N) - (H-h)/(β_2*H))
        # iteration_term: -(N-i)/(β_1*N)
        iteration_term = -(N - iteration) / (self.beta_1 * N)
        
        # horizon_term: -(H-h)/(β_2*H) for each h
        horizon_term = -(H - self.h_indices) / (self.beta_2 * H)  # Shape: [H]
        
        # Combined exponential
        exponent = iteration_term + horizon_term  # Shape: [H]
        sigma_h = self.sigma_base * torch.exp(exponent)  # Shape: [H]
        
        # Clamp to minimum
        sigma_h = torch.clamp(sigma_h, min=self.sigma_min)
        
        # Expand to [H, d_action]
        sigma = sigma_h.unsqueeze(-1).expand(-1, self.d_action)
        
        return sigma
    
    def get_all_variance_schedules(self, is_first_step=False):
        """
        Pre-compute all variance schedules for all iterations.
        
        Returns:
            List of variance tensors, one per iteration
        """
        n_total = self.n_diffuse_init if is_first_step else self.n_diffuse
        schedules = []
        for i in range(n_total):
            schedules.append(self.get_variance_schedule(i, is_first_step))
        return schedules


class DiffusionKnotSampleLib:
    """
    B-spline knot sampling with diffusion-inspired variance scheduling.
    
    Follows the same flow as STORM:
    1. Uniform Halton → Inverse CDF → B-spline fitting → Variance scaling → Add to mean → Clamp
    
    But with variance scaled according to DIAL-MPC's diffusion schedule.
    """
    
    def __init__(self, horizon, d_action, n_knots, degree=2, seed=0,
                 tensor_args={'device': 'cpu', 'dtype': torch.float32},
                 diffusion_config=None):
        """
        Args:
            horizon: Planning horizon H
            d_action: Action dimension
            n_knots: Number of B-spline data points (M)
            degree: B-spline degree
            seed: Random seed
            tensor_args: Tensor arguments
            diffusion_config: Diffusion variance scheduling config
        """
        self.horizon = horizon
        self.d_action = d_action
        self.n_knots = n_knots
        self.degree = degree
        self.seed_val = seed
        self.tensor_args = tensor_args
        self.ndims = n_knots * d_action
        
        # Initialize diffusion variance scheduler
        if diffusion_config is None:
            diffusion_config = {
                'n_diffuse': 4,
                'n_diffuse_init': 10,
                'beta_1': 1.0,  # iteration-level annealing (Equation 7)
                'beta_2': 1.0,  # horizon-level annealing (Equation 7)
                'sigma_base': 1.0,
                'sigma_min': 0.01,
                'temp_sample': 0.05
            }
        self.diffusion_config = diffusion_config
        self.variance_scheduler = DiffusionVarianceScheduler(
            diffusion_config, horizon, d_action, tensor_args
        )
        
        # Pre-generate Halton samples (fixed for reproducibility)
        self.halton_samples = None
        self.current_sample_shape = None
        
    def _generate_base_samples(self, n_samples):
        """
        Generate base Halton samples and convert to Gaussian.
        
        Returns:
            Gaussian samples of shape [n_samples, n_knots, d_action]
        """
        # Step 1: Uniform Halton samples in [0, 1]^(M*A)
        uniform_samples = generate_halton_samples(
            n_samples, self.ndims, 
            seed_val=self.seed_val,
            device=self.tensor_args['device'],
            dtype=self.tensor_args['dtype']
        )
        
        # Step 2: Inverse CDF → Gaussian N(0, 1)
        gaussian_samples = uniform_to_gaussian(uniform_samples)
        
        # Reshape to [N, d_action, n_knots]
        gaussian_samples = gaussian_samples.view(n_samples, self.d_action, self.n_knots)
        
        return gaussian_samples
    
    def _bspline_interpolate(self, knot_samples):
        """
        Interpolate knot samples to full horizon using B-spline.
        
        Args:
            knot_samples: [n_samples, d_action, n_knots]
        
        Returns:
            Interpolated samples: [n_samples, horizon, d_action]
        """
        n_samples = knot_samples.shape[0]
        samples = torch.zeros(
            (n_samples, self.horizon, self.d_action), 
            **self.tensor_args
        )
        
        for i in range(n_samples):
            for j in range(self.d_action):
                samples[i, :, j] = bspline(
                    knot_samples[i, j, :],
                    n=self.horizon,
                    degree=self.degree
                )
        
        return samples
    
    def get_samples(self, sample_shape, iteration=0, is_first_step=False, **kwargs):
        """
        Generate samples with diffusion-inspired variance.
        
        Args:
            sample_shape: Tuple containing number of samples (N,)
            iteration: Current diffusion iteration index
            is_first_step: Whether this is the first control step
        
        Returns:
            Scaled noise samples of shape [N, H, A]
        """
        n_samples = sample_shape[0]
        
        # Generate or reuse base Halton-Gaussian samples
        if self.halton_samples is None or self.current_sample_shape != n_samples:
            self.halton_samples = self._generate_base_samples(n_samples)
            self.current_sample_shape = n_samples
        
        # B-spline interpolation: [N, d_action, n_knots] → [N, H, d_action]
        base_samples = self._bspline_interpolate(self.halton_samples)
        
        # Get variance for this iteration
        sigma = self.variance_scheduler.get_variance_schedule(iteration, is_first_step)
        # sigma shape: [H, d_action]
        
        # Scale samples by variance: samples * σ
        scaled_samples = base_samples * sigma.unsqueeze(0)
        
        return scaled_samples


class DiffusionMPPISampler:
    """
    Complete diffusion-inspired MPPI sampler.
    
    Implements the full sampling pipeline with diffusion variance scheduling:
    1. Halton sequence generation in knot space
    2. Inverse CDF transformation to Gaussian
    3. B-spline interpolation
    4. Diffusion-inspired variance scaling (iteration + horizon level)
    5. Add to mean action
    6. Clamp to action limits
    """
    
    def __init__(self, horizon, d_action, n_particles, knot_scale=5, degree=2,
                 seed=0, tensor_args={'device': 'cpu', 'dtype': torch.float32},
                 diffusion_config=None, action_lows=None, action_highs=None):
        """
        Args:
            horizon: Planning horizon H
            d_action: Action dimension A
            n_particles: Number of particles N
            knot_scale: knot_scale parameter (M = H // knot_scale)
            degree: B-spline degree
            seed: Random seed
            tensor_args: Tensor arguments
            diffusion_config: Diffusion configuration
            action_lows: Lower action bounds
            action_highs: Upper action bounds
        """
        self.horizon = horizon
        self.d_action = d_action
        self.n_particles = n_particles
        self.n_knots = horizon // knot_scale
        self.tensor_args = tensor_args
        
        # Action limits
        if action_lows is None:
            action_lows = -torch.ones(d_action, **tensor_args)
        if action_highs is None:
            action_highs = torch.ones(d_action, **tensor_args)
        self.action_lows = action_lows
        self.action_highs = action_highs
        
        # Initialize diffusion config
        if diffusion_config is None:
            diffusion_config = {
                'n_diffuse': 4,
                'n_diffuse_init': 10,
                'beta_1': 1.0,  # iteration-level annealing (Equation 7)
                'beta_2': 1.0,  # horizon-level annealing (Equation 7)
                'sigma_base': 1.0,
                'sigma_min': 0.01,
                'temp_sample': 0.05
            }
        self.diffusion_config = diffusion_config
        
        # Create knot sample library
        self.knot_sampler = DiffusionKnotSampleLib(
            horizon=horizon,
            d_action=d_action,
            n_knots=self.n_knots,
            degree=degree,
            seed=seed,
            tensor_args=tensor_args,
            diffusion_config=diffusion_config
        )
        
        # Store current iteration state
        self.current_iteration = 0
        self.is_first_step = True
        
    def sample_actions(self, mean_action, cov_scale=None, iteration=None):
        """
        Sample action sequences around the mean with diffusion variance.
        
        Args:
            mean_action: Mean action sequence [H, A]
            cov_scale: Optional additional covariance scale (from STORM's init_cov)
            iteration: Override diffusion iteration
        
        Returns:
            Sampled actions [N, H, A]
        """
        if iteration is None:
            iteration = self.current_iteration
        
        # Get base samples with diffusion variance
        noise = self.knot_sampler.get_samples(
            sample_shape=(self.n_particles,),
            iteration=iteration,
            is_first_step=self.is_first_step
        )
        
        # Apply additional covariance scale if provided
        if cov_scale is not None:
            noise = noise * cov_scale
        
        # Add to mean: a = μ + δ
        mean_expanded = mean_action.unsqueeze(0).expand(self.n_particles, -1, -1)
        actions = mean_expanded + noise
        
        # Clamp to action limits
        actions = torch.clamp(
            actions,
            self.action_lows.view(1, 1, -1),
            self.action_highs.view(1, 1, -1)
        )
        
        return actions
    
    def step_iteration(self):
        """Advance to next diffusion iteration."""
        n_total = (self.diffusion_config['n_diffuse_init'] if self.is_first_step 
                   else self.diffusion_config['n_diffuse'])
        self.current_iteration = min(self.current_iteration + 1, n_total - 1)
    
    def reset_iteration(self):
        """Reset iteration counter for new control step."""
        self.current_iteration = 0
        self.is_first_step = False
    
    def set_first_step(self):
        """Mark as first step (use more iterations)."""
        self.is_first_step = True
        self.current_iteration = 0
    
    def get_n_iterations(self):
        """Get number of diffusion iterations for current step."""
        return (self.diffusion_config['n_diffuse_init'] if self.is_first_step 
                else self.diffusion_config['n_diffuse'])
