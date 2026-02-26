#!/usr/bin/env python3
"""
Test script to verify Equation 7 implementation
σ_{i,h} = exp(-(N-i)/(β_1*N) - (H-h)/(β_2*H))
"""

import torch
import math
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(__file__))

from diffusion_sampler import DiffusionVarianceScheduler

def test_equation7():
    print("="*80)
    print("Testing DIAL-MPC Equation 7 Implementation")
    print("σ_{i,h} = exp(-(N-i)/(β_1*N) - (H-h)/(β_2*H))")
    print("="*80)
    
    # Test configuration
    config = {
        'n_diffuse': 4,
        'n_diffuse_init': 10,
        'beta_1': 1.0,
        'beta_2': 1.0,
        'sigma_base': 1.0,
        'sigma_min': 0.01,
    }
    
    N = config['n_diffuse']
    H = 30
    beta_1 = config['beta_1']
    beta_2 = config['beta_2']
    
    print(f"\nParameters:")
    print(f"  N (iterations) = {N}")
    print(f"  H (horizon) = {H}")
    print(f"  β_1 = {beta_1}")
    print(f"  β_2 = {beta_2}")
    print(f"  sigma_base = {config['sigma_base']}")
    
    tensor_args = {'device': 'cpu', 'dtype': torch.float32}
    scheduler = DiffusionVarianceScheduler(config, horizon=H, d_action=2, tensor_args=tensor_args)
    
    print("\n" + "-"*80)
    print("Variance Schedule across iterations (h=0, h=15, h=29):")
    print("-"*80)
    print(f"{'i':>3} | {'h=0':>10} | {'h=15':>10} | {'h=29':>10}")
    print("-"*80)
    
    for i in range(N):
        sigma = scheduler.get_variance_schedule(i, is_first_step=False)
        print(f"{i:3d} | {sigma[0, 0].item():10.4f} | {sigma[15, 0].item():10.4f} | {sigma[29, 0].item():10.4f}")
    
    print("\n" + "-"*80)
    print("Manual Verification (selected cases):")
    print("-"*80)
    
    # Test case 1: i=0, h=0 (minimum variance)
    i, h = 0, 0
    exponent = -(N-i)/(beta_1*N) - (H-h)/(beta_2*H)
    manual_sigma = math.exp(exponent) * config['sigma_base']
    auto_sigma = scheduler.get_variance_schedule(i, is_first_step=False)[h, 0].item()
    
    print(f"\nCase 1: i={i}, h={h} (early iteration, current timestep)")
    print(f"  Exponent: -({N}-{i})/({beta_1}*{N}) - ({H}-{h})/({beta_2}*{H})")
    print(f"          = -{(N-i)/(beta_1*N):.4f} - {(H-h)/(beta_2*H):.4f} = {exponent:.4f}")
    print(f"  Manual σ: exp({exponent:.4f}) = {manual_sigma:.4f}")
    print(f"  Auto σ:   {auto_sigma:.4f}")
    print(f"  Match: {'✓' if abs(manual_sigma - auto_sigma) < 1e-6 else '✗'}")
    
    # Test case 2: i=3, h=29 (maximum variance)
    i, h = 3, 29
    exponent = -(N-i)/(beta_1*N) - (H-h)/(beta_2*H)
    manual_sigma = math.exp(exponent) * config['sigma_base']
    auto_sigma = scheduler.get_variance_schedule(i, is_first_step=False)[h, 0].item()
    
    print(f"\nCase 2: i={i}, h={h} (late iteration, far timestep)")
    print(f"  Exponent: -({N}-{i})/({beta_1}*{N}) - ({H}-{h})/({beta_2}*{H})")
    print(f"          = -{(N-i)/(beta_1*N):.4f} - {(H-h)/(beta_2*H):.4f} = {exponent:.4f}")
    print(f"  Manual σ: exp({exponent:.4f}) = {manual_sigma:.4f}")
    print(f"  Auto σ:   {auto_sigma:.4f}")
    print(f"  Match: {'✓' if abs(manual_sigma - auto_sigma) < 1e-6 else '✗'}")
    
    # Test case 3: i=2, h=15 (middle)
    i, h = 2, 15
    exponent = -(N-i)/(beta_1*N) - (H-h)/(beta_2*H)
    manual_sigma = math.exp(exponent) * config['sigma_base']
    auto_sigma = scheduler.get_variance_schedule(i, is_first_step=False)[h, 0].item()
    
    print(f"\nCase 3: i={i}, h={h} (middle iteration, middle timestep)")
    print(f"  Exponent: -({N}-{i})/({beta_1}*{N}) - ({H}-{h})/({beta_2}*{H})")
    print(f"          = -{(N-i)/(beta_1*N):.4f} - {(H-h)/(beta_2*H):.4f} = {exponent:.4f}")
    print(f"  Manual σ: exp({exponent:.4f}) = {manual_sigma:.4f}")
    print(f"  Auto σ:   {auto_sigma:.4f}")
    print(f"  Match: {'✓' if abs(manual_sigma - auto_sigma) < 1e-6 else '✗'}")
    
    print("\n" + "="*80)
    print("Key Properties of Equation 7:")
    print("="*80)
    print("1. Variance increases with iteration i (from small to large)")
    print("2. Variance increases with timestep h (from small to large)")
    print("3. Early iterations + near timesteps → minimum variance (precise)")
    print("4. Late iterations + far timesteps → maximum variance (flexible)")
    print("="*80)

if __name__ == '__main__':
    test_equation7()
