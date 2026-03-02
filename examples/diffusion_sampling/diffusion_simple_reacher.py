#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
# Modified for diffusion-inspired sampling based on DIAL-MPC paper.
#

"""
DIAL-MPC Simple Reacher: mirrors examples/simple_reacher.py exactly,
but uses DiffusionSimpleTask (DiffusionMPPI) instead of SimpleTask (MPPI).

Reference:
  DIAL-MPC Equation 7: σ_{i,h} = σ_base * exp(-(N-i)/(β₁N) - (H-h)/(β₂H))
"""

import torch
torch.multiprocessing.set_start_method('spawn', force=True)
import copy
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import time
import yaml
import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from storm_kit.util_file import get_configs_path, get_mpc_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.mpc.task.diffusion_simple_task import DiffusionSimpleTask

traj_log = None


def holonomic_robot(args):
    """Main function – follows examples/simple_reacher.py flow exactly."""

    # ── Setup ──────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")

    tensor_args = {'device': device, 'dtype': torch.float32}

    # Load full config (mppi + diffusion + cost + model)
    config_dir = os.path.dirname(__file__)
    config_path = os.path.join(config_dir, 'config', 'diffusion_simple_reacher.yml')
    full_config = load_yaml(config_path)

    # ── Create task (mirrors: simple_task = SimpleTask(...)) ──
    simple_task = DiffusionSimpleTask(
        robot_file="simple_reacher.yml",
        override_config=full_config,
        tensor_args=tensor_args
    )

    goal_state = [0.4, 0.3]
    simple_task.update_params(goal_state=goal_state)

    current_state = {'position': np.array([0.05, 0.2]),
                     'velocity': np.zeros(2) + 0.0}

    i = 0
    exp_params = simple_task.exp_params
    controller = simple_task.controller
    sim_dt = exp_params['control_dt']

    # ── Logging (same structure as STORM simple_reacher + diffusion extras) ──
    global traj_log
    image = controller.rollout_fn.image_collision_cost.world_coll.im
    extents = np.ravel(exp_params['model']['position_bounds'])

    traj_log = {'position': [], 'velocity': [], 'error': [], 'command': [],
                'des': [], 'acc': [], 'world': image, 'bounds': extents,
                # Diffusion-specific logs
                'noise_scale': [],        # avg diffusion noise per step (all iters)
                'storm_scale_tril': [],   # STORM adaptive scale_tril after optimization
                'iteration_costs': [],    # min cost per diffusion iteration
                'best_cost': [],          # best cost of the final iteration
                'variance_schedule': [],  # full variance schedule per step
                }

    zero_acc = np.zeros(2)
    t_step = 0.0

    filtered_state = copy.deepcopy(current_state)
    plan_length = 200

    # ── Main loop (identical to STORM simple_reacher.py) ──
    while i < plan_length:
        current_state = {'position': current_state['position'],
                         'velocity': current_state['velocity'],
                         'acceleration': current_state['position'] * 0.0}
        filtered_state = current_state

        # get_current_error returns (list, _) – same as BaseTask
        error, _ = simple_task.get_current_error(filtered_state)

        command = simple_task.get_command(t_step, filtered_state, sim_dt, WAIT=True)

        if i == 0:
            top_trajs = simple_task.top_trajs
            traj_log['top_traj'] = top_trajs.cpu().numpy()

        # ── Collect diffusion optimization info ──
        opt_info = getattr(simple_task, '_last_opt_info', {})
        # Variance schedule for this step (noise levels per diffusion iter)
        var_sched = opt_info.get('variance_schedule', [])
        traj_log['variance_schedule'].append(var_sched)
        # Average noise across all diffusion iterations this step
        traj_log['noise_scale'].append(np.mean(var_sched) if var_sched else 0.0)
        # STORM adaptive scale_tril (after optimization)
        traj_log['storm_scale_tril'].append(
            getattr(simple_task, '_last_scale_tril', 0.0))
        # Iteration costs (min cost per diffusion iteration)
        iter_costs = opt_info.get('iteration_costs', [])
        traj_log['iteration_costs'].append(iter_costs)
        # Best cost (last iteration's min cost)
        traj_log['best_cost'].append(iter_costs[-1] if iter_costs else 0.0)

        current_state = command

        print(i, command['position'])
        traj_log['position'].append(filtered_state['position'])
        traj_log['error'].append(error)
        traj_log['velocity'].append(filtered_state['velocity'])
        traj_log['command'].append(command['acceleration'])
        traj_log['acc'].append(command['acceleration'])
        traj_log['des'].append(copy.deepcopy(goal_state))
        t_step += sim_dt
        i += 1

    # Always save plots (headless → save to file, otherwise → show)
    plot_traj(traj_log, save_path=os.path.join(config_dir, 'results'),
              headless=args.headless)


def plot_traj(traj_log, save_path=None, headless=False):
    """Comprehensive diagnostic plots for DIAL-MPC diffusion reacher.
    
    Produces 8 subplots:
      1. XY Position vs time (with desired)
      2. XY Position Error vs time
      3. XY Velocity vs time
      4. Diffusion Noise Scale vs time (per-iter + avg + STORM scale_tril)
      5. Best Cost vs time
      6. Acceleration (command) vs time
      7. 2D Trajectory (with world map)
      8. Per-iteration cost evolution (heatmap-style for first N steps)
    """
    position = np.array(traj_log['position'])
    vel = np.array(traj_log['velocity'])
    err = np.array(traj_log['error'])
    acc = np.array(traj_log['acc'])
    des = np.array(traj_log['des'])
    noise_scale = np.array(traj_log['noise_scale'])
    storm_stril = np.array(traj_log['storm_scale_tril'])
    best_cost = np.array(traj_log['best_cost'])
    steps = np.arange(len(position))

    # Compute XY error separately
    err_x = position[:, 0] - des[:, 0]
    err_y = position[:, 1] - des[:, 1]
    err_norm = np.sqrt(err_x**2 + err_y**2)

    fig, axs = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle('DIAL-MPC Diffusion Simple Reacher — Diagnostics', fontsize=14, fontweight='bold')

    # ── (0,0) XY Position ──
    ax = axs[0, 0]
    ax.set_title('Position vs Time')
    ax.plot(steps, position[:, 0], 'r-', linewidth=1.5, label='x')
    ax.plot(steps, position[:, 1], 'b-', linewidth=1.5, label='y')
    ax.axhline(y=des[0, 0], color='r', linestyle='--', alpha=0.5, label=f'x_des={des[0,0]:.2f}')
    ax.axhline(y=des[0, 1], color='b', linestyle='--', alpha=0.5, label=f'y_des={des[0,1]:.2f}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (0,1) XY Position Error ──
    ax = axs[0, 1]
    ax.set_title('Position Error vs Time')
    ax.plot(steps, err_x, 'r-', linewidth=1.0, label='err_x')
    ax.plot(steps, err_y, 'b-', linewidth=1.0, label='err_y')
    ax.plot(steps, err_norm, 'k-', linewidth=1.5, label='||err||')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,0) XY Velocity ──
    ax = axs[1, 0]
    ax.set_title('Velocity vs Time')
    ax.plot(steps, vel[:, 0], 'r-', linewidth=1.0, label='v_x')
    ax.plot(steps, vel[:, 1], 'b-', linewidth=1.0, label='v_y')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Velocity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (1,1) Diffusion Noise Scale ──
    ax = axs[1, 1]
    ax.set_title('Noise Scale vs Time')
    # Plot per-iteration noise for each step as scatter
    for t, var_sched in enumerate(traj_log['variance_schedule']):
        if var_sched:
            ax.scatter([t] * len(var_sched), var_sched, c='cornflowerblue',
                       s=8, alpha=0.4, zorder=2)
    # Average noise per step
    ax.plot(steps, noise_scale, 'b-', linewidth=1.5, label='avg diffusion noise', zorder=3)
    # STORM adaptive scale_tril
    ax.plot(steps, storm_stril, 'r--', linewidth=1.5, label='STORM scale_tril', zorder=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Noise σ')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # ── (2,0) Best Cost ──
    ax = axs[2, 0]
    ax.set_title('Best Cost vs Time')
    ax.plot(steps, best_cost, 'k-', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Min Cost')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # ── (2,1) Acceleration (Command) ──
    ax = axs[2, 1]
    ax.set_title('Acceleration (Command) vs Time')
    ax.plot(steps, acc[:, 0], 'r-', linewidth=1.0, label='acc_x')
    ax.plot(steps, acc[:, 1], 'b-', linewidth=1.0, label='acc_y')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Acceleration')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── (3,0) 2D Trajectory ──
    ax = axs[3, 0]
    ax.set_title('2D Trajectory')
    extents = (traj_log['bounds'][0], traj_log['bounds'][1],
               traj_log['bounds'][2], traj_log['bounds'][3])
    ax.imshow(traj_log['world'], extent=extents, cmap='gray', alpha=0.4)
    # Start point
    ax.plot(position[0, 0], position[0, 1], 'rX', markersize=14, 
            markeredgewidth=2, label='start', zorder=5)
    # Goal point
    ax.plot(des[0, 0], des[0, 1], 'g*', markersize=16, 
            markeredgewidth=1, label='goal', zorder=5)
    # Trajectory colored by time
    for t in range(len(position) - 1):
        frac = t / max(len(position) - 1, 1)
        color = plt.cm.viridis(frac)
        ax.plot(position[t:t+2, 0], position[t:t+2, 1], '-', color=color,
                linewidth=2.0, zorder=4)
    # Top trajectories from first step
    if 'top_traj' in traj_log:
        for k in range(min(traj_log['top_traj'].shape[0], 5)):
            d = traj_log['top_traj'][k, :, :2]
            ax.plot(d[:, 0], d[:, 1], 'c-', alpha=0.3, linewidth=0.8)
    ax.set_xlim(extents[0], extents[1])
    ax.set_ylim(extents[2], extents[3])
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.2)

    # ── (3,1) Per-iteration cost evolution ──
    ax = axs[3, 1]
    ax.set_title('Per-Iteration Cost (each control step)')
    # Skip the first step (n_diffuse_init iterations) and only plot normal steps (n_diffuse iterations)
    iter_costs_normal = traj_log['iteration_costs'][1:]  # exclude step 0 (init)
    steps_normal = steps[1:]
    if iter_costs_normal:
        n_iters = len(iter_costs_normal[0])  # n_diffuse (e.g. 4)
        cost_matrix = np.full((len(iter_costs_normal), n_iters), np.nan)
        for t, ic in enumerate(iter_costs_normal):
            for j, c in enumerate(ic[:n_iters]):
                cost_matrix[t, j] = c
        for j in range(n_iters):
            col = cost_matrix[:, j]
            valid = ~np.isnan(col)
            if valid.any():
                ax.plot(steps_normal[valid], col[valid], linewidth=1.0, alpha=0.7, label=f'iter {j}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Min Cost')
    ax.set_yscale('log')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig_path = os.path.join(save_path, 'diffusion_reacher_diagnostics.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")

    if not headless:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DIAL-MPC Simple Reacher')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless')
    args = parser.parse_args()

    holonomic_robot(args)
