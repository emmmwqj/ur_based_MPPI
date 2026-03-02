#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
import copy
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

import time
import yaml
import argparse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from storm_kit.geom.geom_types import tensor_circle
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict
from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path
from storm_kit.mpc.rollout.simple_reacher import SimpleReacher
from storm_kit.mpc.control import MPPI
from storm_kit.mpc.utils.state_filter import JointStateFilter, RobotStateFilter
from storm_kit.mpc.utils.mpc_process_wrapper import ControlProcess
from storm_kit.mpc.task.simple_task import SimpleTask

traj_log = None

def holonomic_robot(args):
    # load
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on device: {device}")
    
    tensor_args = {'device': device, 'dtype': torch.float32}
    simple_task = SimpleTask(robot_file="simple_reacher.yml", tensor_args=tensor_args)
    

    goal_state = [0.4,0.3]
    
    simple_task.update_params(goal_state=goal_state)

    curr_state_tensor = torch.zeros((1,4), **tensor_args)
    filter_coeff = {'position':1.0, 'velocity':1.0, 'acceleration':1.0}
    current_state = {'position':np.array([0.05, 0.2]), 'velocity':np.zeros(2) + 0.0}
    
    i = 0
    exp_params = simple_task.exp_params
    controller = simple_task.controller
    sim_dt = exp_params['control_dt']
    
    
    global traj_log
    image = controller.rollout_fn.image_collision_cost.world_coll.im
    extents = np.ravel(exp_params['model']['position_bounds'])

    traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],
                'acc':[], 'world':image, 'bounds':extents}

    zero_acc = np.zeros(2)
    t_step = 0.0
    full_act = None
    curr_state = np.hstack((current_state['position'], current_state['velocity'], zero_acc, t_step))
    curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)

    update_goal = False

    filtered_state = copy.deepcopy(current_state)
    plan_length = 200

    traj_log = {'position':[], 'velocity':[], 'error':[], 'command':[], 'des':[],
                'acc':[], 'world':image, 'bounds':extents,
                'best_cost':[], 'scale_tril':[]}
    

    while(i < plan_length):
        
        current_state = {'position':current_state['position'],
                         'velocity':current_state['velocity'],
                         'acceleration': current_state['position']*0.0}
        filtered_state = current_state
        curr_state = np.hstack((filtered_state['position'], filtered_state['velocity'], filtered_state['acceleration'], t_step))
            

        curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
        
        # 计算 xy 维度的位置误差
        xy_error = np.array(goal_state[:2]) - filtered_state['position']
        
        command = simple_task.get_command(t_step, filtered_state, sim_dt, WAIT=True)
        
        if(i == 0):
            top_trajs = simple_task.top_trajs
            traj_log['top_traj'] = top_trajs.cpu().numpy()

        # 记录 best_cost 和 scale_tril (噪声尺度)
        mppi_controller = simple_task.control_process.controller
        best_cost = mppi_controller.total_costs.min().item()
        scale_tril_diag = mppi_controller.scale_tril.detach().cpu().numpy().copy()
        traj_log['best_cost'].append(best_cost)
        traj_log['scale_tril'].append(scale_tril_diag)

        current_state = command
            
        print(i, command['position'])
        traj_log['position'].append(filtered_state['position'])
        traj_log['error'].append(xy_error.copy())
        traj_log['velocity'].append(filtered_state['velocity'])
        traj_log['command'].append(command['acceleration'])
        traj_log['acc'].append(command['acceleration'])
        traj_log['des'].append(copy.deepcopy(goal_state))
        t_step += sim_dt
        i += 1
        
    matplotlib.use('tkagg')
    plot_traj(traj_log)


def plot_traj(traj_log):
    position = np.array(traj_log['position'])
    vel = np.array(traj_log['velocity'])
    err = np.array(traj_log['error'])
    acc = np.array(traj_log['acc'])
    act = np.array(traj_log['command'])
    des = np.array(traj_log['des'])
    best_cost = np.array(traj_log['best_cost'])
    scale_tril = np.array(traj_log['scale_tril'])

    steps = np.arange(position.shape[0])

    fig, axs = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle('STORM MPPI Simple Reacher', fontsize=14, fontweight='bold')

    # ── (0,0) XY Position ──
    axs[0, 0].set_title('Position')
    axs[0, 0].plot(steps, position[:, 0], 'r', label='x')
    axs[0, 0].plot(steps, position[:, 1], 'g', label='y')
    axs[0, 0].axhline(y=des[0, 0], color='r', linestyle='-.', alpha=0.5, label='x_des')
    axs[0, 0].axhline(y=des[0, 1], color='g', linestyle='-.', alpha=0.5, label='y_des')
    axs[0, 0].set_xlabel('Step')
    axs[0, 0].set_ylabel('Position')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # ── (0,1) XY Error ──
    axs[0, 1].set_title('Position Error (goal - pos)')
    axs[0, 1].plot(steps, err[:, 0], 'r', label='x_err')
    axs[0, 1].plot(steps, err[:, 1], 'g', label='y_err')
    axs[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axs[0, 1].set_xlabel('Step')
    axs[0, 1].set_ylabel('Error')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # ── (1,0) XY Velocity ──
    axs[1, 0].set_title('Velocity')
    axs[1, 0].plot(steps, vel[:, 0], 'r', label='vx')
    axs[1, 0].plot(steps, vel[:, 1], 'g', label='vy')
    axs[1, 0].set_xlabel('Step')
    axs[1, 0].set_ylabel('Velocity')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # ── (1,1) XY Acceleration ──
    axs[1, 1].set_title('Acceleration')
    axs[1, 1].plot(steps, acc[:, 0], 'r', label='ax')
    axs[1, 1].plot(steps, acc[:, 1], 'g', label='ay')
    axs[1, 1].set_xlabel('Step')
    axs[1, 1].set_ylabel('Acceleration')
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    # ── (2,0) Noise (scale_tril) ──
    axs[2, 0].set_title('Noise Scale (scale_tril)')
    if scale_tril.ndim == 1:
        # sigma_I type: scalar per step
        axs[2, 0].plot(steps, scale_tril, 'b', label='σ')
    else:
        # diag_AxA type: per-action-dim
        for d in range(scale_tril.shape[1]):
            axs[2, 0].plot(steps, scale_tril[:, d], label=f'σ_dim{d}')
    axs[2, 0].set_xlabel('Step')
    axs[2, 0].set_ylabel('scale_tril')
    axs[2, 0].set_yscale('log')
    axs[2, 0].legend()
    axs[2, 0].grid(True, alpha=0.3)

    # ── (2,1) Best Cost ──
    axs[2, 1].set_title('Best Cost')
    axs[2, 1].plot(steps, best_cost, 'b', label='best cost')
    axs[2, 1].set_xlabel('Step')
    axs[2, 1].set_ylabel('Cost')
    axs[2, 1].set_yscale('log')
    axs[2, 1].legend()
    axs[2, 1].grid(True, alpha=0.3)

    # ── (3,0) 2D Trajectory ──
    axs[3, 0].set_title('2D Trajectory')
    extents = (traj_log['bounds'][0], traj_log['bounds'][1],
               traj_log['bounds'][2], traj_log['bounds'][3])
    axs[3, 0].imshow(traj_log['world'], extent=extents, cmap='gray', alpha=0.4)
    axs[3, 0].plot(position[0, 0], position[0, 1], 'rX', markersize=15, label='start')
    axs[3, 0].plot(des[0, 0], des[0, 1], 'gX', markersize=15, label='goal')
    axs[3, 0].plot(position[:, 0], position[:, 1], 'k-.', linewidth=2.0)
    if 'top_traj' in traj_log:
        for k in range(traj_log['top_traj'].shape[0]):
            d = traj_log['top_traj'][k, :, :2]
            axs[3, 0].plot(d[:, 0], d[:, 1], alpha=0.3, linewidth=0.5)
    axs[3, 0].set_xlim(traj_log['bounds'][0], traj_log['bounds'][1])
    axs[3, 0].set_ylim(traj_log['bounds'][2], traj_log['bounds'][3])
    axs[3, 0].set_aspect('equal')
    axs[3, 0].legend()
    axs[3, 0].grid(True, alpha=0.3)

    # ── (3,1) Summary text ──
    axs[3, 1].axis('off')
    axs[3, 1].set_title('Summary')
    summary_text = (
        f"Total steps: {len(steps)}\n"
        f"Final position: ({position[-1, 0]:.4f}, {position[-1, 1]:.4f})\n"
        f"Goal: ({des[0, 0]:.4f}, {des[0, 1]:.4f})\n"
        f"Final best cost: {best_cost[-1]:.6f}\n"
        f"Min best cost: {best_cost.min():.6f}\n"
        f"Final scale_tril: {scale_tril[-1]}"
    )
    axs[3, 1].text(0.1, 0.5, summary_text, transform=axs[3, 1].transAxes,
                   fontsize=11, verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = 'simple_reacher_traj_log.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory plot saved to {save_path}")
    plt.show()
if __name__ == '__main__':
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    
    
    holonomic_robot(args)
