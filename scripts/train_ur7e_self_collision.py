import torch
import torch.nn.functional as F
import numpy as np
from storm_kit.mpc.rollout.arm_base import ArmBase
from storm_kit.util_file import get_configs_path, join_path, get_mpc_configs_path, get_weights_path
import yaml
from storm_kit.mpc.control.control_utils import generate_halton_samples
from storm_kit.geom.nn_model.robot_self_collision import RobotSelfCollisionNet
import os

# ----------------------------------------------------------------------------
# 1. 定义数据集类
# ----------------------------------------------------------------------------
class RobotDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, y_gt):
        self.x = x
        self.y = y
        self.y_gt = y_gt
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, idx):
        sample = {'x': self.x[idx,:], 'y': self.y[idx,:], 'y_gt': self.y_gt[idx,:]}
        return sample

def train_ur7e(robot_name='ur7e'):
    # ----------------------------------------------------------------------------
    # 2. 初始化配置
    # ----------------------------------------------------------------------------
    checkpoints_dir = join_path(get_weights_path(), 'robot_self')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    task_file = robot_name + '_reacher.yml'
    device = torch.device('cuda', 0)
    tensor_args = {'device':device, 'dtype':torch.float32}
    mpc_yml_file = join_path(get_mpc_configs_path(), task_file)

    print(f"Loading config from: {mpc_yml_file}")
    with open(mpc_yml_file) as file:
        exp_params = yaml.safe_load(file)

    # ----------------------------------------------------------------------------
    # 3. [关键步骤] 预先计算总采样数并写入配置
    # ----------------------------------------------------------------------------
    # 设定采样规模
    n_random_base = 50000         # 基础随机点数量
    n_random_total = n_random_base * 2 # Halton序列通常生成两倍
    n_special = 2000              # 注入的特殊姿态数量
    
    # 计算总数 (这是为了解决 RuntimeError 维度不匹配的核心修复)
    total_samples = n_random_total + n_special
    print(f"Plan: Random({n_random_total}) + Special({n_special}) = Total({total_samples})")

    # 强制覆盖参数
    if 'robot_collision_params' in exp_params['model']:
        exp_params['model']['robot_collision_params']['self_collision_weights'] = None
    if 'robot_collision_params' in exp_params:
        exp_params['robot_collision_params']['self_collision_weights'] = None
    
    if 'robot_self_collision' in exp_params['cost']:
        exp_params['cost']['robot_self_collision']['weight'] = 1.0
    else:
        exp_params['cost']['robot_self_collision'] = {'weight': 1.0}

    exp_params['robot_params'] = exp_params['model'] 
    exp_params['cost']['primitive_collision']['weight'] = 0.0
    exp_params['control_space'] = 'pos'
    exp_params['mppi']['horizon'] = 2
    
    # [FIX] 将配置中的粒子数设置为真实的 Total Samples
    exp_params['mppi']['num_particles'] = total_samples
    
    # ----------------------------------------------------------------------------
    # 4. 初始化机器人模型
    # ----------------------------------------------------------------------------
    print("Initializing Robot Model for Data Generation...")
    # 此时 ArmBase 内部会分配 size 为 total_samples 的 Tensor，不会再报错了
    rollout_fn = ArmBase(exp_params, tensor_args, world_params=None)
    dof = rollout_fn.dynamics_model.d_action

    # ----------------------------------------------------------------------------
    # 5. 数据生成 (Data Generation)
    # ----------------------------------------------------------------------------
    print(f"Robot DOF: {dof}. Generating samples...")
    
    # A. 随机 Halton 采样
    q_random = generate_halton_samples(n_random_total, dof, use_ghalton=True,
                                        device=tensor_args['device'],
                                        float_dtype=tensor_args['dtype'])
    
    up_bounds = rollout_fn.dynamics_model.state_upper_bounds[:dof]
    low_bounds = rollout_fn.dynamics_model.state_lower_bounds[:dof]
    range_b = up_bounds - low_bounds
    q_random = q_random * range_b + low_bounds
    
    # B. 特殊姿态注入 (Special Poses Injection)
    print("Injecting Special Poses...")
    
    # 1. 全零 (通常是伸直)
    q_zeros = torch.zeros((n_special // 4, dof), **tensor_args)
    
    # 2. 全零微扰
    q_near_zeros = torch.randn((n_special // 4, dof), **tensor_args) * 0.05
    
    # 3. Upright (Base=0, Shoulder=-pi/2, Elbow=0, Wrist1=-pi/2...)
    q_upright = torch.tensor([0.0, -1.57, 0.0, -1.57, 0.0, 0.0], **tensor_args).repeat(n_special // 4, 1)
    if dof != 6: q_upright = torch.zeros((n_special // 4, dof), **tensor_args)

    # 4. 随机伸直噪点
    q_noise_straight = torch.randn((n_special // 4, dof), **tensor_args) * 0.1

    q_special = torch.cat([q_zeros, q_near_zeros, q_upright, q_noise_straight], dim=0)
    q_special = torch.max(torch.min(q_special, up_bounds), low_bounds)

    # C. 合并
    q_all = torch.cat([q_random, q_special], dim=0)
    
    # 再次检查维度一致性
    if q_all.shape[0] != total_samples:
        print(f"Warning: q_all size {q_all.shape[0]} != config size {total_samples}. Adjusting slice.")
        q_all = q_all[:total_samples]
    
    # 调整形状为 (Total, Horizon, DOF)
    q_samples = q_all.unsqueeze(1).repeat(1, 2, 1) # Time=2

    start_state = torch.zeros((rollout_fn.dynamics_model.d_state), **tensor_args)

    # 正运动学解算
    print("Computing Forward Kinematics...")
    # 这里的 q_samples 第一维度是 total_samples，现在与 ArmBase 内部初始化的一致了
    state_dict = rollout_fn.dynamics_model.rollout_open_loop(start_state, q_samples)
    link_pos_seq = state_dict['link_pos_seq']
    link_rot_seq = state_dict['link_rot_seq']
    
    # 计算几何距离
    print("Computing Geometric Distances (Ground Truth)...")
    cost = rollout_fn.robot_self_collision_cost.distance
    dist = cost(link_pos_seq, link_rot_seq)

    # 提取数据
    x = q_samples[:, 0, :]
    y = dist[:, 0].view(-1, 1)

    print(f"Dataset stats -> Min Dist: {torch.min(y).item():.4f}, Max Dist: {torch.max(y).item():.4f}")

    # ----------------------------------------------------------------------------
    # 6. 数据预处理
    # ----------------------------------------------------------------------------
    n_size = x.shape[0]
    nn_model = RobotSelfCollisionNet(n_joints=dof)
    nn_model.model.to(**tensor_args)
    model = nn_model.model

    # 划分数据集
    train_ratio = 0.8
    train_idx = int(n_size * train_ratio)
    
    x_train = x[:train_idx,:]
    y_train = y[:train_idx]
    
    # 碰撞边界数据提取
    mask = y_train[:,0] > -0.05
    x_coll = x_train[mask]
    y_coll = y_train[mask]

    # 归一化参数 (使用真实的 std)
    mean_x = torch.mean(x, dim=0)
    std_x = torch.std(x, dim=0) + 1e-6
    mean_y = torch.mean(y, dim=0)
    std_y = torch.std(y, dim=0) + 1e-6

    # Apply Norm
    x_train = torch.div((x_train - mean_x), std_x)
    if len(x_coll) > 0:
        x_coll = torch.div(x_coll - mean_x, std_x).detach()
        y_coll = torch.div(y_coll - mean_y, std_y).detach()
    
    y_train_true = y_train.clone()
    y_train = torch.div((y_train - mean_y), std_y)

    # Validation Set
    x_val = torch.div(x[train_idx:] - mean_x, std_x)
    y_val = torch.div(y[train_idx:] - mean_y, std_y)
    y_val_real = y[train_idx:]

    train_dataset = RobotDataset(x_train.detach(), y_train.detach(), y_train_true.detach())
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    if len(x_coll) > 0:
        coll_dataset = RobotDataset(x_coll.detach(), y_coll.detach(), y_coll.detach())
        collloader = torch.utils.data.DataLoader(coll_dataset, batch_size=64, shuffle=True)
    else:
        collloader = None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # ----------------------------------------------------------------------------
    # 7. 训练循环
    # ----------------------------------------------------------------------------
    print("Start Training...")
    epochs = 150
    min_loss = 1000.0
    
    for e in range(epochs):
        model.train()
        loss_list = []
        
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            y_batch = data['y'].to(device)
            x_batch = data['x'].to(device)

            y_pred = model(x_batch)
            
            # Base MSE
            loss = F.mse_loss(y_pred, y_batch, reduction='mean')

            # Boundary Enhancement & Safety Constraint
            if collloader:
                try:
                    coll_data = next(iter(collloader))
                except StopIteration:
                    coll_iter = iter(collloader)
                    coll_data = next(coll_iter)
                
                x_c = coll_data['x'].to(device)
                y_c = coll_data['y'].to(device)
                y_c_pred = model(x_c)
                
                loss += 1.5 * F.mse_loss(y_c_pred, y_c, reduction='mean')
                
                # Safety Penalty: 惩罚预测值比真实值更危险(更小)的情况
                diff = y_c - y_c_pred
                penalty = torch.relu(diff - 0.05) 
                loss += 0.5 * torch.mean(penalty**2)

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred = model(x_val)
            val_loss = F.mse_loss(y_val_pred, y_val, reduction='mean')
            
            y_val_pred_real = torch.mul(y_val_pred, std_y) + mean_y
            l1_error = F.l1_loss(y_val_pred_real, y_val_real, reduction='mean')
        
        avg_train_loss = np.mean(loss_list)

        if val_loss < min_loss and e > 10:
            torch.save(
                {
                    'epoch': e,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'norm': {
                        'x': {'mean': mean_x, 'std': std_x},
                        'y': {'mean': mean_y, 'std': std_y}
                    }
                },
                join_path(checkpoints_dir, robot_name + '_self_sdf.pt')
            )
            min_loss = val_loss
        
        if e % 10 == 0:
            print(f"Epoch {e:03d}: Train Loss {avg_train_loss:.5f}, Val Loss {val_loss:.5f}, L1 Error {l1_error:.5f} m")

    # ----------------------------------------------------------------------------
    # 8. 验证
    # ----------------------------------------------------------------------------
    print("\nTraining Finished. Verifying Special Pose (Straight)...")
    test_q = torch.zeros((1, dof), **tensor_args) # 全零测试
    test_q_norm = (test_q - mean_x) / std_x
    
    model.eval()
    with torch.no_grad():
        pred_norm = model(test_q_norm)
        pred_real = pred_norm * std_y + mean_y
    
    print(f"Prediction for Straight Pose (All Zeros): {pred_real.item():.4f} meters")
    save_path = join_path(checkpoints_dir, robot_name + '_self_sdf.pt')
    print(f"Model saved to: {save_path}")

if __name__=='__main__':
    train_ur7e('ur7e')